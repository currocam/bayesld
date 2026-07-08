"""
Shared machinery for inference engine.
"""

import abc
import copy
import pathlib
import time
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from . import _surrogate as sg

if TYPE_CHECKING:
    import xarray as xr

_THREADS_OPTS = {"cpp_options": {"STAN_THREADS": "true"}}

#: Directory holding the engine ``.stan`` files and the ``shared/`` fragments
#: they ``#include``. Added to stanc's include-paths so ``#include shared/*.stan``
#: resolves for both built-in and custom engines.
_STAN_DIR = pathlib.Path(__file__).resolve().parent / "stan"

_DEFAULT_N_QUAD = 8
_DEFAULT_HYPERPRIOR = {
    "hsgp_c": 1.5,
    "hsgp_m_ld": 6,
    "gp_alpha_std": 0.005,
    "lkj_eta": 2.0,
    "log_sigma_y_scale": 1.0,
}


class _BaseEngine(abc.ABC):
    """Base class for data-driven, GP-bias-corrected inference engines."""

    #: Path to the static Stan program; set by each subclass.
    _STAN_FILE: pathlib.Path
    #: Posterior dims that depend on the parameterization (e.g. epoch/boundary).
    _PARAM_DIMS: dict = {}

    #: Posterior dims shared by every engine (the surrogate LD/pi outputs).
    _SHARED_DIMS = {
        "approx_expected_ld": ["bin"],
        "corrected_expected_ld": ["bin"],
        "gp_bias_ld": ["bin"],
        "expected_pi": [],
    }

    def __init__(
        self,
        *,
        n_quad: int = _DEFAULT_N_QUAD,
        **hyperprior: float,
    ) -> None:
        self._gl_nodes, self._gl_weights = np.polynomial.legendre.leggauss(int(n_quad))

        self._hyperprior = dict(_DEFAULT_HYPERPRIOR)
        self._hyperprior.update(hyperprior)

        self._left_bins = None
        self._right_bins = None
        self._data = None
        self._prior = None
        self._bias_points: list[dict] = []
        self._sigma_points: list[dict] = []

        # Freeze and compile the Stan program (cmdstanpy).
        self._stan_code = self._STAN_FILE.read_text()
        import cmdstanpy

        # Resolve ``#include shared/*.stan`` against both the shared fragment
        # directory and the engine file's own directory (so a custom Stan file
        # living elsewhere can still pull in its neighbours).
        include_paths = list(
            dict.fromkeys([str(_STAN_DIR), str(self._STAN_FILE.resolve().parent)])
        )
        self._model = cmdstanpy.CmdStanModel(
            stan_file=str(self._STAN_FILE),
            stanc_options={"include-paths": include_paths},
            **_THREADS_OPTS,
        )

    # ─── Immutable builder plumbing ──────────────────────────────────────────

    def _clone(self) -> "_BaseEngine":
        """Shallow copy sharing the compiled Stan model and read-only arrays.

        Container objects (_bias_points, _sigma_points, _prior, _data,
        _hyperprior) get new wrappers so mutations to one copy don't affect the
        other; the numpy arrays inside them (and _left_bins/_right_bins) are
        shared but treated as immutable. Lets builder steps return an updated
        copy without recompiling Stan or mutating the receiver.
        """
        new = copy.copy(self)
        new._bias_points = list(self._bias_points)
        new._sigma_points = list(self._sigma_points)
        new._prior = dict(self._prior) if self._prior is not None else None
        new._data = dict(self._data) if self._data is not None else None
        new._hyperprior = dict(self._hyperprior)
        return new

    def with_data(
        self,
        mean_diversity: npt.ArrayLike,
        mean_ld: npt.ArrayLike,
        left_bins: npt.ArrayLike,
        right_bins: npt.ArrayLike,
        recombination_rate: float,
        mutation_rate: float,
        num_samples: int,
        sequence_length: float,
        ploidy: int = 2,
    ) -> "_BaseEngine":
        """
        Set the data for sampling.

        Arguments:
        - mean_diversity: mean diversity (sample heterozygosity) across windows
        - mean_ld: mean LD ($X_iX_jY_iY_j$) across windows
        - left_bins: left bins of distances among pairs of loci (Morgans)
        - right_bins: right bins of distances among pairs of loci (Morgans)
        - recombination_rate: recombination rate (per base pair per generation)
        - mutation_rate: mutation rate (per base pair per generation)
        - num_samples: number of samples
        - sequence_length: sequence length (in base pairs)
        - ploidy: ploidy of the samples

        Returns:
        - a new model with the data attached (the receiver is left unchanged). If
          no prior has been set, an empirical-Bayes default (``_default_prior``) is
          installed.
        """
        new = self._clone()
        new._left_bins = np.asarray(left_bins, dtype=float)
        new._right_bins = np.asarray(right_bins, dtype=float)
        new._data = {
            "mean_diversity": np.asarray(mean_diversity, dtype=float),
            "mean_ld": np.asarray(mean_ld, dtype=float),
            "recombination_rate": float(recombination_rate),
            "mutation_rate": float(mutation_rate),
            "num_samples": int(num_samples),
            "sequence_length": float(sequence_length),
            "ploidy": int(ploidy),
        }
        if new._prior is None:
            new._prior = new._default_prior()
        return new

    def with_hyperprior(self, **kwargs: float) -> "_BaseEngine":
        """
        Set the hyperpriors for the model.
        TODO: add documentation for the hyperpriors.

        Returns:
        - a new model with the hyperpriors updated (the receiver is left unchanged)
        """
        unknown = set(kwargs) - set(_DEFAULT_HYPERPRIOR)
        if unknown:
            raise ValueError(f"Unknown hyperprior(s): {sorted(unknown)}")
        new = self._clone()
        new._hyperprior.update(kwargs)
        return new

    def add_synthetic_data(self, points: list[tuple[dict, dict]]) -> "_BaseEngine":
        """Append surrogate bias/sigma points, returning a new model.

        Bias points accumulate across all rounds; sigma points keep only those
        supplied here (the surrogate's locality assumption). The receiver is left
        unchanged.
        """
        for bias_pt, sigma_pt in points:
            sg.validate_bias_point(bias_pt)
            sg.validate_sigma_point(sigma_pt)
        new = self._clone()
        new._bias_points.extend(bias_pt for bias_pt, _ in points)
        new._sigma_points = [sigma_pt for _, sigma_pt in points]
        return new

    # ─── Read-only accessors ─────────────────────────────────────────────────

    @property
    def stan_code(self) -> str:
        return self._stan_code

    @property
    def bias_points(self) -> list[dict]:
        return list(self._bias_points)

    @property
    def sigma_points(self) -> list[dict]:
        return list(self._sigma_points)

    # ─── Stan data ───────────────────────────────────────────────────────────

    def _stan_data(self) -> dict:
        if self._data is None:
            raise RuntimeError(
                "No data attached. Call `.with_data(...)` before building Stan data "
                "or sampling."
            )
        assert self._prior is not None  # with_data always sets it

        div = self._data["mean_diversity"]
        ld = self._data["mean_ld"]
        log_ld_mu, log_ld_sig = sg.compute_standardization(ld)
        log_sigma_y_loc, log_sigma_y_scale = sg.sigma_prior_hyperparams(
            div, ld, self._hyperprior["log_sigma_y_scale"]
        )

        data = {
            "n_bins": int(len(self._left_bins)),
            "num_windows": int(len(div)),
            "left_bins": self._left_bins,
            "right_bins": self._right_bins,
            "mutation_rate": self._data["mutation_rate"],
            "sample_size": self._data["num_samples"],
            "ploidy": self._data["ploidy"],
            "pi_array": div,
            "ld_mat": ld,
            "n_quad": int(len(self._gl_nodes)),
            "gl_nodes": self._gl_nodes,
            "gl_weights": self._gl_weights,
            "hsgp_c": self._hyperprior["hsgp_c"],
            "hsgp_m_ld": self._hyperprior["hsgp_m_ld"],
            "gp_alpha_std": self._hyperprior["gp_alpha_std"],
            "log_ld_mu": log_ld_mu,
            "log_ld_sig": log_ld_sig,
            "log_sigma_y_loc": log_sigma_y_loc,
            "log_sigma_y_scale": log_sigma_y_scale,
            "lkj_eta": self._hyperprior["lkj_eta"],
        }
        data.update(self._prior_stan_data())
        data.update(
            sg.surrogate_payload(
                self._bias_points,
                self._sigma_points,
                self._left_bins,
                self._right_bins,
            )
        )
        return data

    # ─── Posterior labelling ─────────────────────────────────────────────────

    def _dims(self) -> dict:
        return {**self._SHARED_DIMS, **self._PARAM_DIMS}

    def _coords(self) -> dict:
        """Coordinate values for the named dims.

        ``bin`` is labelled by genetic-distance midpoint (Morgans); ``window`` is
        a plain integer range; parameterization-specific coords come from the
        subclass via ``_param_coords``.
        """
        mid = np.round((self._left_bins + self._right_bins) / 2.0, 5)
        coords = {
            "bin": mid,
            "window": np.arange(len(self._data["mean_diversity"])),
        }
        coords.update(self._param_coords())
        return coords

    # ─── Sampling ────────────────────────────────────────────────────────────

    def sample(
        self,
        draws: int = 1000,
        tune: int = 1000,
        chains: int = 1,
        *,
        seed: int | None = None,
        num_workers: int = 1,
        verbose: bool = False,
        **kwargs,
    ):
        """Sample from the posterior using NUTS.

        Arguments:
        - draws: number of draws from the posterior
        - tune: number of warmup draws from the posterior
        - chains: number of chains
        - seed: random seed
        - num_workers: number of workers
        - verbose: whether to print progress
        - kwargs: additional arguments for cmdstanpy.CmdStanModel.sample

        Returns:
        - xr.DataTree: the posterior samples
        """
        if self._data is None:
            raise RuntimeError("Call `.with_data(...)` before sampling.")
        import arviz
        import xarray as xr

        if verbose:
            print(
                f"[bayesld] sampling: {chains} chains x ({tune} warmup + {draws} "
                f"draws), {num_workers} worker(s)/chain"
            )
        start = time.perf_counter()
        kwargs.setdefault("show_console", bool(verbose))
        kwargs.setdefault("threads_per_chain", int(num_workers))
        fit = self._model.sample(
            data=self._stan_data(),
            chains=chains,
            iter_sampling=draws,
            iter_warmup=tune,
            seed=seed,
            **kwargs,
        )
        if verbose:
            print(f"[bayesld] sampling finished in {time.perf_counter() - start:.1f}s")
        idata = arviz.from_cmdstanpy(fit, coords=self._coords(), dims=self._dims())
        ll = self.pointwise_log_lik(idata)
        idata["log_likelihood"] = xr.Dataset({"log_lik": ll})
        return idata

    def pointwise_log_lik(self, idata: "xr.DataTree") -> "xr.DataArray":
        """Per-window log-likelihood from a posterior InferenceData.
        TODO: check this

        Arguments:
        - idata: the posterior xr.DataTree to compute the log-likelihood from

        Returns:
        - xr.DataArray: the log-likelihood with dimensions (chain, draw, window)
        Example:
        ```
            import xarray as xr
            ll = model.pointwise_log_lik(idata)
            idata["log_likelihood"] = xr.Dataset({"log_lik":ll})
            az.loo(idata)
        """
        if self._data is None:
            raise RuntimeError("No data attached. Call `.with_data(...)` first.")
        import xarray as xr

        post = idata.posterior

        # mu_y: (chain, draw, D)
        mu_y = np.concatenate(
            [
                post["expected_pi"].values[..., np.newaxis],
                post["corrected_expected_ld"].values,
            ],
            axis=-1,
        )
        # L_Sigma = diag(sigma_y) @ L_Omega; solve via two steps to avoid
        # materialising L_Sigma and to exploit the known triangular structure.
        sigma_y = np.exp(post["log_sigma_y"].values)  # (chain, draw, D)
        L_Omega = post["L_Omega"].values  # (chain, draw, D, D)

        # y_obs: (W, D)
        y_obs = np.column_stack([self._data["mean_diversity"], self._data["mean_ld"]])
        D = y_obs.shape[1]

        # residuals / sigma_y  →  (chain, draw, D, W)
        residuals = y_obs.T - mu_y[..., np.newaxis]
        z = residuals / sigma_y[..., np.newaxis]

        # Forward solve  L_Omega v = z  →  v = L_Omega^{-1} z
        V = np.linalg.solve(L_Omega, z)  # (chain, draw, D, W)

        quad = (V**2).sum(axis=-2)  # (chain, draw, W)
        log_det = np.log(sigma_y).sum(axis=-1) + np.log(
            np.diagonal(L_Omega, axis1=-2, axis2=-1)
        ).sum(axis=-1)  # (chain, draw)

        values = -0.5 * D * np.log(2.0 * np.pi) - log_det[..., np.newaxis] - 0.5 * quad

        coords = self._coords()
        return xr.DataArray(
            values,
            dims=["chain", "draw", "window"],
            coords={
                "chain": post.coords["chain"].values,
                "draw": post.coords["draw"].values,
                "window": coords["window"],
            },
        )

    def _pathfinder(
        self, draws: int, seed: int, num_workers: int, verbose: bool = False
    ):
        return self._model.pathfinder(
            data=self._stan_data(),
            draws=int(draws),
            seed=int(seed),
            inits=0.5,
            num_threads=int(num_workers),
            show_console=bool(verbose),
            psis_resample=False,
        )

    # ─── Active learning ─────────────────────────────────────────────────────

    def active_learning_round(
        self,
        num_points: int,
        rtol: float,
        min_replicates: int,
        *,
        seed: int | None = None,
        mc_model=None,
        draws: int = 1000,
        num_workers: int = 1,
        verbose: bool = False,
    ) -> "_BaseEngine":
        """
        Run a single active-learning round.
        It consists of (1) proposing points using Pathfinder, (2) evaluating the
        deterministic-vs-MC LD bias at each point using sim_sufficient_stats, and
        (3) accumulating the resulting bias/sigma surrogate points.

        Arguments:
        - num_points: number of points to propose
        - rtol: relative tolerance for the MC replicates
        - min_replicates: minimum number of MC replicates
        - seed: random seed
        - mc_model: the MC model to use
        - draws: number of draws from Pathfinder
        - num_workers: number of workers
        - verbose: whether to print progress
        Returns:
        - a new model with the round's surrogate points accumulated (the receiver
          is left unchanged)
        """
        if self._data is None:
            raise RuntimeError("Call `.with_data(...)` before active_learning_round.")

        rng = np.random.default_rng(seed)

        pf_draws = max(int(draws), int(num_points))
        if verbose:
            print(
                f"[bayesld] active-learning round: Pathfinder ({pf_draws} draws, "
                f"{num_workers} worker(s))"
            )
        start = time.perf_counter()
        pf = self._pathfinder(
            draws=pf_draws,
            seed=int(rng.integers(2**31)),
            num_workers=num_workers,
            verbose=verbose,
        )
        proposals = self._extract_params(pf)
        if verbose:
            print(
                f"[bayesld] Pathfinder done in {time.perf_counter() - start:.1f}s; "
                f"proposing {int(num_points)} point(s)"
            )

        idx = rng.choice(len(proposals), size=int(num_points), replace=False)

        points = []
        for rank, i in enumerate(idx, start=1):
            params = proposals[i]
            det_pi, det_ld = self._expected(params)
            mc_pi, mc_ld = self._sim_mc(
                params, rtol, min_replicates, rng, mc_model, num_workers
            )
            if verbose:
                print(
                    f"[bayesld]   point {rank}/{int(num_points)}: "
                    f"{self._describe_params(params)} -> {len(mc_pi)} MC replicate(s)"
                )
            raw = {
                "det_pi": float(np.asarray(det_pi)),
                "det_ld": np.asarray(det_ld),
                "mc_pi_reps": mc_pi,
                "mc_ld_reps": mc_ld,
            }
            points.append(sg.make_points(raw))

        new = self.add_synthetic_data(points)
        if verbose:
            print(
                f"[bayesld] round complete in {time.perf_counter() - start:.1f}s: "
                f"{len(points)} bias point(s) accumulated "
                f"({len(new._bias_points)} total)"
            )
        return new

    def _sim_mc(
        self,
        params: dict,
        rtol: float,
        min_replicates: int,
        rng: np.random.Generator,
        mc_model,
        num_workers: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """MC sufficient stats for a demography, honouring rtol and min_replicates."""
        from . import sim_sufficient_stats

        demography = self._build_demography(params)
        common = dict(
            demography=demography,
            left_bins=self._left_bins,
            right_bins=self._right_bins,
            mutation_rate=self._data["mutation_rate"],
            recombination_rate=self._data["recombination_rate"],
            sequence_length=self._data["sequence_length"],
            ploidy=self._data["ploidy"],
            num_workers=int(num_workers),
            model=mc_model,
        )
        pi, ld = sim_sufficient_stats(
            self._data["num_samples"],
            random_seed=int(rng.integers(2**31)),
            rtol=rtol,
            **common,
        )
        # Top up to guarantee at least `min_replicates` replicates.
        if min_replicates is not None and len(pi) < int(min_replicates):
            extra = int(min_replicates) - len(pi)
            pi2, ld2 = sim_sufficient_stats(
                self._data["num_samples"],
                random_seed=int(rng.integers(2**31)),
                num_replicates=extra,
                **common,
            )
            pi = np.concatenate([pi, pi2])
            ld = np.concatenate([ld, ld2])
        return np.asarray(pi), np.asarray(ld)

    # ─── Notebook display ────────────────────────────────────────────────────

    @property
    def _title(self) -> str:
        """Heading shown in the repr/HTML summary; subclasses may override."""
        return type(self).__name__

    def _param_rows(self) -> list[tuple[str, str]]:
        """Parameterization-specific (label, value) rows for the summary."""
        return []

    def _describe_params(self, params: dict) -> str:
        """One-line description of a proposal, used in verbose active learning."""
        return ", ".join(f"{k}={np.asarray(v)}" for k, v in params.items())

    def _empirical_rows(self) -> list[tuple[str, str]]:
        """(label, value) rows for the attached empirical data; empty if none."""
        if self._data is None:
            return []
        d = self._data
        n_bins = None if self._left_bins is None else len(self._left_bins)
        return [
            ("Windows", str(len(d["mean_diversity"]))),
            ("LD bins", "unset" if n_bins is None else str(n_bins)),
            ("Samples", f"{d['num_samples']} (ploidy {d['ploidy']})"),
            ("Sequence length", f"{d['sequence_length']:g}"),
            ("Mutation rate", f"{d['mutation_rate']:g}"),
            ("Recombination rate", f"{d['recombination_rate']:g}"),
        ]

    def _prior_rows(self) -> list[tuple[str, str]]:
        """(label, value) rows describing the current prior; subclasses may override."""
        if self._prior is None:
            return [("Status", "not set — call .with_prior(...) or .with_data(...)")]
        return [(k, str(np.asarray(v))) for k, v in self._prior.items()]

    def _synthetic_rows(self) -> list[tuple[str, str]]:
        """(label, value) rows for the accumulated synthetic bias/covariance data.

        The bias dataset uses only synthetic points; the covariance dataset uses
        both the empirical windows and the synthetic sigma sets.
        """
        n_bias = len(self._bias_points)
        n_sigma = len(self._sigma_points)
        if self._data is None:
            cov_str = f"{n_sigma} (sigma sets)"
        else:
            cov_str = f"{len(self._data['mean_diversity'])}+{n_sigma}"
        return [("Bias dataset", str(n_bias)), ("Covariance dataset", cov_str)]

    def _summary_rows(self) -> list[tuple[str, str]]:
        """(label, value) pairs describing the current model state, in display order."""
        rows = list(self._param_rows())
        rows.append(("Quadrature nodes", str(len(self._gl_nodes))))
        if self._data is None:
            rows.append(("Data", "not attached — call .with_data(...)"))
        else:
            rows += self._empirical_rows()
        rows += self._synthetic_rows()
        return rows

    def __repr__(self) -> str:
        inner = ", ".join(f"{k}={v}" for k, v in self._summary_rows())
        return f"{type(self).__name__}({inner})"

    def _repr_html_(self) -> str:
        # Structured HTML summary for jupyter notebook display.
        if self._data is None:
            data_rows = [("Status", "Not attached — call .with_data(...)")]
        else:
            data_rows = self._empirical_rows()
        prior_rows = self._prior_rows()
        synthetic_rows = self._synthetic_rows()

        # Hyperparameters (parameterization rows + quadrature + hyperpriors).
        hyper_rows = list(self._param_rows())
        hyper_rows.append(("Quadrature nodes", str(len(self._gl_nodes))))
        for name, value in self._hyperprior.items():
            hyper_rows.append((name, str(value)))

        def _section(
            title: str, rows: list[tuple[str, str]], *, open_by_default: bool
        ) -> str:
            if not rows:
                return ""
            body = "".join(
                "<tr>"
                f'<td style="padding:2px 8px;text-align:left;">{k}</td>'
                f'<td style="padding:2px 8px;text-align:left;">{v}</td>'
                "</tr>"
                for k, v in rows
            )
            open_attr = " open" if open_by_default else ""
            return (
                "<tr><td colspan='2' style='padding:4px 0;'>"
                f"<details{open_attr}>"
                f"<summary style='cursor:pointer;font-weight:600;'>{title}</summary>"
                "<table style='border-collapse:collapse;margin-top:4px;'>"
                f"{body}"
                "</table>"
                "</details>"
                "</td></tr>"
            )

        cells = (
            _section(
                "Prior",
                prior_rows,
                open_by_default=self._prior is None,
            )
            + _section(
                "Empirical data",
                data_rows,
                open_by_default=self._data is None,
            )
            + _section(
                "Augmented dataset",
                synthetic_rows,
                open_by_default=False,
            )
            + _section(
                "Hyperparameters",
                hyper_rows,
                open_by_default=False,
            )
        )
        return (
            '<div><table style="border-collapse:collapse;">'
            "<thead><tr>"
            '<th style="padding:2px 8px;text-align:left;" colspan="2">'
            f"{self._title}</th></tr></thead>"
            f"<tbody>{cells}</tbody></table></div>"
        )

    # ─── Abstract seams (per parameterization) ───────────────────────────────

    @abc.abstractmethod
    def with_prior(self, *args, **kwargs) -> "_BaseEngine":
        """Set the priors, returning a new model (receiver left unchanged)."""

    @abc.abstractmethod
    def _default_prior(self) -> dict:
        """Empirical-Bayes default prior, computed from the attached data."""

    @abc.abstractmethod
    def _prior_stan_data(self) -> dict:
        """Parameterization + prior entries for the Stan data dict."""

    @abc.abstractmethod
    def sample_prior(self, *args, **kwargs):
        """Draw demographic parameters from the prior."""

    @abc.abstractmethod
    def _build_demography(self, params: dict):
        """Build an ``msprime.Demography`` from a proposal params dict."""

    @abc.abstractmethod
    def _expected(self, params: dict) -> tuple[float, np.ndarray]:
        """Deterministic (expected_pi, expected_ld) for a proposal."""

    @abc.abstractmethod
    def _extract_params(self, fit) -> list[dict]:
        """Turn a Pathfinder fit into a list of per-draw proposal params dicts."""

    @abc.abstractmethod
    def _param_coords(self) -> dict:
        """Coordinate values for the parameterization-specific dims."""
