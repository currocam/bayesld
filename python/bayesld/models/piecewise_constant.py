"""
Piecewise-constant-Ne demographic inference models.

Two classes share a single Stan template:

``TwoEpochDemography``
    Two-epoch specialization with default priors.

``PiecewiseConstantDemography``
    Generic N-epoch model with **no defaults** — the user must supply
    ``n_epochs``, ``parameters``, ``transformed_parameters``, and ``prior``.
"""

import pathlib
import tempfile
from typing import Optional

import numpy as np

from . import _surrogate as sg

_STAN_DIR = pathlib.Path(__file__).resolve().parent.parent / "stan"
_THREADS_OPTS = {"cpp_options": {"STAN_THREADS": "true"}}

_DEFAULT_N_QUAD = 16


def _default_prior(diversity: np.ndarray, mutation_rate: float) -> str:
    ne_hat = float(np.mean(diversity)) / (4.0 * mutation_rate)
    log_ne_mu = float(np.log(ne_hat))
    return (
        f"    log_Ne_c ~ normal({log_ne_mu:.4f}, 1.5);\n"
        f"    log_Ne_a ~ normal({log_ne_mu:.4f}, 1.5);\n"
        f"    log_t0   ~ normal({np.log(50.0):.4f}, 1.5);"
    )


_DEFAULT_PARAMETERS = """\
    real<offset=log_ne_offset> log_Ne_c;
    real<offset=log_ne_offset> log_Ne_a;
    real log_t0;"""

_DEFAULT_TRANSFORMED_PARAMETERS = """\
    real<lower=0> Ne_c = exp(log_Ne_c);
    real<lower=0> Ne_a = exp(log_Ne_a);
    real<lower=0> t0   = exp(log_t0);
    vector[2] Ne_values = [Ne_c, Ne_a]';
    vector[1] t_boundaries = [t0]';"""


def _generate_stan(
    prior: str,
    parameters: str = _DEFAULT_PARAMETERS,
    transformed_parameters: str = _DEFAULT_TRANSFORMED_PARAMETERS,
) -> str:
    """Assemble the complete Stan program for a piecewise-constant model.

    The injected ``transformed_parameters`` must define
    ``vector[n_epochs] Ne_values`` and ``vector[n_epochs - 1] t_boundaries``.
    """
    gp_fn = (_STAN_DIR / "functions" / "gpbasisfun_functions.stan").read_text()
    shared_fn = (_STAN_DIR / "functions" / "shared.stan").read_text()
    model_fn = (_STAN_DIR / "functions" / "piecewise_constant.stan").read_text()

    return f"""\
functions {{
// ---- gpbasisfun_functions.stan ----
{gp_fn}
// ---- shared.stan ----
{shared_fn}
// ---- piecewise_constant.stan ----
{model_fn}
}}

data {{
    int<lower=1> n_bins;
    int<lower=2> num_windows;
    vector[n_bins] left_bins;
    vector[n_bins] right_bins;
    real<lower=0> mutation_rate;
    int<lower=1> sample_size;
    vector[num_windows] pi_array;
    matrix[num_windows, n_bins] ld_mat;
    int<lower=2> n_epochs;
    int<lower=1> n_quad;
    vector[n_quad] gl_nodes;
    vector[n_quad] gl_weights;

{sg.SURROGATE_DATA}}}

transformed data {{
{sg.JOINT_TRANSFORMED_DATA}}}

parameters {{
{parameters}
{sg.SURROGATE_PARAMETERS}}}

transformed parameters {{
{transformed_parameters}
{sg.JOINT_TP_PREFIX}
    real expected_pi = mu_div_piecewise_constant(n_epochs, Ne_values, t_boundaries, mutation_rate);
    vector[n_bins] approx_expected_ld = correct_ld_finite_sample(
        mu_ld_piecewise_constant(n_epochs, Ne_values, t_boundaries,
                                  left_bins, right_bins, n_quad, gl_nodes, gl_weights),
        sample_size
    );
{sg.JOINT_TP_SUFFIX}}}

model {{
{sg.SURROGATE_MODEL}
{prior}

    y_obs ~ multi_normal_cholesky(mu_y, L_Sigma);
}}

generated quantities {{
{sg.JOINT_GENERATED_QUANTITIES}}}
"""


class _PiecewiseConstantBase:
    """Shared implementation for TwoEpochDemography and PiecewiseConstantDemography."""

    def _init_common(
        self,
        diversity: np.ndarray,
        ld: np.ndarray,
        mutation_rate: float,
        recombination_rate: float,
        num_samples: int,
        left_bins: np.ndarray,
        right_bins: np.ndarray,
        n_epochs: int,
        sequence_length: Optional[float],
        ploidy: int,
        num_workers: int,
        hsgp_c: float,
        hsgp_m_u: int,
        hsgp_m_ld: int,
        n_quad: int,
        gp_alpha_std: float,
        lkj_eta: float,
        log_sigma_y_scale: float,
        prior: str,
        parameters: str,
        transformed_parameters: str,
    ):
        self._diversity = np.asarray(diversity, dtype=float)
        self._ld = np.asarray(ld, dtype=float)
        self._mutation_rate = float(mutation_rate)
        self._recombination_rate = float(recombination_rate)
        self._sequence_length = (
            float(sequence_length) if sequence_length is not None else None
        )
        self._ploidy = int(ploidy)
        self._num_samples = int(num_samples)
        self._left_bins = np.asarray(left_bins, dtype=float)
        self._right_bins = np.asarray(right_bins, dtype=float)
        self._n_epochs = int(n_epochs)
        self._num_workers = int(num_workers)
        self._hsgp_c = float(hsgp_c)
        self._hsgp_m_u = int(hsgp_m_u)
        self._hsgp_m_ld = int(hsgp_m_ld)
        self._gp_alpha_std = float(gp_alpha_std)
        self._lkj_eta = float(lkj_eta)
        self._log_sigma_y_scale = float(log_sigma_y_scale)

        self._gl_nodes, self._gl_weights = np.polynomial.legendre.leggauss(n_quad)

        self._synthetic_points: list[dict] = []

        self._prior = prior
        self._parameters = parameters
        self._transformed_parameters = transformed_parameters

        self._tmpdir = pathlib.Path(tempfile.mkdtemp())
        self._stan_code = _generate_stan(
            self._prior, self._parameters, self._transformed_parameters
        )
        self._model = self._compile(self._stan_code)

    # ─── Compilation ──────────────────────────────────────────────────────────

    def _compile(self, code: str):
        import cmdstanpy

        stan_file = self._tmpdir / "piecewise_constant_ne.stan"
        stan_file.write_text(code)
        return cmdstanpy.CmdStanModel(stan_file=str(stan_file), **_THREADS_OPTS)

    # ─── Primitives ───────────────────────────────────────────────────────────

    def get_stan_code(self) -> str:
        return self._stan_code

    @property
    def model(self):
        return self._model

    def stan_data(self) -> dict:
        log_ld_mu, log_ld_sig = sg.compute_standardization(self._ld)
        log_sigma_y_loc = sg.compute_log_sigma_y_loc(self._diversity, self._ld)
        data = {
            "n_bins": int(len(self._left_bins)),
            "num_windows": int(len(self._diversity)),
            "left_bins": self._left_bins,
            "right_bins": self._right_bins,
            "mutation_rate": self._mutation_rate,
            "sample_size": self._num_samples,
            "pi_array": self._diversity,
            "ld_mat": self._ld,
            "n_epochs": self._n_epochs,
            "n_quad": len(self._gl_nodes),
            "gl_nodes": self._gl_nodes,
            "gl_weights": self._gl_weights,
            "hsgp_c": self._hsgp_c,
            "hsgp_m_u": self._hsgp_m_u,
            "hsgp_m_ld": self._hsgp_m_ld,
            "gp_alpha_std": self._gp_alpha_std,
            "log_ld_mu": log_ld_mu,
            "log_ld_sig": log_ld_sig,
            "lkj_eta": self._lkj_eta,
            "log_sigma_y_loc": log_sigma_y_loc,
            "log_sigma_y_scale": self._log_sigma_y_scale
            * np.ones(len(self._left_bins) + 1),
        }
        data.update(
            sg.surrogate_payload(
                self._synthetic_points, self._left_bins, self._right_bins
            )
        )
        return data

    # ─── Data and prior mutation ──────────────────────────────────────────────

    def update_data(
        self,
        diversity: Optional[np.ndarray] = None,
        ld: Optional[np.ndarray] = None,
        mutation_rate: Optional[float] = None,
        recombination_rate: Optional[float] = None,
        num_samples: Optional[int] = None,
        sequence_length: Optional[float] = None,
    ) -> None:
        if diversity is not None:
            self._diversity = np.asarray(diversity, dtype=float)
        if ld is not None:
            self._ld = np.asarray(ld, dtype=float)
        if mutation_rate is not None:
            self._mutation_rate = float(mutation_rate)
        if recombination_rate is not None:
            self._recombination_rate = float(recombination_rate)
        if num_samples is not None:
            self._num_samples = int(num_samples)
        if sequence_length is not None:
            self._sequence_length = float(sequence_length)

    def update_prior(
        self,
        prior: Optional[str] = None,
        parameters: Optional[str] = None,
        transformed_parameters: Optional[str] = None,
        gp_alpha_std: Optional[float] = None,
        lkj_eta: Optional[float] = None,
    ) -> None:
        needs_recompile = False
        if prior is not None and prior != self._prior:
            self._prior = prior
            needs_recompile = True
        if parameters is not None and parameters != self._parameters:
            self._parameters = parameters
            needs_recompile = True
        if (
            transformed_parameters is not None
            and transformed_parameters != self._transformed_parameters
        ):
            self._transformed_parameters = transformed_parameters
            needs_recompile = True
        if gp_alpha_std is not None:
            self._gp_alpha_std = float(gp_alpha_std)
        if lkj_eta is not None:
            self._lkj_eta = float(lkj_eta)

        if needs_recompile:
            self._stan_code = _generate_stan(
                self._prior, self._parameters, self._transformed_parameters
            )
            self._model = self._compile(self._stan_code)

    # ─── Synthetic bias data ──────────────────────────────────────────────────

    @property
    def synthetic_points(self) -> list[dict]:
        return list(self._synthetic_points)

    def add_synthetic_points(self, points: list[dict]) -> None:
        for p in points:
            sg.validate_synthetic_point(p)
        self._synthetic_points.extend(points)

    # ─── MC evaluation ────────────────────────────────────────────────────────

    def _mc_eval(
        self,
        ne_values: np.ndarray,
        t_boundaries: np.ndarray,
        mc_seed: int,
        rtol: float = 0.01,
        num_replicates: Optional[int] = None,
        model=None,
    ) -> dict:
        import msprime

        from .. import deterministic as det
        from .. import montecarlo

        if model is None:
            model = msprime.SMCK(k=1)

        _, det_ld_raw = det.expected_piecewise_constant(
            ne_values,
            t_boundaries,
            self._left_bins,
            self._right_bins,
            self._mutation_rate,
            sample_size=self._num_samples,
            ploidy=self._ploidy,
        )
        det_ld = np.asarray(det_ld_raw)

        mc_kwargs = dict(
            random_seed=mc_seed,
            ploidy=self._ploidy,
            model=model,
            num_workers=self._num_workers,
        )
        if num_replicates is not None:
            mc_kwargs["num_replicates"] = int(num_replicates)
        else:
            mc_kwargs["rtol"] = rtol

        mc_pi_reps, mc_ld_reps = montecarlo.expected_piecewise_constant(
            ne_values,
            t_boundaries,
            self._left_bins,
            self._right_bins,
            self._mutation_rate,
            self._recombination_rate,
            self._sequence_length,
            self._num_samples,
            **mc_kwargs,
        )
        mc_pi_reps = np.asarray(mc_pi_reps)
        mc_ld_reps = np.asarray(mc_ld_reps)
        assert len(mc_ld_reps) > 1, (
            f"MC evaluation returned only {len(mc_ld_reps)} replicate(s); "
            "need at least 2 for a meaningful SE estimate."
        )
        return sg.make_synthetic_point(det_ld, mc_pi_reps, mc_ld_reps)

    # ─── Active learning ──────────────────────────────────────────────────────

    def active_learn_bias(
        self,
        n_points_per_iter: int = 5,
        n_iter: int = 5,
        max_tolerance: float = 0.1,
        num_replicates: Optional[int] = None,
        strategy: str = "pathfinder",
        model=None,
        seed: Optional[int] = None,
        progress_bar: bool = True,
    ) -> None:
        if self._sequence_length is None:
            raise ValueError(
                "sequence_length must be provided at initialisation "
                "to use active_learn_bias."
            )

        from tqdm.auto import tqdm

        rng = np.random.default_rng(seed)

        for iteration in range(n_iter):
            draws = self._get_draws(n_points_per_iter, strategy, rng)
            ne_mean = draws["Ne_values"].mean(axis=0)
            t_mean = draws["t_boundaries"].mean(axis=0)
            ne_str = "  ".join(f"Ne{i + 1}={v:,.0f}" for i, v in enumerate(ne_mean))
            t_str = "  ".join(f"t{i + 1}={v:.1f}" for i, v in enumerate(t_mean))
            print(
                f"[{strategy.upper()} iter={iteration + 1}/{n_iter} "
                f"n_synthetic={len(self._synthetic_points)}]  "
                f"{ne_str}  |  {t_str}"
            )

            iterator = tqdm(
                range(len(draws["Ne_values"])),
                desc=f"Active learning (iter {iteration + 1})",
                disable=not progress_bar,
            )
            for i in iterator:
                mc_seed = int(rng.integers(2**31))
                ne_i = np.asarray(draws["Ne_values"][i])
                t_i = np.asarray(draws["t_boundaries"][i])
                ne_str = " ".join(f"Ne{k + 1}={v:,.0f}" for k, v in enumerate(ne_i))
                t_str = " ".join(f"t{k + 1}={v:.1f}" for k, v in enumerate(t_i))
                iterator.set_postfix_str(f"{ne_str} | {t_str}")
                self._synthetic_points.append(
                    self._mc_eval(
                        draws["Ne_values"][i],
                        draws["t_boundaries"][i],
                        mc_seed,
                        rtol=max_tolerance,
                        num_replicates=num_replicates,
                        model=model,
                    )
                )

    def _get_draws(self, n_draws: int, strategy: str, rng: np.random.Generator) -> dict:
        data = self.stan_data()

        if strategy != "pathfinder":
            raise ValueError(
                f"Unknown strategy {strategy!r}; only 'pathfinder' is supported."
            )

        pf = self._model.pathfinder(
            data=data,
            draws=n_draws,
            seed=int(rng.integers(10_000)),
            num_threads=self._num_workers,
            show_console=False,
            inits=0.5,
        )
        ne_raw = pf.stan_variable("Ne_values")
        t_raw = pf.stan_variable("t_boundaries")
        ne_vals = np.atleast_2d(np.asarray(ne_raw))[:n_draws]
        t_vals = np.atleast_2d(np.asarray(t_raw))[:n_draws]
        return {"Ne_values": ne_vals, "t_boundaries": t_vals}

    # ─── Inference ────────────────────────────────────────────────────────────

    def sample(self, chains: int = 2, **kwargs):
        import arviz
        import xarray as xr

        kwargs.setdefault("threads_per_chain", self._num_workers)
        kwargs.setdefault("show_console", False)
        kwargs.setdefault("inits", 0.5)
        data = self.stan_data()

        fits = []
        for i in range(chains):
            chain_seed = kwargs.get("seed", 12345) + i
            fit = self._model.sample(
                data=data,
                chains=1,
                seed=chain_seed,
                **{k: v for k, v in kwargs.items() if k != "seed"},
            )
            fits.append(fit)

        trees = [arviz.from_cmdstanpy(f) for f in fits]
        if len(trees) == 1:
            return trees[0]

        children = {}
        for group in trees[0].children:
            children[group] = xr.concat([t[group].ds for t in trees], dim="chain")
        return xr.DataTree.from_dict(children)


# ── Public classes ────────────────────────────────────────────────────────────


class TwoEpochDemography(_PiecewiseConstantBase):
    """Two-epoch piecewise-constant model: Ne(t) = Ne_c for t < t0, Ne_a otherwise."""

    def __init__(
        self,
        diversity: np.ndarray,
        ld: np.ndarray,
        mutation_rate: float,
        recombination_rate: float,
        num_samples: int,
        left_bins: np.ndarray,
        right_bins: np.ndarray,
        sequence_length: Optional[float] = None,
        ploidy: int = 2,
        num_workers: int = 1,
        hsgp_c: float = 1.5,
        hsgp_m_u: int = 10,
        hsgp_m_ld: int = 6,
        n_quad: int = _DEFAULT_N_QUAD,
        gp_alpha_std: float = 0.005,
        lkj_eta: float = 2.0,
        log_sigma_y_scale: float = 1.0,
        prior: Optional[str] = None,
        parameters: str = _DEFAULT_PARAMETERS,
        transformed_parameters: str = _DEFAULT_TRANSFORMED_PARAMETERS,
    ):
        diversity = np.asarray(diversity, dtype=float)
        if prior is None:
            prior = _default_prior(diversity, float(mutation_rate))
        self._init_common(
            diversity=diversity,
            ld=ld,
            mutation_rate=mutation_rate,
            recombination_rate=recombination_rate,
            num_samples=num_samples,
            left_bins=left_bins,
            right_bins=right_bins,
            n_epochs=2,
            sequence_length=sequence_length,
            ploidy=ploidy,
            num_workers=num_workers,
            hsgp_c=hsgp_c,
            hsgp_m_u=hsgp_m_u,
            hsgp_m_ld=hsgp_m_ld,
            n_quad=n_quad,
            gp_alpha_std=gp_alpha_std,
            lkj_eta=lkj_eta,
            log_sigma_y_scale=log_sigma_y_scale,
            prior=prior,
            parameters=parameters,
            transformed_parameters=transformed_parameters,
        )


class PiecewiseConstantDemography(_PiecewiseConstantBase):
    """Generic N-epoch piecewise-constant model.

    The injected ``transformed_parameters`` must define
    ``vector[n_epochs] Ne_values`` and ``vector[n_epochs - 1] t_boundaries``.
    """

    def __init__(
        self,
        diversity: np.ndarray,
        ld: np.ndarray,
        mutation_rate: float,
        recombination_rate: float,
        num_samples: int,
        left_bins: np.ndarray,
        right_bins: np.ndarray,
        n_epochs: int,
        parameters: str,
        transformed_parameters: str,
        prior: str,
        sequence_length: Optional[float] = None,
        ploidy: int = 2,
        num_workers: int = 1,
        hsgp_c: float = 1.5,
        hsgp_m_u: int = 10,
        hsgp_m_ld: int = 6,
        n_quad: int = _DEFAULT_N_QUAD,
        gp_alpha_std: float = 0.005,
        lkj_eta: float = 2.0,
        log_sigma_y_scale: float = 1.0,
    ):
        self._init_common(
            diversity=diversity,
            ld=ld,
            mutation_rate=mutation_rate,
            recombination_rate=recombination_rate,
            num_samples=num_samples,
            left_bins=left_bins,
            right_bins=right_bins,
            n_epochs=n_epochs,
            sequence_length=sequence_length,
            ploidy=ploidy,
            num_workers=num_workers,
            hsgp_c=hsgp_c,
            hsgp_m_u=hsgp_m_u,
            hsgp_m_ld=hsgp_m_ld,
            n_quad=n_quad,
            gp_alpha_std=gp_alpha_std,
            lkj_eta=lkj_eta,
            log_sigma_y_scale=log_sigma_y_scale,
            prior=prior,
            parameters=parameters,
            transformed_parameters=transformed_parameters,
        )
