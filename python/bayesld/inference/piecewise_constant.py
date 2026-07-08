"""
Piecewise-constant-Ne inference with lognormal priors and a GP bias surrogate.

```
    model = (
        PiecewiseConstant(num_epochs=2)
        .with_data(mean_diversity=pi, mean_ld=ld,
                   left_bins=lb, right_bins=rb,
                   recombination_rate=r, mutation_rate=mu,
                   num_samples=n, sequence_length=L)
    )

    for _ in range(n_rounds):
        model = model.active_learning_round(num_points=5, rtol=0.1, min_replicates=30)
    idata = model.sample()
```

Everything not specific to the piecewise-constant parameterization lives in
``_base._BaseEngine``; this file supplies the three seams: the Stan file, the
lognormal-on-``Ne``/boundaries prior, and the msprime demography / deterministic
expectation used for Monte-Carlo active learning.
"""

import pathlib

import numpy as np
import numpy.typing as npt

from ._base import _DEFAULT_N_QUAD, _BaseEngine

_STAN_FILE = (
    pathlib.Path(__file__).resolve().parent / "stan" / "piecewise_constant.stan"
)

_DEFAULT_PRIOR_SIGMA = 1.5
_DEFAULT_T_ANCHOR = 50.0  # generations; centre of the default log-t grid


class PiecewiseConstant(_BaseEngine):
    """Piecewise-constant-Ne inference with lognormal priors and a GP bias surrogate."""

    _STAN_FILE = _STAN_FILE
    _PARAM_DIMS = {
        "log_Ne": ["epoch"],
        "Ne_values": ["epoch"],
        "log_t": ["boundary"],
        "t_boundaries": ["boundary"],
    }

    def __init__(
        self,
        num_epochs: int,
        *,
        n_quad: int = _DEFAULT_N_QUAD,
        **hyperprior: float,
    ) -> None:
        if int(num_epochs) < 1:
            raise ValueError(
                f"PiecewiseConstant requires num_epochs >= 1; got {num_epochs}."
            )
        self.num_epochs = int(num_epochs)
        super().__init__(n_quad=n_quad, **hyperprior)

    # ─── Display seam ────────────────────────────────────────────────────────

    @property
    def _title(self) -> str:
        return f"PiecewiseConstant (epochs = {self.num_epochs})"

    def _param_rows(self) -> list[tuple[str, str]]:
        return [("Epochs", str(self.num_epochs))]

    def _describe_params(self, params: dict) -> str:
        ne_s = np.array2string(params["ne"], precision=0, floatmode="fixed")
        t_s = np.array2string(params["t"], precision=0, floatmode="fixed")
        return f"Ne={ne_s} t={t_s}"

    def _prior_rows(self) -> list[tuple[str, str]]:
        if self._prior is None:
            return [("Status", "not set — call .with_prior(...) or .with_data(...)")]
        p = self._prior

        def _fmt(a: npt.ArrayLike) -> str:
            return np.array2string(np.asarray(a), precision=2, floatmode="fixed")

        rows = [
            ("log Ne  μ", _fmt(p["mu_log_ne"])),
            ("log Ne  σ", _fmt(p["sigma_log_ne"])),
        ]
        if self.num_epochs > 1:
            rows += [
                ("log t  μ", _fmt(p["mu_log_t"])),
                ("log t  σ", _fmt(p["sigma_log_t"])),
            ]
        return rows

    # ─── Prior seam ──────────────────────────────────────────────────────────

    def with_prior(
        self,
        mu_log_ne: npt.ArrayLike,
        sigma_log_ne: npt.ArrayLike,
        mu_log_t: npt.ArrayLike,
        sigma_log_t: npt.ArrayLike,
    ) -> "PiecewiseConstant":
        """
        Set the lognormal priors on Ne per epoch and on the epoch boundaries.

        Arguments:
        - mu_log_ne: mean of the lognormal prior on Ne per epoch
        - sigma_log_ne: standard deviation of the lognormal prior on Ne per epoch
        - mu_log_t: mean of the lognormal prior on the epoch boundaries. For a single epoch, pass an empty array.
        - sigma_log_t: standard deviation of the lognormal prior on the epoch boundaries. For a single epoch, pass an empty array.

        Returns:
        - a new model with the prior set (the receiver is left unchanged)

        Example:
        ```
        import numpy as np
        from bayesld.inference.piecewise_constant import PiecewiseConstant
        model = PiecewiseConstant(num_epochs=2).with_prior(
            mu_log_ne=np.log([1000, 2000]), sigma_log_ne=np.array([1.5, 1.0]), mu_log_t=np.log([50.0]), sigma_log_t=np.array([1.0])
        )
        ```
        """
        mu_log_ne = np.asarray(mu_log_ne, dtype=float)
        sigma_log_ne = np.asarray(sigma_log_ne, dtype=float)
        mu_log_t = np.asarray(mu_log_t, dtype=float)
        sigma_log_t = np.asarray(sigma_log_t, dtype=float)
        n, nb = self.num_epochs, self.num_epochs - 1
        if mu_log_ne.shape != (n,) or sigma_log_ne.shape != (n,):
            raise ValueError(
                f"mu_log_ne and sigma_log_ne must have length num_epochs={n}; "
                f"got {mu_log_ne.shape} and {sigma_log_ne.shape}."
            )
        if mu_log_t.shape != (nb,) or sigma_log_t.shape != (nb,):
            raise ValueError(
                f"mu_log_t and sigma_log_t must have length num_epochs-1={nb}; "
                f"got {mu_log_t.shape} and {sigma_log_t.shape}."
            )
        new = self._clone()
        new._prior = {
            "mu_log_ne": mu_log_ne,
            "sigma_log_ne": sigma_log_ne,
            "mu_log_t": mu_log_t,
            "sigma_log_t": sigma_log_t,
        }
        return new

    def _default_prior(self) -> dict:
        """Empirical-Bayes prior: Ne centred on the Watterson estimate pi/(4·mu)."""
        pi = self._data["mean_diversity"]
        mu = self._data["mutation_rate"]
        log_ne_mu = float(np.log(np.mean(pi) / (4.0 * mu)))
        # Log-spaced default boundary grid anchored at _DEFAULT_T_ANCHOR generations.
        nb = self.num_epochs - 1
        if nb == 1:
            log_t = np.array([np.log(_DEFAULT_T_ANCHOR)])
        else:
            log_t = np.log(
                np.geomspace(_DEFAULT_T_ANCHOR, _DEFAULT_T_ANCHOR * 20.0, nb)
            )
        return {
            "mu_log_ne": np.full(self.num_epochs, log_ne_mu),
            "sigma_log_ne": np.full(self.num_epochs, _DEFAULT_PRIOR_SIGMA),
            "mu_log_t": log_t,
            "sigma_log_t": np.full(nb, _DEFAULT_PRIOR_SIGMA),
        }

    def _prior_stan_data(self) -> dict:
        return {
            "n_epochs": self.num_epochs,
            "mu_log_ne": self._prior["mu_log_ne"],
            "sigma_log_ne": self._prior["sigma_log_ne"],
            "mu_log_t": self._prior["mu_log_t"],
            "sigma_log_t": self._prior["sigma_log_t"],
        }

    def sample_prior(
        self,
        draws: int = 1000,
        chains: int = 1,
        seed: int | None = None,
    ):
        """Sample demographic parameters from the prior.

        Arguments:
        - draws: number of draws from the prior
        - chains: number of chains
        - seed: random seed

        Returns:
        - xr.DataTree: the prior samples
        """
        if self._prior is None:
            raise RuntimeError(
                "No prior set. Call `.with_prior(...)` to set one explicitly, or "
                "`.with_data(...)` to use a default empirical-Bayes prior."
            )
        import arviz

        rng = np.random.default_rng(seed)
        total = draws * chains
        nb = self.num_epochs - 1

        log_ne = rng.normal(
            self._prior["mu_log_ne"],
            self._prior["sigma_log_ne"],
            size=(total, self.num_epochs),
        )
        ne_values = np.exp(log_ne)

        if nb > 0:
            log_t = rng.normal(
                self._prior["mu_log_t"],
                self._prior["sigma_log_t"],
                size=(total, nb),
            )
            log_t = np.sort(log_t, axis=-1)
            t_boundaries = np.exp(log_t)
        else:
            log_t = np.empty((total, 0))
            t_boundaries = np.empty((total, 0))

        # Reshape to (chains, draws, ...)
        log_ne = log_ne.reshape(chains, draws, self.num_epochs)
        ne_values = ne_values.reshape(chains, draws, self.num_epochs)
        log_t = log_t.reshape(chains, draws, nb)
        t_boundaries = t_boundaries.reshape(chains, draws, nb)

        posterior = {
            "log_Ne": log_ne,
            "Ne_values": ne_values,
        }
        dims = {
            "log_Ne": ["epoch"],
            "Ne_values": ["epoch"],
        }
        coords = {
            "epoch": np.arange(self.num_epochs),
            "boundary": np.arange(nb),
        }
        if nb > 0:
            posterior["log_t"] = log_t
            posterior["t_boundaries"] = t_boundaries
            dims["log_t"] = ["boundary"]
            dims["t_boundaries"] = ["boundary"]

        return arviz.from_dict({"posterior": posterior}, coords=coords, dims=dims)

    def _param_coords(self) -> dict:
        return {
            "epoch": np.arange(self.num_epochs),
            "boundary": np.arange(self.num_epochs - 1),
        }

    def _extract_params(self, fit) -> list[dict]:
        ne_draws = np.atleast_2d(np.asarray(fit.stan_variable("Ne_values"))).reshape(
            -1, self.num_epochs
        )
        if self.num_epochs > 1:
            t_draws = np.atleast_2d(
                np.asarray(fit.stan_variable("t_boundaries"))
            ).reshape(-1, self.num_epochs - 1)
        else:
            # Single constant epoch: no boundaries to propose.
            t_draws = np.empty((len(ne_draws), 0))
        return [{"ne": ne_draws[i], "t": t_draws[i]} for i in range(len(ne_draws))]

    def _expected(self, params: dict) -> tuple[float, np.ndarray]:
        from .. import deterministic as det

        det_pi, det_ld = det.expected_piecewise_constant(
            params["ne"],
            params["t"],
            self._left_bins,
            self._right_bins,
            self._data["mutation_rate"],
            sample_size=self._data["num_samples"],
            ploidy=self._data["ploidy"],
        )
        return float(np.asarray(det_pi)), np.asarray(det_ld)

    def _build_demography(self, params: dict):
        import msprime

        ne_values = params["ne"]
        t_boundaries = params["t"]
        demography = msprime.Demography()
        demography.add_population(name="pop0", initial_size=float(ne_values[0]))
        for t, ne in zip(t_boundaries, ne_values[1:]):
            demography.add_population_parameters_change(
                time=float(t), initial_size=float(ne), growth_rate=0
            )
        return demography
