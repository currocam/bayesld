"""
Two-phase piecewise-exponential-Ne inference with lognormal priors and a GP bias
surrogate.

The demography is always two epochs::

    Ne(t) = Ne_c * exp(-alpha * t)   for t < t0   (exponential phase)
    Ne(t) = Ne_a                     for t >= t0  (ancient constant phase)

with ``alpha = (log Ne_c - log Ne_a) / t0`` so the two phases meet at ``t0``.
It is parameterized by three scalars:

- ``log_ne_a``       — log ancient population size, ``Ne_a = exp(log_ne_a)``;
- ``log_t``          — log start of the exponential phase, ``t0 = exp(log_t)``;
- ``log_alpha_fold`` — log-fold change of the contemporary size relative to the
  ancient one, ``log Ne_c - log Ne_a``, so ``Ne_c = Ne_a * exp(log_alpha_fold)``.

```
    model = (
        PiecewiseExponential()
        .with_data(mean_diversity=pi, mean_ld=ld,
                   left_bins=lb, right_bins=rb,
                   recombination_rate=r, mutation_rate=mu,
                   num_samples=n, sequence_length=L)
    )

    for _ in range(n_rounds):
        model = model.active_learning_round(num_points=5, rtol=0.1, min_replicates=30)
    idata = model.sample()
```

Everything not specific to this parameterization lives in ``_base._BaseEngine``;
this file supplies the three seams: the Stan file, the lognormal priors on
``(log_ne_a, log_t, log_alpha_fold)``, and the msprime demography / deterministic
expectation used for Monte-Carlo active learning.
"""

import pathlib

import numpy as np
import numpy.typing as npt

from ._base import _DEFAULT_N_QUAD, _BaseEngine

_STAN_FILE = (
    pathlib.Path(__file__).resolve().parent / "stan" / "piecewise_exponential.stan"
)

_DEFAULT_PRIOR_SIGMA = 1.5
_DEFAULT_T_ANCHOR = 50.0  # generations; centre of the default log-t prior


class PiecewiseExponential(_BaseEngine):
    """Two-phase piecewise-exponential-Ne inference with a GP bias surrogate."""

    _STAN_FILE = _STAN_FILE
    _PARAM_DIMS = {}  # all demographic parameters are scalars

    def __init__(
        self,
        *,
        n_quad: int = _DEFAULT_N_QUAD,
        **hyperprior: float,
    ) -> None:
        super().__init__(n_quad=n_quad, **hyperprior)

    # ─── Display seam ────────────────────────────────────────────────────────

    @property
    def _title(self) -> str:
        return "PiecewiseExponential (two-phase)"

    def _describe_params(self, params: dict) -> str:
        return (
            f"Ne_c={params['ne_c']:,.0f} Ne_a={params['ne_a']:,.0f} "
            f"t0={params['t0']:.1f} alpha={params['alpha']:.3g}"
        )

    def _prior_rows(self) -> list[tuple[str, str]]:
        if self._prior is None:
            return [("Status", "not set — call .with_prior(...) or .with_data(...)")]
        p = self._prior
        return [
            ("log Ne_a  μ / σ", f"{p['mu_log_ne_a']:.2f} / {p['sigma_log_ne_a']:.2f}"),
            ("log t  μ / σ", f"{p['mu_log_t']:.2f} / {p['sigma_log_t']:.2f}"),
            (
                "log fold  μ / σ",
                f"{p['mu_log_alpha_fold']:.2f} / {p['sigma_log_alpha_fold']:.2f}",
            ),
        ]

    # ─── Prior seam ──────────────────────────────────────────────────────────

    def with_prior(
        self,
        mu_log_ne_a: float,
        sigma_log_ne_a: float,
        mu_log_t: float,
        sigma_log_t: float,
        mu_log_alpha_fold: float,
        sigma_log_alpha_fold: float,
    ) -> "PiecewiseExponential":
        """
        Set the lognormal priors on the three demographic parameters.

        Arguments:
        - mu_log_ne_a, sigma_log_ne_a: mean/sd of the lognormal prior on the ancient Ne
        - mu_log_t, sigma_log_t: mean/sd of the lognormal prior on the exponential start t0
        - mu_log_alpha_fold, sigma_log_alpha_fold: mean/sd of the normal prior on the
          contemporary-vs-ancient log-fold change (0 => no change; positive => larger
          contemporary size)

        Returns:
        - a new model with the prior set (the receiver is left unchanged)

        Example:
        ```
        import numpy as np
        from bayesld.inference.piecewise_exponential import PiecewiseExponential
        model = PiecewiseExponential().with_prior(
            mu_log_ne_a=np.log(2000), sigma_log_ne_a=1.5,
            mu_log_t=np.log(50.0), sigma_log_t=1.5,
            mu_log_alpha_fold=0.0, sigma_log_alpha_fold=1.5,
        )
        ```
        """
        new = self._clone()
        new._prior = {
            "mu_log_ne_a": float(mu_log_ne_a),
            "sigma_log_ne_a": float(sigma_log_ne_a),
            "mu_log_t": float(mu_log_t),
            "sigma_log_t": float(sigma_log_t),
            "mu_log_alpha_fold": float(mu_log_alpha_fold),
            "sigma_log_alpha_fold": float(sigma_log_alpha_fold),
        }
        return new

    def _default_prior(self) -> dict:
        """Empirical-Bayes prior: Ne_a centred on the Watterson estimate pi/(4·mu)."""
        pi = self._data["mean_diversity"]
        mu = self._data["mutation_rate"]
        log_ne_mu = float(np.log(np.mean(pi) / (4.0 * mu)))
        return {
            "mu_log_ne_a": log_ne_mu,
            "sigma_log_ne_a": _DEFAULT_PRIOR_SIGMA,
            "mu_log_t": float(np.log(_DEFAULT_T_ANCHOR)),
            "sigma_log_t": _DEFAULT_PRIOR_SIGMA,
            "mu_log_alpha_fold": 0.0,
            "sigma_log_alpha_fold": _DEFAULT_PRIOR_SIGMA,
        }

    def _prior_stan_data(self) -> dict:
        return dict(self._prior)

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
        p = self._prior
        shape = (chains, draws)

        log_ne_a = rng.normal(p["mu_log_ne_a"], p["sigma_log_ne_a"], size=shape)
        log_t = rng.normal(p["mu_log_t"], p["sigma_log_t"], size=shape)
        log_alpha_fold = rng.normal(
            p["mu_log_alpha_fold"], p["sigma_log_alpha_fold"], size=shape
        )

        log_ne_c = log_ne_a + log_alpha_fold
        t0 = np.exp(log_t)

        posterior = {
            "log_ne_a": log_ne_a,
            "log_t": log_t,
            "log_alpha_fold": log_alpha_fold,
            "log_ne_c": log_ne_c,
            "Ne_a": np.exp(log_ne_a),
            "Ne_c": np.exp(log_ne_c),
            "t0": t0,
            "alpha": log_alpha_fold / t0,
        }
        return arviz.from_dict({"posterior": posterior})

    def _param_coords(self) -> dict:
        return {}

    def _extract_params(self, fit) -> list[dict]:
        def _draws(name: str) -> np.ndarray:
            return np.atleast_1d(np.asarray(fit.stan_variable(name))).ravel()

        ne_c = _draws("Ne_c")
        ne_a = _draws("Ne_a")
        t0 = _draws("t0")
        alpha = _draws("alpha")
        return [
            {"ne_c": ne_c[i], "ne_a": ne_a[i], "t0": t0[i], "alpha": alpha[i]}
            for i in range(len(ne_c))
        ]

    def _expected(self, params: dict) -> tuple[float, np.ndarray]:
        from .. import deterministic as det

        det_pi, det_ld = det.expected_piecewise_exponential(
            params["ne_c"],
            params["ne_a"],
            params["t0"],
            params["alpha"],
            self._left_bins,
            self._right_bins,
            self._data["mutation_rate"],
            sample_size=self._data["num_samples"],
            ploidy=self._data["ploidy"],
        )
        return float(np.asarray(det_pi)), np.asarray(det_ld)

    def _build_demography(self, params: dict):
        import msprime

        demography = msprime.Demography()
        demography.add_population(
            name="pop0",
            initial_size=float(params["ne_c"]),
            growth_rate=float(params["alpha"]),
        )
        demography.add_population_parameters_change(
            time=float(params["t0"]),
            initial_size=float(params["ne_a"]),
            growth_rate=0,
        )
        return demography
