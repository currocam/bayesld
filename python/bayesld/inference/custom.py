"""
User-facing engine for a *custom* demographic parameterization.

The seams come in three tiers, matching *when* they are needed:

- **Case 1 (``.sample()``):** a Stan file producing ``expected_pi`` +
  ``approx_expected_ld`` and consuming the shared surrogate data (the easiest
  path is to ``#include shared/*.stan`` — see ``inference/stan/piecewise_constant.stan``).
  You also need a prior: either call ``.with_prior(**names)`` explicitly, or pass
  ``default_prior`` so ``.with_data(...)`` can install an empirical-Bayes default.
  ``prior_stan_data`` maps the stored prior dict onto your Stan data names
  (defaults to passing it through unchanged); ``param_coords`` labels any
  parameterization-specific posterior dims (defaults to none).
- **Case 2 (``.active_learning_round(...)``):** additionally supply
  ``build_demography``, ``expected`` and ``extract_params`` so det-vs-MC bias can
  be evaluated at Pathfinder proposals.
- **Case 3 (``.sample_prior(...)``):** additionally supply ``sample_prior`` for
  prior-predictive draws.

Every callable receives the engine instance as its first argument (like an
unbound method), giving read-only access to ``engine._data``, ``engine._prior``,
``engine._left_bins`` / ``engine._right_bins`` etc. Callables left as ``None``
raise a clear error only if the corresponding tier is exercised.

Example (Case 2)::

    import numpy as np, msprime, pathlib
    from bayesld.inference import CustomEngine
    from bayesld import deterministic as det

    engine = CustomEngine(
        stan_file="my_two_epoch.stan",
        param_dims={},  # all scalar params
        default_prior=lambda e: {
            "mu_log_ne": float(np.log(np.mean(e._data["mean_diversity"])
                                      / (4 * e._data["mutation_rate"]))),
            "sigma_log_ne": 1.5,
        },
        build_demography=lambda e, p: msprime.Demography(...),
        expected=lambda e, p: det.expected_constant(...),
        extract_params=lambda e, fit: [{"ne": v} for v in fit.stan_variable("Ne")],
    )
    idata = (engine
             .with_data(mean_diversity=pi, mean_ld=ld, left_bins=lb, right_bins=rb,
                        recombination_rate=r, mutation_rate=mu, num_samples=n,
                        sequence_length=L)
             .active_learning_round(num_points=5, rtol=0.1, min_replicates=30)
             .sample())
"""

import pathlib
from typing import Callable

import numpy as np

from ._base import _DEFAULT_N_QUAD, _BaseEngine


class CustomEngine(_BaseEngine):
    """Bring-your-own Stan file + callable seams over the shared engine machinery."""

    def __init__(
        self,
        stan_file: str | pathlib.Path,
        *,
        param_dims: dict | None = None,
        default_prior: Callable[["CustomEngine"], dict] | None = None,
        prior_stan_data: Callable[["CustomEngine"], dict] | None = None,
        param_coords: Callable[["CustomEngine"], dict] | None = None,
        build_demography: Callable[["CustomEngine", dict], object] | None = None,
        expected: Callable[["CustomEngine", dict], tuple] | None = None,
        extract_params: Callable[["CustomEngine", object], list] | None = None,
        sample_prior: Callable[..., object] | None = None,
        title: str | None = None,
        n_quad: int = _DEFAULT_N_QUAD,
        **hyperprior: float,
    ) -> None:
        """
        Arguments:
        - stan_file: path to a static Stan program (compiled once). ``#include
          shared/*.stan`` resolves against ``inference/stan/`` and the file's own dir.
        - param_dims: posterior dims that depend on the parameterization, e.g.
          ``{"log_Ne": ["epoch"]}`` (see ``_BaseEngine._PARAM_DIMS``). Default: none.
        - default_prior(engine) -> dict: empirical-Bayes prior from ``engine._data``;
          installed by ``.with_data(...)`` when no prior is set. Required only if you
          rely on that default rather than calling ``.with_prior(...)`` yourself.
        - prior_stan_data(engine) -> dict: maps the stored prior onto the Stan data
          names. Default: passes ``engine._prior`` through unchanged.
        - param_coords(engine) -> dict: coordinate values for the param dims. Default: {}.
        - build_demography(engine, params) -> msprime.Demography: for active learning.
        - expected(engine, params) -> (expected_pi, expected_ld): deterministic
          prediction used to form the det-vs-MC bias in active learning.
        - extract_params(engine, fit) -> list[dict]: turn a Pathfinder fit into
          per-draw proposal dicts (keys must match what the above two consume).
        - sample_prior(engine, draws, chains, seed): prior-predictive draws.
        - title: heading shown in the notebook/HTML summary.
        - n_quad, **hyperprior: as in the built-in engines.
        """
        self._STAN_FILE = pathlib.Path(stan_file)
        self._PARAM_DIMS = dict(param_dims or {})
        self._seams: dict[str, Callable | None] = {
            "default_prior": default_prior,
            "prior_stan_data": prior_stan_data,
            "param_coords": param_coords,
            "build_demography": build_demography,
            "expected": expected,
            "extract_params": extract_params,
            "sample_prior": sample_prior,
        }
        self._custom_title = title
        super().__init__(n_quad=n_quad, **hyperprior)

    def _require(self, name: str, tier: str):
        fn = self._seams.get(name)
        if fn is None:
            raise NotImplementedError(
                f"CustomEngine needs `{name}=...` to be provided for {tier}. "
                f"Pass it to the constructor."
            )
        return fn

    # ─── Prior seam ──────────────────────────────────────────────────────────

    def with_prior(self, **prior) -> "CustomEngine":
        """Set the prior as keyword args matching the Stan data names.

        Returns a new engine (the receiver is left unchanged). Scalars and
        array-likes are both accepted and stored verbatim into the Stan data.
        """
        new = self._clone()
        new._prior = dict(prior)
        return new

    def _default_prior(self) -> dict:
        fn = self._seams.get("default_prior")
        if fn is None:
            raise RuntimeError(
                "No prior set and no `default_prior` provided. Either call "
                "`.with_prior(...)` before `.with_data(...)`, or pass "
                "`default_prior=...` to CustomEngine."
            )
        return dict(fn(self))

    def _prior_stan_data(self) -> dict:
        assert self._prior is not None
        fn = self._seams.get("prior_stan_data")
        return dict(fn(self)) if fn is not None else dict(self._prior)

    def sample_prior(self, draws: int = 1000, chains: int = 1, seed=None):
        """Draw demographic parameters from the prior (requires ``sample_prior``)."""
        if self._prior is None:
            raise RuntimeError(
                "No prior set. Call `.with_prior(...)` or `.with_data(...)` first."
            )
        fn = self._require("sample_prior", "prior-predictive sampling")
        return fn(self, draws=draws, chains=chains, seed=seed)

    def _param_coords(self) -> dict:
        fn = self._seams.get("param_coords")
        return dict(fn(self)) if fn is not None else {}

    # ─── MC / demography seam ────────────────────────────────────────────────

    def _build_demography(self, params: dict):
        return self._require("build_demography", "active learning")(self, params)

    def _expected(self, params: dict) -> tuple:
        det_pi, det_ld = self._require("expected", "active learning")(self, params)
        return float(np.asarray(det_pi)), np.asarray(det_ld)

    def _extract_params(self, fit) -> list[dict]:
        return self._require("extract_params", "active learning")(self, fit)

    # ─── Display seam ────────────────────────────────────────────────────────

    @property
    def _title(self) -> str:
        return self._custom_title or "CustomEngine"
