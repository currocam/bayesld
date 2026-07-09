"""
Random-walk-Ne inference over a fixed epoch grid, with a GP bias surrogate.

The epoch boundaries are supplied once, up front, and held **fixed**; only the
population size per epoch is inferred. ``Ne`` follows a log-space random walk
anchored at the most ancient epoch: a lognormal prior sets the ancient size, and
each step towards the present is a Gaussian log-fold change. Concretely, with
epochs ordered ancient → recent,

    log Ne_ancient ~ normal(mu_log_ne, sigma_log_ne)
    step_i         ~ normal(0, sigma_step)
    log Ne_i        = log Ne_ancient + Σ (steps from epoch i up to the ancient one)

i.e. each epoch's log-Ne is the ancient anchor plus the reverse cumulative sum of
the intervening steps.

```
    grid = np.array([50.0, 200.0, 1000.0])  # 3 boundaries → 4 epochs
    model = (
        RandomWalk(grid)
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
this file supplies the three seams: the Stan file, the random-walk prior, and the
(piecewise-constant) msprime demography / deterministic expectation used for
Monte-Carlo active learning.
"""

import pathlib

import numpy as np
import numpy.typing as npt

from ._base import _DEFAULT_N_QUAD, _BaseEngine

_STAN_FILE = pathlib.Path(__file__).resolve().parent / "stan" / "random_walk.stan"

_DEFAULT_PRIOR_SIGMA = 1.5  # sd of the lognormal prior on the ancient Ne
_DEFAULT_STEP_SIGMA = 0.5  # sd of the log-fold random-walk steps


class RandomWalk(_BaseEngine):
    """Random-walk-Ne inference over a fixed epoch grid, with a GP bias surrogate."""

    _STAN_FILE = _STAN_FILE
    _PARAM_DIMS = {
        "log_Ne": ["epoch"],
        "Ne_values": ["epoch"],
        "steps": ["step"],
        "t_boundaries": ["boundary"],
    }

    def __init__(
        self,
        grid: npt.ArrayLike,
        *,
        n_quad: int = _DEFAULT_N_QUAD,
        **hyperprior: float,
    ) -> None:
        """
        Arguments:
        - grid: fixed epoch boundaries in generations, strictly increasing and
          positive. ``len(grid)`` boundaries define ``len(grid) + 1`` epochs.
        - n_quad: number of Gauss-Legendre quadrature nodes.
        - hyperprior: overrides for the GP-surrogate hyperpriors.
        """
        grid = np.asarray(grid, dtype=float)
        if grid.ndim != 1 or grid.size < 1:
            raise ValueError(
                "RandomWalk requires a 1-D grid with at least one boundary "
                f"(giving >= 2 epochs); got shape {grid.shape}."
            )
        if not np.all(grid > 0.0):
            raise ValueError("grid boundaries must be positive (generations).")
        if not np.all(np.diff(grid) > 0.0):
            raise ValueError("grid boundaries must be strictly increasing.")
        self._grid = grid
        self.num_epochs = grid.size + 1
        super().__init__(n_quad=n_quad, **hyperprior)

    # ─── Display seam ────────────────────────────────────────────────────────

    @property
    def _title(self) -> str:
        return f"RandomWalk (epochs = {self.num_epochs})"

    def _param_rows(self) -> list[tuple[str, str]]:
        grid_s = np.array2string(self._grid, precision=0, floatmode="fixed")
        return [("Epochs", str(self.num_epochs)), ("Grid", grid_s)]

    def _describe_params(self, params: dict) -> str:
        ne_s = np.array2string(params["ne"], precision=0, floatmode="fixed")
        t_s = np.array2string(params["t"], precision=0, floatmode="fixed")
        return f"Ne={ne_s} t={t_s}"

    def _prior_rows(self) -> list[tuple[str, str]]:
        if self._prior is None:
            return [("Status", "not set — call .with_prior(...) or .with_data(...)")]
        p = self._prior
        step_s = np.array2string(
            np.asarray(p["sigma_step"]), precision=2, floatmode="fixed"
        )
        return [
            (
                "log Ne (ancient)  μ / σ",
                f"{p['mu_log_ne']:.2f} / {p['sigma_log_ne']:.2f}",
            ),
            ("log-fold step  σ", step_s),
        ]

    # ─── Prior seam ──────────────────────────────────────────────────────────

    def with_prior(
        self,
        mu_log_ne: float,
        sigma_log_ne: float,
        sigma_step: npt.ArrayLike,
    ) -> "RandomWalk":
        """
        Set the random-walk prior on ``Ne``.

        Arguments:
        - mu_log_ne: mean of the lognormal prior on the most ancient epoch's Ne
        - sigma_log_ne: sd of the lognormal prior on the most ancient epoch's Ne
        - sigma_step: sd of the Gaussian log-fold change between adjacent epochs
          (the random-walk innovation scale; larger => rougher histories). Either a
          scalar (shared by every step) or a length ``num_epochs-1`` vector for a
          per-step scale (e.g. to widen the innovation over larger grid gaps).

        Returns:
        - a new model with the prior set (the receiver is left unchanged)

        Example:
        ```
        import numpy as np
        from bayesld.inference.random_walk import RandomWalk
        model = RandomWalk(grid=np.array([50.0, 200.0])).with_prior(
            mu_log_ne=np.log(2000), sigma_log_ne=1.5, sigma_step=0.5
        )
        ```
        """
        if float(sigma_log_ne) <= 0:
            raise ValueError("sigma_log_ne must be positive.")
        new = self._clone()
        new._prior = {
            "mu_log_ne": float(mu_log_ne),
            "sigma_log_ne": float(sigma_log_ne),
            "sigma_step": self._broadcast_sigma_step(sigma_step),
        }
        return new

    def _broadcast_sigma_step(self, sigma_step: npt.ArrayLike) -> np.ndarray:
        """Coerce a scalar or vector step scale to a positive length-(n-1) array."""
        nb = self.num_epochs - 1
        sigma_step = np.asarray(sigma_step, dtype=float)
        if sigma_step.ndim == 0:
            sigma_step = np.repeat(sigma_step, nb)
        if sigma_step.shape != (nb,):
            raise ValueError(
                f"sigma_step must be a scalar or a length num_epochs-1={nb} vector; "
                f"got shape {sigma_step.shape}."
            )
        if np.any(sigma_step <= 0):
            raise ValueError("sigma_step must be positive.")
        return sigma_step

    def _default_prior(self) -> dict:
        """Empirical-Bayes prior: ancient Ne centred on the Watterson estimate pi/(4·mu)."""
        pi = self._data["mean_diversity"]
        mu = self._data["mutation_rate"]
        log_ne_mu = float(np.log(np.mean(pi) / (4.0 * mu)))
        return {
            "mu_log_ne": log_ne_mu,
            "sigma_log_ne": _DEFAULT_PRIOR_SIGMA,
            "sigma_step": np.full(self.num_epochs - 1, _DEFAULT_STEP_SIGMA),
        }

    def _prior_stan_data(self) -> dict:
        return {
            "n_epochs": self.num_epochs,
            "grid": self._grid,
            "mu_log_ne": self._prior["mu_log_ne"],
            "sigma_log_ne": self._prior["sigma_log_ne"],
            "sigma_step": self._prior["sigma_step"],
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
        - xarray.DataTree: the prior samples
        """
        if self._prior is None:
            raise RuntimeError(
                "No prior set. Call `.with_prior(...)` to set one explicitly, or "
                "`.with_data(...)` to use a default empirical-Bayes prior."
            )
        import arviz as az

        rng = np.random.default_rng(seed)
        p = self._prior
        total = draws * chains
        nb = self.num_epochs - 1

        log_ne_a = rng.normal(p["mu_log_ne"], p["sigma_log_ne"], size=total)
        steps = rng.normal(0.0, p["sigma_step"], size=(total, nb))

        # log_Ne[i] = log_ne_a + reverse-cumsum of steps from epoch i (recent) up
        # to the ancient anchor; ancient epoch (last) carries no step.
        rev_cumsum = np.cumsum(steps[:, ::-1], axis=1)[:, ::-1]
        log_ne = np.empty((total, self.num_epochs))
        log_ne[:, -1] = log_ne_a
        log_ne[:, :-1] = log_ne_a[:, None] + rev_cumsum
        ne_values = np.exp(log_ne)

        t_boundaries = np.tile(self._grid, (total, 1))

        log_ne = log_ne.reshape(chains, draws, self.num_epochs)
        ne_values = ne_values.reshape(chains, draws, self.num_epochs)
        steps = steps.reshape(chains, draws, nb)
        t_boundaries = t_boundaries.reshape(chains, draws, nb)

        posterior = {
            "log_Ne": log_ne,
            "Ne_values": ne_values,
            "steps": steps,
            "t_boundaries": t_boundaries,
        }
        dims = {
            "log_Ne": ["epoch"],
            "Ne_values": ["epoch"],
            "steps": ["step"],
            "t_boundaries": ["boundary"],
        }
        return az.from_dict(
            {"posterior": posterior}, coords=self._param_coords(), dims=dims
        )

    def _param_coords(self) -> dict:
        return {
            "epoch": np.arange(self.num_epochs),
            "step": np.arange(self.num_epochs - 1),
            "boundary": np.arange(self.num_epochs - 1),
        }

    def _constant_data_vars(self) -> dict:
        import xarray as xr

        c = self._param_coords()
        return {
            "grid": xr.DataArray(
                np.asarray(self._grid, dtype=float),
                dims=["boundary"],
                coords={"boundary": c["boundary"]},
            ),
        }

    # ─── MC / demography seam ────────────────────────────────────────────────

    def _extract_params(self, fit) -> list[dict]:
        ne_draws = np.atleast_2d(np.asarray(fit.stan_variable("Ne_values"))).reshape(
            -1, self.num_epochs
        )
        return [{"ne": ne_draws[i], "t": self._grid} for i in range(len(ne_draws))]

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
