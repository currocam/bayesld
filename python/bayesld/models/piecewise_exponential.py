"""
Piecewise-exponential demographic inference model.

``PiecewiseExponentialDemography`` models::

    Ne(t) = Ne_c * exp(-alpha * t)   for t < t0
    Ne(t) = Ne_a                     for t >= t0

where ``alpha = log_fold_change / t0``.  The native parameter
``log_fold_change`` captures the total log-ratio of Ne over the exponential
phase and is far less correlated with ``t0`` than the raw rate ``alpha``.

The ``expm1_over_x`` helper in ``piecewise_exponential.stan`` ensures the
formula is numerically stable at alpha = 0 (constant-piecewise limit).

Like ``PiecewiseConstantDemography``, it maintains an approximate model and
a GP-surrogate model that corrects the LD predictions multiplicatively per bin.
"""

import pathlib
import tempfile
import warnings
from typing import Optional

import numpy as np

from ._surrogate import (
    _GP_SURROGATE_DATA,
    _GP_SURROGATE_MODEL,
    _GP_SURROGATE_PARAMS,
    _GP_SURROGATE_TRANSFORMED_DATA,
    _GP_SURROGATE_TRANSFORMED_PARAMS,
    _stan_draw_matrix,
    _stan_vector,
)

_STAN_DIR = pathlib.Path(__file__).resolve().parent.parent.parent.parent / "stan"
_THREADS_OPTS = {"cpp_options": {"STAN_THREADS": "true"}}

_DEFAULT_N_QUAD = 16


def _default_prior(diversity: np.ndarray, mutation_rate: float) -> str:
    ne_hat = float(np.mean(diversity)) / (4.0 * mutation_rate)
    log_ne_mu = float(np.log(ne_hat))
    return (
        f"    log_Ne_c ~ normal({log_ne_mu:.4f}, 1.0);\n"
        f"    log_Ne_a ~ normal({log_ne_mu:.4f}, 1.0);\n"
        f"    log_t0   ~ normal({np.log(200.0):.4f}, 1.0);\n"
        f"    log_fold_change ~ normal(0, 1.0);"
    )


# ──────────────────────────────────────────────────────────────────────────
# Stan source fragments
# ──────────────────────────────────────────────────────────────────────────

_COMMON_DATA = """\
data {
    int<lower=1> n_bins;
    int<lower=2> num_windows;
    vector[n_bins] left_bins;
    vector[n_bins] right_bins;
    real<lower=0> mutation_rate;
    int<lower=1> sample_size;
    vector[num_windows] pi_array;
    matrix[num_windows, n_bins] ld_mat;
    int<lower=1> n_quad;
    vector[n_quad] gl_nodes;
    vector[n_quad] gl_weights;
"""

_COMMON_TRANSFORMED_DATA = """\
transformed data {
    real mean_div = mean(pi_array);
    real<lower=0> sigma_div = sd(pi_array);
    real<lower=0> sem_div   = sigma_div / sqrt(num_windows);

    vector[n_bins] mean_ld;
    vector<lower=0>[n_bins] sigma_ld;
    vector<lower=0>[n_bins] sem_ld;
    for (b in 1:n_bins) {
        mean_ld[b]  = mean(col(ld_mat, b));
        sigma_ld[b] = sd(col(ld_mat, b));
        sem_ld[b]   = sigma_ld[b] / sqrt(num_windows);
    }
    real log_ne_offset = log(mean_div / (4.0 * mutation_rate));
"""

# ──────────────────────────────────────────────────────────────────────────
# Approximate (deterministic) model template
# ──────────────────────────────────────────────────────────────────────────

_MAP_RECT_TRANSFORMED_DATA = """\
    // Pack per-bin data for map_rect parallelism
    array[n_bins, 2 + 2 * n_quad] real mr_bin_data;
    array[n_bins, 1] int mr_bin_int;
    array[n_bins] vector[0] mr_theta;
    for (b in 1:n_bins) {
        mr_bin_data[b, 1] = left_bins[b];
        mr_bin_data[b, 2] = right_bins[b];
        for (k in 1:n_quad) {
            mr_bin_data[b, 2 + k] = gl_nodes[k];
            mr_bin_data[b, 2 + n_quad + k] = gl_weights[k];
        }
        mr_bin_int[b, 1] = n_quad;
    }
"""

_APPROX_TEMPLATE = """\
functions {{
// ---- shared.stan ----
{shared_functions}
// ---- piecewise_exponential.stan ----
{model_functions}
}}

{common_data}}}

{common_transformed_data}
{map_rect_transformed_data}}}

parameters {{
    real<offset=log_ne_offset> log_Ne_c;
    real<offset=log_ne_offset> log_Ne_a;
    real log_t0;
    real log_fold_change;
{extra_parameters}
}}

transformed parameters {{
    real<lower=0> Ne_c = exp(log_Ne_c);
    real<lower=0> Ne_a = exp(log_Ne_a);
    real<lower=0> t0   = exp(log_t0);
    real alpha = log_fold_change / t0;
    real E_pi = mu_div_piecewise_exponential(Ne_c, Ne_a, t0, alpha, mutation_rate,
                                              n_quad, gl_nodes, gl_weights);
    vector[4] mr_phi = [Ne_c, Ne_a, t0, alpha]';
    vector[n_bins] approx_ld = correct_ld_finite_sample(
        map_rect(mu_ld_shard_pe, mr_phi, mr_theta, mr_bin_data, mr_bin_int),
        sample_size
    );
}}

model {{
    // --- user prior ---
{prior_block}
    mean_div ~ normal(E_pi, sem_div);
    target += normal_lpdf(mean_ld | approx_ld, sem_ld) / n_bins;
}}

generated quantities {{
    vector[num_windows] log_lik;
    for (w in 1:num_windows) {{
        log_lik[w] = normal_lpdf(pi_array[w] | E_pi, sigma_div)
                   + normal_lpdf(to_vector(ld_mat[w]) | approx_ld, sigma_ld) / n_bins;
    }}
}}
"""

# ──────────────────────────────────────────────────────────────────────────
# GP-surrogate model template
# ──────────────────────────────────────────────────────────────────────────

_SURROGATE_TEMPLATE = """\
functions {{
// ---- gpbasisfun_functions.stan ----
{gp_functions}
// ---- shared.stan ----
{shared_functions}
// ---- piecewise_exponential.stan ----
{model_functions}
}}

{common_data}
{gp_surrogate_data}}}

{common_transformed_data}
{map_rect_transformed_data}
{gp_surrogate_transformed_data}}}

parameters {{
    real<offset=log_ne_offset> log_Ne_c;
    real<offset=log_ne_offset> log_Ne_a;
    real log_t0;
    real log_fold_change;
{gp_surrogate_params}
{extra_parameters}
}}

transformed parameters {{
    real<lower=0> Ne_c = exp(log_Ne_c);
    real<lower=0> Ne_a = exp(log_Ne_a);
    real<lower=0> t0   = exp(log_t0);
    real alpha = log_fold_change / t0;
{gp_surrogate_transformed_params}
    real E_pi = mu_div_piecewise_exponential(Ne_c, Ne_a, t0, alpha, mutation_rate,
                                              n_quad, gl_nodes, gl_weights);
    vector[4] mr_phi = [Ne_c, Ne_a, t0, alpha]';
    vector[n_bins] approx_ld = correct_ld_finite_sample(
        map_rect(mu_ld_shard_pe, mr_phi, mr_theta, mr_bin_data, mr_bin_int),
        sample_size
    );
    vector[n_bins] corrected_ld = approx_ld .* (1.0 + gp_bias_ld);
}}

model {{
{gp_surrogate_model}
    // --- user prior ---
{prior_block}
    mean_div ~ normal(E_pi, sem_div);
    target += normal_lpdf(mean_ld | corrected_ld, sem_ld) / n_bins;
}}

generated quantities {{
    vector[num_windows] log_lik;
    for (w in 1:num_windows) {{
        log_lik[w] = normal_lpdf(pi_array[w] | E_pi, sigma_div)
                   + normal_lpdf(to_vector(ld_mat[w]) | corrected_ld, sigma_ld) / n_bins;
    }}
}}
"""


class PiecewiseExponentialDemography:
    """
    Bayesian inference of Ne under a piecewise-exponential demographic model.

    Models Ne(t) = Ne_c * exp(-alpha * t) for t < t0, then Ne_a for t >= t0.
    alpha = 0 recovers a two-epoch constant model.

    Maintains two internally compiled Stan programs (approximate and GP-surrogate).

    Parameters
    ----------
    diversity : array-like, shape (num_windows,)
    ld : array-like, shape (num_windows, num_bins)
    mutation_rate : float
    recombination_rate : float
    num_samples : int
    left_bins, right_bins : array-like, shape (num_bins,)
    sequence_length : float or None
    num_workers : int
    hsgp_c : float
    hsgp_m : int
    n_quad : int
    prior : str or None
    extra_parameters : str
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
        sequence_length: Optional[float] = None,
        num_workers: int = 8,
        hsgp_c: float = 1.5,
        hsgp_m: int = 10,
        n_quad: int = _DEFAULT_N_QUAD,
        prior: Optional[str] = None,
        extra_parameters: str = "",
    ):
        self._diversity = np.asarray(diversity, dtype=float)
        self._ld = np.asarray(ld, dtype=float)
        self._mutation_rate = float(mutation_rate)
        self._recombination_rate = float(recombination_rate)
        self._sequence_length = (
            float(sequence_length) if sequence_length is not None else None
        )
        self._num_samples = int(num_samples)
        self._left_bins = np.asarray(left_bins, dtype=float)
        self._right_bins = np.asarray(right_bins, dtype=float)
        self._num_workers = int(num_workers)
        self._hsgp_c = float(hsgp_c)
        self._hsgp_m = int(hsgp_m)

        self._prior = (
            _default_prior(self._diversity, self._mutation_rate)
            if prior is None
            else prior
        )

        gl_nodes, gl_weights = np.polynomial.legendre.leggauss(n_quad)
        self._gl_nodes = gl_nodes
        self._gl_weights = gl_weights

        self._eval_points: list[dict] = []

        self._approx_model = self._compile_approx(self._prior, extra_parameters)
        self._surrogate_model = self._compile_surrogate(self._prior, extra_parameters)

    # ──────────────────────────────────────────────────────────────────────
    # Compilation helpers
    # ──────────────────────────────────────────────────────────────────────

    def _compile_approx(self, prior: str, extra_parameters: str):
        import cmdstanpy

        shared_fn = (_STAN_DIR / "functions" / "shared.stan").read_text()
        model_fn = (_STAN_DIR / "functions" / "piecewise_exponential.stan").read_text()
        code = _APPROX_TEMPLATE.format(
            shared_functions=shared_fn,
            model_functions=model_fn,
            common_data=_COMMON_DATA,
            common_transformed_data=_COMMON_TRANSFORMED_DATA,
            map_rect_transformed_data=_MAP_RECT_TRANSFORMED_DATA,
            extra_parameters=extra_parameters,
            prior_block=prior,
        )
        tmpdir = pathlib.Path(tempfile.mkdtemp())
        (tmpdir / "piecewise_exponential_approx.stan").write_text(code)
        return cmdstanpy.CmdStanModel(
            stan_file=str(tmpdir / "piecewise_exponential_approx.stan"),
            **_THREADS_OPTS,
        )

    def _compile_surrogate(self, prior: str, extra_parameters: str):
        import cmdstanpy

        gp_fn = (_STAN_DIR / "gpbasisfun_functions.stan").read_text()
        shared_fn = (_STAN_DIR / "functions" / "shared.stan").read_text()
        model_fn = (_STAN_DIR / "functions" / "piecewise_exponential.stan").read_text()
        code = _SURROGATE_TEMPLATE.format(
            gp_functions=gp_fn,
            shared_functions=shared_fn,
            model_functions=model_fn,
            common_data=_COMMON_DATA,
            common_transformed_data=_COMMON_TRANSFORMED_DATA,
            map_rect_transformed_data=_MAP_RECT_TRANSFORMED_DATA,
            gp_surrogate_data=_GP_SURROGATE_DATA,
            gp_surrogate_transformed_data=_GP_SURROGATE_TRANSFORMED_DATA,
            gp_surrogate_params=_GP_SURROGATE_PARAMS,
            gp_surrogate_transformed_params=_GP_SURROGATE_TRANSFORMED_PARAMS,
            gp_surrogate_model=_GP_SURROGATE_MODEL,
            extra_parameters=extra_parameters,
            prior_block=prior,
        )
        tmpdir = pathlib.Path(tempfile.mkdtemp())
        (tmpdir / "piecewise_exponential_surrogate.stan").write_text(code)
        return cmdstanpy.CmdStanModel(
            stan_file=str(tmpdir / "piecewise_exponential_surrogate.stan"),
            **_THREADS_OPTS,
        )

    # ──────────────────────────────────────────────────────────────────────
    # Stan data dicts
    # ──────────────────────────────────────────────────────────────────────

    def _base_stan_data(self) -> dict:
        return {
            "n_bins": int(len(self._left_bins)),
            "num_windows": int(len(self._diversity)),
            "left_bins": self._left_bins,
            "right_bins": self._right_bins,
            "mutation_rate": self._mutation_rate,
            "sample_size": self._num_samples,
            "pi_array": self._diversity,
            "ld_mat": self._ld,
            "n_quad": len(self._gl_nodes),
            "gl_nodes": self._gl_nodes,
            "gl_weights": self._gl_weights,
        }

    def _surrogate_stan_data(self) -> dict:
        pts = self._eval_points
        data = self._base_stan_data()
        data.update(
            {
                "n_eval": int(len(pts)),
                "eval_rel_bias": np.array([p["rel_bias"] for p in pts]),
                "eval_eps_rel": np.array([p["eps_rel"] for p in pts]),
                "hsgp_c": self._hsgp_c,
                "hsgp_m": self._hsgp_m,
            }
        )
        return data

    def _active_data(self) -> dict:
        return (
            self._surrogate_stan_data() if self._eval_points else self._base_stan_data()
        )

    def _active_model(self):
        return self._surrogate_model if self._eval_points else self._approx_model

    # ──────────────────────────────────────────────────────────────────────
    # Eval-point management
    # ──────────────────────────────────────────────────────────────────────

    @property
    def eval_points(self) -> list[dict]:
        return list(self._eval_points)

    def add_eval_points(self, points: list[dict]) -> None:
        """
        Inject pre-computed MC evaluation points.

        Each dict must have keys: ``rel_bias`` (1-D, length n_bins),
        ``eps_rel`` (1-D, length n_bins).
        """
        required = {"rel_bias", "eps_rel"}
        for p in points:
            if not required.issubset(p):
                raise ValueError(
                    f"Each eval point must have keys {required}; got {set(p)}"
                )
        self._eval_points.extend(points)

    # ──────────────────────────────────────────────────────────────────────
    # Active learning
    # ──────────────────────────────────────────────────────────────────────

    def surrogate_active_learning(
        self,
        points_per_iter: int = 50,
        max_tolerance: float = 0.01,
        max_replicates: int = 512,
        nuts_warmup: int = 500,
        seed: Optional[int] = None,
        progress_bar: bool = True,
    ):
        """
        Run one round of surrogate active learning.

        Returns
        -------
        cmdstanpy.CmdStanPathfinder
        """
        if self._sequence_length is None:
            raise ValueError(
                "sequence_length must be provided at initialisation to use "
                "surrogate_active_learning."
            )

        from tqdm.auto import tqdm

        from .. import deterministic as det
        from .. import montecarlo as mc2

        rng = np.random.default_rng(seed)

        batch_size = self._num_workers

        def _mc_eval(
            ne_c: float,
            ne_a: float,
            t0: float,
            alpha: float,
            mc_seed: int,
            outer,
        ) -> dict:
            _, det_ld_raw = det.expected_piecewise_exponential(
                ne_c,
                ne_a,
                t0,
                alpha,
                self._left_bins,
                self._right_bins,
                self._mutation_rate,
                sample_size=self._num_samples,
                ploidy=2,
            )
            det_ld = np.asarray(det_ld_raw)

            seed_rng = np.random.default_rng(mc_seed)
            all_mc_ld: list[np.ndarray] = []
            while True:
                _, mc_batch_raw = mc2.expected_piecewise_exponential(
                    ne_c,
                    ne_a,
                    t0,
                    alpha,
                    self._left_bins,
                    self._right_bins,
                    self._mutation_rate,
                    self._recombination_rate,
                    self._sequence_length,
                    self._num_samples,
                    random_seed=int(seed_rng.integers(2**31)),
                    num_replicates=batch_size,
                    ploidy=2,
                    num_workers=self._num_workers,
                )
                all_mc_ld.append(np.asarray(mc_batch_raw))
                mc_ld_reps = np.concatenate(all_mc_ld, axis=0)
                N = mc_ld_reps.shape[0]
                mc_ld_rel = mc_ld_reps / det_ld - 1.0
                rel_bias = mc_ld_rel.mean(axis=0)
                eps_rel = mc_ld_rel.std(axis=0, ddof=1) / np.sqrt(N)
                outer.set_postfix(
                    Ne_c=f"{ne_c:,.0f}", rep=N, max_se=f"{eps_rel.max():.4f}"
                )
                if eps_rel.max() <= max_tolerance or N >= max_replicates:
                    break
            return {"rel_bias": rel_bias, "eps_rel": eps_rel}

        # Anchor Pathfinder at the MAP of whichever model is active.
        try:
            active_map = self._active_model().optimize(
                data=self._active_data(),
                seed=int(rng.integers(10_000)),
                show_console=False,
            )
            map_inits = {
                "log_Ne_c": float(np.log(float(active_map.stan_variable("Ne_c")))),
                "log_Ne_a": float(np.log(float(active_map.stan_variable("Ne_a")))),
                "log_t0": float(np.log(float(active_map.stan_variable("t0")))),
                "log_fold_change": float(active_map.stan_variable("log_fold_change")),
            }
            if self._eval_points:
                map_inits.update(
                    {
                        "gp_rho_r": float(active_map.stan_variable("gp_rho_r")),
                        "gp_alpha": float(active_map.stan_variable("gp_alpha")),
                        "beta_r": np.asarray(
                            active_map.stan_variable("beta_r")
                        ).tolist(),
                    }
                )
        except RuntimeError:
            warnings.warn(
                "Surrogate MAP failed; Pathfinder will use default inits.",
                UserWarning,
                stacklevel=2,
            )
            map_inits = None

        pf_seed = int(rng.integers(10_000))
        pf_kwargs = dict(
            data=self._active_data(),
            draws=points_per_iter,
            seed=pf_seed,
            num_threads=self._num_workers,
            show_console=False,
        )
        if map_inits is not None:
            pf_kwargs["inits"] = map_inits
        try:
            pf = self._active_model().pathfinder(**pf_kwargs)
        except RuntimeError:
            warnings.warn(
                "Pathfinder failed with MAP inits; retrying with defaults.",
                UserWarning,
                stacklevel=2,
            )
            pf_kwargs.pop("inits", None)
            pf_kwargs["seed"] = pf_seed + 1
            pf = self._active_model().pathfinder(**pf_kwargs)

        # Short NUTS chain initialized from Pathfinder draws.
        n_chains = 4
        iter_sampling = max(points_per_iter // n_chains, 10)
        fit = self._active_model().sample(
            data=self._active_data(),
            chains=n_chains,
            iter_warmup=nuts_warmup,
            iter_sampling=iter_sampling,
            inits=pf.create_inits(chains=n_chains),
            seed=int(rng.integers(10_000)),
            threads_per_chain=self._num_workers,
            show_console=False,
        )

        ne_c_draws = np.asarray(fit.stan_variable("Ne_c"))[:points_per_iter]
        ne_a_draws = np.asarray(fit.stan_variable("Ne_a"))[:points_per_iter]
        t0_draws = np.asarray(fit.stan_variable("t0"))[:points_per_iter]
        alpha_draws = np.asarray(fit.stan_variable("alpha"))[:points_per_iter]
        print(
            f"[NUTS n_eval={len(self._eval_points)}]  "
            f"Ne_c={ne_c_draws.mean():,.0f}  Ne_a={ne_a_draws.mean():,.0f}  "
            f"t0={t0_draws.mean():.1f}  alpha={alpha_draws.mean():.4f}"
        )

        iterator = tqdm(
            enumerate(zip(ne_c_draws, ne_a_draws, t0_draws, alpha_draws)),
            total=points_per_iter,
            desc="Active learning",
            disable=not progress_bar,
        )
        for i, (ne_c, ne_a, t0, alpha) in iterator:
            mc_seed = int(rng.integers(2**31))
            self._eval_points.append(
                _mc_eval(
                    float(ne_c),
                    float(ne_a),
                    float(t0),
                    float(alpha),
                    mc_seed,
                    iterator,
                )
            )

        return fit

    def learn_surrogate_likelihood(
        self,
        n_map_iterations: int = 5,
        n_nuts_samples: int = 5,
        n_map_starts: int = 4,
        nuts_warmup: int = 500,
        max_tolerance: float = 0.01,
        max_replicates: int = 512,
        seed: Optional[int] = None,
        progress_bar: bool = True,
    ):
        """
        Learn the surrogate likelihood via MAP warm-up then NUTS sampling.

        Phase 1: ``n_map_iterations`` rounds of MAP (best of
        ``n_map_starts`` restarts), each evaluated and appended to the
        synthetic dataset.

        Phase 2: short NUTS chain (Pathfinder-initialised) producing
        ``n_nuts_samples`` draws, each evaluated and appended.

        Returns
        -------
        cmdstanpy.CmdStanMCMC
        """
        if self._sequence_length is None:
            raise ValueError(
                "sequence_length must be provided at initialisation to use "
                "learn_surrogate_likelihood."
            )

        from tqdm.auto import tqdm

        from .. import deterministic as det
        from .. import montecarlo as mc2

        rng = np.random.default_rng(seed)
        batch_size = self._num_workers

        def _mc_eval(
            ne_c: float,
            ne_a: float,
            t0: float,
            alpha: float,
            mc_seed: int,
            outer,
        ) -> dict:
            _, det_ld_raw = det.expected_piecewise_exponential(
                ne_c,
                ne_a,
                t0,
                alpha,
                self._left_bins,
                self._right_bins,
                self._mutation_rate,
                sample_size=self._num_samples,
                ploidy=2,
            )
            det_ld = np.asarray(det_ld_raw)

            seed_rng = np.random.default_rng(mc_seed)
            all_mc_ld: list[np.ndarray] = []
            while True:
                _, mc_batch_raw = mc2.expected_piecewise_exponential(
                    ne_c,
                    ne_a,
                    t0,
                    alpha,
                    self._left_bins,
                    self._right_bins,
                    self._mutation_rate,
                    self._recombination_rate,
                    self._sequence_length,
                    self._num_samples,
                    random_seed=int(seed_rng.integers(2**31)),
                    num_replicates=batch_size,
                    ploidy=2,
                    num_workers=self._num_workers,
                )
                all_mc_ld.append(np.asarray(mc_batch_raw))
                mc_ld_reps = np.concatenate(all_mc_ld, axis=0)
                N = mc_ld_reps.shape[0]
                mc_ld_rel = mc_ld_reps / det_ld - 1.0
                rel_bias = mc_ld_rel.mean(axis=0)
                eps_rel = mc_ld_rel.std(axis=0, ddof=1) / np.sqrt(N)
                outer.set_postfix(
                    Ne_c=f"{ne_c:,.0f}", rep=N, max_se=f"{eps_rel.max():.4f}"
                )
                if eps_rel.max() <= max_tolerance or N >= max_replicates:
                    break
            return {"rel_bias": rel_bias, "eps_rel": eps_rel}

        # Phase 1: MAP iterations
        iterator = tqdm(
            range(n_map_iterations),
            desc="MAP active learning",
            disable=not progress_bar,
        )
        for iteration in iterator:
            best_map = None
            best_lp = -np.inf
            for _ in range(n_map_starts):
                try:
                    m = self._active_model().optimize(
                        data=self._active_data(),
                        seed=int(rng.integers(10_000)),
                        show_console=False,
                    )
                    lp = float(m.optimized_params_dict["lp__"])
                    if lp > best_lp:
                        best_lp = lp
                        best_map = m
                except RuntimeError:
                    continue

            if best_map is None:
                warnings.warn(
                    f"All MAP starts failed at iteration {iteration}; skipping.",
                    UserWarning,
                    stacklevel=2,
                )
                continue

            ne_c = float(best_map.stan_variable("Ne_c"))
            ne_a = float(best_map.stan_variable("Ne_a"))
            t0 = float(best_map.stan_variable("t0"))
            alpha = float(best_map.stan_variable("alpha"))
            print(
                f"[MAP {iteration + 1}/{n_map_iterations} "
                f"n_eval={len(self._eval_points)}]  "
                f"Ne_c={ne_c:,.0f}  Ne_a={ne_a:,.0f}  "
                f"t0={t0:.1f}  alpha={alpha:.4f}"
            )
            mc_seed = int(rng.integers(2**31))
            self._eval_points.append(_mc_eval(ne_c, ne_a, t0, alpha, mc_seed, iterator))

        # Phase 2: NUTS samples initialised from Pathfinder
        pf = self._active_model().pathfinder(
            data=self._active_data(),
            seed=int(rng.integers(10_000)),
            num_threads=self._num_workers,
            show_console=False,
        )

        n_chains = 4
        iter_sampling = max(n_nuts_samples // n_chains, 10)
        fit = self._active_model().sample(
            data=self._active_data(),
            chains=n_chains,
            iter_warmup=nuts_warmup,
            iter_sampling=iter_sampling,
            inits=pf.create_inits(chains=n_chains),
            seed=int(rng.integers(10_000)),
            threads_per_chain=self._num_workers,
            show_console=False,
        )

        ne_c_draws = np.asarray(fit.stan_variable("Ne_c"))[:n_nuts_samples]
        ne_a_draws = np.asarray(fit.stan_variable("Ne_a"))[:n_nuts_samples]
        t0_draws = np.asarray(fit.stan_variable("t0"))[:n_nuts_samples]
        alpha_draws = np.asarray(fit.stan_variable("alpha"))[:n_nuts_samples]
        print(
            f"[NUTS n_eval={len(self._eval_points)}]  "
            f"Ne_c={ne_c_draws.mean():,.0f}  Ne_a={ne_a_draws.mean():,.0f}  "
            f"t0={t0_draws.mean():.1f}  alpha={alpha_draws.mean():.4f}"
        )

        iterator = tqdm(
            enumerate(zip(ne_c_draws, ne_a_draws, t0_draws, alpha_draws)),
            total=len(ne_c_draws),
            desc="NUTS active learning",
            disable=not progress_bar,
        )
        for i, (ne_c, ne_a, t0, alpha) in iterator:
            mc_seed = int(rng.integers(2**31))
            self._eval_points.append(
                _mc_eval(
                    float(ne_c),
                    float(ne_a),
                    float(t0),
                    float(alpha),
                    mc_seed,
                    iterator,
                )
            )

        return fit

    # ──────────────────────────────────────────────────────────────────────
    # Inference interface
    # ──────────────────────────────────────────────────────────────────────

    def _extract_approx(self, fit) -> dict:
        return {
            "Ne_c": float(_stan_vector(fit.stan_variable("Ne_c"))[0]),
            "Ne_a": float(_stan_vector(fit.stan_variable("Ne_a"))[0]),
            "t0": float(_stan_vector(fit.stan_variable("t0"))[0]),
            "alpha": float(fit.stan_variable("alpha")),
            "log_fold_change": float(fit.stan_variable("log_fold_change")),
            "E_pi": float(fit.stan_variable("E_pi")),
            "approx_ld": np.asarray(fit.stan_variable("approx_ld")),
            "log_lik": np.asarray(fit.stan_variable("log_lik")),
        }

    def _extract_surrogate(self, fit) -> dict:
        result = self._extract_approx(fit)
        result.update(
            {
                "gp_bias_ld": np.asarray(fit.stan_variable("gp_bias_ld")),
                "corrected_ld": np.asarray(fit.stan_variable("corrected_ld")),
                "gp_rho_r": float(fit.stan_variable("gp_rho_r")),
                "gp_alpha": float(fit.stan_variable("gp_alpha")),
            }
        )
        return result

    def optimize(self, **kwargs) -> dict:
        """
        Compute the MAP estimate.

        Returns
        -------
        dict
            Always: ``Ne_c``, ``Ne_a``, ``t0``, ``alpha``,
            ``log_fold_change``, ``E_pi``, ``approx_ld``, ``log_lik``.
            With surrogate: additionally ``gp_bias_ld``, ``corrected_ld``,
            ``gp_rho_r``, ``gp_alpha``.
        """
        fit = self._active_model().optimize(data=self._active_data(), **kwargs)
        return (
            self._extract_surrogate(fit)
            if self._eval_points
            else self._extract_approx(fit)
        )

    def pathfinder(self, **kwargs):
        """Run Pathfinder variational inference."""
        kwargs.setdefault("num_threads", self._num_workers)
        return self._active_model().pathfinder(data=self._active_data(), **kwargs)

    def sample(self, **kwargs):
        """Run NUTS sampling."""
        if not self._eval_points:
            warnings.warn(
                "Using approximate LD predictions. "
                "Check you're in a regime where bias is neglectable.",
                UserWarning,
                stacklevel=2,
            )
        kwargs.setdefault("threads_per_chain", self._num_workers)
        return self._active_model().sample(data=self._active_data(), **kwargs)
