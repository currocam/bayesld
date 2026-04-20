"""
Piecewise-constant-Ne demographic inference model.

``PiecewiseConstantDemography`` internally maintains two Stan programs — a
fast deterministic approximation and a GP-surrogate that learns the relative
LD bias of that approximation — and switches between them automatically
depending on whether a synthetic MC evaluation dataset has been accumulated
via ``surrogate_active_learning``.

The GP surrogate fits the *relative* bias in LD predictions per bin::

    mc_ld(theta, r) / det_ld(theta, r) - 1  ~  GP(r)

where ``r`` is the bin midpoint (standardised).  The corrected LD is then
``det_ld * (1 + GP)``, which is dimensionless and scale-agnostic across bins.

Time-epoch boundaries can either be inferred jointly with effective population
sizes (the default) or fixed a-priori by passing ``t_boundaries``.
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

_DEFAULT_N_EPOCHS = 3
_DEFAULT_N_QUAD = 16
_DEFAULT_LOG_T_PRIOR_SIGMA = 1.0

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
    int<lower=2> n_epochs;
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
"""

# Substitution fragments for free vs. fixed time boundaries.
_FREE_T_EXTRA_DATA = ""
_FREE_T_PARAMS = "    ordered[n_epochs - 1] log_t_boundaries;\n"
_FREE_T_TRANSFORMED = (
    "    vector<lower=0>[n_epochs - 1] t_boundaries = exp(log_t_boundaries);\n"
)

_FIXED_T_EXTRA_DATA = "    vector<lower=0>[n_epochs - 1] t_boundaries;\n"
_FIXED_T_PARAMS = ""
_FIXED_T_TRANSFORMED = ""

# ──────────────────────────────────────────────────────────────────────────
# Approximate (deterministic) model template
# ──────────────────────────────────────────────────────────────────────────

_APPROX_TEMPLATE = """\
functions {{
// ---- shared.stan ----
{shared_functions}
// ---- piecewise_constant.stan ----
{model_functions}
}}

{common_data}{t_extra_data}}}

{common_transformed_data}}}

parameters {{
    vector[n_epochs] log_Ne_values;
{t_params}{extra_parameters}
}}

transformed parameters {{
    vector<lower=0>[n_epochs] Ne_values = exp(log_Ne_values);
{t_transformed}}}

model {{
    // --- user prior ---
{prior_block}
    real mu_div_val = mu_div_piecewise_constant(n_epochs, Ne_values, t_boundaries, mutation_rate);
    vector[n_bins] mu_ld_val = correct_ld_finite_sample(
        mu_ld_piecewise_constant(n_epochs, Ne_values, t_boundaries,
                                  left_bins, right_bins, n_quad, gl_nodes, gl_weights),
        sample_size
    );
    mean_div ~ normal(mu_div_val, sem_div);
    target += normal_lpdf(mean_ld | mu_ld_val, sem_ld) / n_bins;
}}

generated quantities {{
    real E_pi = mu_div_piecewise_constant(n_epochs, Ne_values, t_boundaries, mutation_rate);
    vector<lower=0>[n_bins] approx_ld = correct_ld_finite_sample(
        mu_ld_piecewise_constant(n_epochs, Ne_values, t_boundaries,
                                  left_bins, right_bins, n_quad, gl_nodes, gl_weights),
        sample_size
    );
    vector[num_windows] log_lik;
    for (w in 1:num_windows) {{
        log_lik[w] = normal_lpdf(pi_array[w] | E_pi, sigma_div)
                   + normal_lpdf(to_vector(ld_mat[w]) | approx_ld, sigma_ld) / n_bins;
    }}
}}
"""

# ──────────────────────────────────────────────────────────────────────────
# GP-surrogate model template — LD-bias formulation
#
# GP input: standardised bin midpoint r_std (computed in transformed data)
# GP target: eval_rel_bias[i,b] = mc_ld[i,b] / det_ld[i,b] - 1
# Correction: corrected_ld = approx_ld .* (1 + gp_bias_ld)
# ──────────────────────────────────────────────────────────────────────────

_SURROGATE_TEMPLATE = """\
functions {{
// ---- gpbasisfun_functions.stan ----
{gp_functions}
// ---- shared.stan ----
{shared_functions}
// ---- piecewise_constant.stan ----
{model_functions}
}}

{common_data}{t_extra_data}
{gp_surrogate_data}}}

{common_transformed_data}
{gp_surrogate_transformed_data}}}

parameters {{
    vector[n_epochs] log_Ne_values;
{t_params}
{gp_surrogate_params}
{extra_parameters}
}}

transformed parameters {{
    vector<lower=0>[n_epochs] Ne_values = exp(log_Ne_values);
{t_transformed}
{gp_surrogate_transformed_params}
}}

model {{
{gp_surrogate_model}
    // --- user prior ---
{prior_block}
    real mu_div_val = mu_div_piecewise_constant(n_epochs, Ne_values, t_boundaries, mutation_rate);
    vector[n_bins] approx_ld_val = correct_ld_finite_sample(
        mu_ld_piecewise_constant(n_epochs, Ne_values, t_boundaries,
                                  left_bins, right_bins, n_quad, gl_nodes, gl_weights),
        sample_size
    );
    vector[n_bins] corrected_ld = approx_ld_val .* (1.0 + gp_bias_ld);
    mean_div ~ normal(mu_div_val, sem_div);
    target += normal_lpdf(mean_ld | corrected_ld, sem_ld) / n_bins;
}}

generated quantities {{
    real E_pi = mu_div_piecewise_constant(n_epochs, Ne_values, t_boundaries, mutation_rate);
    vector<lower=0>[n_bins] approx_ld = correct_ld_finite_sample(
        mu_ld_piecewise_constant(n_epochs, Ne_values, t_boundaries,
                                  left_bins, right_bins, n_quad, gl_nodes, gl_weights),
        sample_size
    );
    vector[n_bins] corrected_ld_gq = approx_ld .* (1.0 + gp_bias_ld);
    vector[num_windows] log_lik;
    for (w in 1:num_windows) {{
        log_lik[w] = normal_lpdf(pi_array[w] | E_pi, sigma_div)
                   + normal_lpdf(to_vector(ld_mat[w]) | corrected_ld_gq, sigma_ld) / n_bins;
    }}
}}
"""


def _default_log_t_prior_mu(n_epochs: int) -> np.ndarray:
    return np.linspace(np.log(10.0), np.log(200.0), n_epochs - 1)


def _default_prior(
    n_epochs: int,
    fixed_t: bool,
    diversity: np.ndarray,
    mutation_rate: float,
) -> str:
    ne_hat = float(np.mean(diversity)) / (4.0 * mutation_rate)
    log_ne_mu = float(np.log(ne_hat))
    lines = [f"    log_Ne_values ~ normal({log_ne_mu:.4f}, 1.0);"]
    if not fixed_t:
        for i, mu in enumerate(_default_log_t_prior_mu(n_epochs), start=1):
            lines.append(
                f"    log_t_boundaries[{i}] ~ normal({mu:.4f}, {_DEFAULT_LOG_T_PRIOR_SIGMA});"
            )
    return "\n".join(lines)


class PiecewiseConstantDemography:
    """
    Bayesian inference of Ne under a piecewise-constant demographic model.

    Maintains two internally compiled Stan programs:

    * **Approximate model** — uses a deterministic closed-form LD approximation.
      Active when the synthetic dataset (``eval_points``) is empty.
    * **GP-surrogate model** — augments the approximate model with a 1-D Hilbert-space
      GP that corrects the LD predictions multiplicatively per bin.
      Active once ``eval_points`` is non-empty.

    Parameters
    ----------
    diversity : array-like, shape (num_windows,)
    ld : array-like, shape (num_windows, num_bins)
    mutation_rate : float
    recombination_rate : float
    num_samples : int
    left_bins, right_bins : array-like, shape (num_bins,)
    n_epochs : int
    t_boundaries : array-like of shape (n_epochs - 1,) or None
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
        n_epochs: int = _DEFAULT_N_EPOCHS,
        t_boundaries: Optional[np.ndarray] = None,
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
        self._n_epochs = int(n_epochs)
        self._num_workers = int(num_workers)
        self._hsgp_c = float(hsgp_c)
        self._hsgp_m = int(hsgp_m)

        if t_boundaries is not None:
            t_boundaries = np.asarray(t_boundaries, dtype=float)
            if t_boundaries.shape != (n_epochs - 1,):
                raise ValueError(
                    f"t_boundaries must have shape ({n_epochs - 1},) for "
                    f"n_epochs={n_epochs}; got {t_boundaries.shape}"
                )
            if not np.all(np.diff(t_boundaries) > 0):
                raise ValueError("t_boundaries must be strictly increasing.")
            self._fixed_t = t_boundaries
        else:
            self._fixed_t = None

        self._prior = (
            _default_prior(
                self._n_epochs,
                fixed_t=self._fixed_t is not None,
                diversity=self._diversity,
                mutation_rate=self._mutation_rate,
            )
            if prior is None
            else prior
        )

        gl_nodes, gl_weights = np.polynomial.legendre.leggauss(n_quad)
        self._gl_nodes = gl_nodes
        self._gl_weights = gl_weights

        n = len(self._diversity)
        self._mean_ld = np.mean(self._ld, axis=0)
        self._sem_ld = np.std(self._ld, axis=0, ddof=1) / np.sqrt(n)

        self._eval_points: list[dict] = []

        self._approx_model = self._compile_approx(self._prior, extra_parameters)
        self._surrogate_model = self._compile_surrogate(self._prior, extra_parameters)

    # ──────────────────────────────────────────────────────────────────────
    # Compilation helpers
    # ──────────────────────────────────────────────────────────────────────

    def _t_fragments(self) -> dict:
        if self._fixed_t is None:
            return dict(
                t_extra_data=_FREE_T_EXTRA_DATA,
                t_params=_FREE_T_PARAMS,
                t_transformed=_FREE_T_TRANSFORMED,
            )
        return dict(
            t_extra_data=_FIXED_T_EXTRA_DATA,
            t_params=_FIXED_T_PARAMS,
            t_transformed=_FIXED_T_TRANSFORMED,
        )

    def _compile_approx(self, prior: str, extra_parameters: str):
        import cmdstanpy

        shared_fn = (_STAN_DIR / "functions" / "shared.stan").read_text()
        model_fn = (_STAN_DIR / "functions" / "piecewise_constant.stan").read_text()
        frags = self._t_fragments()
        code = _APPROX_TEMPLATE.format(
            shared_functions=shared_fn,
            model_functions=model_fn,
            common_data=_COMMON_DATA,
            common_transformed_data=_COMMON_TRANSFORMED_DATA,
            extra_parameters=extra_parameters,
            prior_block=prior,
            **frags,
        )
        tmpdir = pathlib.Path(tempfile.mkdtemp())
        (tmpdir / "piecewise_constant_approx.stan").write_text(code)
        return cmdstanpy.CmdStanModel(
            stan_file=str(tmpdir / "piecewise_constant_approx.stan"),
            **_THREADS_OPTS,
        )

    def _compile_surrogate(self, prior: str, extra_parameters: str):
        import cmdstanpy

        gp_fn = (_STAN_DIR / "gpbasisfun_functions.stan").read_text()
        shared_fn = (_STAN_DIR / "functions" / "shared.stan").read_text()
        model_fn = (_STAN_DIR / "functions" / "piecewise_constant.stan").read_text()
        frags = self._t_fragments()
        code = _SURROGATE_TEMPLATE.format(
            gp_functions=gp_fn,
            shared_functions=shared_fn,
            model_functions=model_fn,
            common_data=_COMMON_DATA,
            common_transformed_data=_COMMON_TRANSFORMED_DATA,
            gp_surrogate_data=_GP_SURROGATE_DATA,
            gp_surrogate_transformed_data=_GP_SURROGATE_TRANSFORMED_DATA,
            gp_surrogate_params=_GP_SURROGATE_PARAMS,
            gp_surrogate_transformed_params=_GP_SURROGATE_TRANSFORMED_PARAMS,
            gp_surrogate_model=_GP_SURROGATE_MODEL,
            extra_parameters=extra_parameters,
            prior_block=prior,
            **frags,
        )
        tmpdir = pathlib.Path(tempfile.mkdtemp())
        (tmpdir / "piecewise_constant_surrogate.stan").write_text(code)
        return cmdstanpy.CmdStanModel(
            stan_file=str(tmpdir / "piecewise_constant_surrogate.stan"),
            **_THREADS_OPTS,
        )

    # ──────────────────────────────────────────────────────────────────────
    # Stan data dicts
    # ──────────────────────────────────────────────────────────────────────

    def _base_stan_data(self) -> dict:
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
        }
        if self._fixed_t is not None:
            data["t_boundaries"] = self._fixed_t
        return data

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
        seed: Optional[int] = None,
        progress_bar: bool = True,
    ):
        """
        Run one round of surrogate active learning.

        Draws ``points_per_iter`` parameter vectors via Pathfinder, evaluates
        the Monte Carlo LD at each draw, computes the relative LD bias and its
        SE per bin, and appends the results to the internal dataset.

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
        from .. import montecarlo2 as mc2

        rng = np.random.default_rng(seed)

        batch_size = self._num_workers

        def _mc_eval(
            ne_values: np.ndarray, t_bnd: np.ndarray, mc_seed: int, outer
        ) -> dict:
            _, det_ld_raw = det.expected_piecewise_constant(
                ne_values,
                t_bnd,
                self._left_bins,
                self._right_bins,
                self._mutation_rate,
                sample_size=self._num_samples,
                ploidy=2,
            )
            det_ld = np.asarray(det_ld_raw)

            _ne_str = "/".join(f"{v:,.0f}" for v in ne_values)
            seed_rng = np.random.default_rng(mc_seed)
            all_mc_ld: list[np.ndarray] = []
            while True:
                _, mc_batch_raw = mc2.expected_piecewise_constant(
                    ne_values,
                    t_bnd,
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
                outer.set_postfix(Ne=_ne_str, rep=N, max_se=f"{eps_rel.max():.4f}")
                if eps_rel.max() <= max_tolerance or N >= max_replicates:
                    break
            return {"rel_bias": rel_bias, "eps_rel": eps_rel}

        # Anchor Pathfinder at the MAP of whichever model is active.
        active_map = self._active_model().optimize(
            data=self._active_data(),
            seed=int(rng.integers(10_000)),
            show_console=False,
        )
        map_inits = {"log_Ne_values": np.log(active_map.stan_variable("Ne_values"))}
        if self._fixed_t is None:
            map_inits["log_t_boundaries"] = np.log(
                _stan_vector(active_map.stan_variable("t_boundaries"))
            )
        if self._eval_points:
            map_inits.update(
                {
                    "gp_rho_r": float(active_map.stan_variable("gp_rho_r")),
                    "gp_alpha": float(active_map.stan_variable("gp_alpha")),
                    "beta_r": np.asarray(active_map.stan_variable("beta_r")).tolist(),
                }
            )

        pf_seed = int(rng.integers(10_000))
        pf = self._active_model().pathfinder(
            data=self._active_data(),
            draws=points_per_iter,
            seed=pf_seed,
            num_threads=self._num_workers,
            inits=map_inits,
            show_console=False,
        )

        ne_draws = _stan_draw_matrix(pf.stan_variable("Ne_values"), points_per_iter)[
            :points_per_iter
        ]
        if self._fixed_t is None:
            t_draws = _stan_draw_matrix(
                pf.stan_variable("t_boundaries"), points_per_iter
            )[:points_per_iter]
            _ne_str = "  ".join(
                f"Ne{i + 1}={v:,.0f}" for i, v in enumerate(ne_draws.mean(axis=0))
            )
            _t_str = "  ".join(
                f"t{i + 1}={v:.1f}" for i, v in enumerate(t_draws.mean(axis=0))
            )
            print(
                f"[Pathfinder n_eval={len(self._eval_points)}]  {_ne_str}  |  {_t_str}"
            )
        else:
            t_draws = np.tile(self._fixed_t, (points_per_iter, 1))
            _ne_str = "  ".join(
                f"Ne{i + 1}={v:,.0f}" for i, v in enumerate(ne_draws.mean(axis=0))
            )
            print(f"[Pathfinder n_eval={len(self._eval_points)}]  {_ne_str}  (t fixed)")

        iterator = tqdm(
            enumerate(zip(ne_draws, t_draws)),
            total=min(len(ne_draws), len(t_draws)),
            desc="Active learning",
            disable=not progress_bar,
        )
        for i, (ne, t_bnd) in iterator:
            mc_seed = int(rng.integers(2**31))
            self._eval_points.append(_mc_eval(ne, t_bnd, mc_seed, iterator))

        return pf

    # ──────────────────────────────────────────────────────────────────────
    # Inference interface
    # ──────────────────────────────────────────────────────────────────────

    def _extract_approx(self, fit) -> dict:
        result = {
            "Ne_values": _stan_vector(fit.stan_variable("Ne_values")),
            "E_pi": float(fit.stan_variable("E_pi")),
            "approx_ld": np.asarray(fit.stan_variable("approx_ld")),
            "log_lik": np.asarray(fit.stan_variable("log_lik")),
        }
        if self._fixed_t is None:
            result["t_boundaries"] = _stan_vector(fit.stan_variable("t_boundaries"))
        else:
            result["t_boundaries"] = self._fixed_t
        return result

    def _extract_surrogate(self, fit) -> dict:
        result = self._extract_approx(fit)
        result.update(
            {
                "gp_bias_ld": np.asarray(fit.stan_variable("gp_bias_ld")),
                "corrected_ld": np.asarray(fit.stan_variable("corrected_ld_gq")),
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
            Always: ``Ne_values``, ``t_boundaries``, ``E_pi``, ``approx_ld``,
            ``log_lik``.  With surrogate: additionally ``gp_bias_ld``,
            ``corrected_ld``, ``gp_rho_r``, ``gp_alpha``.
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
