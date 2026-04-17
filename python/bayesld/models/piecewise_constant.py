"""
Piecewise-constant-Ne demographic inference model.

``PiecewiseConstantDemography`` internally maintains two Stan programs — a
fast deterministic approximation and a GP-surrogate that learns the bias of
that approximation — and switches between them automatically depending on
whether a synthetic MC evaluation dataset has been accumulated via
``surrogate_active_learning``.

Time-epoch boundaries can either be inferred jointly with effective population
sizes (the default) or fixed a-priori by passing ``t_boundaries``.
"""

import pathlib
import tempfile
import warnings
from typing import Optional

import numpy as np

_STAN_DIR = pathlib.Path(__file__).resolve().parent.parent.parent.parent / "stan"
_THREADS_OPTS = {"cpp_options": {"STAN_THREADS": "true"}}

_DEFAULT_N_EPOCHS = 3
_DEFAULT_N_QUAD = 16
_DEFAULT_LOG_NE_PRIOR_MU = 11.5
_DEFAULT_LOG_NE_PRIOR_SIGMA = 3.0
_DEFAULT_LOG_T_PRIOR_SIGMA = 2.0

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
#
# Free: t_boundaries are inferred (ordered parameter).
# Fixed: t_boundaries are passed as data, not sampled.
#
# Used in both _APPROX_TEMPLATE and _SURROGATE_TEMPLATE via .format().

_FREE_T_EXTRA_DATA = ""
_FREE_T_PARAMS = "    ordered[n_epochs - 1] log_t_boundaries;\n"
_FREE_T_TRANSFORMED = (
    "    vector<lower=0>[n_epochs - 1] t_boundaries = exp(log_t_boundaries);\n"
)
_FREE_D_EXPR = "2 * n_epochs - 1"
_FREE_LOG_PARAMS_ASSEMBLY = """\
    for (i in 1:n_epochs)
        log_params_vec[i] = log_Ne_values[i];
    for (i in 1:(n_epochs - 1))
        log_params_vec[n_epochs + i] = log_t_boundaries[i];"""

_FIXED_T_EXTRA_DATA = "    vector<lower=0>[n_epochs - 1] t_boundaries;\n"
_FIXED_T_PARAMS = ""
_FIXED_T_TRANSFORMED = ""
_FIXED_D_EXPR = "n_epochs"
_FIXED_LOG_PARAMS_ASSEMBLY = """\
    for (i in 1:n_epochs)
        log_params_vec[i] = log_Ne_values[i];"""

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
# GP-surrogate model template
# ──────────────────────────────────────────────────────────────────────────
# {D_expr}             — Stan expression for dimensionality (substituted by Python)
# {log_params_assembly} — Stan code to fill log_params_vec from current params

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
    // ── GP surrogate evaluation dataset ────────────────────────────────────
    int<lower=2> n_eval;
    array[n_eval] vector[{D_expr}] eval_log_params;
    vector[n_eval] eval_loglik_det;
    vector[n_eval] eval_bc_loglik;
    vector<lower=0>[n_eval] eval_epsilon;
    real<lower=0> hsgp_c;
    int<lower=1>  hsgp_m;
}}

{common_transformed_data}
    // GP: observed biases
    int D = {D_expr};
    vector[n_eval] delta = eval_bc_loglik - eval_loglik_det;

    // Standardise GP inputs per dimension
    vector[D] x_mu = rep_vector(0.0, D);
    for (i in 1:n_eval)
        x_mu += eval_log_params[i];
    x_mu /= n_eval;

    vector[D] x_sig = rep_vector(0.0, D);
    for (i in 1:n_eval)
        x_sig += square(eval_log_params[i] - x_mu);
    x_sig = sqrt(x_sig / (n_eval - 1));

    array[D] vector[n_eval] xn_per_dim;
    for (d in 1:D)
        for (i in 1:n_eval)
            xn_per_dim[d][i] = (eval_log_params[i][d] - x_mu[d]) / x_sig[d];

    array[D] real L_hsgp;
    for (d in 1:D)
        L_hsgp[d] = hsgp_c * fmax(max(xn_per_dim[d]), -min(xn_per_dim[d]));

    array[D] matrix[n_eval, hsgp_m] PHI_eval;
    for (d in 1:D)
        PHI_eval[d] = PHI(n_eval, hsgp_m, L_hsgp[d], xn_per_dim[d]);
}}

parameters {{
    vector[n_epochs] log_Ne_values;
{t_params}
    array[{D_expr}] real<lower=0> gp_rho;
    real<lower=0> gp_alpha;
    array[{D_expr}] vector[hsgp_m] beta;
{extra_parameters}
}}

transformed parameters {{
    vector<lower=0>[n_epochs] Ne_values = exp(log_Ne_values);
{t_transformed}
    // Assemble current log-parameter vector (same order as eval_log_params)
    vector[D] log_params_vec;
{log_params_assembly}

    array[D] real xn_star;
    for (d in 1:D)
        xn_star[d] = (log_params_vec[d] - x_mu[d]) / x_sig[d];

    vector[n_eval] f_eval = rep_vector(0.0, n_eval);
    real gp_bias = 0.0;
    {{
        vector[hsgp_m] spd_d;
        vector[hsgp_m] phi_star_d;
        for (d in 1:D) {{
            spd_d = diagSPD_EQ(gp_alpha, gp_rho[d], L_hsgp[d], hsgp_m);
            f_eval += PHI_eval[d] * (spd_d .* beta[d]);
            for (m in 1:hsgp_m)
                phi_star_d[m] = sin(m * pi() / (2.0 * L_hsgp[d])
                                    * (xn_star[d] + L_hsgp[d])) / sqrt(L_hsgp[d]);
            gp_bias += dot_product(phi_star_d, spd_d .* beta[d]);
        }}
    }}
}}

model {{
    gp_rho   ~ inv_gamma(5, 5);
    gp_alpha ~ normal(0, 0.3);
    for (d in 1:D)
        beta[d] ~ std_normal();
    delta ~ normal(f_eval, eval_epsilon);

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
    target += gp_bias;
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
                   + normal_lpdf(to_vector(ld_mat[w]) | approx_ld, sigma_ld) / n_bins
                   + gp_bias / num_windows;
    }}
}}
"""


def _default_log_t_prior_mu(n_epochs: int) -> np.ndarray:
    """Evenly-spaced (in log-time) boundary priors between 10 and 200 generations."""
    return np.linspace(np.log(10.0), np.log(200.0), n_epochs - 1)


def _default_prior(n_epochs: int, fixed_t: bool) -> str:
    """Default independent normal priors for inferred log-parameters."""
    lines = [
        f"    log_Ne_values ~ normal({_DEFAULT_LOG_NE_PRIOR_MU}, {_DEFAULT_LOG_NE_PRIOR_SIGMA});"
    ]
    if not fixed_t:
        for i, mu in enumerate(_default_log_t_prior_mu(n_epochs), start=1):
            lines.append(
                f"    log_t_boundaries[{i}] ~ normal({mu}, {_DEFAULT_LOG_T_PRIOR_SIGMA});"
            )
    return "\n".join(lines)


def _stan_vector(value) -> np.ndarray:
    """
    Convert a Stan variable to a 1-D array.

    CmdStanPy currently may return scalar values for length-1 vectors in MLE
    outputs. This keeps the public API shape-stable for vector-valued
    parameters such as ``t_boundaries``.
    """
    arr = np.asarray(value)
    return (
        arr.reshape(
            1,
        )
        if arr.ndim == 0
        else arr
    )


def _stan_draw_matrix(value, n_draws: int) -> np.ndarray:
    """
    Convert a Stan draw container to a 2-D ``(draws, dims)`` matrix.

    When Stan dimension is 1, CmdStanPy may collapse results to 0-D or 1-D.
    This helper normalizes those cases so downstream code can always iterate
    over row vectors safely.
    """
    arr = np.asarray(value)
    if arr.ndim == 0:
        return arr.reshape(1, 1)
    if arr.ndim == 1:
        if arr.shape[0] == n_draws:
            return arr.reshape(-1, 1)
        return arr.reshape(1, -1)
    return arr


class PiecewiseConstantDemography:
    """
    Bayesian inference of effective population size under a piecewise-constant-Ne
    demographic model.

    Maintains two internally compiled Stan programs:

    * **Approximate model** — uses a deterministic closed-form LD approximation.
      Active when the synthetic dataset (``eval_points``) is empty.
    * **GP-surrogate model** — augments the approximate model with a Hilbert-space
      GP bias correction fitted jointly to Monte Carlo log-likelihood evaluations.
      Active once ``eval_points`` is non-empty.

    Parameters
    ----------
    diversity : array-like, shape (num_windows,)
        Per-window mean genetic diversity (pi).
    ld : array-like, shape (num_windows, num_bins)
        Per-window mean LD in each recombination-distance bin.
    mutation_rate : float
        Mutation rate per base pair per generation.
    recombination_rate : float
        Recombination rate per base pair per generation.
    num_samples : int
        Number of diploid individuals.
    left_bins : array-like, shape (num_bins,)
        Left edges of LD bins in Morgans.
    right_bins : array-like, shape (num_bins,)
        Right edges of LD bins in Morgans.
    n_epochs : int
        Number of demographic epochs (default 3 — two boundaries at ~10 and
        ~1 000 generations ago).
    t_boundaries : array-like of shape (n_epochs - 1,) or None
        Fixed time boundaries in generations.  When provided, the boundaries
        are not inferred; only ``Ne_values`` is sampled.  When ``None``
        (default), boundaries are inferred jointly with Ne values.
    sequence_length : float or None
        Genomic sequence length (bp) for Monte Carlo simulations.
        Required only when calling ``surrogate_active_learning``.
    num_workers : int
        Number of parallel workers / threads (default 8).
    hsgp_c : float
        HSGP boundary factor (≥1.2; default 1.5).
    hsgp_m : int
        Number of HSGP basis functions per dimension (default 20).
    n_quad : int
        Number of Gauss-Legendre quadrature nodes per bin (default 16).
    prior : str or None
        Stan statements for priors on ``log_Ne_values`` and (when inferred)
        ``log_t_boundaries``, plus any *extra_parameters*. If ``None``,
        defaults to independent normal priors on all inferred log-parameters.
    extra_parameters : str
        Additional Stan ``parameters`` block declarations for complex priors.
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
        hsgp_m: int = 20,
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

        # Fixed vs. free boundaries
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
            _default_prior(self._n_epochs, fixed_t=self._fixed_t is not None)
            if prior is None
            else prior
        )

        # Gauss-Legendre quadrature nodes and weights on [-1, 1]
        gl_nodes, gl_weights = np.polynomial.legendre.leggauss(n_quad)
        self._gl_nodes = gl_nodes
        self._gl_weights = gl_weights

        # LD summary statistics for active learning
        n = len(self._diversity)
        self._mean_ld = np.mean(self._ld, axis=0)
        self._sem_ld = np.std(self._ld, axis=0, ddof=1) / np.sqrt(n)

        # Synthetic dataset of MC log-likelihood evaluations
        self._eval_points: list[dict] = []

        # Compile both Stan models up front
        self._approx_model = self._compile_approx(self._prior, extra_parameters)
        self._surrogate_model = self._compile_surrogate(self._prior, extra_parameters)

    # ──────────────────────────────────────────────────────────────────────
    # Compilation helpers
    # ──────────────────────────────────────────────────────────────────────

    def _t_fragments(self) -> dict:
        """Return the Stan template fragments for the fixed/free boundary case."""
        if self._fixed_t is None:
            return dict(
                t_extra_data=_FREE_T_EXTRA_DATA,
                t_params=_FREE_T_PARAMS,
                t_transformed=_FREE_T_TRANSFORMED,
                D_expr=_FREE_D_EXPR,
                log_params_assembly=_FREE_LOG_PARAMS_ASSEMBLY,
            )
        return dict(
            t_extra_data=_FIXED_T_EXTRA_DATA,
            t_params=_FIXED_T_PARAMS,
            t_transformed=_FIXED_T_TRANSFORMED,
            D_expr=_FIXED_D_EXPR,
            log_params_assembly=_FIXED_LOG_PARAMS_ASSEMBLY,
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
                "eval_log_params": np.array([p["log_params"] for p in pts]),
                "eval_loglik_det": np.array([p["loglik_det"] for p in pts]),
                "eval_bc_loglik": np.array([p["bc_loglik"] for p in pts]),
                "eval_epsilon": np.array([p["epsilon"] for p in pts]),
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
        """Read-only view of the current MC evaluation dataset."""
        return list(self._eval_points)

    def add_eval_points(self, points: list[dict]) -> None:
        """
        Inject pre-computed MC evaluation points into the synthetic dataset.

        Each dict must have keys: ``log_params``, ``loglik_det``,
        ``bc_loglik``, ``epsilon``.  ``log_params`` is a 1-D array of length
        ``D`` (= 2*n_epochs-1 for free boundaries, n_epochs for fixed).
        """
        required = {"log_params", "loglik_det", "bc_loglik", "epsilon"}
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
        num_replicates: int = 200,
        B: int = 200,
        seed: Optional[int] = None,
        progress_bar: bool = True,
    ):
        """
        Run one round of surrogate active learning.

        Draws ``points_per_iter`` parameter vectors via Pathfinder, evaluates
        the Monte Carlo log-likelihood at each draw, and appends the results
        to the internal synthetic dataset.

        Parameters
        ----------
        progress_bar : bool
            Whether to display a tqdm progress bar while evaluating candidate
            points (default True).

        Returns
        -------
        cmdstanpy.CmdStanPathfinder
        """
        if self._sequence_length is None:
            raise ValueError(
                "sequence_length must be provided at initialisation to use "
                "surrogate_active_learning."
            )

        from scipy.stats import norm as sp_norm
        from tqdm.auto import tqdm

        from .. import deterministic as det
        from .. import montecarlo2 as mc2

        rng = np.random.default_rng(seed)
        n_bins = len(self._left_bins)

        def _ld_loglik(pred_ld: np.ndarray) -> float:
            return float(
                np.sum(sp_norm.logpdf(self._mean_ld, pred_ld, self._sem_ld)) / n_bins
            )

        def _mc_eval(ne_values: np.ndarray, t_bnd: np.ndarray, mc_seed: int) -> dict:
            _, _ld_mc = mc2.expected_piecewise_constant(
                ne_values,
                t_bnd,
                self._left_bins,
                self._right_bins,
                self._mutation_rate,
                self._recombination_rate,
                self._sequence_length,
                self._num_samples,
                random_seed=mc_seed,
                num_replicates=num_replicates,
                ploidy=2,
                num_workers=self._num_workers,
            )
            _ld_mc = np.asarray(_ld_mc)
            _, _ld_det = det.expected_piecewise_constant(
                ne_values,
                t_bnd,
                self._left_bins,
                self._right_bins,
                self._mutation_rate,
                sample_size=self._num_samples,
                ploidy=2,
            )
            loglik_det = _ld_loglik(np.asarray(_ld_det))
            T_hat = _ld_loglik(_ld_mc.mean(axis=0))
            boot_lls = np.array(
                [
                    _ld_loglik(
                        _ld_mc[
                            rng.integers(0, num_replicates, size=num_replicates)
                        ].mean(axis=0)
                    )
                    for _ in range(B)
                ]
            )
            bias = boot_lls.mean() - T_hat
            # Build log_params vector: [log(Ne_1), ..., log(Ne_n), log(t_1), ..., log(t_{n-1})]
            if self._fixed_t is None:
                log_params = np.concatenate([np.log(ne_values), np.log(t_bnd)])
            else:
                log_params = np.log(ne_values)
            return {
                "log_params": log_params,
                "loglik_det": loglik_det,
                "bc_loglik": float(T_hat - bias),
                "epsilon": float(boot_lls.std(ddof=1)),
            }

        pf_seed = int(rng.integers(10_000))
        pf = self._active_model().pathfinder(
            data=self._active_data(),
            draws=points_per_iter,
            seed=pf_seed,
            num_threads=self._num_workers,
            show_console=False,
        )

        ne_draws = _stan_draw_matrix(pf.stan_variable("Ne_values"), points_per_iter)[
            :points_per_iter
        ]
        if self._fixed_t is None:
            t_draws = _stan_draw_matrix(
                pf.stan_variable("t_boundaries"), points_per_iter
            )[:points_per_iter]
        else:
            t_draws = np.tile(self._fixed_t, (points_per_iter, 1))

        iterator = tqdm(
            zip(ne_draws, t_draws),
            total=min(len(ne_draws), len(t_draws)),
            desc="Active learning",
            disable=not progress_bar,
        )
        for ne, t_bnd in iterator:
            mc_seed = int(rng.integers(2**31))
            self._eval_points.append(_mc_eval(ne, t_bnd, mc_seed))

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
        return {
            **self._extract_approx(fit),
            "gp_bias": float(fit.stan_variable("gp_bias")),
            "gp_rho": _stan_vector(fit.stan_variable("gp_rho")),
            "gp_alpha": float(fit.stan_variable("gp_alpha")),
        }

    def optimize(self, **kwargs) -> dict:
        """
        Compute the MAP estimate.

        Returns
        -------
        dict
            Always: ``Ne_values``, ``t_boundaries``, ``E_pi``, ``approx_ld``,
            ``log_lik``.  With surrogate: additionally ``gp_bias``,
            ``gp_rho``, ``gp_alpha``.
        """
        fit = self._active_model().optimize(data=self._active_data(), **kwargs)
        return (
            self._extract_surrogate(fit)
            if self._eval_points
            else self._extract_approx(fit)
        )

    def pathfinder(self, **kwargs):
        """
        Run Pathfinder variational inference.

        Returns
        -------
        cmdstanpy.CmdStanPathfinder
        """
        kwargs.setdefault("num_threads", self._num_workers)
        return self._active_model().pathfinder(data=self._active_data(), **kwargs)

    def sample(self, **kwargs):
        """
        Run NUTS sampling.

        Returns
        -------
        cmdstanpy.CmdStanMCMC
        """
        if not self._eval_points:
            warnings.warn(
                "Using approximate LD predictions. "
                "Check you're in a regime where bias is neglectable.",
                UserWarning,
                stacklevel=2,
            )
        kwargs.setdefault("threads_per_chain", self._num_workers)
        return self._active_model().sample(data=self._active_data(), **kwargs)
