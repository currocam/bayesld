"""
Constant-Ne demographic inference model.

``ConstantDemography`` internally maintains two Stan programs — a fast
deterministic approximation and a GP-surrogate that learns the bias of that
approximation — and switches between them automatically depending on whether a
synthetic MC evaluation dataset has been accumulated via
``surrogate_active_learning``.
"""

import pathlib
import tempfile
import warnings
from typing import Optional

import numpy as np

# Path to the stan/ directory relative to this source file.
# TODO: bundle stan files as package data for installed distributions.
_STAN_DIR = pathlib.Path(__file__).resolve().parent.parent.parent.parent / "stan"

_DEFAULT_PRIOR = "    log_Ne ~ normal(11.5, 3.0);"

# Compile both Stan models with threading support so that Pathfinder (and
# future reduce_sum extensions) can exploit multiple cores.
_THREADS_OPTS = {"cpp_options": {"STAN_THREADS": "true"}}

# ---------------------------------------------------------------------------
# Shared Stan source fragments.
# Inserted as replacement *values* into the outer templates, so they must
# contain literal Stan braces { } — not Python-escaped {{ }}.
# ---------------------------------------------------------------------------

_COMMON_DATA = """\
data {
    int<lower=1> n_bins;
    int<lower=2> num_windows;
    vector[n_bins] left_bins;           // left edges of LD bins in Morgans
    vector[n_bins] right_bins;          // right edges of LD bins in Morgans
    real<lower=0> mutation_rate;
    int<lower=1> sample_size;           // number of diploid individuals
    vector[num_windows] pi_array;       // per-window mean diversity
    matrix[num_windows, n_bins] ld_mat; // per-window mean LD per bin
"""

_COMMON_TRANSFORMED_DATA = """\
transformed data {
    // Sufficient statistics for the aggregate likelihood.
    real mean_div = mean(pi_array);
    real<lower=0> sigma_div = sd(pi_array);                  // window-level SD
    real<lower=0> sem_div   = sigma_div / sqrt(num_windows); // SEM of the mean

    vector[n_bins] mean_ld;
    vector<lower=0>[n_bins] sigma_ld;                        // window-level SD per bin
    vector<lower=0>[n_bins] sem_ld;                          // SEM of the mean per bin
    for (b in 1:n_bins) {
        mean_ld[b]  = mean(col(ld_mat, b));
        sigma_ld[b] = sd(col(ld_mat, b));
        sem_ld[b]   = sigma_ld[b] / sqrt(num_windows);
    }
"""

_COMMON_GENQUANT_HEADER = """\
generated quantities {
    real<lower=0> E_pi = 4.0 * Ne * mutation_rate;
    vector<lower=0>[n_bins] approx_ld = correct_ld_finite_sample(
        mu_ld_constant(Ne, left_bins, right_bins),
        sample_size
    );
"""

# ---------------------------------------------------------------------------
# Approximate (deterministic) model template
# ---------------------------------------------------------------------------

_APPROX_TEMPLATE = """\
functions {{
// ---- shared.stan ----
{shared_functions}
// ---- constant.stan ----
{model_functions}
}}

{common_data}}}

{common_transformed_data}}}

parameters {{
    real log_Ne;
{extra_parameters}
}}

transformed parameters {{
    real<lower=0> Ne = exp(log_Ne);
}}

model {{
    // --- user prior ---
{prior_block}

    // --- aggregate likelihood (mean of windows) ---
    real mu_div = 4.0 * Ne * mutation_rate;
    vector[n_bins] mu_ld = correct_ld_finite_sample(
        mu_ld_constant(Ne, left_bins, right_bins),
        sample_size
    );
    mean_div ~ normal(mu_div, sem_div);
    target += normal_lpdf(mean_ld | mu_ld, sem_ld) / n_bins;
}}

{common_genquant_header}
    // Per-window log predictive density (window-level SD, not SEM).
    vector[num_windows] log_lik;
    for (w in 1:num_windows) {{
        log_lik[w] = normal_lpdf(pi_array[w] | E_pi, sigma_div)
                   + normal_lpdf(to_vector(ld_mat[w]) | approx_ld, sigma_ld) / n_bins;
    }}
}}
"""

# ---------------------------------------------------------------------------
# GP-surrogate model template
# ---------------------------------------------------------------------------

_SURROGATE_TEMPLATE = """\
functions {{
// ---- gpbasisfun_functions.stan ----
{gp_functions}
// ---- shared.stan ----
{shared_functions}
// ---- constant.stan ----
{model_functions}
}}

{common_data}
    // ── GP surrogate evaluation dataset ──────────────────────────────────
    int<lower=2> n_eval;
    vector[n_eval] eval_log_ne;
    vector[n_eval] eval_loglik_det;
    vector[n_eval] eval_bc_loglik;
    vector<lower=0>[n_eval] eval_epsilon;

    // ── Hilbert-space GP approximation settings ───────────────────────────
    real<lower=0> hsgp_c;   // boundary factor (≥1.2; recommended 1.5)
    int<lower=1>  hsgp_m;   // number of basis functions
}}

{common_transformed_data}
    // GP: observed biases
    vector[n_eval] delta = eval_bc_loglik - eval_loglik_det;

    // Standardise GP inputs
    real x_mu  = mean(eval_log_ne);
    real x_sig = sd(eval_log_ne);
    vector[n_eval] xn_eval = (eval_log_ne - x_mu) / x_sig;

    // HSGP domain boundary
    real L_hsgp = hsgp_c * fmax(max(xn_eval), -min(xn_eval));

    // Basis-function matrix at eval points (constant)
    matrix[n_eval, hsgp_m] PHI_eval = PHI(n_eval, hsgp_m, L_hsgp, xn_eval);
}}

parameters {{
    real log_Ne;
    real<lower=0> gp_rho;    // EQ kernel length scale (standardised units)
    real<lower=0> gp_alpha;  // EQ kernel signal amplitude
    vector[hsgp_m] beta;     // basis-function coefficients
{extra_parameters}
}}

transformed parameters {{
    real<lower=0> Ne = exp(log_Ne);

    // Standardised query point
    real xn_star = (log_Ne - x_mu) / x_sig;

    // Spectral densities
    vector[hsgp_m] spd = diagSPD_EQ(gp_alpha, gp_rho, L_hsgp, hsgp_m);

    // GP approximation at eval points
    vector[n_eval] f_eval = PHI_eval * (spd .* beta);

    // Basis-function vector at query point
    vector[hsgp_m] phi_star;
    for (m in 1:hsgp_m)
        phi_star[m] = sin(m * pi() / (2.0 * L_hsgp) * (xn_star + L_hsgp)) / sqrt(L_hsgp);

    // GP bias correction at current Ne
    real gp_bias = dot_product(phi_star, spd .* beta);
}}

model {{
    // GP hyperparameter priors
    gp_rho   ~ inv_gamma(5, 5);
    gp_alpha ~ normal(0, 1.0);
    beta     ~ std_normal();

    // HSGP likelihood: fit bias observations
    delta ~ normal(f_eval, eval_epsilon);

    // --- user prior ---
{prior_block}

    // --- aggregate likelihood (mean of windows) ---
    real mu_div = 4.0 * Ne * mutation_rate;
    vector[n_bins] mu_ld = correct_ld_finite_sample(
        mu_ld_constant(Ne, left_bins, right_bins),
        sample_size
    );
    mean_div ~ normal(mu_div, sem_div);
    target += normal_lpdf(mean_ld | mu_ld, sem_ld) / n_bins;

    // Additive bias correction from the GP surrogate
    target += gp_bias;
}}

{common_genquant_header}
    // Per-window log predictive density.
    // gp_bias is a scalar correction to the total log-likelihood;
    // distribute it uniformly across windows.
    vector[num_windows] log_lik;
    for (w in 1:num_windows) {{
        log_lik[w] = normal_lpdf(pi_array[w] | E_pi, sigma_div)
                   + normal_lpdf(to_vector(ld_mat[w]) | approx_ld, sigma_ld) / n_bins
                   + gp_bias / num_windows;
    }}
}}
"""


class ConstantDemography:
    """
    Bayesian inference of effective population size under a constant-Ne model.

    Maintains two internally compiled Stan programs:

    * **Approximate model** — uses a deterministic closed-form LD approximation.
      Active when the synthetic dataset (``eval_points``) is empty.
    * **GP-surrogate model** — augments the approximate model with a Hilbert-space
      GP bias correction fitted jointly to Monte Carlo log-likelihood evaluations.
      Active once ``eval_points`` is non-empty.

    Both models are compiled with ``STAN_THREADS=true``. The same ``num_workers``
    value is used for parallel MC simulation and for ``num_threads`` in Pathfinder
    and ``threads_per_chain`` in NUTS sampling.

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
    sequence_length : float or None
        Genomic sequence length (bp) for Monte Carlo simulations.
        Required only when calling ``surrogate_active_learning``.
    num_workers : int
        Number of parallel workers / threads (default 8). Used for both
        joblib MC simulation and Stan's ``num_threads`` / ``threads_per_chain``.
    hsgp_c : float
        HSGP boundary factor (≥1.2; default 1.5).
    hsgp_m : int
        Number of HSGP basis functions (default 20).
    prior : str
        Stan statements for the prior on ``log_Ne`` and any *extra_parameters*.
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
        sequence_length: Optional[float] = None,
        num_workers: int = 8,
        hsgp_c: float = 1.5,
        hsgp_m: int = 20,
        prior: str = _DEFAULT_PRIOR,
        extra_parameters: str = "",
    ):
        self._diversity = np.asarray(diversity, dtype=float)
        self._ld = np.asarray(ld, dtype=float)
        self._mutation_rate = float(mutation_rate)
        self._recombination_rate = float(recombination_rate)
        self._sequence_length = float(sequence_length) if sequence_length is not None else None
        self._num_samples = int(num_samples)
        self._left_bins = np.asarray(left_bins, dtype=float)
        self._right_bins = np.asarray(right_bins, dtype=float)
        self._num_workers = int(num_workers)
        self._hsgp_c = float(hsgp_c)
        self._hsgp_m = int(hsgp_m)

        # Precompute LD summaries used inside the MC log-likelihood evaluator.
        n = len(self._diversity)
        self._mean_ld = np.mean(self._ld, axis=0)
        self._sem_ld = np.std(self._ld, axis=0, ddof=1) / np.sqrt(n)

        # Synthetic dataset of MC log-likelihood evaluations.
        self._eval_points: list[dict] = []

        # Compile both Stan models up front.
        self._approx_model = self._compile_approx(prior, extra_parameters)
        self._surrogate_model = self._compile_surrogate(prior, extra_parameters)

    # ------------------------------------------------------------------
    # Compilation
    # ------------------------------------------------------------------

    def _compile_approx(self, prior: str, extra_parameters: str):
        import cmdstanpy
        shared_fn = (_STAN_DIR / "functions" / "shared.stan").read_text()
        model_fn = (_STAN_DIR / "functions" / "constant.stan").read_text()
        code = _APPROX_TEMPLATE.format(
            shared_functions=shared_fn,
            model_functions=model_fn,
            common_data=_COMMON_DATA,
            common_transformed_data=_COMMON_TRANSFORMED_DATA,
            common_genquant_header=_COMMON_GENQUANT_HEADER,
            extra_parameters=extra_parameters,
            prior_block=prior,
        )
        tmpdir = pathlib.Path(tempfile.mkdtemp())
        (tmpdir / "constant_ne_approx.stan").write_text(code)
        return cmdstanpy.CmdStanModel(
            stan_file=str(tmpdir / "constant_ne_approx.stan"),
            **_THREADS_OPTS,
        )

    def _compile_surrogate(self, prior: str, extra_parameters: str):
        import cmdstanpy
        gp_fn = (_STAN_DIR / "gpbasisfun_functions.stan").read_text()
        shared_fn = (_STAN_DIR / "functions" / "shared.stan").read_text()
        model_fn = (_STAN_DIR / "functions" / "constant.stan").read_text()
        code = _SURROGATE_TEMPLATE.format(
            gp_functions=gp_fn,
            shared_functions=shared_fn,
            model_functions=model_fn,
            common_data=_COMMON_DATA,
            common_transformed_data=_COMMON_TRANSFORMED_DATA,
            common_genquant_header=_COMMON_GENQUANT_HEADER,
            extra_parameters=extra_parameters,
            prior_block=prior,
        )
        tmpdir = pathlib.Path(tempfile.mkdtemp())
        (tmpdir / "constant_ne_surrogate.stan").write_text(code)
        return cmdstanpy.CmdStanModel(
            stan_file=str(tmpdir / "constant_ne_surrogate.stan"),
            **_THREADS_OPTS,
        )

    # ------------------------------------------------------------------
    # Stan data dicts
    # ------------------------------------------------------------------

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
        }

    def _surrogate_stan_data(self) -> dict:
        pts = self._eval_points
        data = self._base_stan_data()
        data.update({
            "n_eval": int(len(pts)),
            "eval_log_ne": np.array([np.log(p["ne"]) for p in pts]),
            "eval_loglik_det": np.array([p["loglik_det"] for p in pts]),
            "eval_bc_loglik": np.array([p["bc_loglik"] for p in pts]),
            "eval_epsilon": np.array([p["epsilon"] for p in pts]),
            "hsgp_c": self._hsgp_c,
            "hsgp_m": self._hsgp_m,
        })
        return data

    def _active_data(self) -> dict:
        return self._surrogate_stan_data() if self._eval_points else self._base_stan_data()

    def _active_model(self):
        return self._surrogate_model if self._eval_points else self._approx_model

    # ------------------------------------------------------------------
    # Eval-point management
    # ------------------------------------------------------------------

    @property
    def eval_points(self) -> list[dict]:
        """Read-only view of the current MC evaluation dataset."""
        return list(self._eval_points)

    def add_eval_points(self, points: list[dict]) -> None:
        """
        Inject pre-computed MC evaluation points into the synthetic dataset.

        Each dict must have keys: ``ne``, ``loglik_det``, ``bc_loglik``,
        ``epsilon``.  Useful for testing or for supplying evaluations computed
        outside this class.
        """
        required = {"ne", "loglik_det", "bc_loglik", "epsilon"}
        for p in points:
            if not required.issubset(p):
                raise ValueError(
                    f"Each eval point must have keys {required}; got {set(p)}"
                )
        self._eval_points.extend(points)

    # ------------------------------------------------------------------
    # Active learning
    # ------------------------------------------------------------------

    def surrogate_active_learning(
        self,
        points_per_iter: int = 50,
        num_replicates: int = 200,
        B: int = 200,
        seed: Optional[int] = None,
        progress_bar: bool = True,
    ) -> "cmdstanpy.CmdStanPathfinder":  # noqa: F821
        """
        Run one round of surrogate active learning.

        Draws ``points_per_iter`` Ne values via Pathfinder (approximate model
        when ``eval_points`` is empty, GP surrogate otherwise), evaluates the
        Monte Carlo log-likelihood at each draw, and appends the results to the
        internal synthetic dataset.  Call repeatedly to accumulate more points.

        Parameters
        ----------
        points_per_iter : int
            Number of Ne values to draw and evaluate per call.
        num_replicates : int
            Number of Monte Carlo replicates per Ne evaluation.
        B : int
            Number of bootstrap replicates for bias estimation.
        seed : int or None
            RNG seed (Pathfinder seed and MC seeds are both derived from this).
        progress_bar : bool
            Whether to display a tqdm progress bar while evaluating candidate
            points (default True).

        Returns
        -------
        cmdstanpy.CmdStanPathfinder
            The Pathfinder fit used to propose Ne values, useful for
            initialising a subsequent NUTS run.
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

        def _mc_eval(ne: float, mc_seed: int) -> dict:
            _, _ld_mc = mc2.expected_constant(
                float(ne),
                self._left_bins,
                self._right_bins,
                self._mutation_rate,
                self._recombination_rate,
                self._sequence_length,
                self._num_samples,
                random_seed=mc_seed,
                num_replicates=num_replicates,
                ploidy=2,
                model="discretized_smc_prime",
                num_workers=self._num_workers,
            )
            _ld_mc = np.asarray(_ld_mc)
            _, _ld_det = det.expected_constant(
                float(ne),
                self._left_bins,
                self._right_bins,
                self._mutation_rate,
                sample_size=self._num_samples,
                ploidy=2,
            )
            loglik_det = _ld_loglik(np.asarray(_ld_det))
            T_hat = _ld_loglik(_ld_mc.mean(axis=0))
            boot_lls = np.array([
                _ld_loglik(
                    _ld_mc[rng.integers(0, num_replicates, size=num_replicates)].mean(axis=0)
                )
                for _ in range(B)
            ])
            bias = boot_lls.mean() - T_hat
            return {
                "ne": float(ne),
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

        ne_draws = np.asarray(pf.stan_variable("Ne"))[:points_per_iter]
        iterator = tqdm(
            ne_draws,
            total=len(ne_draws),
            desc="Active learning",
            disable=not progress_bar,
        )
        for ne in iterator:
            mc_seed = int(rng.integers(2**31))
            self._eval_points.append(_mc_eval(float(ne), mc_seed))

        return pf

    # ------------------------------------------------------------------
    # Inference interface
    # ------------------------------------------------------------------

    def _extract_approx(self, fit) -> dict:
        return {
            "Ne": float(fit.stan_variable("Ne")),
            "E_pi": float(fit.stan_variable("E_pi")),
            "approx_ld": np.asarray(fit.stan_variable("approx_ld")),
            "log_lik": np.asarray(fit.stan_variable("log_lik")),
        }

    def _extract_surrogate(self, fit) -> dict:
        return {
            **self._extract_approx(fit),
            "gp_bias": float(fit.stan_variable("gp_bias")),
            "gp_rho": float(fit.stan_variable("gp_rho")),
            "gp_alpha": float(fit.stan_variable("gp_alpha")),
        }

    def optimize(self, **kwargs) -> dict:
        """
        Compute the MAP estimate.

        Uses the GP-surrogate model if ``eval_points`` is non-empty, otherwise
        the approximate model.

        Returns
        -------
        dict
            Always: ``Ne``, ``E_pi``, ``approx_ld``, ``log_lik``.
            With surrogate: additionally ``gp_bias``, ``gp_rho``, ``gp_alpha``.
        """
        fit = self._active_model().optimize(data=self._active_data(), **kwargs)
        return self._extract_surrogate(fit) if self._eval_points else self._extract_approx(fit)

    def pathfinder(self, **kwargs):
        """
        Run Pathfinder variational inference.

        Uses the GP-surrogate model if ``eval_points`` is non-empty, otherwise
        the approximate model.  ``num_threads`` defaults to ``self.num_workers``.

        Returns
        -------
        cmdstanpy.CmdStanPathfinder
        """
        kwargs.setdefault("num_threads", self._num_workers)
        return self._active_model().pathfinder(data=self._active_data(), **kwargs)

    def sample(self, **kwargs):
        """
        Run NUTS sampling.

        Uses the GP-surrogate model if ``eval_points`` is non-empty, otherwise
        the approximate model with a warning.  ``threads_per_chain`` defaults to
        ``self.num_workers``.

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
