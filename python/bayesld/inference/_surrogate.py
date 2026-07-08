"""Shared Stan blocks and Python helpers for the GP-surrogate likelihood.

Two synthetic-point streams, learned jointly with the demographic parameters:

* **Bias points** (all accumulated across active-learning iterations) inform an
  independent 1D HSGP per bin over ``log(det_ld)`` that learns the relative LD
  bias ``rel_bias[b] = mc_ld[b] / det_ld[b] - 1`` with per-bin Gaussian noise
  ``eps_rel`` from the MC SE.

* **Sigma points** (only the most recent active-learning iteration — locality
  assumption) contribute a sufficient-statistic multivariate-normal likelihood
  that jointly informs the bias-GP mean and the θ-invariant joint covariance
  ``L_Sigma = diag(sigma_y) · L_Omega`` with ``L_Omega ~ LKJ-Cholesky(eta)``
  and ``log_sigma_y ~ Normal(log_sigma_y_loc, log_sigma_y_scale)``.

The real-window observation likelihood is ``multi_normal_cholesky(mu_y, L_Sigma)``
so observed data also informs ``Sigma`` (sds and correlations).
"""

import numpy as np

_BIAS_KEYS = ("log_det_ld", "rel_bias", "eps_rel")
_SIGMA_KEYS = ("y_bar", "L_S", "N_rep", "det_pi", "det_ld", "log_det_ld")

# ── Stan blocks ────────────────────────────────────────────────────────────────

SURROGATE_DATA = """\
    // ── Bias points (all accumulated): per-bin relative-bias observations ──
    int<lower=0> n_bias;
    array[n_bias] vector[n_bins] log_det_ld_bias;
    array[n_bias] vector[n_bins] rel_bias_bias;
    array[n_bias] vector<lower=0>[n_bins] eps_rel_bias;
    // ── Sigma points (last iteration only): sufficient-stat MVN surrogate ──
    int<lower=0> n_sigma;
    array[n_sigma] vector[n_bins + 1] y_bar_sigma;
    array[n_sigma] matrix[n_bins + 1, n_bins + 1] L_S_sigma;
    array[n_sigma] int<lower=1> N_rep_sigma;
    vector[n_sigma] det_pi_sigma;
    array[n_sigma] vector[n_bins] det_ld_sigma;
    array[n_sigma] vector[n_bins] log_det_ld_sigma;
    // HSGP hyperparameters: shared lengthscale/amplitude across per-bin GPs.
    real<lower=0> hsgp_c;
    int<lower=1>  hsgp_m_ld;
    real<lower=0> gp_alpha_std;
    // Standardization of log(det_ld) axis (from observed windows).
    real log_ld_mu;
    real<lower=0> log_ld_sig;
    // Sigma prior hyperparameters.
    real<lower=0> lkj_eta;
    vector[n_bins + 1] log_sigma_y_loc;
    vector<lower=0>[n_bins + 1] log_sigma_y_scale;
"""

JOINT_TRANSFORMED_DATA = """\
    int D = n_bins + 1;  // joint dim: [pi, ld_1, ..., ld_B]

    array[num_windows] vector[D] y_obs;
    for (w in 1:num_windows) {
        y_obs[w, 1] = pi_array[w];
        for (b in 1:n_bins) y_obs[w, b + 1] = ld_mat[w, b];
    }

    // Initial guess to centre log-Ne parameters around the data.
    real log_ne_offset = log(mean(pi_array) / (4.0 * mutation_rate));

    real L_ld = hsgp_c * 3.0;

    // Pool bias and sigma training inputs per bin so we evaluate the GP once
    // per bin and slice the result for both likelihood terms.
    int n_train = n_bias + n_sigma;
    array[n_bins] matrix[n_train, hsgp_m_ld] PHI_ld_train;
    array[n_bins] vector[n_bias] rel_bias_train;
    array[n_bins] vector<lower=0>[n_bias] eps_rel_train;
    for (b in 1:n_bins) {
        vector[n_train] log_ld_b;
        for (i in 1:n_bias)  log_ld_b[i]            = log_det_ld_bias[i][b];
        for (i in 1:n_sigma) log_ld_b[n_bias + i]   = log_det_ld_sigma[i][b];
        vector[n_train] log_ld_b_std = (log_ld_b - log_ld_mu) / log_ld_sig;
        PHI_ld_train[b] = PHI(n_train, hsgp_m_ld, L_ld, log_ld_b_std);
        for (i in 1:n_bias) {
            rel_bias_train[b, i] = rel_bias_bias[i][b];
            eps_rel_train[b, i]  = eps_rel_bias[i][b];
        }
    }
"""

SURROGATE_PARAMETERS = """\
    real<lower=0> gp_rho;
    real<lower=0> gp_alpha;
    matrix[hsgp_m_ld, n_bins] beta_ld;

    // Theta-invariant joint covariance Sigma = diag(sigma_y) Omega diag(sigma_y).
    vector[n_bins + 1] log_sigma_y;
    cholesky_factor_corr[n_bins + 1] L_Omega;
"""

JOINT_TP_PREFIX = """\
    vector[hsgp_m_ld] spd_ld = diagSPD_EQ(gp_alpha, gp_rho, L_ld, hsgp_m_ld);
    matrix[hsgp_m_ld, n_bins] w_ld = diag_pre_multiply(spd_ld, beta_ld);
"""

JOINT_TP_SUFFIX = """\
    // Evaluate each per-bin GP at its own log(approx_expected_ld[b]).
    vector[n_bins] log_ld_pred_std = (log(approx_expected_ld) - log_ld_mu) / log_ld_sig;
    matrix[n_bins, hsgp_m_ld] PHI_ld_pred = PHI(n_bins, hsgp_m_ld, L_ld, log_ld_pred_std);
    vector[n_bins] gp_bias_ld;
    for (b in 1:n_bins) gp_bias_ld[b] = dot_product(row(PHI_ld_pred, b), col(w_ld, b));
    vector[n_bins] corrected_expected_ld = approx_expected_ld .* (1.0 + gp_bias_ld);

    vector[D] mu_y;
    mu_y[1] = expected_pi;
    mu_y[2:D] = corrected_expected_ld;

    vector<lower=0>[D] sigma_y = exp(log_sigma_y);
    matrix[D, D] L_Sigma = diag_pre_multiply(sigma_y, L_Omega);
"""

SURROGATE_MODEL = """\
    gp_rho   ~ inv_gamma(5, 5);
    gp_alpha ~ normal(0, gp_alpha_std);
    to_vector(beta_ld) ~ std_normal();
    L_Omega     ~ lkj_corr_cholesky(lkj_eta);
    log_sigma_y ~ normal(log_sigma_y_loc, log_sigma_y_scale);

    // Per-bin GP evaluation at training inputs (bias rows 1..n_bias, sigma rows
    // n_bias+1..n_train).
    matrix[n_train, n_bins] gp_bias_train;
    for (b in 1:n_bins) gp_bias_train[, b] = PHI_ld_train[b] * col(w_ld, b);

    if (n_bias > 0) {
        for (b in 1:n_bins) {
            rel_bias_train[b] ~ normal(gp_bias_train[1:n_bias, b], eps_rel_train[b]);
        }
    }

    if (n_sigma > 0) {
        // Sufficient-statistic MVN: each sigma point contributes
        //   log p({y_ij}_{j=1..N_i} | mu_i, Sigma)
        //     = -1/2 [ N_i (ȳ_i - mu_i)' Σ⁻¹ (ȳ_i - mu_i)    // mean term
        //              + tr(Σ⁻¹ S_i)                          // scatter term
        //              + N_i log|Σ| ] + const,
        // with S_i = Σⱼ (y_ij - ȳ_i)(y_ij - ȳ_i)'  =  L_S_i L_S_i'.
        // We accumulate the two quadratic pieces (`quad_mu`, `quad_S`)
        // separately and pay the log-det term once.
        real log_det_Sigma = 2 * sum(log(diagonal(L_Sigma)));

        // Stack residuals (ȳ_i - mu_i) as columns of `resid`; one triangular
        // solve gives V = L_Σ⁻¹ · resid, so dot_self(col(V,i)) is the
        // Mahalanobis distance (ȳ_i - mu_i)' Σ⁻¹ (ȳ_i - mu_i) — the per-point
        // mean-misfit contribution, weighted by N_i replicates.
        matrix[D, n_sigma] resid;
        for (i in 1:n_sigma) {
            vector[D] mu_i;
            mu_i[1] = det_pi_sigma[i];
            mu_i[2:D] = det_ld_sigma[i]
                .* (1.0 + to_vector(gp_bias_train[n_bias + i, ]'));
            resid[, i] = y_bar_sigma[i] - mu_i;
        }
        matrix[D, n_sigma] V = mdivide_left_tri_low(L_Sigma, resid);
        real quad_mu = 0;  // Σᵢ N_i · (ȳ_i - mu_i)' Σ⁻¹ (ȳ_i - mu_i)
        for (i in 1:n_sigma) quad_mu += N_rep_sigma[i] * dot_self(col(V, i));

        // Scatter term: tr(Σ⁻¹ S_i) = ‖L_Σ⁻¹ L_S_i‖_F², informs Σ from the
        // within-sigma-point spread of MC replicates.
        real quad_S = 0;   // Σᵢ tr(Σ⁻¹ S_i)
        for (i in 1:n_sigma) {
            matrix[D, D] M = mdivide_left_tri_low(L_Sigma, L_S_sigma[i]);
            quad_S += sum(square(M));
        }
        target += -0.5 * (quad_mu + quad_S) - 0.5 * sum(N_rep_sigma) * log_det_Sigma;
    }
"""

JOINT_OBS_MODEL = """\
    for (w in 1:num_windows) y_obs[w] ~ multi_normal_cholesky(mu_y, L_Sigma);
"""

JOINT_GENERATED_QUANTITIES = """\
    vector[num_windows] log_lik;
    for (w in 1:num_windows) log_lik[w] = multi_normal_cholesky_lpdf(y_obs[w] | mu_y, L_Sigma);
    matrix[n_bins + 1, n_bins + 1] Omega = multiply_lower_tri_self_transpose(L_Omega);
"""


# ── Python helpers ─────────────────────────────────────────────────────────────


def _mc_stats(mc_reps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (mean, SE-of-mean) along axis 0 of a stack of MC replicates."""
    n = len(mc_reps)
    if n < 2:
        raise ValueError(f"Need ≥ 2 MC replicates; got {n}.")
    return mc_reps.mean(axis=0), mc_reps.std(axis=0, ddof=1) / np.sqrt(n)


def make_bias_point(det_ld: np.ndarray, mc_ld_reps: np.ndarray) -> dict:
    """Bias-only synthetic point (per-bin relative bias + MC SE)."""
    det_ld = np.asarray(det_ld, dtype=float)
    mc_ld_reps = np.asarray(mc_ld_reps, dtype=float)
    mc_mean, mc_se = _mc_stats(mc_ld_reps)
    return {
        "log_det_ld": np.log(det_ld),
        "rel_bias": mc_mean / det_ld - 1.0,
        "eps_rel": mc_se / det_ld,
    }


def make_sigma_point(
    det_pi: float,
    det_ld: np.ndarray,
    mc_pi_reps: np.ndarray,
    mc_ld_reps: np.ndarray,
    ridge_rel: float = 1e-10,
) -> dict:
    """Sufficient-statistic point for the joint (pi, LD) MVN surrogate.

    Returns ``{y_bar, L_S, N_rep, det_pi, det_ld, log_det_ld}`` with
    ``L_S = chol(S + ridge·diag)``, ``S = Σⱼ centeredⱼ centeredⱼ'``.
    The ridge is scaled by the diagonal of ``S`` for numerical robustness.
    """
    mc_pi_reps = np.asarray(mc_pi_reps, dtype=float)
    mc_ld_reps = np.asarray(mc_ld_reps, dtype=float)
    n = len(mc_ld_reps)
    if n != len(mc_pi_reps):
        raise ValueError("mc_pi_reps and mc_ld_reps must have equal length")
    if n < 2:
        raise ValueError(f"Need ≥ 2 MC replicates; got {n}.")

    joint = np.column_stack([mc_pi_reps, mc_ld_reps])
    y_bar = joint.mean(axis=0)
    centered = joint - y_bar
    S = centered.T @ centered
    ridge = max(ridge_rel * float(np.diag(S).max()), 1e-12)
    L_S = np.linalg.cholesky(S + ridge * np.eye(S.shape[0]))
    det_ld = np.asarray(det_ld, dtype=float)
    return {
        "y_bar": y_bar,
        "L_S": L_S,
        "N_rep": int(n),
        "det_pi": float(det_pi),
        "det_ld": det_ld,
        "log_det_ld": np.log(det_ld),
    }


def make_points(raw: dict) -> tuple[dict, dict]:
    """Build (bias_point, sigma_point) from an ``_mc_eval`` raw dict."""
    bias = make_bias_point(raw["det_ld"], raw["mc_ld_reps"])
    sigma = make_sigma_point(
        raw["det_pi"], raw["det_ld"], raw["mc_pi_reps"], raw["mc_ld_reps"]
    )
    return bias, sigma


def _require_keys(p: dict, keys: tuple[str, ...], kind: str) -> None:
    if not isinstance(p, dict):
        raise TypeError(f"{kind} point must be a dict, got {type(p).__name__}")
    missing = set(keys) - set(p)
    if missing:
        raise ValueError(f"{kind} point missing keys: {sorted(missing)}")


def validate_bias_point(p: dict) -> None:
    _require_keys(p, _BIAS_KEYS, "Bias")


def validate_sigma_point(p: dict) -> None:
    _require_keys(p, _SIGMA_KEYS, "Sigma")


# Back-compat aliases for the bias-only era.
make_synthetic_point = make_bias_point
validate_synthetic_point = validate_bias_point


def compute_standardization(ld: np.ndarray) -> tuple[float, float]:
    log_ld_bin = np.log(np.clip(ld.mean(axis=0), 1e-12, None))
    return float(log_ld_bin.mean()), float(max(log_ld_bin.std(ddof=0), 1e-5))


def compute_sigma_emp(diversity: np.ndarray, ld: np.ndarray) -> np.ndarray:
    return np.concatenate([[np.std(diversity, ddof=1)], np.std(ld, axis=0, ddof=1)])


def sigma_prior_hyperparams(
    diversity: np.ndarray,
    ld: np.ndarray,
    log_sigma_y_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Log-Normal prior on Sigma's marginal sds, centred on empirical sds."""
    loc = np.log(compute_sigma_emp(diversity, ld))
    return loc, np.full_like(loc, float(log_sigma_y_scale))


def _stack(points: list[dict], key: str) -> np.ndarray:
    return np.stack([p[key] for p in points])


def surrogate_payload(
    bias_points: list[dict],
    sigma_points: list[dict],
    left_bins: np.ndarray,
    right_bins: np.ndarray,
) -> dict:
    """Stan data payload for both synthetic-point streams."""
    n_bins = len(left_bins)
    D = n_bins + 1

    if bias_points:
        bias = {
            "n_bias": len(bias_points),
            "log_det_ld_bias": _stack(bias_points, "log_det_ld"),
            "rel_bias_bias": _stack(bias_points, "rel_bias"),
            "eps_rel_bias": _stack(bias_points, "eps_rel"),
        }
    else:
        bias = {
            "n_bias": 0,
            "log_det_ld_bias": np.zeros((0, n_bins)),
            "rel_bias_bias": np.zeros((0, n_bins)),
            "eps_rel_bias": np.zeros((0, n_bins)),
        }

    if sigma_points:
        sigma = {
            "n_sigma": len(sigma_points),
            "y_bar_sigma": _stack(sigma_points, "y_bar"),
            "L_S_sigma": _stack(sigma_points, "L_S"),
            "N_rep_sigma": np.array([p["N_rep"] for p in sigma_points], dtype=int),
            "det_pi_sigma": np.array([p["det_pi"] for p in sigma_points], dtype=float),
            "det_ld_sigma": _stack(sigma_points, "det_ld"),
            "log_det_ld_sigma": _stack(sigma_points, "log_det_ld"),
        }
    else:
        sigma = {
            "n_sigma": 0,
            "y_bar_sigma": np.zeros((0, D)),
            "L_S_sigma": np.zeros((0, D, D)),
            "N_rep_sigma": np.zeros(0, dtype=int),
            "det_pi_sigma": np.zeros(0),
            "det_ld_sigma": np.zeros((0, n_bins)),
            "log_det_ld_sigma": np.zeros((0, n_bins)),
        }

    return {**bias, **sigma}
