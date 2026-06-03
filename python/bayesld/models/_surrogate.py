"""Shared Stan blocks and Python helpers for the GP-surrogate likelihood.

* LKJ-Cholesky correlation prior + log-Normal marginal sd on Sigma (D = n_bins+1)
* 2D HSGP over (u, log(det_ld)) for the relative LD bias on the mean
* Per-synthetic-theta sufficient-statistic MVN likelihood: at each synthetic
  theta_i with N_i MC replicates we observe (ȳ_i, S_i) and add the exact
  joint log-likelihood of those N_i i.i.d. MVN draws, with mean mu(theta_i)
  (deterministic prediction + bias GP) and covariance Sigma (theta-independent
  in this version). This simultaneously informs the bias GP via ȳ_i and Sigma
  (diagonal *and* correlations) via S_i.
"""

import numpy as np

_REQUIRED_KEYS = ("y_bar", "L_S", "N_rep", "det_pi", "det_ld", "log_det_ld")

# ── Stan blocks ────────────────────────────────────────────────────────────────

SURROGATE_DATA = """\
    // ── Synthetic theta points (sufficient-statistic surrogate) ────────────
    int<lower=0> n_synthetic;
    array[n_synthetic] vector[n_bins + 1] y_bar_syn;                       // sample mean of joint (pi, LD) MC reps
    array[n_synthetic] matrix[n_bins + 1, n_bins + 1] L_S_syn;             // chol(S_i) with S_i = sum_j (y_ij - ȳ_i)(y_ij - ȳ_i)'
    array[n_synthetic] int<lower=1> N_rep_syn;
    vector[n_synthetic] det_pi_syn;                                        // deterministic pi at theta_i
    array[n_synthetic] vector[n_bins] det_ld_syn;                          // deterministic LD at theta_i
    array[n_synthetic] vector[n_bins] log_det_ld_syn;
    // HSGP hyperparameters (bias GP over (u, log det_ld))
    real<lower=0> hsgp_c;
    int<lower=1>  hsgp_m_u;
    int<lower=1>  hsgp_m_ld;
    real<lower=0> gp_alpha_std;
    // Standardization of log(det_ld) axis (data-derived from observed windows)
    real log_ld_mu;
    real<lower=0> log_ld_sig;
    // Sigma prior hyperparameters
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

    // Initial guess to centre log-Ne parameters around the data
    real log_ne_offset = log(mean(pi_array) / (4.0 * mutation_rate));

    vector[n_bins] bin_midpoints = (left_bins + right_bins) / 2.0;
    real u_mu  = mean(bin_midpoints);
    real u_sig = sd(bin_midpoints);
    vector[n_bins] u_std = (bin_midpoints - u_mu) / u_sig;
    real L_u  = hsgp_c * max(abs(u_std));
    real L_ld = hsgp_c * 3.0;  // log-LD axis standardized; +/-3 sd covers tails

    matrix[n_bins, hsgp_m_u] PHI_u_pred = PHI(n_bins, hsgp_m_u, L_u, u_std);

    // Flatten synthetic (u, log_det_ld) inputs for batched bias-GP evaluation.
    int n_syn_flat = n_synthetic * n_bins;
    vector[n_syn_flat] u_syn_flat;
    vector[n_syn_flat] log_ld_syn_flat;
    for (i in 1:n_synthetic) {
        for (b in 1:n_bins) {
            int k = (i - 1) * n_bins + b;
            u_syn_flat[k]      = bin_midpoints[b];
            log_ld_syn_flat[k] = log_det_ld_syn[i][b];
        }
    }
    vector[n_syn_flat] u_syn_std      = (u_syn_flat - u_mu) / u_sig;
    vector[n_syn_flat] log_ld_syn_std = (log_ld_syn_flat - log_ld_mu) / log_ld_sig;
    matrix[n_syn_flat, hsgp_m_u]  PHI_u_train  = PHI(n_syn_flat, hsgp_m_u,  L_u,  u_syn_std);
    matrix[n_syn_flat, hsgp_m_ld] PHI_ld_train = PHI(n_syn_flat, hsgp_m_ld, L_ld, log_ld_syn_std);
"""

SURROGATE_PARAMETERS = """\
    real<lower=0> gp_rho_u;
    real<lower=0> gp_rho_ld;
    real<lower=0> gp_alpha;
    matrix[hsgp_m_u, hsgp_m_ld] beta_uld;

    vector[n_bins + 1] log_sigma_y;
    cholesky_factor_corr[n_bins + 1] L_Omega;
"""

JOINT_TP_PREFIX = """\
    vector[hsgp_m_u]  spd_u_unit  = diagSPD_EQ(1.0, gp_rho_u,  L_u,  hsgp_m_u);
    vector[hsgp_m_ld] spd_ld_unit = diagSPD_EQ(1.0, gp_rho_ld, L_ld, hsgp_m_ld);
    matrix[hsgp_m_u, hsgp_m_ld] B = gp_alpha
        * diag_pre_multiply(spd_u_unit, diag_post_multiply(beta_uld, spd_ld_unit));
"""

JOINT_TP_SUFFIX = """\
    // Evaluate bias GP at prediction inputs (u_b, log(approx_expected_ld[b])).
    vector[n_bins] log_ld_pred_std = (log(approx_expected_ld) - log_ld_mu) / log_ld_sig;
    matrix[n_bins, hsgp_m_ld] PHI_ld_pred = PHI(n_bins, hsgp_m_ld, L_ld, log_ld_pred_std);
    matrix[n_bins, hsgp_m_ld] tmp_pred = PHI_u_pred * B;
    vector[n_bins] gp_bias_ld;
    for (b in 1:n_bins) {
        gp_bias_ld[b] = dot_product(tmp_pred[b], PHI_ld_pred[b]);
    }
    vector[n_bins] corrected_expected_ld = approx_expected_ld .* (1.0 + gp_bias_ld);

    vector[n_bins + 1] mu_y;
    mu_y[1] = expected_pi;
    for (b in 1:n_bins) mu_y[b + 1] = corrected_expected_ld[b];
    vector<lower=0>[n_bins + 1] sigma_y = exp(log_sigma_y);
    matrix[n_bins + 1, n_bins + 1] L_Sigma = diag_pre_multiply(sigma_y, L_Omega);
"""

SURROGATE_MODEL = """\
    gp_rho_u  ~ inv_gamma(5, 5);
    gp_rho_ld ~ inv_gamma(5, 5);
    gp_alpha  ~ normal(0, gp_alpha_std);
    to_vector(beta_uld) ~ std_normal();
    L_Omega ~ lkj_corr_cholesky(lkj_eta);
    log_sigma_y ~ normal(log_sigma_y_loc, log_sigma_y_scale);

    // Synthetic theta points: per-point sufficient-statistic MVN likelihood.
    // log p({y_ij} | mu_i, Sigma) = -1/2 tr(Sigma^{-1} [S_i + N_i (ȳ_i - mu_i)(ȳ_i - mu_i)'])
    //                              - 1/2 N_i log|Sigma|  + const
    // where  tr(Sigma^{-1} S_i) = sum(square( L_Sigma^{-1} L_S_i ))
    //   and  (ȳ_i - mu_i)' Sigma^{-1} (ȳ_i - mu_i) = dot_self( L_Sigma^{-1} (ȳ_i - mu_i) ).
    if (n_synthetic > 0) {
        matrix[n_syn_flat, hsgp_m_ld] tmp_train = PHI_u_train * B;
        vector[n_syn_flat] gp_bias_flat;
        for (k in 1:n_syn_flat) {
            gp_bias_flat[k] = dot_product(tmp_train[k], PHI_ld_train[k]);
        }
        real log_det_Sigma = 2 * sum(log(diagonal(L_Sigma)));
        for (i in 1:n_synthetic) {
            vector[D] mu_i;
            mu_i[1] = det_pi_syn[i];
            for (b in 1:n_bins) {
                int k = (i - 1) * n_bins + b;
                mu_i[b + 1] = det_ld_syn[i][b] * (1.0 + gp_bias_flat[k]);
            }
            vector[D] v = mdivide_left_tri_low(L_Sigma, y_bar_syn[i] - mu_i);
            matrix[D, D] M = mdivide_left_tri_low(L_Sigma, L_S_syn[i]);
            real quad = N_rep_syn[i] * dot_self(v) + sum(square(M));
            target += -0.5 * quad - 0.5 * N_rep_syn[i] * log_det_Sigma;
        }
    }
"""

JOINT_GENERATED_QUANTITIES = """\
    vector[num_windows] log_lik;
    for (w in 1:num_windows) {
        log_lik[w] = multi_normal_cholesky_lpdf(y_obs[w] | mu_y, L_Sigma);
    }
    matrix[n_bins + 1, n_bins + 1] Omega = multiply_lower_tri_self_transpose(L_Omega);
"""


def make_synthetic_point(
    det_pi: float,
    det_ld: np.ndarray,
    mc_pi_reps: np.ndarray,
    mc_ld_reps: np.ndarray,
    ridge: float = 1e-12,
) -> dict:
    """Build a synthetic-point dict from raw MC outputs at one theta.

    Parameters
    ----------
    det_pi      : scalar deterministic pi prediction at this theta
    det_ld      : (n_bins,) deterministic LD prediction at this theta
    mc_pi_reps  : (N,)         per-replicate MC pi estimates
    mc_ld_reps  : (N, n_bins)  per-replicate MC LD estimates
    ridge       : jitter added to S for numerical Cholesky

    Returns
    -------
    dict with keys:
      y_bar       : (D,)     sample mean of joint (pi, LD) reps
      L_S         : (D, D)   chol( sum_j (y_j - ȳ)(y_j - ȳ)' + ridge*I )
      N_rep       : int      number of MC replicates
      det_pi      : float
      det_ld      : (n_bins,)
      log_det_ld  : (n_bins,)
    """
    N = len(mc_ld_reps)
    if N < 2:
        raise ValueError(f"Need ≥ 2 MC replicates; got {N}.")
    joint = np.concatenate([np.asarray(mc_pi_reps)[:, None], np.asarray(mc_ld_reps)], axis=1)
    D = joint.shape[1]
    y_bar = joint.mean(axis=0)
    centered = joint - y_bar
    S = centered.T @ centered
    L_S = np.linalg.cholesky(S + ridge * np.eye(D))
    det_ld_arr = np.asarray(det_ld, dtype=float)
    return {
        "y_bar": y_bar.astype(float),
        "L_S": L_S.astype(float),
        "N_rep": int(N),
        "det_pi": float(det_pi),
        "det_ld": det_ld_arr,
        "log_det_ld": np.log(det_ld_arr),
    }


def validate_synthetic_point(p: dict) -> None:
    """Raise if ``p`` is not a well-formed synthetic-point dict."""
    if not isinstance(p, dict):
        raise TypeError(f"Synthetic point must be a dict, got {type(p).__name__}")
    missing = set(_REQUIRED_KEYS) - set(p)
    if missing:
        raise ValueError(f"Synthetic point missing keys: {sorted(missing)}")


def compute_standardization(ld: np.ndarray) -> tuple[float, float]:
    log_ld_bin = np.log(np.clip(ld.mean(axis=0), 1e-12, None))
    mu = float(log_ld_bin.mean())
    sig = float(max(log_ld_bin.std(ddof=0), 1e-5))
    return mu, sig


def compute_log_sigma_y_loc(diversity: np.ndarray, ld: np.ndarray) -> np.ndarray:
    emp_sd_pi = float(np.std(diversity, ddof=1))
    emp_sd_ld = np.std(ld, axis=0, ddof=1)
    return np.log(np.concatenate([[emp_sd_pi], emp_sd_ld]))


def surrogate_payload(
    synthetic_points: list[dict],
    left_bins: np.ndarray,
    right_bins: np.ndarray,
) -> dict:
    n_bins = len(left_bins)
    D = n_bins + 1
    K = len(synthetic_points)

    if K == 0:
        return {
            "n_synthetic": 0,
            "y_bar_syn": np.zeros((0, D)),
            "L_S_syn": np.zeros((0, D, D)),
            "N_rep_syn": np.zeros(0, dtype=int),
            "det_pi_syn": np.zeros(0),
            "det_ld_syn": np.zeros((0, n_bins)),
            "log_det_ld_syn": np.zeros((0, n_bins)),
        }

    return {
        "n_synthetic": K,
        "y_bar_syn": np.stack([p["y_bar"] for p in synthetic_points]),
        "L_S_syn": np.stack([p["L_S"] for p in synthetic_points]),
        "N_rep_syn": np.array([p["N_rep"] for p in synthetic_points], dtype=int),
        "det_pi_syn": np.array([p["det_pi"] for p in synthetic_points], dtype=float),
        "det_ld_syn": np.stack([p["det_ld"] for p in synthetic_points]),
        "log_det_ld_syn": np.stack([p["log_det_ld"] for p in synthetic_points]),
    }
