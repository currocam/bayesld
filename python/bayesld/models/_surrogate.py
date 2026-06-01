"""Shared Stan blocks and Python helpers for the GP-surrogate likelihood.

* LKJ-Cholesky correlation prior + log-Normal marginal sd on Sigma (D = n_bins+1)
* 2D HSGP over (u, log(det_ld)) for the relative LD bias
* Wishart sufficient-statistic surrogate that pools sample covariances from MC
  evaluations at synthetic theta points.
"""

import numpy as np

_REQUIRED_KEYS = ("rel_bias", "eps_rel", "log_det_ld", "S_mc", "N_rep")

# ── Stan blocks ────────────────────────────────────────────────────────────────

SURROGATE_DATA = """\
    // ── GP surrogate: synthetic relative-bias observations ─────────────────
    int<lower=0> n_synthetic_obs;
    vector[n_synthetic_obs] u_synthetic_obs;       // bin midpoint
    vector[n_synthetic_obs] log_ld_synthetic_obs;  // log(det_ld(theta_syn, bin))
    vector[n_synthetic_obs] synthetic_obs;         // mc_ld/det_ld - 1
    vector<lower=0>[n_synthetic_obs] bias_se;      // SE of synthetic_obs
    // Wishart sufficient stats for joint (pi, LD) Sigma
    int<lower=0> n_sum;
    matrix[n_bins + 1, n_bins + 1] S_sum;
    // HSGP hyperparameters
    real<lower=0> hsgp_c;
    int<lower=1>  hsgp_m_u;
    int<lower=1>  hsgp_m_ld;
    real<lower=0> gp_alpha_std;
    // Standardization of log(det_ld) axis (data-derived)
    real log_ld_mu;
    real<lower=0> log_ld_sig;
    // LKJ + log-Normal prior hyperparameters
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

    vector[n_synthetic_obs] u_synthetic_std     = (u_synthetic_obs - u_mu) / u_sig;
    vector[n_synthetic_obs] log_ld_synthetic_std = (log_ld_synthetic_obs - log_ld_mu) / log_ld_sig;
    matrix[n_synthetic_obs, hsgp_m_u]  PHI_u_train  = PHI(n_synthetic_obs, hsgp_m_u,  L_u,  u_synthetic_std);
    matrix[n_synthetic_obs, hsgp_m_ld] PHI_ld_train = PHI(n_synthetic_obs, hsgp_m_ld, L_ld, log_ld_synthetic_std);

    matrix[D, D] L_S_sum = cholesky_decompose(n_sum > 0 ? S_sum : diag_matrix(rep_vector(1.0, D)));
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
    // Evaluate GP at prediction inputs (u_b, log(approx_expected_ld[b])).
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

    // GP surrogate: synthetic relative-bias observations
    if (n_synthetic_obs > 0) {
        matrix[n_synthetic_obs, hsgp_m_ld] tmp_train = PHI_u_train * B;
        vector[n_synthetic_obs] gp_bias_train;
        for (k in 1:n_synthetic_obs) {
            gp_bias_train[k] = dot_product(tmp_train[k], PHI_ld_train[k]);
        }
        synthetic_obs ~ normal(gp_bias_train, bias_se);
    }

    // Wishart sufficient-statistic likelihood for Sigma
    if (n_sum > 0) {
        matrix[D, D] A = mdivide_left_tri_low(L_Sigma, L_S_sum);
        real log_det_Sigma = 2 * sum(log(diagonal(L_Sigma)));
        target += -0.5 * sum(square(A)) - 0.5 * n_sum * log_det_Sigma;
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
    det_ld: np.ndarray, mc_pi_reps: np.ndarray, mc_ld_reps: np.ndarray
) -> dict:
    """Build a synthetic-point dict from raw MC outputs at one theta.

    Parameters
    ----------
    det_ld : (n_bins,) deterministic LD prediction at this theta
    mc_pi_reps : (N,) per-window MC pi estimates
    mc_ld_reps : (N, n_bins) per-window MC LD estimates

    Returns
    -------
    dict with keys:
      rel_bias    : (n_bins,) mean of mc_ld/det_ld - 1
      eps_rel     : (n_bins,) SE of rel_bias across replicates
      log_det_ld  : (n_bins,) log(det_ld) at this theta
      S_mc        : (D, D)   sample covariance of joint (pi, LD)
      N_rep       : int      number of MC replicates
    """
    N = len(mc_ld_reps)
    rel = mc_ld_reps / det_ld - 1.0
    joint = np.concatenate([mc_pi_reps[:, None], mc_ld_reps], axis=1)
    S_mc = np.cov(joint, rowvar=False, ddof=1)
    return {
        "rel_bias": rel.mean(axis=0),
        "eps_rel": rel.std(axis=0, ddof=1) / np.sqrt(N),
        "log_det_ld": np.log(det_ld),
        "S_mc": S_mc,
        "N_rep": int(N),
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
    u_mid = 0.5 * (np.asarray(left_bins) + np.asarray(right_bins))

    if not synthetic_points:
        return {
            "n_synthetic_obs": 0,
            "u_synthetic_obs": np.zeros(0),
            "log_ld_synthetic_obs": np.zeros(0),
            "synthetic_obs": np.zeros(0),
            "bias_se": np.ones(0),
            "n_sum": 0,
            "S_sum": np.eye(D),
        }

    synthetic_obs = np.concatenate([p["rel_bias"] for p in synthetic_points])
    bias_se = np.concatenate([p["eps_rel"] for p in synthetic_points])
    log_ld_synthetic_obs = np.concatenate([p["log_det_ld"] for p in synthetic_points])
    u_synthetic_obs = np.tile(u_mid, len(synthetic_points))
    S_sum = sum((p["N_rep"] - 1) * p["S_mc"] for p in synthetic_points)
    n_sum = int(sum(p["N_rep"] - 1 for p in synthetic_points))
    return {
        "n_synthetic_obs": int(len(synthetic_obs)),
        "u_synthetic_obs": u_synthetic_obs,
        "log_ld_synthetic_obs": log_ld_synthetic_obs,
        "synthetic_obs": synthetic_obs,
        "bias_se": bias_se,
        "n_sum": n_sum,
        "S_sum": S_sum,
    }
