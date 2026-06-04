"""Shared Stan blocks and Python helpers for the GP-bias likelihood.

Simplified scaffold:

* Observation likelihood is independent normals per component, with fixed
  empirical per-component sd ``sigma_emp`` (computed once from observed
  windows upstream). No LKJ / log-Normal sd learning.
* One independent 1D HSGP per bin over ``log(det_ld)`` learns the relative
  LD bias: ``bias[b] = f_b(log_det_ld[b])``. The ``f_b`` share priors on
  lengthscale and amplitude but have independent realisations.
  ``corrected_ld = det_ld * (1 + bias)``.
* Synthetic theta points contribute a *per-bin* Gaussian likelihood on
  ``rel_bias = mc_ld / det_ld - 1`` with sd ``eps_rel`` derived from the MC SE.
  No Wishart / sufficient-statistic surrogate; pi is not used in the surrogate.
"""

import numpy as np

_REQUIRED_KEYS = ("log_det_ld", "rel_bias", "eps_rel")

# ── Stan blocks ────────────────────────────────────────────────────────────────

SURROGATE_DATA = """\
    // ── Synthetic theta points (per-bin relative-bias observations) ────────
    int<lower=0> n_synthetic;
    array[n_synthetic] vector[n_bins] log_det_ld_syn;
    array[n_synthetic] vector[n_bins] rel_bias_syn;
    array[n_synthetic] vector<lower=0>[n_bins] eps_rel_syn;
    // HSGP hyperparameters: one 1D GP per bin over log(det_ld), with
    // priors on lengthscale and amplitude shared across bins.
    real<lower=0> hsgp_c;
    int<lower=1>  hsgp_m_ld;
    real<lower=0> gp_alpha_std;
    // Standardization of log(det_ld) axis (data-derived from observed windows)
    real log_ld_mu;
    real<lower=0> log_ld_sig;
    // Fixed per-component empirical sds (computed upstream from windows).
    vector<lower=0>[n_bins + 1] sigma_emp;
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

    real L_ld = hsgp_c * 3.0;  // log-LD axis standardized; +/-3 sd covers tails

    // Per-bin training inputs: stack synthetic log_det_ld values for each bin.
    array[n_bins] matrix[n_synthetic, hsgp_m_ld] PHI_ld_train;
    array[n_bins] vector[n_synthetic] rel_bias_train;
    array[n_bins] vector<lower=0>[n_synthetic] eps_rel_train;
    for (b in 1:n_bins) {
        vector[n_synthetic] log_ld_b;
        vector[n_synthetic] rb_b;
        vector[n_synthetic] er_b;
        for (i in 1:n_synthetic) {
            log_ld_b[i] = log_det_ld_syn[i][b];
            rb_b[i]     = rel_bias_syn[i][b];
            er_b[i]     = eps_rel_syn[i][b];
        }
        vector[n_synthetic] log_ld_b_std = (log_ld_b - log_ld_mu) / log_ld_sig;
        PHI_ld_train[b]  = PHI(n_synthetic, hsgp_m_ld, L_ld, log_ld_b_std);
        rel_bias_train[b] = rb_b;
        eps_rel_train[b]  = er_b;
    }
"""

SURROGATE_PARAMETERS = """\
    // Shared lengthscale and amplitude across the n_bins independent 1D GPs.
    real<lower=0> gp_rho;
    real<lower=0> gp_alpha;
    matrix[hsgp_m_ld, n_bins] beta_ld;
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
    for (b in 1:n_bins) {
        gp_bias_ld[b] = dot_product(row(PHI_ld_pred, b), col(w_ld, b));
    }
    vector[n_bins] corrected_expected_ld = approx_expected_ld .* (1.0 + gp_bias_ld);

    vector[n_bins + 1] mu_y;
    mu_y[1] = expected_pi;
    for (b in 1:n_bins) mu_y[b + 1] = corrected_expected_ld[b];
"""

SURROGATE_MODEL = """\
    gp_rho   ~ inv_gamma(5, 5);
    gp_alpha ~ normal(0, gp_alpha_std);
    to_vector(beta_ld) ~ std_normal();

    // Synthetic theta points: per-bin Gaussian likelihood on relative bias.
    if (n_synthetic > 0) {
        for (b in 1:n_bins) {
            vector[n_synthetic] gp_bias_b = PHI_ld_train[b] * col(w_ld, b);
            rel_bias_train[b] ~ normal(gp_bias_b, eps_rel_train[b]);
        }
    }
"""

JOINT_OBS_MODEL = """\
    // Weight pi and LD equally in total: each LD bin contributes 1/n_bins.
    for (w in 1:num_windows) {
        target += normal_lpdf(y_obs[w, 1] | mu_y[1], sigma_emp[1]);
        target += normal_lpdf(y_obs[w, 2:(n_bins + 1)] | mu_y[2:(n_bins + 1)], sigma_emp[2:(n_bins + 1)]) / n_bins;
    }
"""

JOINT_GENERATED_QUANTITIES = """\
    vector[num_windows] log_lik;
    for (w in 1:num_windows) {
        log_lik[w] = normal_lpdf(y_obs[w, 1] | mu_y[1], sigma_emp[1])
                   + normal_lpdf(y_obs[w, 2:(n_bins + 1)] | mu_y[2:(n_bins + 1)], sigma_emp[2:(n_bins + 1)]) / n_bins;
    }
"""


def make_synthetic_point(
    det_ld: np.ndarray,
    mc_ld_reps: np.ndarray,
) -> dict:
    """Build a synthetic-point dict from raw MC LD outputs at one theta.

    Parameters
    ----------
    det_ld      : (n_bins,)    deterministic LD prediction at this theta
    mc_ld_reps  : (N, n_bins)  per-replicate MC LD estimates

    Returns
    -------
    dict with keys:
      log_det_ld : (n_bins,)
      rel_bias   : (n_bins,)  mc_mean / det_ld - 1
      eps_rel    : (n_bins,)  MC SE of mc_mean, in the rel_bias scale
    """
    mc_ld_reps = np.asarray(mc_ld_reps, dtype=float)
    N = len(mc_ld_reps)
    if N < 2:
        raise ValueError(f"Need ≥ 2 MC replicates; got {N}.")
    det_ld_arr = np.asarray(det_ld, dtype=float)
    mc_mean = mc_ld_reps.mean(axis=0)
    mc_se = mc_ld_reps.std(axis=0, ddof=1) / np.sqrt(N)
    return {
        "log_det_ld": np.log(det_ld_arr),
        "rel_bias": mc_mean / det_ld_arr - 1.0,
        "eps_rel": mc_se / det_ld_arr,
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


def compute_sigma_emp(diversity: np.ndarray, ld: np.ndarray) -> np.ndarray:
    emp_sd_pi = float(np.std(diversity, ddof=1))
    emp_sd_ld = np.std(ld, axis=0, ddof=1)
    return np.concatenate([[emp_sd_pi], emp_sd_ld])


def surrogate_payload(
    synthetic_points: list[dict],
    left_bins: np.ndarray,
    right_bins: np.ndarray,
) -> dict:
    n_bins = len(left_bins)
    K = len(synthetic_points)

    if K == 0:
        return {
            "n_synthetic": 0,
            "log_det_ld_syn": np.zeros((0, n_bins)),
            "rel_bias_syn": np.zeros((0, n_bins)),
            "eps_rel_syn": np.zeros((0, n_bins)),
        }

    return {
        "n_synthetic": K,
        "log_det_ld_syn": np.stack([p["log_det_ld"] for p in synthetic_points]),
        "rel_bias_syn": np.stack([p["rel_bias"] for p in synthetic_points]),
        "eps_rel_syn": np.stack([p["eps_rel"] for p in synthetic_points]),
    }
