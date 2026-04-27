"""Shared utilities and Stan GP-surrogate template fragments.

All four demographic models (constant, piecewise_constant, piecewise_exponential,
carrying_capacity) use the same 1-D HSGP GP surrogate that learns the relative
LD bias per bin:

    rel_bias[i, b] = mc_ld[i, b] / det_ld[i, b] - 1

The corrected LD is then ``approx_ld .* (1 + gp_bias_ld)``.

Stan blocks
-----------
These are literal Stan code strings — no Python ``{{ }}`` escaping needed because
they contain no curly braces that would interfere with str.format().
"""

import numpy as np

# ── Stan GP surrogate blocks ───────────────────────────────────────────────────

_GP_SURROGATE_DATA = """\
    // ── GP surrogate evaluation dataset (LD-bias) ──────────────────────────
    int<lower=1> n_eval;
    matrix[n_eval, n_bins] eval_rel_bias;
    matrix[n_eval, n_bins] eval_eps_rel;
    real<lower=0> hsgp_c;
    int<lower=1>  hsgp_m;
"""

_GP_SURROGATE_TRANSFORMED_DATA = """\
    // GP: bin midpoints and standardised r
    vector[n_bins] bin_midpoints = (left_bins + right_bins) / 2.0;
    real r_mu  = mean(bin_midpoints);
    real r_sig = sd(bin_midpoints);
    vector[n_bins] r_std = (bin_midpoints - r_mu) / r_sig;
    real L_r = hsgp_c * max(abs(r_std));
    matrix[n_bins, hsgp_m] PHI_r = PHI(n_bins, hsgp_m, L_r, r_std);
"""

_GP_SURROGATE_PARAMS = """\
    real<lower=0> gp_rho_r;
    real<lower=0> gp_alpha;
    vector[hsgp_m] beta_r;
"""

_GP_SURROGATE_TRANSFORMED_PARAMS = """\
    vector[hsgp_m] spd_r = diagSPD_EQ(gp_alpha, gp_rho_r, L_r, hsgp_m);
    vector[n_bins] gp_bias_ld = PHI_r * (spd_r .* beta_r);
"""

_GP_SURROGATE_MODEL = """\
    gp_rho_r ~ inv_gamma(5, 5);
    gp_alpha ~ normal(0, 0.005);
    beta_r ~ std_normal();
    to_vector(eval_rel_bias) ~ normal(
        to_vector(rep_matrix(to_row_vector(gp_bias_ld), n_eval)),
        to_vector(eval_eps_rel));
"""


# ── Python utilities ───────────────────────────────────────────────────────────

def _stan_vector(value) -> np.ndarray:
    """Convert a Stan variable to a 1-D array.

    CmdStanPy may return scalar values for length-1 vectors in MLE outputs.
    """
    arr = np.asarray(value)
    return arr.reshape(1) if arr.ndim == 0 else arr


def _stan_draw_matrix(value, n_draws: int) -> np.ndarray:
    """Convert a Stan draw container to a 2-D (draws, dims) matrix.

    When Stan dimension is 1, CmdStanPy may collapse results to 0-D or 1-D.
    """
    arr = np.asarray(value)
    if arr.ndim == 0:
        return arr.reshape(1, 1)
    if arr.ndim == 1:
        if arr.shape[0] == n_draws:
            return arr.reshape(-1, 1)
        return arr.reshape(1, -1)
    return arr
