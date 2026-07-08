// ---- generated_quantities.stan ----
// log_lik omitted: all ingredients (expected_pi, corrected_expected_ld,
// log_sigma_y, L_Omega) are saved as parameters/transformed parameters so
// log_lik can be computed post-hoc in Python without re-running Stan.
matrix[n_bins + 1, n_bins + 1] Omega = multiply_lower_tri_self_transpose(L_Omega);
