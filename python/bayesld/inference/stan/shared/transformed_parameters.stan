// ---- transformed_parameters.stan ----
// Shared tail of the transformed parameters block. The demography-specific part
// (included *above* this) must have already defined:
//   real         expected_pi;
//   vector[n_bins] approx_expected_ld;   // finite-sample corrected
// This applies the per-bin HSGP LD-bias correction and assembles the joint mean
// mu_y = [expected_pi, corrected_expected_ld] and Cholesky L_Sigma.

vector[hsgp_m_ld] spd_ld = diagSPD_EQ(gp_alpha, gp_rho, L_ld, hsgp_m_ld);
matrix[hsgp_m_ld, n_bins] w_ld = diag_pre_multiply(spd_ld, beta_ld);

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
