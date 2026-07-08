// ---- model.stan ----
// Shared tail of the model block: GP-surrogate priors, the two synthetic-point
// likelihood terms (per-bin bias GP + sufficient-statistic MVN), and the joint
// MVN observation model over the real windows. The demography-specific priors
// are placed *above* this include.

// ── GP-surrogate priors ──
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
  //     = -1/2 [ N_i (ȳ_i - mu_i)' Σ⁻¹ (ȳ_i - mu_i)
  //              + tr(Σ⁻¹ S_i) + N_i log|Σ| ] + const,
  // with S_i = Σⱼ (y_ij - ȳ_i)(y_ij - ȳ_i)' = L_S_i L_S_i'.
  real log_det_Sigma = 2 * sum(log(diagonal(L_Sigma)));

  matrix[D, n_sigma] resid;
  for (i in 1:n_sigma) {
    vector[D] mu_i;
    mu_i[1] = det_pi_sigma[i];
    mu_i[2:D] = det_ld_sigma[i]
      .* (1.0 + to_vector(gp_bias_train[n_bias + i, ]'));
    resid[, i] = y_bar_sigma[i] - mu_i;
  }
  matrix[D, n_sigma] V = mdivide_left_tri_low(L_Sigma, resid);
  real quad_mu = 0;
  for (i in 1:n_sigma) quad_mu += N_rep_sigma[i] * dot_self(col(V, i));

  real quad_S = 0;
  for (i in 1:n_sigma) {
    matrix[D, D] M = mdivide_left_tri_low(L_Sigma, L_S_sigma[i]);
    quad_S += sum(square(M));
  }
  target += -0.5 * (quad_mu + quad_S) - 0.5 * sum(N_rep_sigma) * log_det_Sigma;
}

// ── Observation model: joint MVN over real windows ──
for (w in 1:num_windows) y_obs[w] ~ multi_normal_cholesky(mu_y, L_Sigma);
