// ---- transformed_data.stan ----
// Fully parameterization-independent: stacks the observed windows into y_obs,
// derives the Watterson-based log-Ne offset used to centre the demographic
// parameters, and precomputes the per-bin HSGP design matrices for the training
// (bias + sigma) points. Include as the entire transformed data block body.

int D = n_bins + 1;  // joint dim: [pi, ld_1, ..., ld_B]

array[num_windows] vector[D] y_obs;
for (w in 1:num_windows) {
  y_obs[w, 1] = pi_array[w];
  for (b in 1:n_bins) y_obs[w, b + 1] = ld_mat[w, b];
}

// Initial guess to centre the log-Ne parameters around the data.
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
  for (i in 1:n_bias)  log_ld_b[i]          = log_det_ld_bias[i][b];
  for (i in 1:n_sigma) log_ld_b[n_bias + i] = log_det_ld_sigma[i][b];
  vector[n_train] log_ld_b_std = (log_ld_b - log_ld_mu) / log_ld_sig;
  PHI_ld_train[b] = PHI(n_train, hsgp_m_ld, L_ld, log_ld_b_std);
  for (i in 1:n_bias) {
    rel_bias_train[b, i] = rel_bias_bias[i][b];
    eps_rel_train[b, i]  = eps_rel_bias[i][b];
  }
}
