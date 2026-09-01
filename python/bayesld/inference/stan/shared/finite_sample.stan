// Fournier 2023 finite-sample correction factors
real ld_fs_beta(real S) {
  return 1.0 / square(S - 1.0);
}
real ld_fs_alpha(real S) {
  return square(square(S) - S + 2.0) / square(square(S) - 3.0 * S + 2.0);
}

vector correct_ld_finite_sample(vector mu, int sample_size, int ploidy) {
  if (ploidy != 2) return mu;
  real S = 2.0 * sample_size;
  real beta = ld_fs_beta(S);
  real alpha_corr = ld_fs_alpha(S);
  return (alpha_corr - beta) * mu + 4.0 * beta;
}

// Take an LD prediction corrected for sample size S and
// change the correction factor to  `S_eff`. 
vector rescale_ld_effective_sample(vector mu_S, real S, vector S_eff) {
  real beta_S = ld_fs_beta(S);
  real alpha_S = ld_fs_alpha(S);
  vector[num_elements(mu_S)] mu_inf = (mu_S - 4.0 * beta_S) / (alpha_S - beta_S);

  int n = num_elements(S_eff);
  vector[n] beta_eff;
  vector[n] alpha_eff;
  for (b in 1:n) {
    beta_eff[b] = ld_fs_beta(S_eff[b]);
    alpha_eff[b] = ld_fs_alpha(S_eff[b]);
  }
  return (alpha_eff - beta_eff) .* mu_inf + 4.0 * beta_eff;
}
