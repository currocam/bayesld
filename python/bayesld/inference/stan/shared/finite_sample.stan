// ---- finite_sample.stan ----
// Finite-sample bias correction for LD. Only diploid data (ploidy == 2) carries
// this bias; for any other ploidy the input is returned unchanged (identity).
// S = 2 * sample_size (haploid count for diploid individuals).
vector correct_ld_finite_sample(vector mu, int sample_size, int ploidy) {
  if (ploidy != 2) return mu;
  int S = 2 * sample_size;
  real beta = 1.0 / square(S - 1);
  real alpha_corr = square(square(S) - S + 2.0) / square(square(S) - 3.0 * S + 2.0);
  return (alpha_corr - beta) * mu + 4.0 * beta;
}
