// From Fournier 2023, the correction factor they apply that seems
// to work well in diploids only
vector correct_ld_finite_sample(vector mu, int sample_size, int ploidy) {
  if (ploidy != 2) return mu;
  int S = 2 * sample_size;
  real beta = 1.0 / square(S - 1);
  real alpha_corr = square(square(S) - S + 2.0) / square(square(S) - 3.0 * S + 2.0);
  return (alpha_corr - beta) * mu + 4.0 * beta;
}
