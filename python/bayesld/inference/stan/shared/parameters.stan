// Shared parameters to be inferred
// GP surrogate
real<lower=0> gp_rho;
real<lower=0> gp_alpha;
matrix[hsgp_m_ld, n_bins] beta_ld;

// Fitting the Stan’s Cholesky factor of teh covariance matrix
// https://mc-stan.org/docs/reference-manual/transforms.html#cholesky-factor-of-covariance-matrix-transform
vector[n_bins + 1] log_sigma_y;
cholesky_factor_corr[n_bins + 1] L_Omega;
