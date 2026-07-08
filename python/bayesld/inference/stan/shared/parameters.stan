// ---- parameters.stan ----
// GP-surrogate parameters shared by every engine. Include at the *bottom* of the
// parameters block, after the demography-specific parameters.

// ── GP surrogate ──
real<lower=0> gp_rho;
real<lower=0> gp_alpha;
matrix[hsgp_m_ld, n_bins] beta_ld;

// Theta-invariant joint covariance Sigma = diag(sigma_y) Omega diag(sigma_y).
vector[n_bins + 1] log_sigma_y;
cholesky_factor_corr[n_bins + 1] L_Omega;
