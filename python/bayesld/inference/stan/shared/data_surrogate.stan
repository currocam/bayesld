// All models share (1) optional synthetic points to learn bias 
// and covariance matrix and (2) (hyper)-priors for the GP surrogate
// We neglect heteroscedasticity and assume that the covariance matrix
// is constant in the region of high posterior probability. Because early 
// active rounds might be explorering a different region, we keep track of 
// data points for learning the bias and the covariance separately
// Bias points : per-bin relative-bias observations.
int<lower=0> n_bias;
array[n_bias] vector[n_bins] log_det_ld_bias;
array[n_bias] vector[n_bins] rel_bias_bias;
// We have a stderr for the observed bias
array[n_bias] vector<lower=0>[n_bins] eps_rel_bias;
// Sigma points:
// Each sigma point summarises N_rep_sigma[i] MC replicates drawn at a fixed
// demography with deterministic predictions (det_pi_sigma, det_ld_sigma).
// The per-replicate observations y_ij ~ MVN(mu_i, Sigma) are summarised by
// their sample mean y_bar_sigma[i] and scatter matrix L_S_sigma[i] (the
// Cholesky of S_i = Σ_j (y_ij - y_bar_i)(y_ij - y_bar_i)^T). Together with
// N_rep_sigma[i], these three quantities are sufficient statistics for the
// Gaussian log-density and allow the covariance Sigma to be learned without
// storing all individual replicates.
int<lower=0> n_sigma;
array[n_sigma] vector[n_bins + 1] y_bar_sigma;
array[n_sigma] matrix[n_bins + 1, n_bins + 1] L_S_sigma;
array[n_sigma] int<lower=1> N_rep_sigma;
vector[n_sigma] det_pi_sigma;
array[n_sigma] vector[n_bins] det_ld_sigma;
array[n_sigma] vector[n_bins] log_det_ld_sigma;

// We fit a Hilbert Space Gaussian process
// HSGP hyperparameters: shared lengthscale/amplitude across per-bin GPs.
real<lower=0> hsgp_c;
int<lower=1>  hsgp_m_ld;
real<lower=0> gp_alpha_std;
// Standardization of log(det_ld) axis (from observed windows).
real log_ld_mu;
real<lower=0> log_ld_sig;
// Covariance prior.
real<lower=0> lkj_eta;
vector[n_bins + 1] log_sigma_y_loc;
vector<lower=0>[n_bins + 1] log_sigma_y_scale;