// ---- data_surrogate.stan ----
// GP-surrogate synthetic points + HSGP hyperparameters. Include at the *bottom*
// of the data block. These entries are populated by _surrogate.surrogate_payload
// and the engine hyperprior; they are parameterization-independent.

// Bias points (all accumulated): per-bin relative-bias observations.
int<lower=0> n_bias;
array[n_bias] vector[n_bins] log_det_ld_bias;
array[n_bias] vector[n_bins] rel_bias_bias;
array[n_bias] vector<lower=0>[n_bins] eps_rel_bias;
// Sigma points (last iteration only): sufficient-stat MVN surrogate.
int<lower=0> n_sigma;
array[n_sigma] vector[n_bins + 1] y_bar_sigma;
array[n_sigma] matrix[n_bins + 1, n_bins + 1] L_S_sigma;
array[n_sigma] int<lower=1> N_rep_sigma;
vector[n_sigma] det_pi_sigma;
array[n_sigma] vector[n_bins] det_ld_sigma;
array[n_sigma] vector[n_bins] log_det_ld_sigma;

// HSGP hyperparameters: shared lengthscale/amplitude across per-bin GPs.
real<lower=0> hsgp_c;
int<lower=1>  hsgp_m_ld;
real<lower=0> gp_alpha_std;
// Standardization of log(det_ld) axis (from observed windows).
real log_ld_mu;
real<lower=0> log_ld_sig;
// Sigma prior hyperparameters.
real<lower=0> lkj_eta;
vector[n_bins + 1] log_sigma_y_loc;
vector<lower=0>[n_bins + 1] log_sigma_y_scale;
