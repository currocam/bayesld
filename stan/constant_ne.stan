functions {
#include functions/shared.stan
#include functions/constant.stan
}

data {
    int<lower=1> n_bins;
    vector[n_bins] left_bins;   // left edges of LD bins in Morgans
    vector[n_bins] right_bins;  // right edges of LD bins in Morgans
    real<lower=0> mutation_rate;
    int<lower=1> sample_size;   // number of diploid individuals (for finite-sample correction)

    // Observed summaries (mean and std-error across replicates/windows)
    real<lower=0> mean_div;
    real<lower=0> est_sigma_div;
    vector<lower=0>[n_bins] mean_ld;
    vector<lower=0>[n_bins] est_sigma_ld;

    // Lognormal prior hyperparameters for Ne
    real log_Ne_prior_mu;
    real<lower=0> log_Ne_prior_sigma;
}

parameters {
    real log_Ne;
}

transformed parameters {
    real<lower=0> Ne = exp(log_Ne);
}

model {
    // Lognormal prior on Ne (i.e. log_Ne ~ Normal)
    log_Ne ~ normal(log_Ne_prior_mu, log_Ne_prior_sigma);

    // Deterministic predictions
    real mu_div = 4.0 * Ne * mutation_rate;
    vector[n_bins] mu_ld = correct_ld_finite_sample(
        mu_ld_constant(Ne, left_bins, right_bins),
        sample_size
    );

    // Diversity likelihood (1 term)
    mean_div ~ normal(mu_div, est_sigma_div);

    // LD likelihood: each bin contributes 1/n_bins so total LD weight == diversity weight
    target += normal_lpdf(mean_ld | mu_ld, est_sigma_ld) / n_bins;
}

generated quantities {
    real E_pi = 4.0 * Ne * mutation_rate;
    vector[n_bins] approx_ld = correct_ld_finite_sample(
        mu_ld_constant(Ne, left_bins, right_bins), sample_size
    );
}
