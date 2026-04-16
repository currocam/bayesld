functions {
    // Deterministic expected LD under constant Ne (diploid, ploidy=2).
    // Ne_internal = Ne (because Ne/2 * 2 = Ne).
    // Formula: integral of coalescent-based LD decay over a recombination-distance bin.
    vector mu_ld_constant(real Ne, vector left_bins, vector right_bins) {
        int n_bins = num_elements(left_bins);
        vector[n_bins] result;
        for (i in 1:n_bins) {
            real u_i = left_bins[i];
            real u_j = right_bins[i];
            result[i] = (-log(4.0 * Ne * u_i + 1.0) + log(4.0 * Ne * u_j + 1.0))
                        / (4.0 * Ne * (u_j - u_i));
        }
        return result;
    }

    // Finite-sample bias correction for LD (Fournier et al. 2023).
    // Corrects upward bias in r^2-based LD estimates with limited sample size.
    // S = 2 * sample_size (haploid count for diploid individuals).
    vector correct_ld_finite_sample(vector mu, int sample_size) {
        int S = 2 * sample_size;
        real beta = 1.0 / square(S - 1);
        real alpha_corr = square(square(S) - S + 2.0) / square(square(S) - 3.0 * S + 2.0);
        return (alpha_corr - beta) * mu + 4.0 * beta;
    }
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
    for (b in 1:n_bins) {
        target += normal_lpdf(mean_ld[b] | mu_ld[b], est_sigma_ld[b]) / n_bins;
    }
}
