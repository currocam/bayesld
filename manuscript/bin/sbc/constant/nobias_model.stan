functions {
// ---- shared.stan ----
// expm1(x)/x, stable at x=0 (limit = 1). expm1 is accurate for all x != 0.
real expm1_over_x(real x) {
    if (x == 0.0)
        return 1.0;
    return expm1(x) / x;
}

// Finite-sample bias correction for LD (Fournier et al. 2023).
// S = 2 * sample_size (haploid count for diploid individuals).
vector correct_ld_finite_sample(vector mu, int sample_size) {
    int S = 2 * sample_size;
    real beta = 1.0 / square(S - 1);
    real alpha_corr = square(square(S) - S + 2.0) / square(square(S) - 3.0 * S + 2.0);
    return (alpha_corr - beta) * mu + 4.0 * beta;
}

// Scalar overload for use inside reduce_sum partial functions.
real correct_ld_finite_sample_scalar(real mu, int sample_size) {
    int S = 2 * sample_size;
    real beta = 1.0 / square(S - 1);
    real alpha_corr = square(square(S) - S + 2.0) / square(square(S) - 3.0 * S + 2.0);
    return (alpha_corr - beta) * mu + 4.0 * beta;
}

// ---- constant.stan ----
// Expected LD under constant Ne (diploid, ploidy=2).
// Closed-form integral of coalescent-based LD decay over a recombination-distance bin.
vector mu_ld_constant(real Ne, vector left_bins, vector right_bins) {
    int n_bins = num_elements(left_bins);
    vector[n_bins] result;
    for (i in 1:n_bins) {
        real u_i = left_bins[i];
        real u_j = right_bins[i];
        // Use log1p for numerical stability when 4*Ne*u is large
        result[i] = (log1p(4.0 * Ne * u_j) - log1p(4.0 * Ne * u_i))
                    / (4.0 * Ne * (u_j - u_i));
    }
    return result;
}

}

data {
    int<lower=1> n_bins;
    int<lower=2> num_windows;
    vector[n_bins] left_bins;
    vector[n_bins] right_bins;
    real<lower=0> mutation_rate;
    int<lower=1> sample_size;
    vector[num_windows] pi_array;
    matrix[num_windows, n_bins] ld_mat;
    // Prior
    real mu_log_ne_prior;
    real<lower=0> sigma_log_ne_prior;
}

transformed data {
    real mean_div = mean(pi_array);
    real<lower=0> sigma_div = sd(pi_array);
    real<lower=0> sem_div   = sigma_div / sqrt(num_windows);

    vector[n_bins] mean_ld;
    vector<lower=0>[n_bins] sigma_ld;
    vector<lower=0>[n_bins] sem_ld;
    for (b in 1:n_bins) {
        mean_ld[b]  = mean(col(ld_mat, b));
        sigma_ld[b] = sd(col(ld_mat, b));
        sem_ld[b]   = sigma_ld[b] / sqrt(num_windows);
    }
    real log_ne_offset = log(mean_div / (4.0 * mutation_rate));
}

parameters {
    real<offset=log_ne_offset> log_Ne;
}

transformed parameters {
    real<lower=0> Ne = exp(log_Ne);
    real expected_pi = 4.0 * Ne * mutation_rate;
    vector[n_bins] approx_expected_ld = correct_ld_finite_sample(
        mu_ld_constant(Ne, left_bins, right_bins), sample_size
    );
}

model {
    // --- user prior ---
    log_Ne ~ normal(mu_log_ne_prior, sigma_log_ne_prior);
    mean_div ~ normal(expected_pi, sem_div);
    target += normal_lpdf(mean_ld | approx_expected_ld, sem_ld) / n_bins;
}

generated quantities {
    vector[num_windows] log_lik;
    for (w in 1:num_windows) {
        log_lik[w] = normal_lpdf(pi_array[w] | expected_pi, sigma_div)
                   + normal_lpdf(to_vector(ld_mat[w]) | approx_expected_ld, sigma_ld) / n_bins;
    }
}
