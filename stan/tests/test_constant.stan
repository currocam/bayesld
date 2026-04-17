functions {
    #include ../functions/shared.stan
    #include ../functions/constant.stan
}
data {
    int<lower=1> n_bins;
    vector[n_bins] left_bins;
    vector[n_bins] right_bins;
    real<lower=0> Ne;
    real<lower=0> mutation_rate;
    int<lower=1> sample_size;
}
parameters {}
model {}
generated quantities {
    real mu_div = 4.0 * Ne * mutation_rate;
    vector[n_bins] mu_ld = correct_ld_finite_sample(
        mu_ld_constant(Ne, left_bins, right_bins),
        sample_size
    );
}
