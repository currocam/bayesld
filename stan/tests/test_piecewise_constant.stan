functions {
    #include ../functions/shared.stan
    #include ../functions/piecewise_constant.stan
}
data {
    int<lower=1> n_bins;
    vector[n_bins] left_bins;
    vector[n_bins] right_bins;
    int<lower=1> n_epochs;
    vector[n_epochs] Ne_values;
    vector[n_epochs - 1] t_boundaries;
    real<lower=0> mutation_rate;
    int<lower=1> sample_size;
    int<lower=1> n_quad;
    vector[n_quad] gl_nodes;
    vector[n_quad] gl_weights;
}
parameters {}
model {}
generated quantities {
    real mu_div = mu_div_piecewise_constant(n_epochs, Ne_values, t_boundaries, mutation_rate);
    vector[n_bins] mu_ld = correct_ld_finite_sample(
        mu_ld_piecewise_constant(n_epochs, Ne_values, t_boundaries, left_bins, right_bins,
                                 n_quad, gl_nodes, gl_weights),
        sample_size
    );
}
