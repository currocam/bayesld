functions {
    #include ../functions/shared.stan
    #include ../functions/piecewise_constant.stan
}
data {
    int<lower=1> n_bins;
    vector[n_bins] left_bins;
    vector[n_bins] right_bins;
    vector[n_bins] obs_ld;
    vector[n_bins] est_sigma_ld;
    int<lower=1> n_epochs;
    vector[n_epochs] Ne_values;
    vector[n_epochs - 1] t_boundaries;
    int<lower=1> sample_size;
    int<lower=1> n_quad;
    vector[n_quad] gl_nodes;
    vector[n_quad] gl_weights;
}
parameters {}
model {}
generated quantities {
    real lp = ld_lp_piecewise_constant(
        obs_ld, est_sigma_ld, sample_size,
        n_epochs, Ne_values, t_boundaries,
        left_bins, right_bins,
        n_quad, gl_nodes, gl_weights
    );
}
