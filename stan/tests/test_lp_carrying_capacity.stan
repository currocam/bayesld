functions {
    #include ../functions/shared.stan
    #include ../functions/carrying_capacity.stan
}
data {
    int<lower=1> n_bins;
    vector[n_bins] left_bins;
    vector[n_bins] right_bins;
    vector[n_bins] obs_ld;
    vector[n_bins] est_sigma_ld;
    real Ne_c;
    real Ne_a;
    real t0;
    real t1;
    real alpha;
    int<lower=1> sample_size;
    int<lower=1> n_quad;
    vector[n_quad] gl_nodes;
    vector[n_quad] gl_weights;
}
parameters {}
model {}
generated quantities {
    real lp = ld_lp_carrying_capacity(
        obs_ld, est_sigma_ld, sample_size,
        Ne_c, Ne_a, t0, t1, alpha,
        left_bins, right_bins,
        n_quad, gl_nodes, gl_weights
    );
}
