// GL nodes/weights are passed as data so Python and Stan use identical quadrature points.
functions {
    #include ../functions/shared.stan
    #include ../functions/carrying_capacity.stan
}
data {
    int<lower=1> n_bins;
    vector[n_bins] left_bins;
    vector[n_bins] right_bins;
    real Ne_c;
    real Ne_a;
    real t0;
    real t1;
    real alpha;
    real<lower=0> mutation_rate;
    int<lower=1> sample_size;
    int<lower=1> n_quad;
    vector[n_quad] gl_nodes;
    vector[n_quad] gl_weights;
}
parameters {}
model {}
generated quantities {
    real mu_div = mu_div_carrying_capacity(Ne_c, Ne_a, t0, t1, alpha, mutation_rate,
                                           n_quad, gl_nodes, gl_weights);
    vector[n_bins] mu_ld = correct_ld_finite_sample(
        mu_ld_carrying_capacity(Ne_c, Ne_a, t0, t1, alpha, left_bins, right_bins,
                                n_quad, gl_nodes, gl_weights),
        sample_size
    );
}
