// Piecewise-exponential demography:
// Ne(t) = Ne_c * exp(-alpha * t)   for t < t0
// Ne(t) = Ne_a                     for t >= t0
// Diploid (ploidy=2): Ne_c, Ne_a enter unchanged.
//
// All expressions involving alpha use expm1_over_x (defined in shared.stan) to
// remove the removable singularity at alpha=0; the formulas are valid for all alpha.

// Survival probability at genetic distance u.
// i.e. the probability that two loci separated by distance u have not experienced
// a recombination since the common ancestor.
// piece1 [0, t0]: Gauss-Legendre quadrature | piece2 [t0, ∞): closed form
real S_u_piecewise_exponential(real u, real Ne_c, real Ne_a, real t0, real alpha,
                               int n_quad, vector gl_nodes, vector gl_weights) {
    real half_t0 = t0 / 2.0;
    real sum1 = 0.0;
    for (k in 1:n_quad) {
        real t_k    = half_t0 * gl_nodes[k] + half_t0;
        real exp_arg = t_k * alpha - 2.0*t_k*u - t_k * expm1_over_x(t_k * alpha) / (2.0*Ne_c);
        sum1 += gl_weights[k] * exp(exp_arg);
    }
    real piece1 = half_t0 * sum1 / (Ne_c * 2.0);
    real piece2_exp = -2.0*u*t0 - t0 * expm1_over_x(t0 * alpha) / (2.0*Ne_c);
    real piece2 = exp(piece2_exp) / (4.0*u*Ne_a + 1.0);
    return piece1 + piece2;
}

// Expected LD per bin via GL quadrature over bin width.
vector mu_ld_piecewise_exponential(real Ne_c, real Ne_a, real t0, real alpha,
                                   vector left_bins, vector right_bins,
                                   int n_quad, vector gl_nodes, vector gl_weights) {
    int n_bins = num_elements(left_bins);
    vector[n_bins] result;
    for (b in 1:n_bins) {
        real mid        = (left_bins[b] + right_bins[b]) / 2.0;
        real half_width = (right_bins[b] - left_bins[b]) / 2.0;
        real s = 0.0;
        for (k in 1:n_quad) {
            real u_k = mid + half_width * gl_nodes[k];
            s += gl_weights[k] * S_u_piecewise_exponential(u_k, Ne_c, Ne_a, t0, alpha,
                                                            n_quad, gl_nodes, gl_weights);
        }
        result[b] = 0.5 * s;  // average over bin: (1/width) * (width/2) * sum(w_k f(u_k))
    }
    return result;
}

real partial_ld_lp_pe(array[] real mean_ld_slice, int start, int end,
                      vector g, vector t_nodes, real C_t0, real t0, real Ne_a,
                      vector left_bins, vector right_bins, vector est_sigma_ld,
                      vector gl_nodes, vector gl_weights, int sample_size) {
    real lp = 0.0;
    int n_quad = num_elements(g);
    for (i in 1:(end - start + 1)) {
        int b = start + i - 1;
        real mid = (left_bins[b] + right_bins[b]) / 2.0;
        real hw  = (right_bins[b] - left_bins[b]) / 2.0;
        real s = 0.0;
        for (j in 1:n_quad) {
            real u      = mid + hw * gl_nodes[j];
            real piece1 = dot_product(g, exp(-2.0 * t_nodes * u));
            real piece2 = C_t0 * exp(-2.0 * u * t0) / (4.0 * u * Ne_a + 1.0);
            s += gl_weights[j] * (piece1 + piece2);
        }
        real mu_b = correct_ld_finite_sample_scalar(0.5 * s, sample_size);
        lp += normal_lpdf(mean_ld_slice[i] | mu_b, est_sigma_ld[b]);
    }
    return lp;
}

real ld_lp_piecewise_exponential(vector mean_ld, vector est_sigma_ld, int sample_size,
                                  real Ne_c, real Ne_a, real t0, real alpha,
                                  vector left_bins, vector right_bins,
                                  int n_quad, vector gl_nodes, vector gl_weights) {
    real half_t0 = t0 / 2.0;
    real scale   = half_t0 / (Ne_c * 2.0);
    vector[n_quad] g;
    vector[n_quad] t_nodes;
    for (k in 1:n_quad) {
        real t_k   = half_t0 * gl_nodes[k] + half_t0;
        t_nodes[k] = t_k;
        g[k] = scale * gl_weights[k]
               * exp(t_k * alpha - t_k * expm1_over_x(t_k * alpha) / (2.0 * Ne_c));
    }
    real C_t0 = exp(-t0 * expm1_over_x(t0 * alpha) / (2.0 * Ne_c));
    array[num_elements(mean_ld)] real mean_ld_arr = to_array_1d(mean_ld);
    return reduce_sum(partial_ld_lp_pe, mean_ld_arr, 1,
                      g, t_nodes, C_t0, t0, Ne_a,
                      left_bins, right_bins, est_sigma_ld,
                      gl_nodes, gl_weights, sample_size);
}

// Expected genetic diversity.
real mu_div_piecewise_exponential(real Ne_c, real Ne_a, real t0, real alpha,
                                   real mutation_rate,
                                   int n_quad, vector gl_nodes, vector gl_weights) {
    real half_t0 = t0 / 2.0;
    real sum1 = 0.0;
    for (k in 1:n_quad) {
        real t_k    = half_t0 * gl_nodes[k] + half_t0;
        real exp_arg = t_k * alpha - t_k * expm1_over_x(t_k * alpha) / (2.0*Ne_c);
        sum1 += gl_weights[k] * t_k * exp(exp_arg);
    }
    real piece1 = half_t0 * sum1 / (Ne_c * 2.0);
    real piece2 = (2.0*Ne_a + t0) * exp(-t0 * expm1_over_x(t0 * alpha) / (2.0*Ne_c));
    return (piece1 + piece2) * 2.0 * mutation_rate;
}
