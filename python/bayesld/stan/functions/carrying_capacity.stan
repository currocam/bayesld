// Exponential carrying-capacity demography:
// Ne(t) = Ne_c                          for t < t0     (recent constant)
// Ne(t) = Ne_c * exp(-alpha * (t - t0)) for t0 <= t < t1 (exponential growth backwards)
// Ne(t) = Ne_a                          for t >= t1    (ancestral constant)
//
// Piece 1 [0, t0]  — closed form.
// Piece 2 [t0, t1] — Gauss-Legendre quadrature (integrand has no closed form).
// Piece 3 [t1, ∞)  — closed form.

// Survival probability at genetic distance u.
// i.e. the probability that two loci separated by distance u have not experienced
// a recombination since the common ancestor.
real S_u_carrying_capacity(real u, real Ne_c, real Ne_a, real t0, real t1, real alpha,
                            int n_quad, vector gl_nodes, vector gl_weights) {
    real dt = t1 - t0;

    // Piece 1: [0, t0] — constant Ne_c, closed form (no alpha dependence)
    real piece1 = (1.0 - exp(-t0 * (4.0*Ne_c*u + 1.0) / (Ne_c * 2.0)))
                  / (4.0*Ne_c*u + 1.0);

    // Piece 2: [t0, t1] — exponential growth phase, GL quadrature
    real half_dt = dt / 2.0;
    real mid_t   = (t0 + t1) / 2.0;
    real sum2 = 0.0;
    for (k in 1:n_quad) {
        real t_k    = half_dt * gl_nodes[k] + mid_t;
        real s      = t_k - t0;
        real exp_arg = (-s * expm1_over_x(s * alpha) + 2.0*s*Ne_c*alpha
                        - 4.0*Ne_c*t_k*u - t0) / (Ne_c * 2.0);
        sum2 += gl_weights[k] * exp(exp_arg);
    }
    real piece2 = half_dt * sum2 / (Ne_c * 2.0);

    // Piece 3: [t1, ∞) — constant Ne_a, closed form
    real piece3_exp = (-dt * expm1_over_x(dt * alpha) - (4.0*Ne_c*t1*u + t0)) / (Ne_c * 2.0);
    real piece3 = exp(piece3_exp) / (4.0*Ne_a*u + 1.0);

    return piece1 + piece2 + piece3;
}

// map_rect shard: computes raw LD for a single bin (parallel across bins).
// phi = [Ne_c, Ne_a, t0, t1, alpha]
// x_r = [left_bin, right_bin, gl_nodes[1..n_quad], gl_weights[1..n_quad]]
// x_i = [n_quad]
vector mu_ld_shard_cc(vector phi, vector theta,
                      data array[] real x_r, data array[] int x_i) {
    real Ne_c  = phi[1];
    real Ne_a  = phi[2];
    real t0    = phi[3];
    real t1    = phi[4];
    real alpha = phi[5];
    int n_quad = x_i[1];

    vector[n_quad] gl_nodes;
    vector[n_quad] gl_weights;
    for (k in 1:n_quad) {
        gl_nodes[k]   = x_r[2 + k];
        gl_weights[k] = x_r[2 + n_quad + k];
    }

    real mid = (x_r[1] + x_r[2]) / 2.0;
    real hw  = (x_r[2] - x_r[1]) / 2.0;
    real s = 0.0;
    for (k in 1:n_quad) {
        real u_k = mid + hw * gl_nodes[k];
        s += gl_weights[k] * S_u_carrying_capacity(u_k, Ne_c, Ne_a, t0, t1, alpha,
                                                    n_quad, gl_nodes, gl_weights);
    }
    return [0.5 * s]';
}

// Expected LD per bin via GL quadrature over bin width (serial fallback).
vector mu_ld_carrying_capacity(real Ne_c, real Ne_a, real t0, real t1, real alpha,
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
            s += gl_weights[k] * S_u_carrying_capacity(u_k, Ne_c, Ne_a, t0, t1, alpha,
                                                        n_quad, gl_nodes, gl_weights);
        }
        result[b] = 0.5 * s;  // average over bin: (1/width) * (width/2) * sum(w_k f(u_k))
    }
    return result;
}

real partial_ld_lp_cc(array[] real mean_ld_slice, int start, int end,
                      real C1, real t0, real Ne_c,
                      vector g, vector t_nodes,
                      real C3, real t1, real Ne_a,
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
            real piece1 = (1.0 - C1 * exp(-2.0 * t0 * u)) / (4.0 * Ne_c * u + 1.0);
            real piece2 = dot_product(g, exp(-2.0 * t_nodes * u));
            real piece3 = C3 * exp(-2.0 * t1 * u) / (4.0 * Ne_a * u + 1.0);
            s += gl_weights[j] * (piece1 + piece2 + piece3);
        }
        real mu_b = correct_ld_finite_sample_scalar(0.5 * s, sample_size);
        lp += normal_lpdf(mean_ld_slice[i] | mu_b, est_sigma_ld[b]);
    }
    return lp;
}

real ld_lp_carrying_capacity(vector mean_ld, vector est_sigma_ld, int sample_size,
                              real Ne_c, real Ne_a, real t0, real t1, real alpha,
                              vector left_bins, vector right_bins,
                              int n_quad, vector gl_nodes, vector gl_weights) {
    real dt      = t1 - t0;
    real half_dt = dt / 2.0;
    real mid_t   = (t0 + t1) / 2.0;
    real scale   = half_dt / (Ne_c * 2.0);

    // Piece 1 prefactor
    real C1 = exp(-t0 / (2.0 * Ne_c));

    // Piece 2 weights and nodes (the expensive expm1_over_x calls happen here, once)
    vector[n_quad] g;
    vector[n_quad] t_nodes;
    for (k in 1:n_quad) {
        real t_k = half_dt * gl_nodes[k] + mid_t;
        real s_k = t_k - t0;
        t_nodes[k] = t_k;
        g[k] = scale * gl_weights[k]
               * exp((-s_k * expm1_over_x(s_k * alpha) + 2.0*s_k*Ne_c*alpha - t0) / (Ne_c * 2.0));
    }

    // Piece 3 prefactor
    real C3 = exp((-dt * expm1_over_x(dt * alpha) - t0) / (Ne_c * 2.0));

    array[num_elements(mean_ld)] real mean_ld_arr = to_array_1d(mean_ld);
    return reduce_sum(partial_ld_lp_cc, mean_ld_arr, 1,
                      C1, t0, Ne_c,
                      g, t_nodes,
                      C3, t1, Ne_a,
                      left_bins, right_bins, est_sigma_ld,
                      gl_nodes, gl_weights, sample_size);
}

// Expected genetic diversity.
real mu_div_carrying_capacity(real Ne_c, real Ne_a, real t0, real t1, real alpha,
                               real mutation_rate,
                               int n_quad, vector gl_nodes, vector gl_weights) {
    real dt = t1 - t0;

    // Piece 2: GL quadrature over [t0, t1]
    real half_dt  = dt / 2.0;
    real mid_t    = (t0 + t1) / 2.0;
    real int_piece = 0.0;
    for (k in 1:n_quad) {
        real t_k    = half_dt * gl_nodes[k] + mid_t;
        real s      = t_k - t0;
        real exp_arg = (-s * expm1_over_x(s * alpha) + 2.0*s*Ne_c*alpha - t0) / (Ne_c * 2.0);
        int_piece += gl_weights[k] * t_k * exp(exp_arg);
    }
    int_piece *= half_dt;

    // Piece 3 prefactor exp(-Gamma(t1))
    real exp_t1 = exp(-(t0 + dt * expm1_over_x(dt * alpha)) / (Ne_c * 2.0));

    // Piece 1 closed-form contribution
    real exp_t0 = exp(-t0 / (2.0 * Ne_c));

    real expected_tmrca = (int_piece
                           + (2.0*t1 + 4.0*Ne_a) * Ne_c * exp_t1
                           + (-4.0*square(Ne_c) - 2.0*Ne_c*t0) * exp_t0
                           + 4.0*square(Ne_c))
                          / (Ne_c * 2.0);

    return expected_tmrca * 2.0 * mutation_rate;
}
