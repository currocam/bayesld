// Piecewise-constant demography with n_epochs epochs.
// Ne(t) = Ne_values[i]         for t in [t_boundaries[i-1], t_boundaries[i])  (i = 1..n_epochs-1)
// Ne(t) = Ne_values[n_epochs]  for t >= t_boundaries[n_epochs-1]
//
// Stan 1-based indexing: Ne_values[1..n_epochs], t_boundaries[1..n_epochs-1].
//
// All per-epoch integrals have closed forms — no quadrature needed.

// Survival probability at genetic distance u.
// i.e. the probability that two loci separated by distance u have not experienced
// a recombination since the common ancestor.
real S_u_piecewise_constant(real u, int n_epochs, vector Ne_values, vector t_boundaries) {
    real total     = 0.0;
    real Gamma_prev = 0.0;   // cumulative coalescent hazard Gamma(t_prev)
    real t_prev    = 0.0;
    for (i in 1:n_epochs - 1) {
        real Ne_i   = Ne_values[i];
        real t_curr = t_boundaries[i];
        real dt     = t_curr - t_prev;
        real c      = 2.0*u + 1.0 / (2.0*Ne_i);  // combined rate at this u
        total += exp(-Gamma_prev - 2.0*u*t_prev) * (1.0 - exp(-c * dt)) / (4.0*Ne_i*u + 1.0);
        Gamma_prev += dt / (2.0 * Ne_i);
        t_prev = t_curr;
    }
    // Last semi-infinite epoch [t_boundaries[n_epochs-1], ∞)
    real Ne_last = Ne_values[n_epochs];
    total += exp(-Gamma_prev - 2.0*u*t_prev) / (4.0*Ne_last*u + 1.0);
    return total;
}

// Expected LD per bin via GL quadrature over bin width.
vector mu_ld_piecewise_constant(int n_epochs, vector Ne_values, vector t_boundaries,
                                 vector left_bins, vector right_bins,
                                 int n_quad, vector gl_nodes, vector gl_weights) {
    int n_bins = num_elements(left_bins);
    vector[n_bins] result;
    for (b in 1:n_bins) {
        real mid       = (left_bins[b] + right_bins[b]) / 2.0;
        real half_width = (right_bins[b] - left_bins[b]) / 2.0;
        real s = 0.0;
        for (k in 1:n_quad) {
            real u_k = mid + half_width * gl_nodes[k];
            s += gl_weights[k] * S_u_piecewise_constant(u_k, n_epochs, Ne_values, t_boundaries);
        }
        result[b] = 0.5 * s;  // average over bin: (1/width) * (width/2) * sum(w_k f(u_k))
    }
    return result;
}

real partial_ld_lp_piecewise_constant(array[] real mean_ld_slice, int start, int end,
                                       vector left_bins, vector right_bins,
                                       vector est_sigma_ld,
                                       int n_epochs, vector Ne_values, vector t_boundaries,
                                       int n_quad, vector gl_nodes, vector gl_weights,
                                       int sample_size) {
    real lp = 0.0;
    for (i in 1:(end - start + 1)) {
        int b = start + i - 1;
        real mid        = (left_bins[b] + right_bins[b]) / 2.0;
        real half_width = (right_bins[b] - left_bins[b]) / 2.0;
        real su = 0.0;
        for (k in 1:n_quad) {
            real u_k = mid + half_width * gl_nodes[k];
            su += gl_weights[k] * S_u_piecewise_constant(u_k, n_epochs, Ne_values, t_boundaries);
        }
        real mu_b = correct_ld_finite_sample_scalar(0.5 * su, sample_size);
        lp += normal_lpdf(mean_ld_slice[i] | mu_b, est_sigma_ld[b]);
    }
    return lp;
}

real ld_lp_piecewise_constant(vector mean_ld, vector est_sigma_ld, int sample_size,
                               int n_epochs, vector Ne_values, vector t_boundaries,
                               vector left_bins, vector right_bins,
                               int n_quad, vector gl_nodes, vector gl_weights) {
    array[num_elements(mean_ld)] real mean_ld_arr = to_array_1d(mean_ld);
    return reduce_sum(partial_ld_lp_piecewise_constant, mean_ld_arr, 1,
                      left_bins, right_bins, est_sigma_ld,
                      n_epochs, Ne_values, t_boundaries,
                      n_quad, gl_nodes, gl_weights, sample_size);
}

// Expected genetic diversity.
real mu_div_piecewise_constant(int n_epochs, vector Ne_values, vector t_boundaries,
                                real mutation_rate) {
    real tmrca     = 0.0;
    real Gamma_prev = 0.0;
    real t_prev    = 0.0;
    for (i in 1:n_epochs - 1) {
        real Ne_i   = Ne_values[i];
        real t_curr = t_boundaries[i];
        real dt     = t_curr - t_prev;
        tmrca += exp(-Gamma_prev) * ((t_prev + 2.0*Ne_i)
                                     - (t_curr + 2.0*Ne_i) * exp(-dt / (2.0*Ne_i)));
        Gamma_prev += dt / (2.0 * Ne_i);
        t_prev = t_curr;
    }
    // Last epoch
    real Ne_last = Ne_values[n_epochs];
    tmrca += exp(-Gamma_prev) * (t_prev + 2.0 * Ne_last);
    return tmrca * 2.0 * mutation_rate;
}
