functions {
// ---- shared.stan ----
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

// ---- piecewise_constant.stan ----
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
//
// Numerically stable via log-sum-exp accumulation:
//   log_term[i] = -Gamma_prev - 2u*t_prev + log1m_exp(-c*dt) - log1p(4*Ne_i*u)
// This avoids (1) catastrophic cancellation in (1-exp(-c*dt)) when c*dt≪1, and
// (2) underflow of exp(-Gamma_prev - 2u*t_prev) when the cumulative hazard is large.
real S_u_piecewise_constant(real u, int n_epochs, vector Ne_values, vector t_boundaries) {
    array[n_epochs] real log_terms;
    real Gamma_prev = 0.0;   // cumulative coalescent hazard Gamma(t_prev)
    real t_prev     = 0.0;
    for (i in 1:n_epochs - 1) {
        real Ne_i   = Ne_values[i];
        real t_curr = t_boundaries[i];
        real dt     = t_curr - t_prev;
        real c      = 2.0*u + 1.0 / (2.0*Ne_i);  // combined rate at this u
        // log1m_exp(-c*dt) = log(1 - exp(-c*dt)); argument -c*dt < 0 always.
        log_terms[i] = -Gamma_prev - 2.0*u*t_prev
                       + log1m_exp(-c * dt)
                       - log1p(4.0*Ne_i*u);
        Gamma_prev += dt / (2.0 * Ne_i);
        t_prev = t_curr;
    }
    // Last semi-infinite epoch [t_boundaries[n_epochs-1], ∞): 1 - exp(-∞) = 1.
    real Ne_last = Ne_values[n_epochs];
    log_terms[n_epochs] = -Gamma_prev - 2.0*u*t_prev - log1p(4.0*Ne_last*u);
    return exp(log_sum_exp(to_vector(log_terms)));
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
//
// Numerically stable via log-sum-exp; per-epoch log-contribution:
//   log_term[i] = -Gamma_prev + log_diff_exp(log(t_prev+2Ne_i),
//                                             log(t_curr+2Ne_i) - dt/(2Ne_i))
// log_diff_exp avoids cancellation when dt/(2Ne_i) is small (inner terms nearly equal).
// exp(-Gamma_prev) underflow is avoided by keeping Gamma_prev in log space.
real mu_div_piecewise_constant(int n_epochs, vector Ne_values, vector t_boundaries,
                                real mutation_rate) {
    array[n_epochs] real log_terms;
    real Gamma_prev = 0.0;
    real t_prev     = 0.0;
    for (i in 1:n_epochs - 1) {
        real Ne_i   = Ne_values[i];
        real t_curr = t_boundaries[i];
        real dt     = t_curr - t_prev;
        // log((t_prev+2Ne_i) - (t_curr+2Ne_i)*exp(-dt/(2Ne_i)))
        //   = log_diff_exp(log(t_prev+2Ne_i), log(t_curr+2Ne_i) - dt/(2Ne_i))
        log_terms[i] = -Gamma_prev
                       + log_diff_exp(log(t_prev + 2.0*Ne_i),
                                      log(t_curr + 2.0*Ne_i) - dt / (2.0*Ne_i));
        Gamma_prev += dt / (2.0 * Ne_i);
        t_prev = t_curr;
    }
    // Last epoch: semi-infinite, contribution = exp(-Gamma_prev) * (t_prev + 2*Ne_last)
    real Ne_last = Ne_values[n_epochs];
    log_terms[n_epochs] = -Gamma_prev + log(t_prev + 2.0 * Ne_last);
    return exp(log_sum_exp(to_vector(log_terms))) * 2.0 * mutation_rate;
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
    int<lower=2> n_epochs;
    int<lower=1> n_quad;
    vector[n_quad] gl_nodes;
    vector[n_quad] gl_weights;
    // Prior
    real mu_log_ne1_prior;
    real mu_log_ne2_prior;
    real<lower=0> sigma_log_ne_prior;
    real mu_log_t0_prior;
    real<lower=0> sigma_log_t0_prior;
    // Joint MVN noise prior (matches corrected model's likelihood scaffold)
    real<lower=0> lkj_eta;
    vector[n_bins + 1] log_sigma_y_loc;
    vector<lower=0>[n_bins + 1] log_sigma_y_scale;
}

transformed data {
    int D = n_bins + 1;
    array[num_windows] vector[D] y_obs;
    for (w in 1:num_windows) {
        y_obs[w, 1] = pi_array[w];
        for (b in 1:n_bins) y_obs[w, b + 1] = ld_mat[w, b];
    }
    real log_ne_offset = log(mean(pi_array) / (4.0 * mutation_rate));
}

parameters {
    real<offset=log_ne_offset> log_Ne_c;
    real<offset=log_ne_offset> log_Ne_a;
    real log_t0;
    vector[n_bins + 1] log_sigma_y;
    cholesky_factor_corr[n_bins + 1] L_Omega;
}

transformed parameters {
    real<lower=0> Ne_c = exp(log_Ne_c);
    real<lower=0> Ne_a = exp(log_Ne_a);
    real<lower=0> t0   = exp(log_t0);
    vector[2] Ne_values = [Ne_c, Ne_a]';
    vector[1] t_boundaries = [t0]';

    real expected_pi = mu_div_piecewise_constant(n_epochs, Ne_values, t_boundaries, mutation_rate);
    vector[n_bins] approx_expected_ld = correct_ld_finite_sample(
        mu_ld_piecewise_constant(n_epochs, Ne_values, t_boundaries,
                                  left_bins, right_bins, n_quad, gl_nodes, gl_weights),
        sample_size
    );
    vector[n_bins + 1] mu_y;
    mu_y[1] = expected_pi;
    for (b in 1:n_bins) mu_y[b + 1] = approx_expected_ld[b];
    vector<lower=0>[n_bins + 1] sigma_y = exp(log_sigma_y);
    matrix[n_bins + 1, n_bins + 1] L_Sigma = diag_pre_multiply(sigma_y, L_Omega);
}

model {
    log_Ne_c ~ normal(mu_log_ne1_prior, sigma_log_ne_prior);
    log_Ne_a ~ normal(mu_log_ne2_prior, sigma_log_ne_prior);
    log_t0   ~ normal(mu_log_t0_prior, sigma_log_t0_prior);
    L_Omega ~ lkj_corr_cholesky(lkj_eta);
    log_sigma_y ~ normal(log_sigma_y_loc, log_sigma_y_scale);
    y_obs ~ multi_normal_cholesky(mu_y, L_Sigma);
}

generated quantities {
    vector[num_windows] log_lik;
    for (w in 1:num_windows) {
        log_lik[w] = multi_normal_cholesky_lpdf(y_obs[w] | mu_y, L_Sigma);
    }
    matrix[n_bins + 1, n_bins + 1] Omega = multiply_lower_tri_self_transpose(L_Omega);
}
