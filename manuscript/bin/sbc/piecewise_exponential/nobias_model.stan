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

// ---- piecewise_exponential.stan ----
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
//
// Numerically stable via log-sum-exp:
//   piece1 GL sum:  log_sum_exp(log(w_k) + exp_arg_k) avoids underflow when 2*t_k*u is large.
//   piece2:         exp_arg - log1p(4*u*Ne_a) avoids (exp * division) precision loss.
//   piece1 + piece2: log_sum_exp([log_piece1, log_piece2]) handles magnitude mismatch.
real S_u_piecewise_exponential(real u, real Ne_c, real Ne_a, real t0, real alpha,
                               int n_quad, vector gl_nodes, vector gl_weights) {
    real half_t0 = t0 / 2.0;
    vector[n_quad] log_terms;
    for (k in 1:n_quad) {
        real t_k     = half_t0 * gl_nodes[k] + half_t0;
        real exp_arg = t_k * alpha - 2.0*t_k*u - t_k * expm1_over_x(t_k * alpha) / (2.0*Ne_c);
        log_terms[k] = log(gl_weights[k]) + exp_arg;
    }
    real log_piece1 = log(half_t0) - log(Ne_c * 2.0) + log_sum_exp(log_terms);
    real piece2_exp = -2.0*u*t0 - t0 * expm1_over_x(t0 * alpha) / (2.0*Ne_c);
    real log_piece2 = piece2_exp - log1p(4.0*u*Ne_a);
    return exp(log_sum_exp([log_piece1, log_piece2]'));
}

// map_rect shard: computes raw LD for a single bin (parallel across bins).
// phi = [Ne_c, Ne_a, t0, alpha]
// x_r = [left_bin, right_bin, gl_nodes[1..n_quad], gl_weights[1..n_quad]]
// x_i = [n_quad]
vector mu_ld_shard_pe(vector phi, vector theta,
                      data array[] real x_r, data array[] int x_i) {
    real Ne_c  = phi[1];
    real Ne_a  = phi[2];
    real t0    = phi[3];
    real alpha = phi[4];
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
        s += gl_weights[k] * S_u_piecewise_exponential(u_k, Ne_c, Ne_a, t0, alpha,
                                                        n_quad, gl_nodes, gl_weights);
    }
    return [0.5 * s]';
}

// Expected LD per bin via GL quadrature over bin width (serial fallback).
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

// Calls stable S_u_piecewise_exponential directly — no precomputed linear-space
// intermediates that could underflow before reaching the inner loop.
real partial_ld_lp_pe(array[] real mean_ld_slice, int start, int end,
                      real Ne_c, real Ne_a, real t0, real alpha,
                      vector left_bins, vector right_bins, vector est_sigma_ld,
                      int n_quad, vector gl_nodes, vector gl_weights, int sample_size) {
    real lp = 0.0;
    for (i in 1:(end - start + 1)) {
        int b = start + i - 1;
        real mid = (left_bins[b] + right_bins[b]) / 2.0;
        real hw  = (right_bins[b] - left_bins[b]) / 2.0;
        real s = 0.0;
        for (k in 1:n_quad) {
            real u_k = mid + hw * gl_nodes[k];
            s += gl_weights[k] * S_u_piecewise_exponential(u_k, Ne_c, Ne_a, t0, alpha,
                                                            n_quad, gl_nodes, gl_weights);
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
    array[num_elements(mean_ld)] real mean_ld_arr = to_array_1d(mean_ld);
    return reduce_sum(partial_ld_lp_pe, mean_ld_arr, 1,
                      Ne_c, Ne_a, t0, alpha,
                      left_bins, right_bins, est_sigma_ld,
                      n_quad, gl_nodes, gl_weights, sample_size);
}

// Expected genetic diversity.
//
// Numerically stable via log-sum-exp:
//   piece1 GL sum: log(w_k) + log(t_k) + exp_arg_k avoids underflow.
//   piece2: log(2*Ne_a + t0) + coal_exp combines cleanly.
real mu_div_piecewise_exponential(real Ne_c, real Ne_a, real t0, real alpha,
                                   real mutation_rate,
                                   int n_quad, vector gl_nodes, vector gl_weights) {
    real half_t0 = t0 / 2.0;
    vector[n_quad] log_terms;
    for (k in 1:n_quad) {
        real t_k     = half_t0 * gl_nodes[k] + half_t0;
        real exp_arg = t_k * alpha - t_k * expm1_over_x(t_k * alpha) / (2.0*Ne_c);
        log_terms[k] = log(gl_weights[k]) + log(t_k) + exp_arg;
    }
    real log_piece1 = log(half_t0) - log(Ne_c * 2.0) + log_sum_exp(log_terms);
    real log_piece2 = log(2.0*Ne_a + t0) - t0 * expm1_over_x(t0 * alpha) / (2.0*Ne_c);
    return exp(log_sum_exp([log_piece1, log_piece2]')) * 2.0 * mutation_rate;
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
    int<lower=1> n_quad;
    vector[n_quad] gl_nodes;
    vector[n_quad] gl_weights;
    // Prior
    real mu_log_ne1_prior;
    real mu_log_ne2_prior;
    real<lower=0> sigma_log_ne_prior;
    real mu_log_t0_prior;
    real<lower=0> sigma_log_t0_prior;
    // Fixed per-component noise sds (empirical, computed upstream).
    vector<lower=0>[n_bins + 1] sigma_emp;
}

transformed data {
    int D = n_bins + 1;
    array[num_windows] vector[D] y_obs;
    for (w in 1:num_windows) {
        y_obs[w, 1] = pi_array[w];
        for (b in 1:n_bins) y_obs[w, b + 1] = ld_mat[w, b];
    }
    real log_ne_offset = log(mean(pi_array) / (4.0 * mutation_rate));

    // Pack per-bin data for map_rect parallelism
    array[n_bins, 2 + 2 * n_quad] real mr_bin_data;
    array[n_bins, 1] int mr_bin_int;
    array[n_bins] vector[0] mr_theta;
    for (b in 1:n_bins) {
        mr_bin_data[b, 1] = left_bins[b];
        mr_bin_data[b, 2] = right_bins[b];
        for (k in 1:n_quad) {
            mr_bin_data[b, 2 + k] = gl_nodes[k];
            mr_bin_data[b, 2 + n_quad + k] = gl_weights[k];
        }
        mr_bin_int[b, 1] = n_quad;
    }
}

parameters {
    real<offset=log_ne_offset> log_Ne_c;
    real<offset=log_ne_offset> log_Ne_a;
    real log_t0;
}

transformed parameters {
    real<lower=0> Ne_c = exp(log_Ne_c);
    real<lower=0> Ne_a = exp(log_Ne_a);
    real<lower=0> t0   = exp(log_t0);
    real alpha = (log_Ne_c - log_Ne_a) / t0;

    real expected_pi = mu_div_piecewise_exponential(Ne_c, Ne_a, t0, alpha, mutation_rate,
                                                    n_quad, gl_nodes, gl_weights);
    vector[4] mr_phi = [Ne_c, Ne_a, t0, alpha]';
    vector[n_bins] approx_expected_ld = correct_ld_finite_sample(
        map_rect(mu_ld_shard_pe, mr_phi, mr_theta, mr_bin_data, mr_bin_int),
        sample_size
    );
    vector[n_bins + 1] mu_y;
    mu_y[1] = expected_pi;
    for (b in 1:n_bins) mu_y[b + 1] = approx_expected_ld[b];
}

model {
    log_Ne_c ~ normal(mu_log_ne1_prior, sigma_log_ne_prior);
    log_Ne_a ~ normal(mu_log_ne2_prior, sigma_log_ne_prior);
    log_t0   ~ normal(mu_log_t0_prior, sigma_log_t0_prior);
    for (w in 1:num_windows) y_obs[w] ~ normal(mu_y, sigma_emp);
}

generated quantities {
    vector[num_windows] log_lik;
    for (w in 1:num_windows) {
        log_lik[w] = normal_lpdf(y_obs[w] | mu_y, sigma_emp);
    }
}
