// Two-phase piecewise-exponential-Ne inference engine.
//
// Everything not specific to this parameterization is #included from shared/*.stan
// (the GP-bias surrogate, the joint-MVN observation model, and the standardized
// data/transformed-data machinery). Only the two-phase demography lives here:
//   Ne(t) = Ne_c * exp(-alpha * t)   for t < t0
//   Ne(t) = Ne_a                     for t >= t0
// with alpha = (log Ne_c - log Ne_a) / t0 so the phases meet at t0. It is
// parameterized by three scalars: log_ne_a, log_t and log_alpha_fold (with
// Ne_c = Ne_a * exp(log_alpha_fold) and alpha = log_alpha_fold / t0).
functions {
  #include shared/gpbasisfun.stan
  #include shared/finite_sample.stan

  // ---- piecewise_exponential demography ----
  // All alpha-dependent expressions use expm1_over_x to remove the removable
  // singularity at alpha = 0; the formulas are valid for all alpha.

  // expm1(x)/x, stable at x=0 (limit = 1). expm1 is accurate for all x != 0.
  real expm1_over_x(real x) {
    if (x == 0.0) return 1.0;
    return expm1(x) / x;
  }

  // Survival probability at genetic distance u (numerically stable log-sum-exp).
  // piece1 [0, t0]: Gauss-Legendre quadrature | piece2 [t0, inf): closed form.
  real S_u_piecewise_exponential(real u, real Ne_c, real Ne_a, real t0, real alpha,
                                 int n_quad, vector gl_nodes, vector gl_weights) {
    real half_t0 = t0 / 2.0;
    vector[n_quad] log_terms;
    for (k in 1:n_quad) {
      real t_k     = half_t0 * gl_nodes[k] + half_t0;
      real exp_arg = t_k * alpha - 2.0 * t_k * u - t_k * expm1_over_x(t_k * alpha) / (2.0 * Ne_c);
      log_terms[k] = log(gl_weights[k]) + exp_arg;
    }
    real log_piece1 = log(half_t0) - log(Ne_c * 2.0) + log_sum_exp(log_terms);
    real piece2_exp = -2.0 * u * t0 - t0 * expm1_over_x(t0 * alpha) / (2.0 * Ne_c);
    real log_piece2 = piece2_exp - log1p(4.0 * u * Ne_a);
    return exp(log_sum_exp([log_piece1, log_piece2]'));
  }

  // Expected LD per bin via Gauss-Legendre quadrature over the bin width. Plain
  // serial loop over bins: each bin is a bin-averaged survival probability,
  // 0.5 * sum_k w_k S_u(u_k).
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

  // Expected genetic diversity (numerically stable log-sum-exp).
  real mu_div_piecewise_exponential(real Ne_c, real Ne_a, real t0, real alpha,
                                    real mutation_rate,
                                    int n_quad, vector gl_nodes, vector gl_weights) {
    real half_t0 = t0 / 2.0;
    vector[n_quad] log_terms;
    for (k in 1:n_quad) {
      real t_k     = half_t0 * gl_nodes[k] + half_t0;
      real exp_arg = t_k * alpha - t_k * expm1_over_x(t_k * alpha) / (2.0 * Ne_c);
      log_terms[k] = log(gl_weights[k]) + log(t_k) + exp_arg;
    }
    real log_piece1 = log(half_t0) - log(Ne_c * 2.0) + log_sum_exp(log_terms);
    real log_piece2 = log(2.0 * Ne_a + t0) - t0 * expm1_over_x(t0 * alpha) / (2.0 * Ne_c);
    return exp(log_sum_exp([log_piece1, log_piece2]')) * 2.0 * mutation_rate;
  }
}

data {
  #include shared/data_stats.stan

  // ── Lognormal priors (data-driven; scalar per parameter) ──
  //   log_ne_a       ~ normal(mu_log_ne_a,       sigma_log_ne_a)
  //   log_t          ~ normal(mu_log_t,          sigma_log_t)
  //   log_alpha_fold ~ normal(mu_log_alpha_fold, sigma_log_alpha_fold)
  real         mu_log_ne_a;
  real<lower=0> sigma_log_ne_a;
  real         mu_log_t;
  real<lower=0> sigma_log_t;
  real         mu_log_alpha_fold;
  real<lower=0> sigma_log_alpha_fold;

  #include shared/data_surrogate.stan
}

transformed data {
  #include shared/transformed_data.stan
}

parameters {
  // ── Demography: ancient Ne, exponential start, log-fold change to contemporary ──
  real<offset=log_ne_offset> log_ne_a;
  real log_t;
  real log_alpha_fold;

  #include shared/parameters.stan
}

transformed parameters {
  // ── Demography: expected_pi + finite-sample-corrected approx_expected_ld ──
  real<lower=0> Ne_a = exp(log_ne_a);
  real<lower=0> t0   = exp(log_t);
  real log_ne_c      = log_ne_a + log_alpha_fold;
  real<lower=0> Ne_c = exp(log_ne_c);
  // alpha = (log Ne_c - log Ne_a) / t0 makes the two phases meet at t0.
  real alpha = log_alpha_fold / t0;

  real expected_pi = mu_div_piecewise_exponential(Ne_c, Ne_a, t0, alpha, mutation_rate,
                                                  n_quad, gl_nodes, gl_weights);
  vector[n_bins] approx_expected_ld = correct_ld_finite_sample(
    mu_ld_piecewise_exponential(Ne_c, Ne_a, t0, alpha,
                                left_bins, right_bins, n_quad, gl_nodes, gl_weights),
    sample_size, ploidy
  );

  #include shared/transformed_parameters.stan
}

model {
  // ── Demographic priors (lognormal) ──
  log_ne_a       ~ normal(mu_log_ne_a,       sigma_log_ne_a);
  log_t          ~ normal(mu_log_t,          sigma_log_t);
  log_alpha_fold ~ normal(mu_log_alpha_fold, sigma_log_alpha_fold);

  #include shared/model.stan
}

generated quantities {
  #include shared/generated_quantities.stan
}
