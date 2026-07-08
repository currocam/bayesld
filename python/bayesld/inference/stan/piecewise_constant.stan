// Piecewise-constant-Ne inference engine.
//
// Everything not specific to this parameterization is #included from shared/*.stan
// (the GP-bias surrogate, the joint-MVN observation model, and the standardized
// data/transformed-data machinery). Only the piecewise-constant demography lives
// here: the log-sum-exp expectation functions, the per-epoch lognormal priors,
// and the ordered log-boundary parameterization.
functions {
  #include shared/gpbasisfun.stan
  #include shared/finite_sample.stan

  // ---- piecewise_constant demography ----
  // Survival probability at genetic distance u (numerically stable log-sum-exp).
  real S_u_piecewise_constant(real u, int n_epochs, vector Ne_values, vector t_boundaries) {
    array[n_epochs] real log_terms;
    real Gamma_prev = 0.0;
    real t_prev     = 0.0;
    for (i in 1:n_epochs - 1) {
      real Ne_i   = Ne_values[i];
      real t_curr = t_boundaries[i];
      real dt     = t_curr - t_prev;
      real c      = 2.0 * u + 1.0 / (2.0 * Ne_i);
      log_terms[i] = -Gamma_prev - 2.0 * u * t_prev
                     + log1m_exp(-c * dt)
                     - log1p(4.0 * Ne_i * u);
      Gamma_prev += dt / (2.0 * Ne_i);
      t_prev = t_curr;
    }
    real Ne_last = Ne_values[n_epochs];
    log_terms[n_epochs] = -Gamma_prev - 2.0 * u * t_prev - log1p(4.0 * Ne_last * u);
    return exp(log_sum_exp(to_vector(log_terms)));
  }

  // Expected LD per bin via Gauss-Legendre quadrature over the bin width. Plain
  // serial loop over bins (no map_rect marshalling): each bin is a bin-averaged
  // survival probability, 0.5 * sum_k w_k S_u(u_k).
  vector mu_ld_piecewise_constant(int n_epochs, vector Ne_values, vector t_boundaries,
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
        s += gl_weights[k] * S_u_piecewise_constant(u_k, n_epochs, Ne_values, t_boundaries);
      }
      result[b] = 0.5 * s;  // average over bin: (1/width) * (width/2) * sum(w_k f(u_k))
    }
    return result;
  }

  // Expected genetic diversity (numerically stable log-sum-exp / log-diff-exp).
  real mu_div_piecewise_constant(int n_epochs, vector Ne_values, vector t_boundaries,
                                 real mutation_rate) {
    array[n_epochs] real log_terms;
    real Gamma_prev = 0.0;
    real t_prev     = 0.0;
    for (i in 1:n_epochs - 1) {
      real Ne_i   = Ne_values[i];
      real t_curr = t_boundaries[i];
      real dt     = t_curr - t_prev;
      log_terms[i] = -Gamma_prev
                     + log_diff_exp(log(t_prev + 2.0 * Ne_i),
                                    log(t_curr + 2.0 * Ne_i) - dt / (2.0 * Ne_i));
      Gamma_prev += dt / (2.0 * Ne_i);
      t_prev = t_curr;
    }
    real Ne_last = Ne_values[n_epochs];
    log_terms[n_epochs] = -Gamma_prev + log(t_prev + 2.0 * Ne_last);
    return exp(log_sum_exp(to_vector(log_terms))) * 2.0 * mutation_rate;
  }
}

data {
  #include shared/data_stats.stan

  // ── Demographic dimension + lognormal priors (data-driven, per epoch) ──
  int<lower=1> n_epochs;
  vector[n_epochs]         mu_log_ne;
  vector<lower=0>[n_epochs] sigma_log_ne;
  vector[n_epochs - 1]         mu_log_t;
  vector<lower=0>[n_epochs - 1] sigma_log_t;

  #include shared/data_surrogate.stan
}

transformed data {
  #include shared/transformed_data.stan
}

parameters {
  // ── Demography: lognormal Ne per epoch, ordered lognormal boundaries ──
  vector<offset=log_ne_offset>[n_epochs] log_Ne;
  ordered[n_epochs - 1] log_t;

  #include shared/parameters.stan
}

transformed parameters {
  // ── Demography: expected_pi + finite-sample-corrected approx_expected_ld ──
  vector<lower=0>[n_epochs]     Ne_values    = exp(log_Ne);
  vector<lower=0>[n_epochs - 1] t_boundaries = exp(log_t);

  real expected_pi = mu_div_piecewise_constant(n_epochs, Ne_values, t_boundaries, mutation_rate);
  vector[n_bins] approx_expected_ld = correct_ld_finite_sample(
    mu_ld_piecewise_constant(n_epochs, Ne_values, t_boundaries,
                             left_bins, right_bins, n_quad, gl_nodes, gl_weights),
    sample_size, ploidy
  );

  #include shared/transformed_parameters.stan
}

model {
  // ── Demographic priors (lognormal) ──
  log_Ne ~ normal(mu_log_ne, sigma_log_ne);
  log_t  ~ normal(mu_log_t,  sigma_log_t);

  #include shared/model.stan
}

generated quantities {
  #include shared/generated_quantities.stan
}
