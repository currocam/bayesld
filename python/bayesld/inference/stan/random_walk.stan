// Random-walk-Ne inference engine.
functions {
  #include shared/gpbasisfun.stan
  #include shared/finite_sample.stan

  // Survival probability under the SMC_prime
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

  // Numerical integration in bins
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

  // Expected genetic diversity
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

  // We fix the epoch boundaries
  int<lower=2> n_epochs;
  vector<lower=0>[n_epochs - 1] grid;

  real         mu_log_ne;
  real<lower=0> sigma_log_ne;
  vector<lower=0>[n_epochs - 1] sigma_step;

  #include shared/data_surrogate.stan
}

transformed data {
  #include shared/transformed_data.stan
}

parameters {
  real<offset=log_ne_offset> log_ne_a;
  vector[n_epochs - 1] steps;

  #include shared/parameters.stan
}

transformed parameters {
  vector[n_epochs] log_Ne;
  log_Ne[n_epochs] = log_ne_a;
  for (i in 1:n_epochs - 1) {
    int idx = n_epochs - i;             // n_epochs-1, n_epochs-2, ..., 1
    log_Ne[idx] = log_Ne[idx + 1] + steps[idx];
  }
  vector<lower=0>[n_epochs]     Ne_values    = exp(log_Ne);
  vector<lower=0>[n_epochs - 1] t_boundaries = grid;

  real expected_pi = mu_div_piecewise_constant(n_epochs, Ne_values, t_boundaries, mutation_rate);
  vector[n_bins] approx_expected_ld = correct_ld_finite_sample(
    mu_ld_piecewise_constant(n_epochs, Ne_values, t_boundaries,
                             left_bins, right_bins, n_quad, gl_nodes, gl_weights),
    sample_size, ploidy
  );

  #include shared/transformed_parameters.stan
}

model {
  // Demographic priors
  log_ne_a ~ normal(mu_log_ne, sigma_log_ne);
  steps    ~ normal(0, sigma_step);

  #include shared/model.stan
}

generated quantities {
  #include shared/generated_quantities.stan
}
