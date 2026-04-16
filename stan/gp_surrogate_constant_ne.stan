functions {
    vector mu_ld_constant(real Ne, vector left_bins, vector right_bins) {
        int n_bins = num_elements(left_bins);
        vector[n_bins] result;
        for (i in 1:n_bins) {
            real u_i = left_bins[i];
            real u_j = right_bins[i];
            result[i] = (-log(4.0 * Ne * u_i + 1.0) + log(4.0 * Ne * u_j + 1.0))
                        / (4.0 * Ne * (u_j - u_i));
        }
        return result;
    }

    vector correct_ld_finite_sample(vector mu, int sample_size) {
        int S = 2 * sample_size;
        real beta = 1.0 / square(S - 1);
        real alpha_corr = square(square(S) - S + 2.0) / square(square(S) - 3.0 * S + 2.0);
        return (alpha_corr - beta) * mu + 4.0 * beta;
    }
}

data {
    // ── Inference data ──────────────────────────────────────────────────────
    int<lower=1> n_bins;
    vector[n_bins] left_bins;
    vector[n_bins] right_bins;
    real<lower=0> mutation_rate;
    int<lower=1> sample_size;
    real<lower=0> mean_div;
    real<lower=0> est_sigma_div;
    vector<lower=0>[n_bins] mean_ld;
    vector<lower=0>[n_bins] est_sigma_ld;
    real log_Ne_prior_mu;
    real<lower=0> log_Ne_prior_sigma;

    // ── GP surrogate evaluation dataset ────────────────────────────────────
    int<lower=2> n_eval;
    vector[n_eval] eval_log_ne;           // log(Ne) at each eval point
    vector[n_eval] eval_loglik_det;       // deterministic LD log-lik
    vector[n_eval] eval_bc_loglik;        // bias-corrected MC log-lik
    vector<lower=0>[n_eval] eval_epsilon; // bootstrap std error (known noise)
}

transformed data {
    // Observed biases: what the GP is fitted to
    vector[n_eval] delta = eval_bc_loglik - eval_loglik_det;

    // Standardize GP inputs by the eval distribution.
    // This makes rho interpretable as "correlation length in std-devs of log_Ne".
    // gp_exp_quad_cov requires array[] real, not vector.
    real x_mu  = mean(eval_log_ne);
    real x_sig = sd(eval_log_ne);
    array[n_eval] real x_eval;
    for (i in 1:n_eval) {
        x_eval[i] = (eval_log_ne[i] - x_mu) / x_sig;
    }
}

parameters {
    real log_Ne;
    real<lower=0> gp_rho;    // GP length scale in standardised log_Ne space
    real<lower=0> gp_alpha;  // GP signal amplitude (scale of the bias)
}

transformed parameters {
    real Ne = exp(log_Ne);

    // Standardised query point (current parameter value)
    real x_star = (log_Ne - x_mu) / x_sig;

    // K(X, X) with heteroscedastic noise + jitter — do Cholesky once
    matrix[n_eval, n_eval] K = gp_exp_quad_cov(x_eval, gp_alpha, gp_rho);
    for (i in 1:n_eval) {
        K[i, i] += square(eval_epsilon[i]) + 1e-9;
    }
    matrix[n_eval, n_eval] L_K = cholesky_decompose(K);

    // k(x_star, X): cross-covariance with the current parameter
    vector[n_eval] K_star;
    for (i in 1:n_eval) {
        K_star[i] = square(gp_alpha)
                    * exp(-0.5 * square((x_star - x_eval[i]) / gp_rho));
    }

    // GP posterior mean at x_star:
    //   mu* = K_star' K_noise^{-1} delta
    // K = L L^T  →  K^{-1} delta = L^{-T} (L^{-1} delta)
    // mdivide_left_spd handles the SPD solve cleanly via its own Cholesky;
    // we pass K (already computed) directly.
    real gp_bias = dot_product(K_star, mdivide_left_spd(K, delta));
}

model {
    // GP hyperparameter priors
    // rho ~ inv_gamma(5, 5): mean ≈ 1.25 in standardised space (sensible default)
    gp_rho   ~ inv_gamma(5, 5);
    // alpha: half-normal, scale reflects expected bias magnitude
    gp_alpha ~ normal(0, 0.5);

    // GP marginal likelihood — fits hyperparameters to the eval dataset
    delta ~ multi_normal_cholesky(rep_vector(0, n_eval), L_K);

    // Ne prior (lognormal)
    log_Ne ~ normal(log_Ne_prior_mu, log_Ne_prior_sigma);

    // Data likelihood
    real mu_div_val = 4.0 * Ne * mutation_rate;
    vector[n_bins] mu_ld_val = correct_ld_finite_sample(
        mu_ld_constant(Ne, left_bins, right_bins), sample_size
    );

    mean_div ~ normal(mu_div_val, est_sigma_div);
    for (b in 1:n_bins) {
        target += normal_lpdf(mean_ld[b] | mu_ld_val[b], est_sigma_ld[b]) / n_bins;
    }

    // GP posterior mean as additive bias correction
    target += gp_bias;
}
