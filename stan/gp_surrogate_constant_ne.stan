functions {
#include gpbasisfun_functions.stan
#include functions/shared.stan
#include functions/constant.stan
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
    vector[n_eval] eval_log_ne;
    vector[n_eval] eval_loglik_det;
    vector[n_eval] eval_bc_loglik;
    vector<lower=0>[n_eval] eval_epsilon;

    // ── Hilbert-space GP approximation settings ─────────────────────────────
    real<lower=0> hsgp_c;   // boundary factor (≥1.2; recommended 1.5)
    int<lower=1>  hsgp_m;   // number of basis functions (more → accurate, slower)
}

transformed data {
    // Observed biases: what the HSGP is fitted to
    vector[n_eval] delta = eval_bc_loglik - eval_loglik_det;

    // Standardise GP inputs (makes gp_rho interpretable as correlation length in σ units)
    real x_mu  = mean(eval_log_ne);
    real x_sig = sd(eval_log_ne);
    vector[n_eval] xn_eval = (eval_log_ne - x_mu) / x_sig;

    // HSGP domain boundary L = c * max(|xn|)
    // Use max/min on vector (avoids abs(vector) version fragility)
    real L_hsgp = hsgp_c * fmax(max(xn_eval), -min(xn_eval));

    // Pre-compute basis-function matrix at eval points (constant — not parameter-dependent)
    matrix[n_eval, hsgp_m] PHI_eval = PHI(n_eval, hsgp_m, L_hsgp, xn_eval);
}

parameters {
    real log_Ne;
    real<lower=0> gp_rho;    // EQ kernel length scale in standardised log_Ne units
    real<lower=0> gp_alpha;  // EQ kernel signal amplitude
    vector[hsgp_m] beta;     // basis-function coefficients (standard-normal prior)
}

transformed parameters {
    real Ne = exp(log_Ne);

    // Standardised query point (current parameter value)
    real xn_star = (log_Ne - x_mu) / x_sig;

    // Spectral densities for the EQ (squared-exponential) kernel
    vector[hsgp_m] spd = diagSPD_EQ(gp_alpha, gp_rho, L_hsgp, hsgp_m);

    // GP approximation at eval points: f ≈ Φ (spd ⊙ β)
    vector[n_eval] f_eval = PHI_eval * (spd .* beta);

    // Basis-function vector at query point (single point — inlined PHI formula)
    vector[hsgp_m] phi_star;
    for (m in 1:hsgp_m)
        phi_star[m] = sin(m * pi() / (2.0 * L_hsgp) * (xn_star + L_hsgp)) / sqrt(L_hsgp);

    // GP bias correction at current Ne
    real gp_bias = dot_product(phi_star, spd .* beta);
}

model {
    // GP hyperparameter priors
    gp_rho   ~ inv_gamma(5, 5);
    // Bias corrections are in log-lik units (typically small); keep amplitude tight
    gp_alpha ~ normal(0, 0.3);
    // Basis coefficients: standard normal ↔ GP prior
    beta ~ std_normal();

    // HSGP likelihood: fit bias observations with known heteroscedastic noise
    delta ~ normal(f_eval, eval_epsilon);

    // Ne prior (lognormal)
    log_Ne ~ normal(log_Ne_prior_mu, log_Ne_prior_sigma);

    // Data likelihood (deterministic predictions)
    real mu_div_val = 4.0 * Ne * mutation_rate;
    vector[n_bins] mu_ld_val = correct_ld_finite_sample(
        mu_ld_constant(Ne, left_bins, right_bins), sample_size
    );
    mean_div ~ normal(mu_div_val, est_sigma_div);
    for (b in 1:n_bins)
        target += normal_lpdf(mean_ld[b] | mu_ld_val[b], est_sigma_ld[b]) / n_bins;

    // HSGP posterior mean as additive bias correction
    target += gp_bias;
}

generated quantities {
    real E_pi = 4.0 * Ne * mutation_rate;
    vector[n_bins] approx_ld = correct_ld_finite_sample(
        mu_ld_constant(Ne, left_bins, right_bins), sample_size
    );
}
