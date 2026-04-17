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
