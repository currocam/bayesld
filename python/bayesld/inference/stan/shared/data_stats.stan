// ---- data_stats.stan ----
// Observed sufficient statistics and the Gauss-Legendre quadrature grid used by
// every engine. Include at the *top* of the data block; the demography-specific
// dimension and priors follow, then #include shared/data_surrogate.stan.

// ── Observed sufficient statistics ──
int<lower=1> n_bins;
int<lower=1> num_windows;
vector[n_bins] left_bins;
vector[n_bins] right_bins;
real<lower=0> mutation_rate;
int<lower=1> sample_size;
int<lower=1, upper=2> ploidy;
vector[num_windows] pi_array;
matrix[num_windows, n_bins] ld_mat;

// ── Gauss-Legendre quadrature over LD bins ──
int<lower=1> n_quad;
vector[n_quad] gl_nodes;
vector[n_quad] gl_weights;
