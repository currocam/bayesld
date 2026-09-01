// Genomic windows
int<lower=1> num_windows;
// Data consist of mean LD across n_bins and pi 
int<lower=1> n_bins;
vector[n_bins] left_bins;
vector[n_bins] right_bins;
// To link pi with Ne
real<lower=0> mutation_rate;
int<lower=1> sample_size;
int<lower=1, upper=2> ploidy;
// Sufficient summary statistics
vector[num_windows] pi_array;
matrix[num_windows, n_bins] ld_mat;

// Optional per-(window, bin) missingness correction (diploid only).
int<lower=0, upper=1> use_missingness;
matrix<lower=0, upper=1>[num_windows, n_bins] missingness;

// Gauss-Legendre quadrature for numerical integration
int<lower=1> n_quad;
vector[n_quad] gl_nodes;
vector[n_quad] gl_weights;
