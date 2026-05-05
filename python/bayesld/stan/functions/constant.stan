// Expected LD under constant Ne (diploid, ploidy=2).
// Closed-form integral of coalescent-based LD decay over a recombination-distance bin.
vector mu_ld_constant(real Ne, vector left_bins, vector right_bins) {
    int n_bins = num_elements(left_bins);
    vector[n_bins] result;
    for (i in 1:n_bins) {
        real u_i = left_bins[i];
        real u_j = right_bins[i];
        // Use log1p for numerical stability when 4*Ne*u is large
        result[i] = (log1p(4.0 * Ne * u_j) - log1p(4.0 * Ne * u_i))
                    / (4.0 * Ne * (u_j - u_i));
    }
    return result;
}
