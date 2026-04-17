// Expected LD under constant Ne (diploid, ploidy=2).
// Closed-form integral of coalescent-based LD decay over a recombination-distance bin.
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
