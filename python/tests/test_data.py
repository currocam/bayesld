import bayesld
import numpy as np


def standardize_genotypes(genotypes, allele_frequency):
    """Standardize genotypes assuming Hardy-Weinberg equilibrium"""
    mean = 2.0 * allele_frequency
    std_dev = np.sqrt(2.0 * allele_frequency * (1.0 - allele_frequency))
    return (genotypes - mean) / std_dev


def compute_genetic_diversity(genotypes):
    """Compute genetic diversity (pi) from diploid genotypes"""
    n_ref = np.sum(2 - genotypes)
    n_alt = np.sum(genotypes)
    n_total = n_ref + n_alt

    if n_total < 2:
        return 0.0

    pi = (2.0 * n_ref * n_alt) / (n_total * (n_total - 1))
    return pi


def compute_allele_frequency(genotypes):
    """Compute allele frequency from diploid genotypes"""
    n_ref = np.sum(2 - genotypes)
    n_alt = np.sum(genotypes)
    n_total = n_ref + n_alt

    if n_total < 2:
        return 0.0

    return n_alt / n_total


def compute_minor_allele_frequency(genotypes):
    """Compute minor allele frequency from diploid genotypes"""
    n_ref = np.sum(2 - genotypes)
    n_alt = np.sum(genotypes)
    n_total = n_ref + n_alt

    if n_total < 2:
        return 0.0

    return min(n_ref, n_alt) / n_total


def naive_linkage_disequilibrium(x, y):
    """
    Naive implementation of linkage disequilibrium
    E[X_iY_iX_jY_j] where i != j
    """
    n = len(x)
    acc = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            acc += x[i] * y[i] * x[j] * y[j]
    return acc / (n * (n - 1) / 2.0)


def compute_naive_binned_ld(genotypes, positions, left_bins, right_bins):
    """
    Compute naive binned LD for comparison with streaming implementation.

    Args:
        genotypes: Array of genotypes (n_variants, n_samples)
        positions: Array of variant positions
        left_bins: Left bin boundaries
        right_bins: Right bin boundaries

    Returns:
        Tuple of (avg_ld, count_ld) arrays
    """
    n_genotypes = len(genotypes)
    acc_ld = np.zeros(len(left_bins))
    count_ld = np.zeros(len(left_bins), dtype=int)

    for i in range(n_genotypes):
        genotypes1 = genotypes[i]
        maf1 = compute_minor_allele_frequency(genotypes1)
        if maf1 < 0.25:
            continue

        af1 = compute_allele_frequency(genotypes1)
        x = standardize_genotypes(genotypes1, af1)
        pos_x = positions[i]

        for j in range(i + 1, n_genotypes):
            genotypes2 = genotypes[j]
            maf2 = compute_minor_allele_frequency(genotypes2)
            if maf2 < 0.25:
                continue

            af2 = compute_allele_frequency(genotypes2)
            y = standardize_genotypes(genotypes2, af2)
            pos_y = positions[j]

            ld = naive_linkage_disequilibrium(x, y)
            distance = pos_y - pos_x

            for k_bin in range(len(left_bins)):
                if left_bins[k_bin] <= distance <= right_bins[k_bin]:
                    acc_ld[k_bin] += ld
                    count_ld[k_bin] += 1

    # Compute average LD
    avg_ld = np.zeros(len(left_bins))
    for k_bin in range(len(left_bins)):
        if count_ld[k_bin] > 0:
            avg_ld[k_bin] = acc_ld[k_bin] / count_ld[k_bin]
        else:
            avg_ld[k_bin] = np.nan

    return avg_ld, count_ld


def test_binned_ld_random():
    """
    Test that StreamingStatsDiploid produces the same results as naive implementation
    for randomly generated genotype data.
    """

    # Simulate 100 unique positions between [1, 10_000]
    region_span = 10_000
    positions = np.random.choice(np.arange(1, region_span + 1), size=100, replace=False)
    positions = np.sort(positions).astype(np.int32)

    # Simulate 100 genotypes with sample size between 5 and 200
    sample_size = np.random.randint(5, 201)
    genotypes = np.random.randint(0, 3, size=(100, sample_size), dtype=np.int32)

    # Set up bins: left_bins [100, 4000), right_bins [4000, 100_000]
    # Bins are disjoint (do not overlap)
    left_bins = np.array([100.0, 4000.0])
    right_bins = np.array([3999.0, 100_000.0])

    # Compute diversity and LD using StreamingStatsDiploid
    streaming = bayesld.StreamingStatsDiploid(left_bins, right_bins)
    streaming.add_batch(genotypes, positions, region_span)
    streaming_results = streaming.finalize()

    # Extract results: streaming_results has shape (n_bins+1, 2) where columns are [mean, count]
    streaming_genetic_diversity = streaming_results[0, 0]
    streaming_ld = streaming_results[1:, 0]
    streaming_ld_count = streaming_results[1:, 1]

    # Compute naive genetic diversity
    naive_genetic_diversity = (
        np.sum([compute_genetic_diversity(genotypes[i]) for i in range(len(genotypes))])
        / region_span
    )

    # Compute naive binned LD
    avg_ld, count_ld = compute_naive_binned_ld(
        genotypes, positions, left_bins, right_bins
    )

    # Assert genetic diversity matches
    assert abs(naive_genetic_diversity - streaming_genetic_diversity) < 1e-6, (
        f"Genetic diversity mismatch: {naive_genetic_diversity} vs {streaming_genetic_diversity}"
    )

    # Assert binned LD matches
    for k_bin in range(len(left_bins)):
        assert np.isnan(avg_ld[k_bin]) == np.isnan(streaming_ld[k_bin]), (
            f"Bin {k_bin} LD mismatch: {avg_ld[k_bin]} vs {streaming_ld[k_bin]}"
        )
        assert streaming_ld_count[k_bin] == count_ld[k_bin], (
            f"Bin {k_bin} count mismatch: {count_ld[k_bin]} vs {streaming_ld_count[k_bin]}"
        )
        assert np.isclose(avg_ld[k_bin], streaming_ld[k_bin], rtol=1e-3), (
            f"Bin {k_bin} LD mismatch: {avg_ld[k_bin]} vs {streaming_ld[k_bin]}"
        )


def test_msprime():
    import bayesld
    import msprime

    Ne = np.random.uniform(1000, 3000)
    demo = msprime.Demography.isolated_model([Ne])
    ts = msprime.sim_ancestry(
        samples=1000, demography=demo, recombination_rate=1e-8, sequence_length=1e8
    )
    mts = msprime.sim_mutations(ts, rate=1e-8)
    left_bins, right_bins = bayesld.linear_bins()
    data = bayesld.data_from_tree_sequence(
        mts,
        left_bins_morgan=left_bins,
        right_bins_morgan=right_bins,
        recombination_rate=1e-8,
    )
    # Invariant 1: LD should always be non-negative
    assert np.all(data["mean_linkage_disequilibrium"] >= 0), (
        "LD should always be non-negative"
    )
    # Invariant 2: LD should be monotonically decreasing
    assert np.all(np.diff(data["mean_linkage_disequilibrium"][:5]) <= 0), (
        "LD should be monotonically decreasing",
        data["mean_linkage_disequilibrium"],
    )
    # Estimated genetic diversity should match the one computed by tskit
    assert np.isclose(data["mean_genetic_diversity"], mts.diversity(), rtol=1e-3), (
        f"Genetic diversity mismatch: {data['mean_genetic_diversity']} vs {mts.diversity()}"
    )


def test_msprime_haploid():
    import bayesld
    import msprime

    Ne = np.random.uniform(1000, 3000)
    demo = msprime.Demography.isolated_model([Ne])
    ts = msprime.sim_ancestry(
        samples=1000,
        demography=demo,
        recombination_rate=1e-8,
        sequence_length=1e8,
        ploidy=1,
    )
    mts = msprime.sim_mutations(ts, rate=1e-8)
    left_bins, right_bins = bayesld.linear_bins()
    data = bayesld.data_from_tree_sequence(
        mts,
        left_bins_morgan=left_bins,
        right_bins_morgan=right_bins,
        recombination_rate=1e-8,
        ploidy=1,
    )
    # Invariant 1: LD should always be non-negative
    assert np.all(data["mean_linkage_disequilibrium"] >= 0), (
        "LD should always be non-negative"
    )
    # Invariant 2: LD should be monotonically decreasing
    assert np.all(np.diff(data["mean_linkage_disequilibrium"][:5]) <= 0), (
        "LD should be monotonically decreasing",
        data["mean_linkage_disequilibrium"],
    )
    # Estimated genetic diversity should match the one computed by tskit
    assert np.isclose(data["mean_genetic_diversity"], mts.diversity(), rtol=1e-3), (
        f"Genetic diversity mismatch: {data['mean_genetic_diversity']} vs {mts.diversity()}"
    )
