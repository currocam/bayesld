"""
Generate reference data from the current PyTensor-based code.

Run this script ONCE before porting to JAX, to record golden outputs.
"""

import json
import numpy as np
import bayesld
from bayesld import deterministic, montecarlo

LEFT_BINS, RIGHT_BINS = bayesld.linear_bins()
MUTATION_RATE = 1e-8
RECOMBINATION_RATE = 1e-8
SEQUENCE_LENGTH = RIGHT_BINS[-1] * 2 / RECOMBINATION_RATE
SAMPLE_SIZE = 10
PLOIDY = 2

DET_KWARGS = dict(
    left_bins=LEFT_BINS,
    right_bins=RIGHT_BINS,
    mutation_rate=MUTATION_RATE,
    sample_size=SAMPLE_SIZE,
    ploidy=PLOIDY,
    evaluate=True,
)

MC_KWARGS = dict(
    left_bins=LEFT_BINS,
    right_bins=RIGHT_BINS,
    mutation_rate=MUTATION_RATE,
    recombination_rate=RECOMBINATION_RATE,
    sequence_length=SEQUENCE_LENGTH,
    sample_size=SAMPLE_SIZE,
    random_seed=42,
    num_replicates=2,
    ploidy=PLOIDY,
    model="hudson",
    cores=1,
    progressbar=False,
)


def to_serializable(x):
    """Convert numpy arrays/scalars to JSON-serializable types."""
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return float(x)
    return x


def main():
    data = {"deterministic": {}, "montecarlo": {}}

    # === Deterministic ===
    print("Computing deterministic reference data...")

    # Constant
    pi, ld = deterministic.expected_constant(Ne=1000.0, **DET_KWARGS)
    data["deterministic"]["constant"] = {
        "pi": to_serializable(pi),
        "ld": to_serializable(ld),
    }
    print("  constant: done")

    # Piecewise exponential
    pi, ld = deterministic.expected_piecewise_exponential(
        Ne_c=500.0, Ne_a=2000.0, t0=50.0, alpha=0.01, **DET_KWARGS
    )
    data["deterministic"]["piecewise_exponential"] = {
        "pi": to_serializable(pi),
        "ld": to_serializable(ld),
    }
    print("  piecewise_exponential: done")

    # Exponential carrying capacity
    pi, ld = deterministic.expected_exponential_carrying_capacity(
        Ne_c=500.0, Ne_a=2000.0, t0=20.0, t1=60.0, alpha=0.01, **DET_KWARGS
    )
    data["deterministic"]["exponential_carrying_capacity"] = {
        "pi": to_serializable(pi),
        "ld": to_serializable(ld),
    }
    print("  exponential_carrying_capacity: done")

    # Piecewise constant
    pi, ld = deterministic.expected_piecewise_constant(
        Ne_values=[500.0, 1000.0, 2000.0],
        t_boundaries=[30.0, 60.0],
        **DET_KWARGS,
    )
    data["deterministic"]["piecewise_constant"] = {
        "pi": to_serializable(pi),
        "ld": to_serializable(ld),
    }
    print("  piecewise_constant: done")

    # Secondary introduction
    pi, ld = deterministic.expected_secondary_introduction(
        Ne_1=3000.0, Ne_2=3000.0, Ne_a=6000.0, t0=20.0, t1=60.0,
        migration_rate=0.1, **DET_KWARGS,
    )
    data["deterministic"]["secondary_introduction"] = {
        "pi": to_serializable(pi),
        "ld": to_serializable(ld),
    }
    print("  secondary_introduction: done")

    # === Monte Carlo ===
    print("Computing montecarlo reference data...")

    # Constant
    pi_mean, ld_mean, pi_vec, ld_mat = montecarlo.expected_constant(
        Ne=1000.0, **MC_KWARGS
    )
    data["montecarlo"]["constant"] = {
        "pi_mean": to_serializable(pi_mean),
        "ld_mean": to_serializable(ld_mean),
        "pi_vec": to_serializable(pi_vec),
        "ld_mat": to_serializable(ld_mat),
    }
    print("  constant: done")

    # Piecewise exponential
    pi_mean, ld_mean, pi_vec, ld_mat = montecarlo.expected_piecewise_exponential(
        Ne_c=500.0, Ne_a=2000.0, t0=50.0, alpha=0.01, **MC_KWARGS
    )
    data["montecarlo"]["piecewise_exponential"] = {
        "pi_mean": to_serializable(pi_mean),
        "ld_mean": to_serializable(ld_mean),
        "pi_vec": to_serializable(pi_vec),
        "ld_mat": to_serializable(ld_mat),
    }
    print("  piecewise_exponential: done")

    # Exponential carrying capacity
    pi_mean, ld_mean, pi_vec, ld_mat = montecarlo.expected_exponential_carrying_capacity(
        Ne_c=500.0, Ne_a=2000.0, t0=20.0, t1=60.0, alpha=0.01, **MC_KWARGS
    )
    data["montecarlo"]["exponential_carrying_capacity"] = {
        "pi_mean": to_serializable(pi_mean),
        "ld_mean": to_serializable(ld_mean),
        "pi_vec": to_serializable(pi_vec),
        "ld_mat": to_serializable(ld_mat),
    }
    print("  exponential_carrying_capacity: done")

    # Piecewise constant
    pi_mean, ld_mean, pi_vec, ld_mat = montecarlo.expected_piecewise_constant(
        Ne_values=[500.0, 1000.0, 2000.0],
        t_boundaries=[30.0, 60.0],
        **MC_KWARGS,
    )
    data["montecarlo"]["piecewise_constant"] = {
        "pi_mean": to_serializable(pi_mean),
        "ld_mean": to_serializable(ld_mean),
        "pi_vec": to_serializable(pi_vec),
        "ld_mat": to_serializable(ld_mat),
    }
    print("  piecewise_constant: done")

    # Secondary introduction
    pi_mean, ld_mean, pi_vec, ld_mat = montecarlo.expected_secondary_introduction(
        Ne_1=3000.0, Ne_2=3000.0, Ne_a=6000.0, t0=20.0, t1=60.0,
        migration_rate=0.1, **MC_KWARGS,
    )
    data["montecarlo"]["secondary_introduction"] = {
        "pi_mean": to_serializable(pi_mean),
        "ld_mean": to_serializable(ld_mean),
        "pi_vec": to_serializable(pi_vec),
        "ld_mat": to_serializable(ld_mat),
    }
    print("  secondary_introduction: done")

    # Save to JSON
    outpath = "python/tests/reference_data.json"
    with open(outpath, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nSaved reference data to {outpath}")


if __name__ == "__main__":
    main()
