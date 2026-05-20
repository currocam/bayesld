#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "bayesld @ git+https://github.com/currocam/bayesld.git@ae4d14b",
#     "msprime==1.4.0",
#     "numpy==2.2.6",
#     "joblib==1.5.3",
#     "tqdm==4.67.3",
# ]
# ///
"""Deterministic vs Monte Carlo expected LD/pi under constant Ne

Usage:
    error_constant.py <recombination_rate> <num_samples> <ploidy> <num_workers> [output.pkl]
"""

import gzip
import pickle
import sys

import msprime
import numpy as np
from bayesld import deterministic, linear_bins, montecarlo

# --- parameters ---
# Mutation rate is set per Ne so the expected number of segregating sites
# stays ~constant across the grid. Watterson: S = theta * a_n, with
# theta = 2 * ploidy * Ne * mu * L.
TARGET_SEG_SITES = 100_000
RANDOM_SEED = 9362178
NUM_REPLICATES = 20
SMC_PRIME_MODEL = msprime.SMCK(k=1)
NE_VALUES = np.logspace(np.log10(100), np.log10(100_000), 50)


def main():
    if len(sys.argv) < 5:
        print(__doc__)
        sys.exit(1)
    recombination_rate = float(sys.argv[1])
    num_samples = int(sys.argv[2])
    ploidy = int(sys.argv[3])
    num_workers = int(sys.argv[4])
    out_path = sys.argv[5] if len(sys.argv) > 5 else "error_constant.pkl"

    left_bins, right_bins = linear_bins()
    sequence_length = right_bins[-1] * 2 / recombination_rate
    # a_n uses haploid sample size
    n_hap = num_samples * ploidy
    a_n = sum(1.0 / i for i in range(1, n_hap))
    # Match coalescent timescale between haploid and diploid: haploid Ne is
    # doubled so 2*Ne_hap == Ne_dip in chromosomes.
    ne_values = NE_VALUES * (2 if ploidy == 1 else 1)

    import tqdm

    results = []
    # Process from largest Ne to smallest (cheapest MC first)
    order = np.argsort(ne_values)[::-1]
    for i in tqdm.tqdm(order, desc="Ne values"):
        Ne = ne_values[i]
        mutation_rate = TARGET_SEG_SITES / (
            a_n * 2 * ploidy * float(Ne) * sequence_length
        )
        pi_det, ld_det = deterministic.expected_constant(
            Ne=float(Ne),
            left_bins=left_bins,
            right_bins=right_bins,
            mutation_rate=mutation_rate,
            sample_size=num_samples,
            ploidy=ploidy,
        )
        pi_det_inf, ld_det_inf = deterministic.expected_constant(
            Ne=float(Ne),
            left_bins=left_bins,
            right_bins=right_bins,
            mutation_rate=mutation_rate,
            sample_size=None,
            ploidy=ploidy,
        )
        pi_mc, ld_mc = montecarlo.expected_constant(
            Ne=float(Ne),
            left_bins=left_bins,
            right_bins=right_bins,
            mutation_rate=mutation_rate,
            recombination_rate=recombination_rate,
            sequence_length=sequence_length,
            sample_size=num_samples,
            random_seed=RANDOM_SEED + int(i),
            num_replicates=NUM_REPLICATES,
            model=SMC_PRIME_MODEL,
            num_workers=num_workers,
            ploidy=ploidy,
        )
        results.append(
            {
                "Ne": float(Ne),
                "mutation_rate": mutation_rate,
                "pi_det": float(pi_det),
                "ld_det": np.asarray(ld_det),
                "pi_det_inf": float(pi_det_inf),
                "ld_det_inf": np.asarray(ld_det_inf),
                "pi_mc": np.asarray(pi_mc),
                "ld_mc": np.asarray(ld_mc),
            }
        )

    data = {
        "results": results,
        "Ne_values": np.asarray(ne_values),
        "left_bins": left_bins,
        "right_bins": right_bins,
        "params": {
            "target_seg_sites": TARGET_SEG_SITES,
            "recombination_rate": recombination_rate,
            "sequence_length": sequence_length,
            "num_samples": num_samples,
            "ploidy": ploidy,
            "a_n": a_n,
            "num_replicates": NUM_REPLICATES,
            "model": "SMCK(k=1)",
        },
    }

    with gzip.open(out_path, "wb") as f:
        pickle.dump(data, f)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()

