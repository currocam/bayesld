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
    error_constant.py <recombination_rate> <num_samples> [output.pkl]
"""

import gzip
import pickle
import sys

import msprime
import numpy as np
from bayesld import deterministic, linear_bins, montecarlo

# --- parameters ---
MUTATION_RATE = 1e-7
RANDOM_SEED = 9362178
NUM_WORKERS = 8
RTOL = 0.1
SMC_PRIME_MODEL = msprime.SMCK(k=1)

# Ne sweep: 40 log-spaced points from 10 to 100_000
NE_VALUES = np.logspace(np.log10(10), np.log10(100_000), 40)


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    recombination_rate = float(sys.argv[1])
    num_samples = int(sys.argv[2])
    out_path = sys.argv[3] if len(sys.argv) > 3 else "error_constant.pkl"

    left_bins, right_bins = linear_bins()
    sequence_length = right_bins[-1] * 2 / recombination_rate

    import tqdm

    results = []
    for i, Ne in enumerate(tqdm.tqdm(NE_VALUES, desc="Ne values")):
        pi_det, ld_det = deterministic.expected_constant(
            Ne=float(Ne),
            left_bins=left_bins,
            right_bins=right_bins,
            mutation_rate=MUTATION_RATE,
            sample_size=num_samples,
        )
        pi_mc, ld_mc = montecarlo.expected_constant(
            Ne=float(Ne),
            left_bins=left_bins,
            right_bins=right_bins,
            mutation_rate=MUTATION_RATE,
            recombination_rate=recombination_rate,
            sequence_length=sequence_length,
            sample_size=num_samples,
            random_seed=RANDOM_SEED + i,
            rtol=RTOL,
            model=SMC_PRIME_MODEL,
            num_workers=NUM_WORKERS,
        )
        results.append(
            {
                "Ne": float(Ne),
                "pi_det": float(pi_det),
                "ld_det": np.asarray(ld_det),
                "pi_mc": np.asarray(pi_mc),
                "ld_mc": np.asarray(ld_mc),
            }
        )

    data = {
        "results": results,
        "Ne_values": np.asarray(NE_VALUES),
        "left_bins": left_bins,
        "right_bins": right_bins,
        "params": {
            "mutation_rate": MUTATION_RATE,
            "recombination_rate": recombination_rate,
            "sequence_length": sequence_length,
            "num_samples": num_samples,
            "rtol": RTOL,
            "model": "SMCK(k=1)",
        },
    }

    with gzip.open(out_path, "wb") as f:
        pickle.dump(data, f)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
