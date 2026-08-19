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
import time

import msprime
import numpy as np
from bayesld import deterministic, linear_bins, montecarlo


def log(msg):
    print(msg, flush=True)


MUTATION_RATE = 1e-8
RANDOM_SEED = 9362178
RTOL = 0.1
SMC_PRIME_MODEL = msprime.SMCK(k=1)
NE_VALUES = np.logspace(np.log10(100), np.log10(50_000), 50)


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
    ne_values = NE_VALUES * (2 if ploidy == 1 else 1)

    results = []
    order = np.argsort(ne_values)[::-1]
    for i in order:
        Ne = ne_values[i]
        tag = f"[Ne={Ne:.0f}]"

        t0 = time.perf_counter()
        pi_det, ld_det = deterministic.expected_constant(
            Ne=float(Ne),
            left_bins=left_bins,
            right_bins=right_bins,
            mutation_rate=MUTATION_RATE,
            sample_size=num_samples,
            ploidy=ploidy,
        )
        pi_det_inf, ld_det_inf = deterministic.expected_constant(
            Ne=float(Ne),
            left_bins=left_bins,
            right_bins=right_bins,
            mutation_rate=MUTATION_RATE,
            sample_size=None,
            ploidy=ploidy,
        )
        log(f"{tag} det        {time.perf_counter() - t0:7.2f}s")

        t0 = time.perf_counter()
        pi_mc, ld_mc = montecarlo.expected_constant(
            Ne=float(Ne),
            left_bins=left_bins,
            right_bins=right_bins,
            mutation_rate=MUTATION_RATE,
            recombination_rate=recombination_rate,
            sequence_length=sequence_length,
            sample_size=num_samples,
            random_seed=RANDOM_SEED + int(i),
            rtol=RTOL,
            model=SMC_PRIME_MODEL,
            num_workers=num_workers,
            ploidy=ploidy,
        )
        log(f"{tag} MC         {time.perf_counter() - t0:7.2f}s")

        results.append(
            {
                "Ne": float(Ne),
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
            "mutation_rate": MUTATION_RATE,
            "recombination_rate": recombination_rate,
            "sequence_length": sequence_length,
            "num_samples": num_samples,
            "ploidy": ploidy,
            "rtol": RTOL,
            "model": "SMCK(k=1)",
        },
    }

    with gzip.open(out_path, "wb") as f:
        pickle.dump(data, f)
    log(f"Saved to {out_path}")


if __name__ == "__main__":
    main()

