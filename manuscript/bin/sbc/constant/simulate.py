#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "bayesld @ git+https://github.com/currocam/bayesld.git@8e48ea0",
#     "msprime==1.4.0",
#     "numpy==2.2.6",
# ]
# ///
"""
SBC simulate — constant Ne model.

Usage:
    simulate.py <output.pkl> --prior-ne 5000 --prior-sigma 0.5
                --batch-size 50 --sample-size 50 --num-windows 50 --seed 321736
"""

import argparse
import pickle
import sys

import msprime
import numpy as np
from bayesld import linear_bins
from bayesld import montecarlo as mc

MUTATION_RATE = RECOMBINATION_RATE = 1e-8
NUM_WORKERS = 8


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    parser.add_argument("--prior-ne", type=float, required=True)
    parser.add_argument("--prior-sigma", type=float, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--sample-size", type=int, required=True)
    parser.add_argument("--num-windows", type=int, required=True)
    parser.add_argument("--seed", type=int, default=321736)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    num_draws = rng.lognormal(
        mean=np.log(args.prior_ne), sigma=args.prior_sigma, size=args.batch_size
    )

    left_bins, right_bins = linear_bins()
    window_length = right_bins[-1] * 2 / RECOMBINATION_RATE

    datasets = []
    for i, ne in enumerate(num_draws):
        pi, ld = mc.expected_constant(
            ne,
            left_bins,
            right_bins,
            MUTATION_RATE,
            RECOMBINATION_RATE,
            window_length,
            args.sample_size,
            random_seed=args.seed * 10000 + i,
            num_replicates=args.num_windows,
            ploidy=2,
            num_workers=NUM_WORKERS,
            model=msprime.SMCK(k=1),
        )
        datasets.append((np.array(pi), np.array(ld)))

    with open(args.output, "wb") as f:
        pickle.dump(
            {
                "num_draws": num_draws,
                "datasets": datasets,
                "left_bins": left_bins,
                "right_bins": right_bins,
                "mutation_rate": MUTATION_RATE,
                "recombination_rate": RECOMBINATION_RATE,
                "window_length": window_length,
                "sample_size": args.sample_size,
                "prior": {"prior_ne": args.prior_ne, "prior_sigma": args.prior_sigma},
            },
            f,
        )

    print(f"Simulated {args.batch_size} datasets → {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
