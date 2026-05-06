#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "bayesld @ git+https://github.com/currocam/bayesld.git@2cd0e46",
#     "msprime==1.4.0",
#     "numpy==2.2.6",
# ]
# ///
"""
SBC simulate — piecewise exponential (2-epoch) model.

Draws Ne1 ~ LogNormal(log(prior_ne1), prior_sigma_ne),
      Ne2 ~ LogNormal(log(prior_ne2), prior_sigma_ne),
      t0  ~ LogNormal(log(prior_t0), prior_sigma_t0).

Usage:
    simulate.py <output.pkl> --prior-ne1 5000 --prior-sigma-ne 0.5
                --prior-ne2 5000 --prior-t0 50 --prior-sigma-t0 1.0
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
    parser.add_argument("--prior-ne1", type=float, required=True)
    parser.add_argument("--prior-sigma-ne", type=float, required=True)
    parser.add_argument("--prior-ne2", type=float, required=True)
    parser.add_argument("--prior-t0", type=float, required=True)
    parser.add_argument("--prior-sigma-t0", type=float, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--sample-size", type=int, required=True)
    parser.add_argument("--num-windows", type=int, required=True)
    parser.add_argument("--seed", type=int, default=321736)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    ne1_draws = rng.lognormal(
        mean=np.log(args.prior_ne1), sigma=args.prior_sigma_ne, size=args.batch_size
    )
    ne2_draws = rng.lognormal(
        mean=np.log(args.prior_ne2), sigma=args.prior_sigma_ne, size=args.batch_size
    )
    t0_draws = rng.lognormal(
        mean=np.log(args.prior_t0), sigma=args.prior_sigma_t0, size=args.batch_size
    )
    alpha_draws = (np.log(ne1_draws) - np.log(ne2_draws)) / t0_draws
    assert np.all(np.isclose(ne1_draws * np.exp(-alpha_draws * t0_draws), ne2_draws)), (
        "Wrong growth rates calculation"
    )

    left_bins, right_bins = linear_bins()
    window_length = right_bins[-1] * 2 / RECOMBINATION_RATE

    datasets = []
    for i in range(args.batch_size):
        pi, ld = mc.expected_piecewise_exponential(
            ne1_draws[i],
            ne2_draws[i],
            t0_draws[i],
            alpha_draws[i],
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
            model=msprime.SMCK(k=0),
        )
        datasets.append((np.array(pi), np.array(ld)))

    with open(args.output, "wb") as f:
        pickle.dump(
            {
                "ne1_draws": ne1_draws,
                "ne2_draws": ne2_draws,
                "t0_draws": t0_draws,
                "alpha_draws": alpha_draws,
                "datasets": datasets,
                "left_bins": left_bins,
                "right_bins": right_bins,
                "mutation_rate": MUTATION_RATE,
                "recombination_rate": RECOMBINATION_RATE,
                "window_length": window_length,
                "sample_size": args.sample_size,
                "prior": {
                    "prior_ne1": args.prior_ne1,
                    "prior_ne2": args.prior_ne2,
                    "prior_sigma_ne": args.prior_sigma_ne,
                    "prior_t0": args.prior_t0,
                    "prior_sigma_t0": args.prior_sigma_t0,
                },
            },
            f,
        )

    print(f"Simulated {args.batch_size} datasets → {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
