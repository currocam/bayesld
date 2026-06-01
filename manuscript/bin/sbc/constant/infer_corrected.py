#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "bayesld @ git+https://github.com/currocam/bayesld.git@152060c",
#     "numpy",
# ]
# ///
"""
SBC infer corrected — constant Ne model.

Constructs a fresh ConstantDemography per dataset (CmdStanPy caches the
compiled binary by Stan code hash, so no recompilation after the first).
Prior parameters are read from the batch pkl.

Usage:
    infer_corrected.py <batch.pkl> <output.pkl> [--n-points-per-iter 3] [--n-iter 10]
"""

import argparse
import pickle
import sys

import numpy as np
from bayesld.models import ConstantDemography

NUM_WORKERS = 8


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("batch_pkl")
    parser.add_argument("output")
    parser.add_argument("--n-points-per-iter", type=int, default=3)
    parser.add_argument("--n-iter", type=int, default=10)
    args = parser.parse_args()

    with open(args.batch_pkl, "rb") as f:
        batch = pickle.load(f)

    prior = batch["prior"]
    prior_str = f"    log_Ne ~ normal({np.log(prior['prior_ne']):.6f}, {prior['prior_sigma']:.6f});"

    idatas = []
    for i, (pi, ld) in enumerate(batch["datasets"]):
        model = ConstantDemography(
            diversity=pi,
            ld=ld,
            mutation_rate=batch["mutation_rate"],
            recombination_rate=batch["recombination_rate"],
            num_samples=batch["sample_size"],
            left_bins=batch["left_bins"],
            right_bins=batch["right_bins"],
            sequence_length=batch["window_length"],
            prior=prior_str,
            num_workers=NUM_WORKERS,
        )
        model.active_learn_bias(
            n_points_per_iter=args.n_points_per_iter,
            n_iter=args.n_iter,
            strategy="pathfinder",
        )
        idatas.append(model.sample(iter_warmup=2000, iter_sampling=2000, chains=2))
        print(f"  {i + 1}/{len(batch['datasets'])}", file=sys.stderr)

    with open(args.output, "wb") as f:
        pickle.dump({"idatas": idatas}, f)


if __name__ == "__main__":
    main()
