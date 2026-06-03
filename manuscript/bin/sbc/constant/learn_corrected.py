#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "arviz>=0.23.4",
#     "bayesld @ git+https://github.com/currocam/bayesld.git@c56ee31",
#     "numpy",
# ]
# ///
# update 2
"""
SBC stage A (constant Ne): active learning + final pathfinder for one dataset.

Usage:
    learn_corrected.py <batch.pkl> <dataset_idx> <output.pkl>
                       [--n-points-per-iter 3] [--n-iter 10]
"""

import argparse
import pickle

import arviz as az
import numpy as np
from bayesld.models import ConstantDemography

NUM_WORKERS = 8
SEED = 788277
PATHFINDER_DRAWS = 1000


def _build_model(batch_meta, pi, ld):
    prior = batch_meta["prior"]
    prior_str = (
        f"    log_Ne ~ normal({np.log(prior['prior_ne']):.6f}, "
        f"{prior['prior_sigma']:.6f});"
    )
    return ConstantDemography(
        diversity=pi,
        ld=ld,
        mutation_rate=batch_meta["mutation_rate"],
        recombination_rate=batch_meta["recombination_rate"],
        num_samples=batch_meta["sample_size"],
        left_bins=batch_meta["left_bins"],
        right_bins=batch_meta["right_bins"],
        sequence_length=batch_meta["window_length"],
        prior=prior_str,
        num_workers=NUM_WORKERS,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("batch_pkl")
    parser.add_argument("dataset_idx", type=int)
    parser.add_argument("output")
    parser.add_argument("--n-points-per-iter", type=int, default=3)
    parser.add_argument("--n-iter", type=int, default=10)
    args = parser.parse_args()

    with open(args.batch_pkl, "rb") as f:
        batch = pickle.load(f)

    pi, ld = batch["datasets"][args.dataset_idx]
    batch_meta = {k: v for k, v in batch.items() if k != "datasets"}

    model = _build_model(batch_meta, pi, ld)
    model.active_learn_bias(
        n_points_per_iter=args.n_points_per_iter,
        n_iter=args.n_iter,
        strategy="pathfinder",
        num_replicates=50,
        seed=SEED,
    )

    pf = model.model.pathfinder(
        data=model.stan_data(),
        draws=PATHFINDER_DRAWS,
        seed=SEED,
        num_threads=NUM_WORKERS,
        show_console=False,
        inits=0.5,
    )
    # Pathfinder fits don't have a chain dim; az.from_cmdstanpy expects one.
    # Wrap stan_variables() output with a singleton chain axis.
    idata_pathfinder = az.from_dict(
        {"posterior": {k: v[np.newaxis, ...] for k, v in pf.stan_variables().items()}}
    )

    bundle = {
        "batch_meta": batch_meta,
        "dataset": (pi, ld),
        "dataset_idx": args.dataset_idx,
        "synthetic_points": model.synthetic_points,
        "idata_pathfinder": idata_pathfinder,
    }
    with open(args.output, "wb") as f:
        pickle.dump(bundle, f)


if __name__ == "__main__":
    main()
