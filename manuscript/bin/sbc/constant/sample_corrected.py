#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "arviz>=1.1",
#     "bayesld @ git+https://github.com/currocam/bayesld.git@cf118d3",
#     "numpy",
# ]
# ///
"""
SBC stage B (constant Ne): MCMC sampling for one dataset, reusing the
synthetic points learned in stage A.

Usage:
    sample_corrected.py <learn.pkl> <output.pkl>
"""

import argparse
import pickle

import numpy as np
from bayesld.models import ConstantDemography

NUM_WORKERS = 1
SEED = 788277


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
    parser.add_argument("learn_pkl")
    parser.add_argument("output")
    args = parser.parse_args()

    with open(args.learn_pkl, "rb") as f:
        bundle = pickle.load(f)

    pi, ld = bundle["dataset"]
    model = _build_model(bundle["batch_meta"], pi, ld)
    model.add_synthetic_points(bundle["synthetic_points"])

    idata = model.sample(
        iter_warmup=2000, iter_sampling=2000, chains=1, seed=SEED
    )

    with open(args.output, "wb") as f:
        pickle.dump({"idata": idata, "dataset_idx": bundle["dataset_idx"]}, f)


if __name__ == "__main__":
    main()
