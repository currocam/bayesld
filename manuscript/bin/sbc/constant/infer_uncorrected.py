#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "bayesld @ git+https://github.com/currocam/bayesld.git@152060c",
#     "numpy",
# ]
# ///
"""
SBC infer uncorrected — constant Ne model.

Compiles the Stan model once, then loops over datasets with update_data + sample.
Prior parameters are read from the batch pkl.

Usage:
    infer_uncorrected.py <batch.pkl> <output.pkl>
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
    args = parser.parse_args()

    with open(args.batch_pkl, "rb") as f:
        batch = pickle.load(f)

    prior = batch["prior"]
    prior_str = f"    log_Ne ~ normal({np.log(prior['prior_ne']):.6f}, {prior['prior_sigma']:.6f});"

    pi0, ld0 = batch["datasets"][0]
    model = ConstantDemography(
        diversity=pi0,
        ld=ld0,
        mutation_rate=batch["mutation_rate"],
        recombination_rate=batch["recombination_rate"],
        num_samples=batch["sample_size"],
        left_bins=batch["left_bins"],
        right_bins=batch["right_bins"],
        sequence_length=batch["window_length"],
        prior=prior_str,
        num_workers=NUM_WORKERS,
    )

    idatas = []
    for i, (pi, ld) in enumerate(batch["datasets"]):
        model.update_data(diversity=pi, ld=ld)
        idatas.append(model.sample(iter_warmup=2000, iter_sampling=2000, chains=2))
        print(f"  {i + 1}/{len(batch['datasets'])}", file=sys.stderr)

    with open(args.output, "wb") as f:
        pickle.dump({"idatas": idatas}, f)


if __name__ == "__main__":
    main()
