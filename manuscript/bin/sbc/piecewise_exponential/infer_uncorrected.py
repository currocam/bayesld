#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "bayesld @ git+https://github.com/currocam/bayesld.git@152060c",
#     "numpy",
# ]
# ///
"""
SBC infer uncorrected — piecewise exponential (2-epoch) model.

Usage:
    infer_uncorrected.py <batch.pkl> <output.pkl>
"""

import argparse
import pickle
import sys

import numpy as np
from bayesld.models import PiecewiseExponentialDemography

NUM_WORKERS = 8
SEED = 361278


def _build_model(batch, pi, ld):
    prior = batch["prior"]

    parameters = """\
    real<offset=log_ne_offset> log_Ne_c;
    real<offset=log_ne_offset> log_Ne_a;
    real log_t0;"""

    transformed_parameters = """\
    real<lower=0> Ne_c = exp(log_Ne_c);
    real<lower=0> Ne_a = exp(log_Ne_a);
    real<lower=0> t0   = exp(log_t0);
    real log_fold_change = log_Ne_c - log_Ne_a;
    real alpha = log_fold_change / t0;"""

    prior_str = (
        f"    log_Ne_c ~ normal({np.log(prior['prior_ne1']):.6f}, {prior['prior_sigma_ne']:.6f});\n"
        f"    log_Ne_a ~ normal({np.log(prior['prior_ne2']):.6f}, {prior['prior_sigma_ne']:.6f});\n"
        f"    log_t0  ~ normal({np.log(prior['prior_t0']):.6f}, {prior['prior_sigma_t0']:.6f});"
    )

    return PiecewiseExponentialDemography(
        diversity=pi,
        ld=ld,
        mutation_rate=batch["mutation_rate"],
        recombination_rate=batch["recombination_rate"],
        num_samples=batch["sample_size"],
        left_bins=batch["left_bins"],
        right_bins=batch["right_bins"],
        sequence_length=batch["window_length"],
        parameters=parameters,
        transformed_parameters=transformed_parameters,
        prior=prior_str,
        num_workers=NUM_WORKERS,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("batch_pkl")
    parser.add_argument("output")
    args = parser.parse_args()

    with open(args.batch_pkl, "rb") as f:
        batch = pickle.load(f)

    pi0, ld0 = batch["datasets"][0]
    model = _build_model(batch, pi0, ld0)

    idatas = []
    for i, (pi, ld) in enumerate(batch["datasets"]):
        model.update_data(diversity=pi, ld=ld)
        idatas.append(
            model.sample(iter_warmup=6000, iter_sampling=2000, chains=2, seed=SEED)
        )
        print(f"  {i + 1}/{len(batch['datasets'])}", file=sys.stderr)

    with open(args.output, "wb") as f:
        pickle.dump({"idatas": idatas}, f)


if __name__ == "__main__":
    main()
