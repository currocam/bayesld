#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "bayesld @ git+https://github.com/currocam/bayesld.git@c56ee31",
#     "numpy",
# ]
# ///
"""
SBC stage B (piecewise exponential, 2-epoch): MCMC sampling for one dataset,
reusing the synthetic points learned in stage A.

Usage:
    sample_corrected.py <learn.pkl> <output.pkl>
"""

import argparse
import pickle

import numpy as np
from bayesld.models import PiecewiseExponentialDemography

NUM_WORKERS = 1
SEED = 736187


def _build_model(batch_meta, pi, ld):
    prior = batch_meta["prior"]
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
        mutation_rate=batch_meta["mutation_rate"],
        recombination_rate=batch_meta["recombination_rate"],
        num_samples=batch_meta["sample_size"],
        left_bins=batch_meta["left_bins"],
        right_bins=batch_meta["right_bins"],
        sequence_length=batch_meta["window_length"],
        parameters=parameters,
        transformed_parameters=transformed_parameters,
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
        iter_warmup=6000, iter_sampling=2000, chains=1, seed=SEED
    )

    with open(args.output, "wb") as f:
        pickle.dump({"idata": idata, "dataset_idx": bundle["dataset_idx"]}, f)


if __name__ == "__main__":
    main()
