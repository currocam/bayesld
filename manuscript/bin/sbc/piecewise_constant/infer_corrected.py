#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "bayesld @ git+https://github.com/currocam/bayesld.git@152060c",
#     "numpy",
# ]
# ///
"""
SBC infer corrected — piecewise constant (2-epoch) model.

Usage:
    infer_corrected.py <batch.pkl> <output.pkl> [--n-points-per-iter 3] [--n-iter 10]
"""

import argparse
import pickle
import sys

import numpy as np
from bayesld.models import PiecewiseConstantDemography

NUM_WORKERS = 8
SEED = 391237


def _build_model(batch, pi, ld):
    prior = batch["prior"]

    parameters = """\
    real log_Ne1;
    real log_Ne2;
    real log_t0;"""

    transformed_parameters = """\
    real<lower=0> Ne1 = exp(log_Ne1);
    real<lower=0> Ne2 = exp(log_Ne2);
    real<lower=0> t0  = exp(log_t0);
    vector[2] Ne_values = [Ne1, Ne2]';
    vector[1] t_boundaries = [t0]';"""

    prior_str = (
        f"    log_Ne1 ~ normal({np.log(prior['prior_ne1']):.6f}, {prior['prior_sigma_ne']:.6f});\n"
        f"    log_Ne2 ~ normal({np.log(prior['prior_ne2']):.6f}, {prior['prior_sigma_ne']:.6f});\n"
        f"    log_t0  ~ normal({np.log(prior['prior_t0']):.6f}, {prior['prior_sigma_t0']:.6f});"
    )

    return PiecewiseConstantDemography(
        diversity=pi,
        ld=ld,
        mutation_rate=batch["mutation_rate"],
        recombination_rate=batch["recombination_rate"],
        num_samples=batch["sample_size"],
        left_bins=batch["left_bins"],
        right_bins=batch["right_bins"],
        sequence_length=batch["window_length"],
        n_epochs=2,
        parameters=parameters,
        transformed_parameters=transformed_parameters,
        prior=prior_str,
        num_workers=NUM_WORKERS,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("batch_pkl")
    parser.add_argument("output")
    parser.add_argument("--n-points-per-iter", type=int, default=6)
    parser.add_argument("--n-iter", type=int, default=10)
    args = parser.parse_args()

    with open(args.batch_pkl, "rb") as f:
        batch = pickle.load(f)

    idatas = []
    for i, (pi, ld) in enumerate(batch["datasets"]):
        model = _build_model(batch, pi, ld)
        model.active_learn_bias(
            n_points_per_iter=args.n_points_per_iter,
            n_iter=args.n_iter,
            strategy="pathfinder",
            num_replicates=50,
            seed=SEED,
        )
        idatas.append(
            model.sample(iter_warmup=2000, iter_sampling=2000, chains=2, seed=SEED)
        )
        print(f"  {i + 1}/{len(batch['datasets'])}", file=sys.stderr)

    with open(args.output, "wb") as f:
        pickle.dump({"idatas": idatas}, f)


if __name__ == "__main__":
    main()
