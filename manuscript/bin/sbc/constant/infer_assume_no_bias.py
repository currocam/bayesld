#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "bayesld @ git+https://github.com/currocam/bayesld.git@152060c",
#     "numpy",
# ]
# ///
"""
SBC infer naive model with no bias — constant Ne model.

Usage:
    infer_assume_no_bias.py <stan_file> <batch.pkl> <output.pkl>
"""

import argparse
import pickle
import sys

import arviz as az
import numpy as np
from cmdstanpy import CmdStanModel

NUM_WORKERS = 8
SEED = 932187


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stan_file")
    parser.add_argument("batch_pkl")
    parser.add_argument("output")
    args = parser.parse_args()

    # Compile the Stan model
    model = CmdStanModel(stan_file=args.stan_file)

    with open(args.batch_pkl, "rb") as f:
        batch = pickle.load(f)

    left_bins = batch["left_bins"]
    right_bins = batch["right_bins"]
    mutation_rate = batch["mutation_rate"]
    sample_size = batch["sample_size"]
    prior = batch["prior"]

    n_bins = int(len(left_bins))
    base_data = {
        "n_bins": n_bins,
        "left_bins": left_bins,
        "right_bins": right_bins,
        "mutation_rate": mutation_rate,
        "sample_size": sample_size,
        # Prior
        "mu_log_ne_prior": np.log(prior["prior_ne"]),
        "sigma_log_ne_prior": prior["prior_sigma"],
        # Joint MVN noise hyperparameters (match corrected model defaults)
        "lkj_eta": 2.0,
        "log_sigma_y_scale": np.ones(n_bins + 1),
    }

    idatas = []
    for i, (pi, ld) in enumerate(batch["datasets"]):
        emp_sd = np.concatenate(
            [[float(np.std(pi, ddof=1))], np.std(ld, axis=0, ddof=1)]
        )
        data = {
            **base_data,
            "num_windows": int(len(pi)),
            "pi_array": pi,
            "ld_mat": ld,
            "log_sigma_y_loc": np.log(emp_sd),
        }
        fit = model.sample(
            data=data,
            iter_warmup=2000,
            iter_sampling=2000,
            chains=2,
            parallel_chains=2,
            seed=SEED,
            threads_per_chain=NUM_WORKERS // 2,
        )
        idatas.append(az.from_cmdstanpy(fit))
        print(f"  {i + 1}/{len(batch['datasets'])}", file=sys.stderr)

    with open(args.output, "wb") as f:
        pickle.dump({"idatas": idatas}, f)


if __name__ == "__main__":
    main()
