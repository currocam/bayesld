#!/usr/bin/env -S uv run --script
# /// script
# dependencies = [
#     "bayesld==0.1.0",
#     "numpy==2.4.4",
#     "arviz==1.1.0",
#     "h5netcdf==1.8.1",
#     "h5py==3.16.0",
#     "netcdf4==1.7.4",
# ]
# requires-python = ">=3.12"
#
# [tool.uv.sources]
# bayesld = { git = "https://github.com/currocam/bayesld.git", rev = "363b723" }
# ///

"""Fit a log-random-walk Ne(t) model over a fixed epoch grid.

Usage:
    fit_random_walk.py <data.pkl> <recombination_rate> <mutation_rate>
                        <num_workers> <out.nc>
"""

import pickle
import sys

import numpy as np
from bayesld import inference

SEED = 20250821


def main():
    if len(sys.argv) != 6:
        print(__doc__, file=sys.stderr)
        sys.exit(1)

    data_pkl = sys.argv[1]
    recombination_rate = float(sys.argv[2])
    mutation_rate = float(sys.argv[3])
    num_workers = int(sys.argv[4])
    out_nc = sys.argv[5]

    with open(data_pkl, "rb") as f:
        data = pickle.load(f)

    grid = np.unique(np.round(np.geomspace(1, 1000, num=25)).astype(int))
    dt = np.diff(np.append([0], grid))
    sigma_step = 0.5 * np.sqrt(dt)

    model = inference.RandomWalk(grid=grid).with_data(
        mean_diversity=data["mean_genetic_diversity"],
        mean_ld=data["mean_linkage_disequilibrium"],
        left_bins=data["left_bins_morgan"],
        right_bins=data["right_bins_morgan"],
        recombination_rate=recombination_rate,
        mutation_rate=mutation_rate,
        num_samples=data["sample_size"],
        sequence_length=data["window_length"],
    ).with_prior(
        mu_log_ne=np.log(30_000),
        sigma_log_ne=1.5,
        sigma_step=sigma_step,
    )

    prior = model.sample_prior()

    for i in range(4):
        model = model.active_learning_round(
            25,
            rtol=0.1,
            min_replicates=20,
            seed=SEED + i,
            num_workers=num_workers,
            verbose=True,
        )

    idata = model.sample(
        tune=2000,
        draws=4000,
        chains=4,
        parallel_chains=4,
        num_workers=4,
        seed=SEED,
    )
    idata["prior"] = prior["posterior"]

    idata.to_netcdf(out_nc)


if __name__ == "__main__":
    main()
