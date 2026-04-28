# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "bayesld",
#     "msprime",
#     "numpy",
#     "joblib",
#     "tqdm",
# ]
#
# [tool.uv.sources]
# bayesld = { path = "../../" }
# ///
import gzip
import pickle
import sys

import msprime
import numpy as np
from bayesld import data_from_tree_sequence, linear_bins
from joblib import Parallel, delayed

# --- parameters ---
NE1 = 500  # Contemporary Ne
NE2 = 5000  # Ancient Ne
RECOMBINATION_RATE = MUTATION_RATE = 1e-8
NUM_WINDOWS = 5 * 20
NUM_SAMPLES = 50
RANDOM_SEED = 62387687
NUM_WORKERS = 8

left_bins, right_bins = linear_bins()
SEQUENCE_LENGTH = right_bins[-1] * 2 / RECOMBINATION_RATE

# Few points for quick iteration
TIMES = np.arange(1, 101)


def make_demo(t0):
    demo = msprime.Demography.isolated_model(initial_size=[NE1])
    demo.add_population_parameters_change(initial_size=NE2, time=t0)
    return demo


def sim_replicates(demo=None, population_size=None):
    seeds = np.arange(RANDOM_SEED, RANDOM_SEED + NUM_WINDOWS)
    if demo is not None:
        ts_list = Parallel(n_jobs=NUM_WORKERS)(
            delayed(msprime.sim_ancestry)(
                samples=NUM_SAMPLES,
                sequence_length=SEQUENCE_LENGTH,
                recombination_rate=RECOMBINATION_RATE,
                demography=demo,
                random_seed=int(s),
            )
            for s in seeds
        )
    else:
        # Baseline
        ts_list = Parallel(n_jobs=NUM_WORKERS)(
            delayed(msprime.sim_ancestry)(
                samples=NUM_SAMPLES,
                sequence_length=SEQUENCE_LENGTH,
                recombination_rate=RECOMBINATION_RATE,
                population_size=population_size,
                model=[
                    msprime.DiscreteTimeWrightFisher(duration=500),
                    msprime.StandardCoalescent(),
                ],
                random_seed=int(s),
            )
            for s in seeds
        )
    return [
        msprime.sim_mutations(
            ts,
            random_seed=int(s),
            rate=MUTATION_RATE,
            model=msprime.BinaryMutationModel(),
        )
        for ts, s in zip(ts_list, seeds)
    ]


def compute_linkage(ts):
    return data_from_tree_sequence(
        ts=ts,
        recombination_rate=RECOMBINATION_RATE,
        left_bins_morgan=left_bins,
        right_bins_morgan=right_bins,
    )["mean_linkage_disequilibrium"]


def compute_stats(ts_list):
    """Compute all summary statistics and discard tree sequences."""
    div = np.array([ts.diversity() for ts in ts_list])
    afs = np.array(
        [
            ts.allele_frequency_spectrum(polarised=True, span_normalise=True)
            for ts in ts_list
        ]
    )
    linkage = np.array([compute_linkage(ts) for ts in ts_list])
    return div, afs, linkage


def main():
    import tqdm.auto as tqdm

    out_path = sys.argv[1] if len(sys.argv) > 1 else "conceptual_data.pkl"

    print("Simulating baseline (constant Ne)...")
    baseline_div, baseline_afs, baseline_linkage = compute_stats(
        sim_replicates(population_size=NE2)
    )

    print(f"Simulating {len(TIMES)} demographic scenarios...")
    sim_div, sim_afs, sim_linkage = [], [], []
    for t0 in tqdm.tqdm(TIMES):
        div, afs, linkage = compute_stats(sim_replicates(demo=make_demo(t0)))
        sim_div.append(div)
        sim_afs.append(afs)
        sim_linkage.append(linkage)

    sim_div = np.array(sim_div)
    sim_afs = np.array(sim_afs)
    sim_linkage = np.array(sim_linkage)

    data = {
        "baseline_div": baseline_div,
        "sim_div": sim_div,
        "baseline_afs": baseline_afs,
        "sim_afs": sim_afs,
        "baseline_linkage": baseline_linkage,
        "sim_linkage": sim_linkage,
        "times": TIMES,
        "params": {
            "ne1": NE1,
            "ne2": NE2,
            "recombination_rate": RECOMBINATION_RATE,
            "mutation_rate": MUTATION_RATE,
            "num_windows": NUM_WINDOWS,
            "num_samples": NUM_SAMPLES,
            "sequence_length": SEQUENCE_LENGTH,
        },
    }

    with gzip.open(out_path, "wb") as f:
        pickle.dump(data, f)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
