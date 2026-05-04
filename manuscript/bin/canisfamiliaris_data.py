#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "bayesld @ git+https://github.com/currocam/bayesld.git@ae4d14b",
#     "msprime==1.4.0",
#     "numpy==2.2.6",
#     "stdpopsim==0.3.0",
#     "joblib==1.5.3",
#     "tqdm==4.67.3",
# ]
# ///
"""
Simulate windows from Canis familiaris under constant Ne using stdpopsim
genome parameters. Vary Ne from 13000 down to 13 (dividing by 10).

Uses msprime directly with SMCK(k=1) ancestry model and genome parameters
(recombination/mutation rates, chromosome lengths) from stdpopsim.

Outputs
-------
  {out_prefix}.pkl.gz  – pickled dict with per-window, per-Ne LD measurements
"""

import gzip
import pickle
import sys

import msprime
import numpy as np
import stdpopsim
from bayesld import data_from_tree_sequence, linear_bins
from joblib import Parallel, delayed

# --- parameters ---
NUM_SAMPLES = 30
RANDOM_SEED = 81263947
NUM_WORKERS = 8
CHROMOSOMES = [str(i) for i in range(1, 39)]  # 1..38 autosomes
WINDOW_MORGAN = 0.2  # 20 cM
NE_VALUES = np.geomspace(30_000, 50, num=100).tolist()

left_bins, right_bins = linear_bins()


def build_windows(species):
    """Tile each chromosome into non-overlapping windows of WINDOW_MORGAN.

    Extracts plain scalars from stdpopsim so the returned dicts are picklable.
    """
    windows = []
    for chrom_id in CHROMOSOMES:
        chrom = species.genome.get_chromosome(chrom_id)
        rec_rate = chrom.recombination_rate
        mut_rate = chrom.mutation_rate
        window_bp = int(WINDOW_MORGAN / rec_rate)
        left = 0
        while left + window_bp <= chrom.length:
            windows.append(
                {
                    "chrom": chrom_id,
                    "left": left,
                    "right": left + window_bp,
                    "recombination_rate": rec_rate,
                    "mutation_rate": mut_rate,
                }
            )
            left += window_bp
    assert len(windows) > 0, "Ups, window size too large for chromosome length"
    return windows


def simulate(window, ne, seed):
    """Simulate one window with msprime SMCK(k=1) and measure LD."""
    ts = msprime.sim_ancestry(
        samples=NUM_SAMPLES,
        sequence_length=window["right"] - window["left"],
        recombination_rate=window["recombination_rate"],
        population_size=ne,
        model=msprime.SMCK(k=1),
        random_seed=seed,
    )
    ts = msprime.sim_mutations(
        ts,
        rate=window["mutation_rate"],
        random_seed=seed,
        model=msprime.BinaryMutationModel(),
    )
    ld_data = data_from_tree_sequence(
        ts=ts,
        recombination_rate=window["recombination_rate"],
        left_bins_morgan=left_bins,
        right_bins_morgan=right_bins,
    )
    return {
        "chrom": window["chrom"],
        "left": window["left"],
        "right": window["right"],
        "sequence_length": ts.sequence_length,
        "num_sites": ts.num_sites,
        "num_trees": ts.num_trees,
        **ld_data,
    }


def main():
    import tqdm

    out_prefix = sys.argv[1] if len(sys.argv) > 1 else "canisfamiliaris"

    species = stdpopsim.get_species("CanFam")
    windows = build_windows(species)
    print(
        f"Prepared {len(windows)} windows of {WINDOW_MORGAN * 100:.0f} cM "
        f"across {len(CHROMOSOMES)} chromosomes"
    )
    print(f"Ne values: {NE_VALUES}")

    rng = np.random.RandomState(RANDOM_SEED)

    all_results = {}
    for ne in NE_VALUES:
        print(f"\n--- Ne = {ne:,} ---")
        seeds = rng.randint(1, 2**31, size=len(windows))
        results = list(
            tqdm.tqdm(
                Parallel(n_jobs=NUM_WORKERS, return_as="generator", prefer="threads")(
                    delayed(simulate)(w, ne, int(s)) for w, s in zip(windows, seeds)
                ),
                total=len(windows),
                desc=f"Ne={ne:,}",
            )
        )
        all_results[ne] = results

    # --- save ---
    pkl_path = f"{out_prefix}.pkl.gz"
    data = {
        "results": all_results,
        "left_bins": left_bins,
        "right_bins": right_bins,
        "params": {
            "num_samples": NUM_SAMPLES,
            "random_seed": RANDOM_SEED,
            "chromosomes": CHROMOSOMES,
            "window_morgan": WINDOW_MORGAN,
            "ne_values": NE_VALUES,
            "species_id": "CanFam",
        },
    }
    with gzip.open(pkl_path, "wb") as f:
        pickle.dump(data, f)
    print(
        f"\nSaved {sum(len(v) for v in all_results.values())} total windows to {pkl_path}"
    )


if __name__ == "__main__":
    main()
