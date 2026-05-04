#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "bayesld @ git+https://github.com/currocam/bayesld.git@2e20fa7",
#     "msprime==1.4.0",
#     "numpy==2.2.6",
#     "stdpopsim==0.3.0",
#     "joblib==1.5.3",
#     "tqdm==4.67.3",
# ]
# ///
"""
Simulate windows from Holstein-Friesian cattle using stdpopsim. Then measure LD in distance bins with bayesld.

Outputs
-------
  {out_prefix}.pkl.gz  – pickled dict with per-window LD measurements
"""

import gzip
import pickle
import sys

import numpy as np
import stdpopsim
from bayesld import data_from_tree_sequence, linear_bins
from joblib import Parallel, delayed

# --- parameters ---
NUM_SAMPLES = 30
RANDOM_SEED = 48172635
NUM_WORKERS = 8
CHROMOSOMES = [str(i) for i in range(1, 10)]  # 1..29 autosomes
WINDOW_MORGAN = 0.2  # 20 cM

left_bins, right_bins = linear_bins()


def build_windows(species, model):
    """Tile each chromosome into non-overlapping windows of WINDOW_MORGAN."""
    window_bp = int(WINDOW_MORGAN / model.recombination_rate)
    windows = []
    for chrom_id in CHROMOSOMES:
        chrom = species.genome.get_chromosome(chrom_id)
        chrom_len = chrom.length
        left = 0
        while left + window_bp <= chrom_len:
            windows.append({"chrom": chrom_id, "left": left, "right": left + window_bp})
            left += window_bp
    assert len(windows) > 0, "Ups, window size too large for chromosome length"
    return windows


def simulate(window, model, species, engine, seed):
    contig = species.get_contig(
        chromosome=window["chrom"],
        left=window["left"],
        right=window["right"],
        mutation_rate=model.mutation_rate,
        recombination_rate=model.recombination_rate,
    )
    samples = {"Holstein_Friesian": NUM_SAMPLES}
    ts = engine.simulate(
        demographic_model=model,
        contig=contig,
        samples=samples,
        seed=seed,
    )
    ld_data = data_from_tree_sequence(
        ts=ts,
        recombination_rate=model.recombination_rate,
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

    out_prefix = sys.argv[1] if len(sys.argv) > 1 else "holsteinfriesian"

    species = stdpopsim.get_species("BosTau")
    model = species.get_demographic_model("HolsteinFriesian_1M13")
    engine = stdpopsim.get_engine("msprime")

    windows = build_windows(species, model)
    print(
        f"Prepared {len(windows)} windows of {WINDOW_MORGAN * 100:.0f} cM across {len(CHROMOSOMES)} chromosomes"
    )

    rng = np.random.RandomState(RANDOM_SEED)
    seeds = rng.randint(1, 2**31, size=len(windows))

    results = list(
        tqdm.tqdm(
            Parallel(n_jobs=NUM_WORKERS, return_as="generator")(
                delayed(simulate)(w, model, species, engine, int(s))
                for w, s in zip(windows, seeds)
            ),
            total=len(windows),
            desc="Windows",
        )
    )

    # --- save LD data ---
    pkl_path = f"{out_prefix}.pkl.gz"
    data = {
        "results": results,
        "left_bins": left_bins,
        "right_bins": right_bins,
        "params": {
            "num_samples": NUM_SAMPLES,
            "random_seed": RANDOM_SEED,
            "chromosomes": CHROMOSOMES,
            "window_morgan": WINDOW_MORGAN,
            "model_id": "HolsteinFriesian_1M13",
            "species_id": "BosTau",
            "mutation_rate": model.mutation_rate,
            "recombination_rate": model.recombination_rate,
        },
    }
    with gzip.open(pkl_path, "wb") as f:
        pickle.dump(data, f)
    print(f"Saved {len(results)} windows to {pkl_path}")


if __name__ == "__main__":
    main()
