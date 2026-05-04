#!/usr/bin/env -S uv run --script --isolated
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "bayesld @ git+https://github.com/currocam/bayesld.git@a78bf21",
#     "msprime==1.4.0",
#     "numpy==2.2.6",
#     "stdpopsim==0.3.0",
#     "joblib==1.5.3",
#     "tqdm==4.67.3",
# ]
# ///
"""
Simulate windows from Vaquita using stdpopsim. Then measure LD in distance bins with bayesld.

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
RANDOM_SEED = 56218473
NUM_WORKERS = 8
CHROMOSOMES = [str(i) for i in range(1, 22)]  # 1..21 autosomes
WINDOW_MORGAN = 0.2  # 20 cM

left_bins, right_bins = linear_bins()


def build_windows(species):
    """Tile each chromosome into non-overlapping windows of WINDOW_MORGAN."""
    windows = []
    for chrom_id in CHROMOSOMES:
        chrom = species.genome.get_chromosome(chrom_id)
        rec_rate = chrom.recombination_rate
        window_bp = int(WINDOW_MORGAN / rec_rate)
        left = 0
        while left + window_bp <= chrom.length:
            windows.append({
                "chrom": chrom_id,
                "left": left,
                "right": left + window_bp,
                "recombination_rate": rec_rate,
            })
            left += window_bp
    assert len(windows) > 0, "Ups, window size too large for chromosome length"
    return windows


def simulate(window, model, species, engine, seed):
    contig = species.get_contig(
        chromosome=window["chrom"],
        left=window["left"],
        right=window["right"],
        mutation_rate=model.mutation_rate,
    )
    samples = {"Vaquita": NUM_SAMPLES}
    ts = engine.simulate(
        demographic_model=model,
        contig=contig,
        samples=samples,
        seed=seed,
    )
    rec_rate = window["recombination_rate"]
    ld_data = data_from_tree_sequence(
        ts=ts,
        recombination_rate=rec_rate,
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

    out_prefix = sys.argv[1] if len(sys.argv) > 1 else "vaquita"

    species = stdpopsim.get_species("PhoSin")
    model = species.get_demographic_model("Vaquita2Epoch_1R22")
    engine = stdpopsim.get_engine("msprime")

    windows = build_windows(species)
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
            "model_id": "Vaquita2Epoch_1R22",
            "species_id": "PhoSin",
            "mutation_rate": model.mutation_rate,
        },
    }
    with gzip.open(pkl_path, "wb") as f:
        pickle.dump(data, f)
    print(f"Saved {len(results)} windows to {pkl_path}")


if __name__ == "__main__":
    main()
