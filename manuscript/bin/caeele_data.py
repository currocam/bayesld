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
Simulate windows from C. elegans using stdpopsim (constant Ne).
Then measure LD in distance bins with bayesld.

Outputs
-------
  {out_prefix}.pkl.gz  - pickled dict with per-window LD measurements
"""

import gzip
import pickle
import sys

import numpy as np
import stdpopsim
from bayesld import data_from_tree_sequence
from joblib import Parallel, delayed

# --- parameters ---
NUM_SAMPLES = 30
RANDOM_SEED = 56218473
NUM_WORKERS = 8
CHROMOSOMES = ["I", "II", "III", "IV", "V"]  # autosomes only; skip X and MtDNA

# 5 linear bins from 0.01 cM to 0.04 cM (in Morgan: 1e-4 to 4e-4)
_edges = np.linspace(1e-4, 4e-4, 6)
left_bins = _edges[:-1]
right_bins = _edges[1:]


def build_windows(species):
    """One window per full chromosome."""
    windows = []
    for chrom_id in CHROMOSOMES:
        chrom = species.genome.get_chromosome(chrom_id)
        windows.append(
            {
                "chrom": chrom_id,
                "left": 0,
                "right": chrom.length,
                "recombination_rate": chrom.recombination_rate,
                "mutation_rate": chrom.mutation_rate,
            }
        )
    return windows


def simulate(window, model, species, engine, seed):
    contig = species.get_contig(
        chromosome=window["chrom"],
        left=window["left"],
        right=window["right"],
        mutation_rate=window["mutation_rate"],
    )
    samples = {"pop_0": NUM_SAMPLES}
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

    out_prefix = sys.argv[1] if len(sys.argv) > 1 else "caeele"

    species = stdpopsim.get_species("CaeEle")
    model = stdpopsim.PiecewiseConstantSize(species.population_size)
    engine = stdpopsim.get_engine("msprime")

    # All autosomes + X share the same mutation rate in CaeEle
    mut_rates = {species.genome.get_chromosome(c).mutation_rate for c in CHROMOSOMES}
    assert len(mut_rates) == 1, f"Mixed mutation rates: {mut_rates}"
    mutation_rate = mut_rates.pop()

    windows = build_windows(species)
    print(
        f"Prepared {len(windows)} whole-chromosome windows "
        f"across {len(CHROMOSOMES)} chromosomes"
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

    pkl_path = f"{out_prefix}.pkl.gz"
    data = {
        "results": results,
        "left_bins": left_bins,
        "right_bins": right_bins,
        "params": {
            "num_samples": NUM_SAMPLES,
            "random_seed": RANDOM_SEED,
            "chromosomes": CHROMOSOMES,
            "model_id": "PiecewiseConstant",
            "species_id": "CaeEle",
            "Ne": species.population_size,
            "mutation_rate": mutation_rate,
        },
    }
    with gzip.open(pkl_path, "wb") as f:
        pickle.dump(data, f)
    print(f"Saved {len(results)} windows to {pkl_path}")


if __name__ == "__main__":
    main()
