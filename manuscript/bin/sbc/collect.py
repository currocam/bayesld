#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "arviz==0.21.0",
#     "numpy==2.2.6",
# ]
# ///
"""
SBC collect — merges all batches for one (model, prior) into a single pkl.

Batch pkls supply num_draws, datasets, metadata, and a model-specific "prior" dict.
Uncorrected/corrected pkls supply idatas lists.
All three lists must be provided in matching batch order (sort by filename).

Usage:
    collect.py <output.pkl>
               --batches     batch_0.pkl batch_1.pkl ...
               --uncorrected unc_0.pkl   unc_1.pkl   ...
               --corrected   corr_0.pkl  corr_1.pkl  ...
"""

import argparse
import pickle
import sys

import numpy as np

NUM_WORKERS = 8


def load(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    parser.add_argument("--batches", nargs="+", required=True)
    parser.add_argument("--uncorrected", nargs="+", required=True)
    parser.add_argument("--corrected", nargs="+", required=True)
    args = parser.parse_args()

    batches = [load(p) for p in sorted(args.batches)]
    uncorrected = [load(p) for p in sorted(args.uncorrected)]
    corrected = [load(p) for p in sorted(args.corrected)]

    assert len(batches) == len(uncorrected) == len(corrected), (
        "Mismatch in number of batch / uncorrected / corrected files"
    )

    result = {
        "num_draws": np.concatenate([b["num_draws"] for b in batches]),
        "datasets": [ds for b in batches for ds in b["datasets"]],
        "idatas_uncorrected": [idata for u in uncorrected for idata in u["idatas"]],
        "idatas_corrected": [idata for c in corrected for idata in c["idatas"]],
        # metadata from first batch (identical across batches)
        "left_bins": batches[0]["left_bins"],
        "right_bins": batches[0]["right_bins"],
        "mutation_rate": batches[0]["mutation_rate"],
        "recombination_rate": batches[0]["recombination_rate"],
        "window_length": batches[0]["window_length"],
        "sample_size": batches[0]["sample_size"],
        "prior": batches[0]["prior"],
    }

    with open(args.output, "wb") as f:
        pickle.dump(result, f)

    n = len(result["num_draws"])
    print(
        f"Collected {len(batches)} batches → {n} datasets → {args.output}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
