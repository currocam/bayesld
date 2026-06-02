#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "arviz>=0.23.4",
# ]
# ///
"""
SBC collect — merges all batches for one (model, prior) into a single pkl.

Batch pkls supply draw arrays, datasets, metadata, and a model-specific "prior" dict.
Learn pkls are per-dataset and supply `idata_pathfinder`.
Corrected pkls are per-dataset and supply the MCMC `idata`.
No-bias pkls are per-batch and supply `idatas` lists.

Per-dataset filenames embed batch_idx and ds_idx zero-padded, so filename sort
matches the dataset order produced by flattening batches in batch_idx order.

Usage:
    collect.py <output.pkl>
               --batches   batch_0.pkl   batch_1.pkl   ...
               --learn     learn_*.pkl
               --corrected corr_*.pkl
               --no-bias   nb_0.pkl      nb_1.pkl      ...
"""

import argparse
import pickle
import sys

import numpy as np


def load(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    parser.add_argument("--batches", nargs="+", required=True)
    parser.add_argument("--learn", nargs="+", required=True)
    parser.add_argument("--corrected", nargs="+", required=True)
    parser.add_argument("--no-bias", nargs="+", required=True)
    args = parser.parse_args()

    batches = [load(p) for p in sorted(args.batches)]
    learn = [load(p) for p in sorted(args.learn)]
    corrected = [load(p) for p in sorted(args.corrected)]
    no_bias = [load(p) for p in sorted(args.no_bias)]

    n_datasets = sum(len(b["datasets"]) for b in batches)
    assert len(corrected) == n_datasets, (
        f"Expected {n_datasets} corrected pkls (one per dataset), got {len(corrected)}"
    )
    assert len(learn) == n_datasets, (
        f"Expected {n_datasets} learn pkls (one per dataset), got {len(learn)}"
    )
    assert len(no_bias) == len(batches), (
        f"Expected {len(batches)} no-bias pkls (one per batch), got {len(no_bias)}"
    )

    result = {
        "datasets": [ds for b in batches for ds in b["datasets"]],
        "idatas_corrected": [c["idata"] for c in corrected],
        "idatas_pathfinder": [l["idata_pathfinder"] for l in learn],
        "idatas_no_bias": [idata for n in no_bias for idata in n["idatas"]],
    }

    # Dynamically collect remaining keys from the first batch
    first_batch = batches[0]
    for key in first_batch:
        if key in result:
            continue  # already handled above

        first_val = first_batch[key]
        if isinstance(first_val, list):
            # flatten across batches
            result[key] = [item for b in batches for item in b[key]]
        elif isinstance(first_val, np.ndarray):
            # concatenate across batches (e.g. ne_draws, ne1_draws, t0_draws)
            result[key] = np.concatenate([b[key] for b in batches])
        else:
            # metadata — assert identical across batches
            for b in batches[1:]:
                if b[key] != first_val:
                    raise ValueError(
                        f"Mismatch in batch metadata key '{key}' across batches"
                    )
            result[key] = first_val

    with open(args.output, "wb") as f:
        pickle.dump(result, f)

    print(
        f"Collected {len(batches)} batches → {n_datasets} datasets → {args.output}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
