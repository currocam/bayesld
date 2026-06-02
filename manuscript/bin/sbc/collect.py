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
Corrected/no-bias pkls supply idatas lists.
All lists must be provided in matching batch order (sort by filename).

Usage:
    collect.py <output.pkl>
               --batches     batch_0.pkl batch_1.pkl ...
               --corrected   corr_0.pkl  corr_1.pkl  ...
               --no-bias     nb_0.pkl    nb_1.pkl    ...
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
    parser.add_argument("--corrected", nargs="+", required=True)
    parser.add_argument("--no-bias", nargs="+", required=True)
    args = parser.parse_args()

    batches = [load(p) for p in sorted(args.batches)]
    corrected = [load(p) for p in sorted(args.corrected)]
    no_bias = [load(p) for p in sorted(args.no_bias)]

    assert len(batches) == len(corrected) == len(no_bias), (
        "Mismatch in number of batch / corrected / no-bias files"
    )

    result = {
        "datasets": [ds for b in batches for ds in b["datasets"]],
        "idatas_corrected": [idata for c in corrected for idata in c["idatas"]],
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

    n = len(result["datasets"])
    print(
        f"Collected {len(batches)} batches → {n} datasets → {args.output}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
