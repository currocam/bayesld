#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "arviz>=0.23.4",
# ]
# ///
"""
SBC collect — merges all batches for one (model, prior) into a single pkl.

Usage:
    collect.py <output.pkl>
               --batches   batch_0.pkl   batch_1.pkl   ...
               --learn     learn_*.pkl
               --corrected corr_*.pkl
               --no-bias   nb_0.pkl      nb_1.pkl      ...
"""

import argparse
import os
import pickle
import re
import sys

import numpy as np

_BATCH_RE = re.compile(r"_(\d+)\.pkl$")
_DS_RE = re.compile(r"_(\d+)_(\d+)\.pkl$")


# This should be much easier, but I'm learning nextflow ...
def load(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _parse_batch(path):
    m = _BATCH_RE.search(os.path.basename(str(path)))
    if not m:
        raise ValueError(f"Cannot parse trailing _<batch>.pkl from {path}")
    return int(m.group(1))


def _parse_batch_ds(path):
    m = _DS_RE.search(os.path.basename(str(path)))
    if not m:
        raise ValueError(f"Cannot parse trailing _<batch>_<ds>.pkl from {path}")
    return int(m.group(1)), int(m.group(2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    parser.add_argument("--batches", nargs="+", required=True)
    parser.add_argument("--learn", nargs="+", required=True)
    parser.add_argument("--corrected", nargs="+", required=True)
    parser.add_argument("--no-bias", nargs="+", required=True)
    args = parser.parse_args()

    batch_paths = sorted(args.batches, key=_parse_batch)
    nobias_paths = sorted(args.no_bias, key=_parse_batch)
    learn_paths = sorted(args.learn, key=_parse_batch_ds)
    corrected_paths = sorted(args.corrected, key=_parse_batch_ds)

    batches = [load(p) for p in batch_paths]
    no_bias = [load(p) for p in nobias_paths]
    learn = [load(p) for p in learn_paths]
    corrected = [load(p) for p in corrected_paths]

    # Verify per-batch files form a contiguous 0..N-1 sequence.
    n_batches = len(batches)
    batch_indices = [_parse_batch(p) for p in batch_paths]
    nobias_indices = [_parse_batch(p) for p in nobias_paths]
    expected_batches = list(range(n_batches))
    assert batch_indices == expected_batches, (
        f"Batch filenames are not contiguous 0..{n_batches - 1}: {batch_indices}"
    )
    assert nobias_indices == expected_batches, (
        f"No-bias filenames don't match batch indices: {nobias_indices} vs {expected_batches}"
    )

    batch_size = len(batches[0]["datasets"])
    for b in batches:
        assert len(b["datasets"]) == batch_size, (
            "Batch sizes are not uniform across batches — collect assumes they are"
        )
    n_datasets = n_batches * batch_size

    expected_pairs = [(b, d) for b in range(n_batches) for d in range(batch_size)]
    learn_pairs = [_parse_batch_ds(p) for p in learn_paths]
    corrected_pairs = [_parse_batch_ds(p) for p in corrected_paths]
    assert learn_pairs == expected_pairs, (
        f"learn filenames don't form a complete (batch, ds) grid; "
        f"got {len(learn_pairs)} files, expected {len(expected_pairs)}"
    )
    assert corrected_pairs == expected_pairs, (
        f"corrected filenames don't form a complete (batch, ds) grid; "
        f"got {len(corrected_pairs)} files, expected {len(expected_pairs)}"
    )

    # Cross-check filename ds_idx against the dataset_idx embedded by
    # learn_corrected.py / sample_corrected.py.
    for (_, ds_idx), bundle, path in zip(learn_pairs, learn, learn_paths):
        if "dataset_idx" in bundle and bundle["dataset_idx"] != ds_idx:
            raise AssertionError(
                f"learn pkl {path}: filename ds_idx={ds_idx} but "
                f"bundle['dataset_idx']={bundle['dataset_idx']}"
            )
    for (_, ds_idx), bundle, path in zip(corrected_pairs, corrected, corrected_paths):
        if "dataset_idx" in bundle and bundle["dataset_idx"] != ds_idx:
            raise AssertionError(
                f"corrected pkl {path}: filename ds_idx={ds_idx} but "
                f"bundle['dataset_idx']={bundle['dataset_idx']}"
            )

    # Per-batch no_bias lists are flattened in batch order — verify count.
    for b_idx, nb in enumerate(no_bias):
        if len(nb["idatas"]) != batch_size:
            raise AssertionError(
                f"no_bias batch {b_idx} has {len(nb['idatas'])} idatas, "
                f"expected {batch_size}"
            )

    result = {
        "datasets": [ds for b in batches for ds in b["datasets"]],
        "idatas_corrected": [c["idata"] for c in corrected],
        "idatas_pathfinder": [l["idata_pathfinder"] for l in learn],
        "idatas_no_bias": [idata for n in no_bias for idata in n["idatas"]],
    }
    assert len(result["datasets"]) == n_datasets
    assert len(result["idatas_corrected"]) == n_datasets
    assert len(result["idatas_pathfinder"]) == n_datasets
    assert len(result["idatas_no_bias"]) == n_datasets

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
        f"Collected {n_batches} batches → {n_datasets} datasets → {args.output}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()

