#!/usr/bin/env -S uv run --script
# /// script
# dependencies = [
#     "bayesld==0.1.0",
#     "numpy==2.4.4",
#     "joblib==1.5.3",
#     "tqdm==4.67.3",
# ]
# requires-python = ">=3.12"
#
# [tool.uv.sources]
# bayesld = { git = "https://github.com/currocam/bayesld.git", rev = "363b723" }
# ///

"""Split a VCF into windows of length 20 cM and compute per-window summary statistics.

Usage:
    bayesld_data.py <vcf.gz> <recombination_rate> <num_workers> <out.pkl>
"""

import pickle
import subprocess
import sys

import bayesld
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm


def contigs_and_lengths(vcf_path):
    header = subprocess.run(
        ["bcftools", "view", "-h", vcf_path], check=True, capture_output=True, text=True
    ).stdout
    out = []
    for line in header.splitlines():
        if not line.startswith("##contig"):
            continue
        fields = dict(
            kv.split("=", 1)
            for kv in line[line.index("<") + 1 : line.rindex(">")].split(",")
        )
        out.append((fields["ID"], int(fields["length"])))
    return out


def build_windows(vcf_path, window_length):
    windows = []
    for contig, length in contigs_and_lengths(vcf_path):
        n_windows = int(length // window_length)
        offset = (length - n_windows * window_length) / 2
        for i in range(n_windows):
            start = int(offset + i * window_length)
            end = int(offset + (i + 1) * window_length)
            windows.append({"chromosome": contig, "start": start, "end": end})
    return windows


def collect_data(vcf_path, recombination_rate, num_workers):
    left_bins, right_bins = bayesld.linear_bins()
    window_length = right_bins[-1] * 2 / recombination_rate

    windows = build_windows(vcf_path, window_length)

    def process_window(row):
        return bayesld.data_from_vcf(
            vcf_path=vcf_path,
            recombination_rate=recombination_rate,
            left_bins_morgan=left_bins,
            right_bins_morgan=right_bins,
            contig=row["chromosome"],
            start_bp=row["start"],
            end_bp=row["end"],
            chunk_size=10_000,
            progress_bar=False,
        )

    results = list(
        tqdm(
            Parallel(return_as="generator", n_jobs=num_workers)(
                delayed(process_window)(row) for row in windows
            ),
            total=len(windows),
        )
    )

    return {
        "sample_size": results[0]["sample_size"],
        "left_bins_morgan": results[0]["left_bins_morgan"],
        "right_bins_morgan": results[0]["right_bins_morgan"],
        "mean_linkage_disequilibrium": np.array(
            [r["mean_linkage_disequilibrium"] for r in results]
        ),
        "num_pairs_linkage_disequilibrium": np.array(
            [r["num_pairs_linkage_disequilibrium"] for r in results]
        ),
        "mean_genetic_diversity": np.array(
            [r["mean_genetic_diversity"] for r in results]
        ),
        "num_sites_genetic_diversity": np.array(
            [r["num_sites_genetic_diversity"] for r in results]
        ),
        "windows": np.array(
            [[w["chromosome"], w["start"], w["end"]] for w in windows]
        ),
        "window_length": window_length,
    }


def main():
    if len(sys.argv) != 5:
        print(__doc__, file=sys.stderr)
        sys.exit(1)

    vcf_path = sys.argv[1]
    recombination_rate = float(sys.argv[2])
    num_workers = int(sys.argv[3])
    out_pkl = sys.argv[4]

    data = collect_data(vcf_path, recombination_rate, num_workers)

    with open(out_pkl, "wb") as f:
        pickle.dump(data, f)


if __name__ == "__main__":
    main()
