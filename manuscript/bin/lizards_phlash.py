#!/usr/bin/env -S uv run --script
# /// script
# requires-python = "==3.12.*"
# dependencies = [
#     "ipython",
#     "numpy",
#     "pandas",
#     "phlash[gpu]",
# ]
#
# [tool.uv.sources]
# phlash = { git = "https://github.com/jthlab/phlash" , rev = "96a6e3f8e01053271e88d21c97862198422d9ae0"}
# ///
import pickle
import sys

import numpy as np
import pandas as pd
import phlash
import phlash.data


def main():
    vcf_path = sys.argv[1]
    sequence_report = sys.argv[2]
    samples_csv = sys.argv[3]
    pkl_path = sys.argv[4]
    npz_path = sys.argv[5]

    samples = samples_csv.split(",")

    contigs = pd.read_csv(sequence_report, sep="\t")
    contigs = contigs[
        contigs["Sequence name"].str.startswith("SUPER")
        & ~contigs["Sequence name"].str.startswith("SUPER_W")
        & ~contigs["Sequence name"].str.startswith("SUPER_Z")
    ]
    print(f"Found {len(contigs)} contigs to process")

    chroms = []
    for _, row in contigs.iterrows():
        contig_name = str(row["GenBank seq accession"])
        length = int(row["Seq length"])
        chroms.append(
            phlash.data.VcfContig(
                vcf_path,
                samples=samples,
                contig=contig_name,
                interval=(1, length),
            )
        )
        print(f"  {contig_name} ({length:,} bp)")

    # From https://academic.oup.com/mbe/article/39/1/msab311/6413643
    generation_time = 2
    mutation_rate_per_year = 1.98e-9
    mutation_rate = mutation_rate_per_year * generation_time

    test_data = chroms[0]
    train_data = chroms[1:]
    results = phlash.fit(
        data=train_data, test_data=test_data, mutation_rate=mutation_rate
    )

    with open(pkl_path, "wb") as f:
        pickle.dump(results, f)

    times = np.array([dm.eta.t[1:] for dm in results])
    T = np.geomspace(times.min(), times.max(), 1000)
    Nes = np.array([dm.eta(T, Ne=True) for dm in results])
    np.savez(npz_path, T=T, Nes=Nes)


if __name__ == "__main__":
    main()
