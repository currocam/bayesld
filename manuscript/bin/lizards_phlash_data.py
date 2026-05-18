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

import pandas as pd
import phlash.data

vcf_path = sys.argv[1]
sequence_report = sys.argv[2]
samples_csv = sys.argv[3]
pkl_path = sys.argv[4]

samples = samples_csv.split(",")

# Filter autosomes from sequence report (exclude sex chromosomes)
contigs = pd.read_csv(sequence_report, sep="\t")
contigs = contigs[
    contigs["Sequence name"].str.startswith("SUPER")
    & ~contigs["Sequence name"].str.startswith("SUPER_W")
    & ~contigs["Sequence name"].str.startswith("SUPER_Z")
]
print(f"Found {len(contigs)} contigs to process")

# Create VcfContig objects with actual sequence lengths
chroms = []
for _, row in contigs.iterrows():
    contig_name = str(row["GenBank seq accession"])
    length = int(row["Seq length"])
    contig_obj = phlash.data.VcfContig(
        vcf_path,
        samples=samples,
        contig=contig_name,
        interval=(1, length),
    )
    chroms.append(contig_obj)
    print(f"  {contig_name} ({length:,} bp)")

with open(pkl_path, "wb") as f:
    pickle.dump(chroms, f)
