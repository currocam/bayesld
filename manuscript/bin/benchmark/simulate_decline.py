#!/usr/bin/env -S uv run --script
# /// script
# dependencies = [
#     "msprime==1.4.1",
#     "demes==0.2.3",
#     "numpy==2.4.4",
# ]
# requires-python = ">=3.12"
# ///

"""Simulate a recent decline: Ne=5000 for t >= 5 generations, Ne=500 for t < 5.

Usage:
    simulate_decline.py <num_chroms> <chrom_length_bp> <recombination_rate>
                         <mutation_rate> <num_individuals> <seed> <out_prefix>

Writes ``<out_prefix>.vcf.gz`` (+ ``.csi``) and ``<out_prefix>.demes.yaml``.
"""

import subprocess
import sys
from pathlib import Path

import os

import demes
import msprime
import numpy as np
import tskit

NE_RECENT, NE_ANCESTRAL, T_CHANGE = 500, 5000, 5


def demography():
    d = msprime.Demography()
    d.add_population(name="pop", initial_size=NE_RECENT)
    d.add_population_parameters_change(time=T_CHANGE, initial_size=NE_ANCESTRAL)
    return d

def write_vcf_gz(ts: tskit.TreeSequence, outfile: str, individual_names: list[str], contig_id: str):
    read_fd, write_fd = os.pipe()
    write_pipe = os.fdopen(write_fd, "w")
    with open(outfile, "w") as bcf_file:
        proc = subprocess.Popen(
            ["bcftools", "view", "-O", "z"], stdin=read_fd, stdout=bcf_file
        )
        ts.write_vcf(
            write_pipe,
            individual_names=individual_names,
            contig_id=contig_id,
            allow_position_zero=True,
            )
        write_pipe.close()
        os.close(read_fd)
        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError("bcftools failed with status:", proc.returncode)


def main():
    if len(sys.argv) != 8:
        print(__doc__, file=sys.stderr)
        sys.exit(1)

    num_chroms = int(sys.argv[1])
    chrom_length_bp = int(sys.argv[2])
    recombination_rate = float(sys.argv[3])
    mutation_rate = float(sys.argv[4])
    num_individuals = int(sys.argv[5])
    seed = int(sys.argv[6])
    out_prefix = sys.argv[7]

    out_path = Path(out_prefix)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(seed)
    individual_names = [f"ind{j}" for j in range(num_individuals)]

    demog = demography()
    per_chrom_vcf_gz = []
    replicates = msprime.sim_ancestry(
            samples=num_individuals,
            demography=demog,
            sequence_length=chrom_length_bp,
            recombination_rate=recombination_rate,
            ploidy=2,
            model="hudson",
            num_replicates=num_chroms,
            discrete_genome=True,
            random_seed=rng.randint(1, 2**31 - 1),
        )
    for i in range(1, num_chroms + 1):
        contig_id = f"chr{i}"
        ts = next(replicates)
        ts = msprime.sim_mutations(
            ts, rate=mutation_rate, random_seed=rng.randint(1, 2**31 - 1)
        )
        outfile = f"{out_prefix}.{contig_id}.vcf.gz"
        write_vcf_gz(ts, outfile, individual_names, contig_id)
        subprocess.run(["bcftools", "index", outfile], check=True)
        per_chrom_vcf_gz.append(outfile)

    out_vcf_gz = f"{out_prefix}.vcf.gz"
    subprocess.run(
        ["bcftools", "concat", "-a", "-O", "z", "-o", out_vcf_gz, *per_chrom_vcf_gz],
        check=True,
    )
    subprocess.run(["bcftools", "index", out_vcf_gz], check=True)

    demes.dump(demog.to_demes(), f"{out_prefix}.demes.yaml")


if __name__ == "__main__":
    main()
