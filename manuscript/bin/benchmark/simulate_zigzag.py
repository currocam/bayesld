#!/usr/bin/env -S uv run --script
# /// script
# dependencies = [
#     "msprime==1.4.1",
#     "demes==0.2.3",
#     "numpy==2.4.4",
# ]
# requires-python = ">=3.12"
# ///

"""Simulate a zigzag demography. 
Usage:
    simulate_zigzag.py <num_chroms> <chrom_length_bp> <recombination_rate>
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

# Adapted from `stdpopsim` Zigzag_1S14 model.
N_SMALL = 500.0
N_LARGE = 5000.0
PLATEAU_GENERATIONS = 600.0

RATE_CHANGES = {
    0.00232905: -329.546,
    0.00931619: 82.3865,
    0.0372648: -20.5966,
    0.149059: 5.14916,
}
ANCIENT_KEY = 0.596236


def demography():
    keys = sorted(RATE_CHANGES)
    ne_time = PLATEAU_GENERATIONS / (4 * ANCIENT_KEY)

    signed_rates = [0.0] + [-RATE_CHANGES[k] for k in keys]
    boundary_keys = [0.0] + keys

    log_ratio = [0.0]
    for i in range(1, len(boundary_keys)):
        d_key = boundary_keys[i] - boundary_keys[i - 1]
        log_ratio.append(log_ratio[-1] - signed_rates[i - 1] * d_key)

    raw_min, raw_max = min(log_ratio), max(log_ratio)
    b = np.log(N_LARGE / N_SMALL) / (raw_max - raw_min)
    base = N_SMALL * np.exp(-b * raw_min)

    d = msprime.Demography()
    d.add_population(name="pop", initial_size=base, growth_rate=0)
    for i, key in enumerate(keys, start=1):
        d.add_population_parameters_change(
            time=key * 4 * ne_time,
            growth_rate=signed_rates[i] * b / (4 * ne_time),
            population="pop",
        )
    d.add_population_parameters_change(
        time=ANCIENT_KEY * 4 * ne_time,
        growth_rate=0,
        initial_size=N_LARGE,
        population="pop",
    )
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
