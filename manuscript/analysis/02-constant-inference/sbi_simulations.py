import bayesld.models
import tempfile
import subprocess
import pandas as pd
import msprime
import multiprocess
import tqdm
import numpy as np
import pickle

def process(ts, recombination_rate):
    # Use a temporary directory for intermediate files
    with tempfile.TemporaryDirectory() as tmpdir:
        trees_path = f"{tmpdir}/example.trees"
        bcf_path = f"{tmpdir}/example.bcf"
        ts.dump(trees_path)
        subprocess.run(
            f"tskit vcf --allow-position-zero {trees_path} | bcftools view -O b > {bcf_path}",
            shell=True,
            check=True,
        )
        subprocess.run(["bcftools", "index", bcf_path], check=True)
        mean, variance, n, left, right, _ = bayesld.compute_ld(
            bcf_path, "1", recombination_rate
        )
    return mean.astype("float32")

def simulator(Ne, seed):
    demo = msprime.Demography.isolated_model(initial_size=[Ne])
    ts = msprime.sim_ancestry(
        samples=50,
        demography=demo,
        sequence_length=2e7,
        recombination_rate=1e-8,
        random_seed=seed,
        model=[
            msprime.DiscreteTimeWrightFisher(duration=500),
            msprime.StandardCoalescent(),
        ],
    )
    mts = msprime.sim_mutations(ts, rate=1e-8, random_seed=seed)
    return process(mts, recombination_rate=1e-8)

def main(seed, num_samples):
    rng = np.random.default_rng(seed)
    Nes = rng.uniform(0, 20_000, size=num_samples)

    # independent seeds
    seeds = rng.integers(1, 2**31 - 1, size=num_samples)

    def worker(args):
        Ne, s = args
        return simulator(Ne, s)

    # Parallel simulations
    with multiprocess.Pool(8) as pool:
        results = list(tqdm.tqdm(pool.imap(worker, zip(Nes, seeds)), total=num_samples))

    data = {"theta": Nes.reshape(-1, 1), "x": np.asarray(results)}
    pd.to_pickle(data, f"constant_training_{seed}.pkl")

if __name__ == "__main__":
    seed=98796
    num_samples=2_000
    main(seed, num_samples)
