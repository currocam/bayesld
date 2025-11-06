import bayesld.models
import tempfile
import subprocess
import pandas as pd
import msprime
import multiprocess
import tqdm
import numpy as np
import pickle

Ne_c = 5_000  # Current effective population size
Ne_a = 6_000  # Ancestral effective population size
t_inv = 50  # Time of invasions in generations


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
        mean, variance, n, left, right = bayesld.compute_ld(
            bcf_path, "1", recombination_rate
        )
    return pd.DataFrame(
        {
            "mean": mean,
            "variance": variance,
            "n": n,
            "left": left / 100,
            "right": right / 100,
        }
    )


def simulator(Ne_f, seed):
    alpha = (np.log(Ne_c) - np.log(Ne_f)) / t_inv
    demo = msprime.Demography.isolated_model(initial_size=[Ne_c], growth_rate=[alpha])
    demo.add_population_parameters_change(
        time=t_inv, initial_size=Ne_a, growth_rate=0, population=0
    )
    seeds = np.arange(50) + seed

    def worker(seed):
        ts = msprime.sim_ancestry(
            samples=50,
            demography=demo,
            sequence_length=3e8,
            recombination_rate=1e-8,
            random_seed=seed,
            model=[
                msprime.DiscreteTimeWrightFisher(duration=500),
                msprime.StandardCoalescent(),
            ],
        )
        mts = msprime.sim_mutations(ts, rate=1e-8, random_seed=seed)
        return process(mts, recombination_rate=1e-8)

    # Process in parallel using worker and annotate with tqdm
    with multiprocess.Pool(4) as pool:
        results = list(
            tqdm.tqdm(pool.imap(worker, seeds), total=len(seeds))
        )
    return results


def main(Ne_f, seed):
    results = simulator(Ne_f, seed)
    outfile = f"simulation_Nef{Ne_f}_seed{seed}.pkl"
    with open(outfile, "wb") as f:
        pickle.dump(results, f)


if __name__ == "__main__":
    seed = 131562
    for Ne_f in [10, 50, 100, 500]:
        main(Ne_f=Ne_f, seed=seed) 
