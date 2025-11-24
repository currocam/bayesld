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
    num_windows = 5
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
        mean, variance, n, left, right, window_index = bayesld.compute_ld(
            bcf_path, "1", recombination_rate, num_windows
        )
    return pd.DataFrame(
        {
            "mean": mean,
            "variance": variance,
            "n": n,
            "left": left / 100,
            "right": right / 100,
            "window_index" : window_index
        }
    )


def simulator(Ne_c, Ne_a, t0, seed):
    demo = msprime.Demography.isolated_model(initial_size=[Ne_c])
    demo.add_population_parameters_change(
        time=t0, initial_size=Ne_a, growth_rate=0, population=0
    )
    seeds = np.arange(50) + seed

    def worker(seed):
        ts = msprime.sim_ancestry(
            samples=50,
            demography=demo,
            sequence_length=1e8,
            recombination_rate=1e-8,
            random_seed=seed,
            model=[
                msprime.DiscreteTimeWrightFisher(duration=500),
                msprime.StandardCoalescent(),
            ],
        )
        mts = msprime.sim_mutations(ts, rate=1e-8, random_seed=seed)
        return mts, process(mts, recombination_rate=1e-8)

    # Process in parallel using worker and annotate with tqdm
    with multiprocess.Pool(6) as pool:
        results = list(
            tqdm.tqdm(pool.imap(worker, seeds), total=len(seeds))
        )
    return results


def run(Ne_c, Ne_a, t0, seed):
    results = simulator(Ne_c, Ne_a, t0, seed)
    outfile = f"steps/simulation_Ne_c{Ne_c}_Ne_a{Ne_a}_t0{t0}_seed{seed}.pkl"
    with open(outfile, "wb") as f:
        pickle.dump(results, f)


if __name__ == "__main__":
    seed = 468214376
    params = [
      [500, 500, 100], # Constant scenario (relatively small)
      [10_000, 10_000, 100], # Constant scenario (relatively large)
      [1_000, 10_000, 1000], # Long-time ago change (relatively large)
      [1_000, 10_000, 200], # Not so long-time ago change (relatively large)
      [1_000, 10_000, 50], # Recent-time ago change (relatively large)
    ]
    for param in params:
        run(Ne_c=param[0], Ne_a=param[1], t0=param[2], seed=seed) 
