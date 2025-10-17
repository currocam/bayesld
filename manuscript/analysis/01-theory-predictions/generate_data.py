import msprime
import numpy as np
import tempfile
import subprocess
import os
import bayesld
import bayesld.models
import pymc as pm
import arviz as az
import multiprocess as mp
from tqdm import tqdm
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constants
N_SAMPLES = 20
RECOMBINATION_RATE = MUTATION_RATE = 1e-8
SEED = 1234
CONTIG_LENGTH = 1e8
CPU = 8

# Demographic scenarios
Ne_modern = 10_000
Ne_high = 5_000
msprime_demos = []
for alpha in [0.0, 0.05]:
    demo = msprime.Demography()
    demo.add_population(name="pop_0", initial_size=Ne_modern, growth_rate=alpha)
    demo.add_population_parameters_change(population="pop_0", time=50, growth_rate=0)
    msprime_demos.append(demo)

# High constant population size
demo_high = msprime.Demography()
demo_high.add_population(name="pop_0", initial_size=Ne_high, growth_rate=0.0)
msprime_demos.append(demo_high)


def process(ts):
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
            bcf_path, "1", RECOMBINATION_RATE
        )
    return {
        "mean": mean,
        "variance": variance,
        "n": n,
        "left": left / 100,
        "right": right / 100,
    }


def simulate_and_process_single(args):
    """Process a single tree sequence replicate."""
    demo, replicate_idx = args
    ts = msprime.sim_ancestry(
        samples={"pop_0": N_SAMPLES},
        recombination_rate=RECOMBINATION_RATE,
        sequence_length=CONTIG_LENGTH,
        random_seed=SEED + replicate_idx,
        model=[
            msprime.DiscreteTimeWrightFisher(duration=100),
            msprime.StandardCoalescent(),
        ],
        demography=demo,
    )
    mts = msprime.sim_mutations(
        ts, rate=MUTATION_RATE, random_seed=SEED + replicate_idx
    )
    return process(mts)


def simulate_treeseq(demo, num_replicates):
    """Simulate tree sequences in parallel."""
    logger.info(
        f"Starting simulation with {num_replicates} replicates using {CPU} CPUs"
    )
    args = [(demo, i) for i in range(num_replicates)]
    with mp.Pool(CPU) as pool:
        results = list(
            tqdm(
                pool.imap(simulate_and_process_single, args),
                total=num_replicates,
                desc="Simulating",
            )
        )
    return results


def expected_prediction(datasets, alpha, Ne=None):
    if Ne is None:
        Ne = Ne_modern
    coords = {"bins": np.arange(len(datasets[0]["left"]))}
    empirical_sigma = np.nanstd(np.asarray([x["mean"] for x in datasets]), axis=0)
    with pm.Model(coords=coords) as model:
        if alpha == 0:
            # There are better approximations
            LD = pm.Deterministic(
                "LD",
                bayesld.models.expected_ld_constant(
                    Ne,
                    datasets[0]["left"],
                    datasets[0]["right"],
                    sample_size=N_SAMPLES,
                ),
                dims="bins",
            )
        else:
            LD = pm.Deterministic(
                "LD",
                bayesld.models.expected_ld_exponential(
                    Ne,
                    Ne * np.exp(-alpha * 50),
                    50,
                    alpha,
                    datasets[0]["left"],
                    datasets[0]["right"],
                    sample_size=N_SAMPLES,
                ),
                dims="bins",
            )
        # We don't have theoretical predictions for `sigma`
        pm.Normal("x", LD, sigma=empirical_sigma)
        trace = pm.sample(progressbar=False)
    return az.extract(trace)


def run_scenario(args):
    """Run a complete scenario: simulation and prediction."""
    demo, alpha, Ne, name = args
    logger.info(f"Processing {name} scenario")
    sims = simulate_treeseq(demo, 10)
    idata = expected_prediction(sims, alpha, Ne)
    logger.info(f"Completed {name} scenario")
    return name, sims, idata


if __name__ == "__main__":
    logger.info("Starting data generation pipeline")

    # Define scenarios
    scenarios = [
        (msprime_demos[0], 0, None, "constant"),
        (msprime_demos[1], 0.05, None, "growth"),
        (msprime_demos[2], 0, Ne_high, "high_constant"),
    ]

    # Run scenarios sequentially (parallelization happens within each scenario)
    logger.info(f"Running {len(scenarios)} scenarios")
    results = {}
    for scenario in tqdm(scenarios, desc="Scenarios"):
        name, sims, idata = run_scenario(scenario)
        results[name] = {"sims": sims, "idata": idata}

    # Save results
    logger.info("Saving results to file")
    np.savez("analytical_data.npz", **results)
    logger.info("Data generation complete. Results saved to analytical_data.npz")
