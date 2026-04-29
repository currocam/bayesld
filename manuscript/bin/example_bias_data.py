#!/usr/bin/env -S uv run --script --isolated
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "bayesld @ git+https://github.com/currocam/bayesld.git@a78bf21",
#     "msprime==1.4.0",
#     "numpy==2.2.6",
#     "joblib==1.5.3",
#     "tqdm==4.67.3",
# ]
# ///
import gzip
import pickle
import sys

import msprime
import numpy as np
from bayesld import deterministic, linear_bins, montecarlo

# --- parameters ---
MUTATION_RATE = RECOMBINATION_RATE = 1e-8
NUM_SAMPLES = 50
RANDOM_SEED = 9362178
NUM_WORKERS = 8
RTOL = 0.025

DTWF_DURATION = 200
SMC_MODEL = msprime.SMCK(k=0)  # SMC
SMC_PRIME_MODEL = msprime.SMCK(k=1)  # SMC'
DTWF_MODEL = [
    msprime.DiscreteTimeWrightFisher(duration=DTWF_DURATION),
    msprime.StandardCoalescent(),
]

left_bins, right_bins = linear_bins()
SEQUENCE_LENGTH = right_bins[-1] * 2 / RECOMBINATION_RATE

# --- scenarios ---
SCENARIOS = {
    "small_piecewise": {
        "type": "piecewise_constant",
        "Ne_values": [1_000.0, 100.0, 1_000.0],
        "t_boundaries": [25.0, 75.0],
    },
    "large_piecewise": {
        "type": "piecewise_constant",
        "Ne_values": [10_000.0, 1_000.0, 10_000.0],
        "t_boundaries": [25.0, 75.0],
    },
    "small_exponential": {
        "type": "piecewise_exponential",
        "Ne_c": 5_000.0,
        "Ne_a": 500.0,
        "t0": 100.0,
    },
    "large_exponential": {
        "type": "piecewise_exponential",
        "Ne_c": 50_000.0,
        "Ne_a": 5_000.0,
        "t0": 100.0,
    },
}


def _run_mc(spec, seed, model):
    """Run MC for a scenario with a given ancestry model."""
    if spec["type"] == "piecewise_constant":
        return montecarlo.expected_piecewise_constant(
            Ne_values=spec["Ne_values"],
            t_boundaries=spec["t_boundaries"],
            left_bins=left_bins,
            right_bins=right_bins,
            mutation_rate=MUTATION_RATE,
            recombination_rate=RECOMBINATION_RATE,
            sequence_length=SEQUENCE_LENGTH,
            sample_size=NUM_SAMPLES,
            random_seed=seed,
            rtol=RTOL,
            model=model,
            num_workers=NUM_WORKERS,
        )
    elif spec["type"] == "piecewise_exponential":
        alpha = np.log(spec["Ne_c"] / spec["Ne_a"]) / spec["t0"]
        return montecarlo.expected_piecewise_exponential(
            Ne_c=spec["Ne_c"],
            Ne_a=spec["Ne_a"],
            t0=spec["t0"],
            alpha=alpha,
            left_bins=left_bins,
            right_bins=right_bins,
            mutation_rate=MUTATION_RATE,
            recombination_rate=RECOMBINATION_RATE,
            sequence_length=SEQUENCE_LENGTH,
            sample_size=NUM_SAMPLES,
            random_seed=seed,
            rtol=RTOL,
            model=model,
            num_workers=NUM_WORKERS,
        )
    else:
        raise ValueError(f"Unknown scenario type: {spec['type']}")


def _run_det(spec):
    """Run deterministic prediction for a scenario."""
    if spec["type"] == "piecewise_constant":
        return deterministic.expected_piecewise_constant(
            Ne_values=spec["Ne_values"],
            t_boundaries=spec["t_boundaries"],
            left_bins=left_bins,
            right_bins=right_bins,
            mutation_rate=MUTATION_RATE,
            sample_size=NUM_SAMPLES,
        )
    elif spec["type"] == "piecewise_exponential":
        alpha = np.log(spec["Ne_c"] / spec["Ne_a"]) / spec["t0"]
        return deterministic.expected_piecewise_exponential(
            Ne_c=spec["Ne_c"],
            Ne_a=spec["Ne_a"],
            t0=spec["t0"],
            alpha=alpha,
            left_bins=left_bins,
            right_bins=right_bins,
            mutation_rate=MUTATION_RATE,
            sample_size=NUM_SAMPLES,
        )
    else:
        raise ValueError(f"Unknown scenario type: {spec['type']}")


def run_scenario(name, spec, seed_offset):
    pi_det, ld_det = _run_det(spec)
    pi_smc, ld_smc = _run_mc(spec, RANDOM_SEED + seed_offset, SMC_MODEL)
    pi_smc_prime, ld_smc_prime = _run_mc(
        spec, RANDOM_SEED + seed_offset + 300, SMC_PRIME_MODEL
    )
    pi_dtwf, ld_dtwf = _run_mc(spec, RANDOM_SEED + seed_offset + 600, DTWF_MODEL)

    return {
        "spec": spec,
        "pi_det": float(pi_det),
        "ld_det": np.asarray(ld_det),
        "pi_smc": np.asarray(pi_smc),
        "ld_smc": np.asarray(ld_smc),
        "pi_smc_prime": np.asarray(pi_smc_prime),
        "ld_smc_prime": np.asarray(ld_smc_prime),
        "pi_dtwf": np.asarray(pi_dtwf),
        "ld_dtwf": np.asarray(ld_dtwf),
    }


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "example_bias_data.pkl"

    import tqdm

    results = {}
    for i, (name, spec) in enumerate(tqdm.tqdm(SCENARIOS.items(), desc="Scenarios")):
        results[name] = run_scenario(name, spec, seed_offset=i * 1000)

    data = {
        "results": results,
        "left_bins": left_bins,
        "right_bins": right_bins,
        "params": {
            "mutation_rate": MUTATION_RATE,
            "recombination_rate": RECOMBINATION_RATE,
            "sequence_length": SEQUENCE_LENGTH,
            "num_samples": NUM_SAMPLES,
            "rtol": RTOL,
            "dtwf_duration": DTWF_DURATION,
        },
    }

    with gzip.open(out_path, "wb") as f:
        pickle.dump(data, f)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
