#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "bayesld @ git+https://github.com/currocam/bayesld.git@ae4d14b",
#     "msprime==1.4.0",
#     "numpy==2.2.6",
#     "joblib==1.5.3",
#     "tqdm==4.67.3",
# ]
# ///

import gzip
import pickle
import sys
import time

import msprime
import numpy as np
import tqdm
from bayesld import deterministic, linear_bins, montecarlo

# --- shared parameters ---
MUTATION_RATE = RECOMBINATION_RATE = 1e-8
NUM_SAMPLES = 50
RANDOM_SEED = 9362178
NUM_WORKERS = 8
RTOL = 0.05
DTWF_DURATION = 200

DTWF_MODEL = [
    msprime.DiscreteTimeWrightFisher(duration=DTWF_DURATION),
    msprime.StandardCoalescent(),
]

LEFT_BINS, RIGHT_BINS = linear_bins()
SEQUENCE_LENGTH = RIGHT_BINS[-1] * 2 / RECOMBINATION_RATE


# --- variant factories: each returns (label, det_fn, mc_fn, kwargs, epochs) ---
def constant(label, Ne):
    return dict(
        label=label,
        det_fn=deterministic.expected_piecewise_constant,
        mc_fn=montecarlo.expected_piecewise_constant,
        kwargs=dict(Ne_values=[Ne], t_boundaries=[]),
        epochs=[{"end_time": 0, "start_size": Ne}],
    )


def piecewise_exp(label, *, Ne_c, Ne_a, t0, Nf=None):
    """Continuous at t0 when Nf is None; otherwise discontinuous jump Ne_a -> Nf."""
    if Nf is None:
        alpha = np.log(Ne_c / Ne_a) / t0
        exp_start = Ne_a
    else:
        alpha = (np.log(Ne_c) - np.log(Nf)) / t0
        exp_start = Nf
    return dict(
        label=label,
        det_fn=deterministic.expected_piecewise_exponential,
        mc_fn=montecarlo.expected_piecewise_exponential,
        kwargs=dict(Ne_c=Ne_c, Ne_a=Ne_a, t0=t0, alpha=alpha),
        epochs=[
            {"end_time": t0, "start_size": Ne_a},
            {"end_time": 0, "start_size": exp_start, "end_size": Ne_c},
        ],
    )


def carrying_capacity(label, *, Ne_c, Ne_a, Nf, t0, t1):
    alpha = (np.log(Ne_c) - np.log(Nf)) / (t1 - t0)
    return dict(
        label=label,
        det_fn=deterministic.expected_exponential_carrying_capacity,
        mc_fn=montecarlo.expected_exponential_carrying_capacity,
        kwargs=dict(Ne_c=Ne_c, Ne_a=Ne_a, t0=t0, t1=t1, alpha=alpha),
        epochs=[
            {"end_time": t1, "start_size": Ne_a},
            {"end_time": t0, "start_size": Nf, "end_size": Ne_c},
            {"end_time": 0, "start_size": Ne_c},
        ],
    )


SCENARIOS = {
    "constant": dict(
        smc_k=1,
        variants=[
            constant("Nc=5e2", 500),
            constant("Nc=1e3", 1_000),
            constant("Nc=2e3", 2_000),
            constant("Nc=1e4", 10_000),
        ],
    ),
    "decline": dict(
        smc_k=0,
        variants=[
            piecewise_exp("Nc=1e3, t0=25", Ne_c=1_000, Ne_a=10_000, t0=25),
            piecewise_exp("Nc=5e3, t0=25", Ne_c=5_000, Ne_a=10_000, t0=25),
            piecewise_exp("Nc=1e3, t0=50", Ne_c=1_000, Ne_a=10_000, t0=50),
            piecewise_exp("Nc=5e3, t0=50", Ne_c=5_000, Ne_a=10_000, t0=50),
        ],
    ),
    "invasion": dict(
        smc_k=1,
        variants=[
            piecewise_exp("Nf=10, t0=25", Ne_c=5_000, Ne_a=10_000, t0=25, Nf=10),
            piecewise_exp("Nf=100, t0=25", Ne_c=5_000, Ne_a=10_000, t0=25, Nf=100),
            piecewise_exp("Nf=10, t0=50", Ne_c=5_000, Ne_a=10_000, t0=50, Nf=10),
            piecewise_exp("Nf=100, t0=50", Ne_c=5_000, Ne_a=10_000, t0=50, Nf=100),
        ],
    ),
    "carrying_capacity": dict(
        smc_k=0,
        variants=[
            carrying_capacity(
                "Nf=100, t0=25", Ne_c=5_000, Ne_a=10_000, Nf=100, t0=25, t1=75
            ),
            carrying_capacity(
                "Nf=10, t0=25", Ne_c=5_000, Ne_a=10_000, Nf=10, t0=25, t1=75
            ),
            carrying_capacity(
                "Nf=10, t0=50", Ne_c=5_000, Ne_a=10_000, Nf=10, t0=50, t1=75
            ),
            carrying_capacity(
                "Nf=100, t0=50", Ne_c=5_000, Ne_a=10_000, Nf=100, t0=50, t1=75
            ),
        ],
    ),
    "growth": dict(
        smc_k=0,
        variants=[
            piecewise_exp("Nc=1e4, t0=25", Ne_c=10_000, Ne_a=1_000, t0=25),
            piecewise_exp("Nc=5e3, t0=25", Ne_c=5_000, Ne_a=1_000, t0=25),
            piecewise_exp("Nc=1e4, t0=50", Ne_c=10_000, Ne_a=1_000, t0=50),
            piecewise_exp("Nc=5e3, t0=50", Ne_c=5_000, Ne_a=1_000, t0=50),
        ],
    ),
}


DET_KWARGS = dict(
    left_bins=LEFT_BINS,
    right_bins=RIGHT_BINS,
    mutation_rate=MUTATION_RATE,
    sample_size=NUM_SAMPLES,
)
MC_KWARGS = dict(
    left_bins=LEFT_BINS,
    right_bins=RIGHT_BINS,
    mutation_rate=MUTATION_RATE,
    recombination_rate=RECOMBINATION_RATE,
    sequence_length=SEQUENCE_LENGTH,
    sample_size=NUM_SAMPLES,
    rtol=RTOL,
    num_workers=NUM_WORKERS,
)


def run_variant(variant, smc_k, seed, scenario_name=""):
    kw = variant["kwargs"]
    label = variant["label"]
    tag = f"[{scenario_name} | {label}]"

    t0 = time.perf_counter()
    pi_det, ld_det = variant["det_fn"](**kw, **DET_KWARGS)
    tqdm.tqdm.write(f"{tag} det          {time.perf_counter() - t0:7.2f}s")

    t0 = time.perf_counter()
    pi_smc, ld_smc = variant["mc_fn"](
        **kw, **MC_KWARGS, random_seed=seed, model=msprime.SMCK(k=smc_k)
    )
    tqdm.tqdm.write(f"{tag} SMC(k={smc_k})     {time.perf_counter() - t0:7.2f}s")

    t0 = time.perf_counter()
    pi_dtwf, ld_dtwf = variant["mc_fn"](
        **kw, **MC_KWARGS, random_seed=seed + 300, model=DTWF_MODEL
    )
    tqdm.tqdm.write(f"{tag} DTWF         {time.perf_counter() - t0:7.2f}s")
    return {
        "label": variant["label"],
        "epochs": variant["epochs"],
        "pi_det": float(pi_det),
        "ld_det": np.asarray(ld_det),
        "pi_smc": np.asarray(pi_smc),
        "ld_smc": np.asarray(ld_smc),
        "pi_dtwf": np.asarray(pi_dtwf),
        "ld_dtwf": np.asarray(ld_dtwf),
    }


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "example_bias_data.pkl"

    results = {}
    seed = 0
    for name, scenario in tqdm.tqdm(SCENARIOS.items(), desc="Scenarios"):
        variants = []
        for v in tqdm.tqdm(scenario["variants"], desc=name, leave=False):
            variants.append(
                run_variant(v, scenario["smc_k"], RANDOM_SEED + seed * 1000, name)
            )
            seed += 1
        results[name] = {"smc_k": scenario["smc_k"], "variants": variants}

    data = {
        "results": results,
        "left_bins": LEFT_BINS,
        "right_bins": RIGHT_BINS,
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
