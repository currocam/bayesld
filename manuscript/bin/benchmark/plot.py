#!/usr/bin/env -S uv run --script
# /// script
# dependencies = [
#     "arviz==1.1.0",
#     "demes==0.2.3",
#     "matplotlib==3.10.9",
#     "numpy==2.4.4",
#     "pandas==3.0.2",
#     "netcdf4==1.7.4",
#     "h5netcdf==1.8.1",
#     "h5py==3.16.0",
# ]
# requires-python = ">=3.12"
# ///

"""Plot bayesld / GONE2 / HapNe-LD Ne(t) estimates against the true demography.

Usage:
    plot.py <name> <demes.yaml> <bayesld.nc> <gone_ne_file> <hapne_csv> <out_prefix>
"""

import sys
from pathlib import Path

import arviz as az
import demes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MAX_GENERATIONS = 200
TABLE_GENERATIONS = [1, 5, 10, 50]


def pred_piecewise(idata, max_t):
    t_b = az.extract(idata, var_names="t_boundaries").values
    Ne = az.extract(idata, var_names="Ne_values").values

    t = np.arange(0, max_t)
    n_draws = Ne.shape[1]
    matrix = np.empty((n_draws, t.size))
    matrix[:] = Ne[-1][:, None]
    for i in reversed(range(t_b.shape[0])):
        mask = t[None, :] < t_b[i][:, None]
        matrix = np.where(mask, Ne[i][:, None], matrix)
    return matrix


def true_step(graph, max_t):
    deme = graph.demes[0]
    t = np.arange(0, max_t)
    ne = np.empty(max_t)
    for epoch in deme.epochs:
        mask = (t >= epoch.end_time) & (t < epoch.start_time)
        ne[mask] = epoch.start_size
    return t, ne


def true_ne_at(graph, generations):
    deme = graph.demes[0]
    out = []
    for g in generations:
        for epoch in deme.epochs:
            if epoch.end_time <= g < epoch.start_time:
                out.append(epoch.start_size)
                break
    return np.array(out)


def main():
    if len(sys.argv) != 7:
        print(__doc__, file=sys.stderr)
        sys.exit(1)

    name = sys.argv[1]
    demes_yaml = sys.argv[2]
    bayesld_nc = sys.argv[3]
    gone_ne_file = sys.argv[4]
    hapne_csv = sys.argv[5]
    out_prefix = sys.argv[6]

    plt.style.use(Path(__file__).parent.parent / "theme.mplstyle")
    plt.rc("figure", autolayout=True)
    plt.rcParams["pgf.texsystem"] = "pdflatex"

    graph = demes.load(demes_yaml)
    idata = az.from_netcdf(bayesld_nc)
    gone = pd.read_csv(gone_ne_file, sep="\t")
    hapne = pd.read_csv(hapne_csv)

    bayesld_mat = pred_piecewise(idata, MAX_GENERATIONS)
    bayesld_q = pd.DataFrame(
        {
            "TIME": np.arange(0, MAX_GENERATIONS),
            "Q0.025": np.quantile(bayesld_mat, 0.025, axis=0),
            "Q0.5": np.quantile(bayesld_mat, 0.5, axis=0),
            "Q0.975": np.quantile(bayesld_mat, 0.975, axis=0),
        }
    )

    true_t, true_ne = true_step(graph, MAX_GENERATIONS)

    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=300)
    ax.plot(true_t, true_ne, color="black", linestyle="--", lw=2, label="Truth")
    ax.fill_between(
        bayesld_q["TIME"], bayesld_q["Q0.025"], bayesld_q["Q0.975"],
        color="C0", alpha=0.15, linewidth=0,
    )
    ax.plot(bayesld_q["TIME"], bayesld_q["Q0.5"], color="C0", lw=2, label="bayesld")
    ax.plot(gone["Generation"], gone["Ne_diploids"], color="C2", lw=2, label="GONE2")
    ax.plot(hapne["TIME"], hapne["Q0.5"], color="C3", lw=2, label="HapNe-LD")
    ax.fill_between(
        hapne["TIME"], hapne["Q0.025"], hapne["Q0.975"],
        color="C3", alpha=0.15, linewidth=0,
    )

    ax.set_xlim(1, MAX_GENERATIONS)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Time ago (generations)")
    ax.set_ylabel(r"Effective population size $N_e$")
    ax.set_title(name)
    ax.legend()

    fig.savefig(f"{out_prefix}.pdf")
    fig.savefig(f"{out_prefix}.pgf")

    true_at = true_ne_at(graph, TABLE_GENERATIONS)
    gone_at = np.interp(TABLE_GENERATIONS, gone["Generation"], gone["Ne_diploids"])
    hapne_at = np.interp(TABLE_GENERATIONS, hapne["TIME"], hapne["Q0.5"])
    bayesld_at = np.interp(TABLE_GENERATIONS, bayesld_q["TIME"], bayesld_q["Q0.5"])

    table = pd.DataFrame(
        {
            "Generations ago": TABLE_GENERATIONS,
            "Truth": true_at,
            "bayesld": bayesld_at,
            "GONE2": gone_at,
            "HapNe-LD": hapne_at,
        }
    )
    table.to_latex(f"{out_prefix}.tex", index=False, float_format="%.1f")


if __name__ == "__main__":
    main()
