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
#     "jinja2",
# ]
# requires-python = ">=3.12"
# ///

"""Plot bayesld / GONE2 / HapNe-LD Ne(t) estimates against the true demography.

One or more bayesld model fits can be overlaid, each as ``<model>=<path.nc>``.

Usage:
    plot.py <name> <demes.yaml> <gone_ne_file> <hapne_csv> <out_prefix>
            <model1>=<nc1> [<model2>=<nc2> ...]
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
# C2/C3 are reserved for GONE2/HapNe-LD below, so bayesld models cycle through
# the rest.
BAYESLD_COLORS = ["C0", "C1", "C4", "C5", "C6", "C7", "C8", "C9"]


def pred_piecewise(idata, max_t, group="posterior"):
    t_b = az.extract(idata, group=group, var_names="t_boundaries").values
    Ne = az.extract(idata, group=group, var_names="Ne_values").values

    t = np.arange(0, max_t)
    n_draws = Ne.shape[1]
    matrix = np.empty((n_draws, t.size))
    matrix[:] = Ne[-1][:, None]
    for i in reversed(range(t_b.shape[0])):
        mask = t[None, :] < t_b[i][:, None]
        matrix = np.where(mask, Ne[i][:, None], matrix)
    return matrix


def ne_quantiles(idata, group="posterior"):
    mat = pred_piecewise(idata, MAX_GENERATIONS, group=group)
    return pd.DataFrame(
        {
            "TIME": np.arange(0, MAX_GENERATIONS),
            "Q0.025": np.quantile(mat, 0.025, axis=0),
            "Q0.5": np.quantile(mat, 0.5, axis=0),
            "Q0.975": np.quantile(mat, 0.975, axis=0),
        }
    )


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
    if len(sys.argv) < 7:
        print(__doc__, file=sys.stderr)
        sys.exit(1)

    name = sys.argv[1]
    demes_yaml = sys.argv[2]
    gone_ne_file = sys.argv[3]
    hapne_csv = sys.argv[4]
    out_prefix = sys.argv[5]
    bayesld_specs = [arg.split("=", 1) for arg in sys.argv[6:]]

    plt.style.use(Path(__file__).parent.parent / "theme.mplstyle")
    plt.rc("figure", autolayout=True)
    plt.rcParams["pgf.texsystem"] = "pdflatex"

    graph = demes.load(demes_yaml)
    gone = pd.read_csv(gone_ne_file, sep="\t")
    hapne = pd.read_csv(hapne_csv)

    bayesld_idata = {model: az.from_netcdf(nc) for model, nc in bayesld_specs}
    bayesld_q = {model: ne_quantiles(idata) for model, idata in bayesld_idata.items()}
    bayesld_prior_q = {
        model: ne_quantiles(idata, group="prior") for model, idata in bayesld_idata.items()
    }

    true_t, true_ne = true_step(graph, MAX_GENERATIONS)

    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=300)
    ax.plot(true_t, true_ne, color="black", linestyle="--", lw=2, label="Truth")

    for (model, q), color in zip(bayesld_q.items(), BAYESLD_COLORS):
        ax.fill_between(
            q["TIME"], q["Q0.025"], q["Q0.975"], color=color, alpha=0.15, linewidth=0,
        )
        ax.plot(q["TIME"], q["Q0.5"], color=color, lw=2, label=f"bayesld ({model})")

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

    # Prior vs. truth: GONE2/HapNe-LD have no tweakable prior to sanity-check
    # against, so this is bayesld-only — it shows whether the prior is broad
    # enough not to have already baked in the answer.
    fig_prior, ax_prior = plt.subplots(figsize=(6, 4.5), dpi=300)
    ax_prior.plot(true_t, true_ne, color="black", linestyle="--", lw=2, label="Truth")

    for (model, q), color in zip(bayesld_prior_q.items(), BAYESLD_COLORS):
        ax_prior.fill_between(
            q["TIME"], q["Q0.025"], q["Q0.975"], color=color, alpha=0.15, linewidth=0,
        )
        ax_prior.plot(q["TIME"], q["Q0.5"], color=color, lw=2, label=f"bayesld ({model})")

    ax_prior.set_xlim(1, MAX_GENERATIONS)
    ax_prior.set_xscale("log")
    ax_prior.set_yscale("log")
    ax_prior.set_xlabel("Time ago (generations)")
    ax_prior.set_ylabel(r"Effective population size $N_e$")
    ax_prior.set_title(f"{name} (prior)")
    ax_prior.legend()

    fig_prior.savefig(f"{out_prefix}_prior.pdf")
    fig_prior.savefig(f"{out_prefix}_prior.pgf")

    true_at = true_ne_at(graph, TABLE_GENERATIONS)
    gone_at = np.interp(TABLE_GENERATIONS, gone["Generation"], gone["Ne_diploids"])
    hapne_at = np.interp(TABLE_GENERATIONS, hapne["TIME"], hapne["Q0.5"])

    table = pd.DataFrame(
        {
            "Generations ago": TABLE_GENERATIONS,
            "Truth": true_at,
            **{
                f"bayesld ({model})": np.interp(TABLE_GENERATIONS, q["TIME"], q["Q0.5"])
                for model, q in bayesld_q.items()
            },
            "GONE2": gone_at,
            "HapNe-LD": hapne_at,
        }
    )
    table.to_latex(f"{out_prefix}.tex", index=False, float_format="%.1f")


if __name__ == "__main__":
    main()
