#!/usr/bin/env -S uv run --script
# /// script
# dependencies = [
#     "arviz==1.1.0",
#     "demes==0.2.3",
#     "matplotlib==3.10.9",
#     "msprime==1.4.1",
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

Usage:
    plot.py <name> <demes.yaml> <gone_ne_file> <hapne_csv> <out_prefix>
            <model1>=<nc1> [<model2>=<nc2> ...]

Writes ``<out_prefix>.{pdf,pgf}`` (posterior, with the competing methods) and
``<out_prefix>_prior.{pdf,pgf}``.
"""

import sys
from pathlib import Path

import arviz as az
import demes
import matplotlib.pyplot as plt
import msprime
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

DEFAULT_GENERATIONS = 200
# The zigzag keeps changing well beyond the horizon the other scenarios need.
SCENARIO_GENERATIONS = {"zigzag": 600}
CI_PROB = 0.95
# The piecewise trajectories are evaluated on this grid; the change points do not
# land on integer generations, so a one-generation step visibly rounds them.
TIME_STEP = 0.05
N_DRAWS = 30
DRAW_ALPHA = 0.4
SEED = 1234
# I fitted more models, but I think those are the most interesting.
SCENARIO_MODELS = {
    "growth": "two_epoch",
    "decline": "two_epoch",
    "growth_75": "two_epoch",
    "decline_75": "two_epoch",
    "zigzag": "random_walk",
}


def pred_piecewise(idata, max_t, group="posterior"):
    t_b = az.extract(idata, group=group, var_names="t_boundaries").values
    Ne = az.extract(idata, group=group, var_names="Ne_values").values

    t = np.arange(0, max_t, TIME_STEP)
    n_draws = Ne.shape[1]
    matrix = np.empty((n_draws, t.size))
    matrix[:] = Ne[-1][:, None]
    for i in reversed(range(t_b.shape[0])):
        mask = t[None, :] < t_b[i][:, None]
        matrix = np.where(mask, Ne[i][:, None], matrix)
    return matrix


def ne_ci(mat):
    tail = (1 - CI_PROB) / 2
    return np.quantile(mat, [tail, 1 - tail], axis=0)


def true_trajectory(graph, times):
    dbg = msprime.Demography.from_demes(graph).debug()
    return dbg.population_size_trajectory(np.asarray(times, dtype=float))[:, 0]


def draw_ensemble(ax, t, mat, rng, color="C0"):
    low, high = ne_ci(mat)
    ax.fill_between(t, low, high, color=color, alpha=0.2, linewidth=0)
    draws = mat[rng.choice(mat.shape[0], N_DRAWS, replace=False)]
    ax.plot(t, draws.T, color=color, lw=0.8, alpha=DRAW_ALPHA)


def dress_axis(ax, max_generations):
    ax.set_xlim(1, max_generations)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Time ago (generations)")
    ax.set_ylabel(r"Effective population size $N_e$")


def main():
    if len(sys.argv) < 7:
        print(__doc__, file=sys.stderr)
        sys.exit(1)

    name = sys.argv[1]
    demes_yaml = sys.argv[2]
    gone_ne_file = sys.argv[3]
    hapne_csv = sys.argv[4]
    out_prefix = sys.argv[5]
    specs = [arg.split("=", 1) for arg in sys.argv[6:]]
    wanted = SCENARIO_MODELS.get(name)
    if wanted is not None:
        specs = [spec for spec in specs if spec[0] == wanted]
    assert len(specs) == 1, f"expected exactly one model for {name!r}, got {specs}"
    (model, model_nc), = specs

    plt.style.use(Path(__file__).parent.parent / "theme.mplstyle")
    plt.rcParams["pgf.texsystem"] = "pdflatex"

    graph = demes.load(demes_yaml)
    gone = pd.read_csv(gone_ne_file, sep="\t")
    hapne = pd.read_csv(hapne_csv)

    max_generations = SCENARIO_GENERATIONS.get(name, DEFAULT_GENERATIONS)

    idata = az.from_netcdf(model_nc)
    post_mat = pred_piecewise(idata, max_generations)
    prior_mat = pred_piecewise(idata, max_generations, group="prior")
    rng = np.random.default_rng(SEED)

    true_t = np.arange(0, max_generations, TIME_STEP)
    true_ne = true_trajectory(graph, true_t)

    fig_prior, ax_prior = plt.subplots(figsize=(5, 4), dpi=300)
    draw_ensemble(ax_prior, true_t, prior_mat, rng)
    ax_prior.plot(true_t, true_ne, color="black", linestyle="--", lw=2, label="Truth")
    ax_prior.set_title("Prior")
    dress_axis(ax_prior, max_generations)

    fig_post, ax_post = plt.subplots(figsize=(5, 4), dpi=300)
    draw_ensemble(ax_post, true_t, post_mat, rng)
    ax_post.plot(gone["Generation"], gone["Ne_diploids"], color="C2", lw=2, label="GONE2")
    ax_post.plot(hapne["TIME"], hapne["Q0.5"], color="C3", lw=2, label="HapNe-LD")
    ax_post.fill_between(
        hapne["TIME"], hapne["Q0.025"], hapne["Q0.975"],
        color="C3", alpha=0.15, linewidth=0,
    )
    ax_post.plot(true_t, true_ne, color="black", linestyle="--", lw=2, label="Truth")
    ax_post.set_title("Posterior")
    dress_axis(ax_post, max_generations)

    # The draws are too faint to read as a legend swatch, so bayesld gets an
    # opaque proxy line in both figures.
    for fig, ax in ((fig_prior, ax_prior), (fig_post, ax_post)):
        handles, labels = ax.get_legend_handles_labels()
        handles.insert(0, Line2D([], [], color="C0", lw=1))
        labels.insert(0, f"bayesld ({model})")
        fig.legend(handles, labels, loc="outside lower center", ncol=len(labels))
        fig.suptitle(name)

    fig_prior.savefig(f"{out_prefix}_prior.pdf")
    fig_prior.savefig(f"{out_prefix}_prior.pgf")
    fig_post.savefig(f"{out_prefix}.pdf")
    fig_post.savefig(f"{out_prefix}.pgf")


if __name__ == "__main__":
    main()
