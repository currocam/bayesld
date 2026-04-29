#!/usr/bin/env -S uv run --script --isolated
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "bayesld @ git+https://github.com/currocam/bayesld.git@a78bf21",
#     "marimo==0.23.3",
#     "numpy==2.2.6",
#     "matplotlib==3.10.9",
#     "msprime==1.4.0",
#     "demesdraw==0.4.1",
#     "matplotlib-label-lines==0.8.1",
# ]
# ///
import marimo

__generated_with = "0.23.3"
app = marimo.App(width="medium")

with app.setup:
    import gzip
    import pickle
    from pathlib import Path

    import marimo as mo
    import matplotlib.patheffects as pe
    import matplotlib.pyplot as plt
    import numpy as np

    # Theme settings
    plt.style.use(Path(__file__).parent / "theme.mplstyle")
    plt.rc("figure", autolayout=True)

    ONE_MM = 1 / 25.4
    SINGLE_COL = 85 * ONE_MM
    DOUBLE_COL = SINGLE_COL * 2
    ONE_HALF_COL = SINGLE_COL * 1.5


@app.cell
def _():
    from bayesld import linear_bins

    return (linear_bins,)


@app.cell
def _(linear_bins):
    left_bins, right_bins = linear_bins()
    return left_bins, right_bins


@app.cell
def _():
    pkl_path = mo.cli_args().get("pkl", "conceptual_data.pkl")
    with gzip.open(pkl_path, "rb") as f:
        data = pickle.load(f)

    baseline_div = data["baseline_div"]
    sim_div = data["sim_div"]
    baseline_linkage = data["baseline_linkage"]
    sim_linkage = data["sim_linkage"]
    times = data["times"]
    params = data["params"]

    ne1 = params["ne1"]
    ne2 = params["ne2"]
    mutation_rate = params["mutation_rate"]
    return (
        baseline_linkage,
        mutation_rate,
        ne1,
        ne2,
        sim_div,
        sim_linkage,
        times,
    )


@app.cell
def _(mutation_rate, ne2):
    expected_div = 4 * ne2 * mutation_rate  # 4 * Ne * mu for diploid
    return (expected_div,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Panel A: Two demographies
    """)
    return


@app.cell
def _(ne1, ne2):
    _t_change = 5

    fig_a, ax_a = plt.subplots(
        figsize=(ONE_HALF_COL, ONE_HALF_COL * 0.75), dpi=300, constrained_layout=True
    )

    import demesdraw
    import msprime
    from matplotlib.lines import Line2D

    demo = msprime.Demography()
    demo.add_population(name="Constant", initial_size=ne2)
    demo.add_population(name="Decline", initial_size=ne1)
    demo.add_population_parameters_change(
        population=1, initial_size=ne2, time=_t_change
    )

    demesdraw.size_history(
        demo.to_demes(),
        ax=ax_a,
        inf_label=True,
        colours={"Constant": "C5", "Decline": "C1"},
    )

    # Add custom legend
    custom_legend = [
        Line2D([0], [0], color="C5", lw=2, label="Constant"),
        Line2D([0], [0], color="C1", lw=2, label="Bottleneck"),
    ]
    ax_a.legend(handles=custom_legend, title="")
    ax_a.set_xlabel("Time ago (generations)")
    ax_a.set_ylabel("Population size $N_e(t)$", rotation=90)
    fig_a.savefig("conceptual_figure_panel_a.pdf")
    fig_a
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Panel Ba: Histogram of per-window nucleotide diversity
    """)
    return


@app.cell
def _(baseline_div, sim_div, times):
    fig_ba, ax_ba = plt.subplots(
        figsize=(ONE_HALF_COL, ONE_HALF_COL * 0.75), dpi=300, constrained_layout=True
    )
    t5_idx = np.where(times == 5)[0][0]

    bplot = ax_ba.boxplot(
        [baseline_div, sim_div[t5_idx]],
        labels=["Constant", "Bottleneck"],
        patch_artist=True,
        widths=0.5,
    )
    for patch, color in zip(bplot["boxes"], ["C5", "C1"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    for median in bplot["medians"]:
        median.set_color("black")
    ax_ba.set_ylabel(r"Nucleotide diversity ($\pi$)")

    fig_ba.savefig("conceptual_figure_panel_ba.pdf")
    fig_ba
    return (t5_idx,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Panel Bb: Full LD curve
    """)
    return


@app.cell
def _(baseline_linkage, left_bins, right_bins, sim_linkage, t5_idx):
    fig_bb, ax_bb = plt.subplots(
        figsize=(ONE_HALF_COL, ONE_HALF_COL * 0.75), dpi=300, constrained_layout=True
    )
    _x = (left_bins + right_bins) / 2 * 100
    _jitter_scale = (_x[1] - _x[0]) * 0.15
    _rng = np.random.default_rng(42)

    def _plot_jittered(ax, x, data, color, label):
        n_rep = data.shape[0]
        for i in range(n_rep):
            _jitter = _rng.normal(0, _jitter_scale, size=len(x))
            ax.plot(x + _jitter, data[i], ".", color=color, alpha=0.2, markersize=3)
        ax.plot(x, data.mean(axis=0), "-", color=color, lw=2, label=label)

    _plot_jittered(ax_bb, _x, baseline_linkage, "C5", "Constant")
    _plot_jittered(ax_bb, _x, sim_linkage[t5_idx], "C1", "Bottleneck")

    ax_bb.set_xlabel("Genetic distance (centimorgan)")
    ax_bb.set_ylabel(r"Linkage disequilibrium $\mathbb E[X_iX_jY_iY_j]$")
    ax_bb.legend()
    fig_bb.savefig("conceptual_figure_panel_bb.pdf")
    fig_bb
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Panel C: Diversity
    """)
    return


@app.cell
def _():
    from labellines import labelLine

    return (labelLine,)


@app.cell
def _(expected_div, labelLine, sim_div, times):
    fig_c, ax_c = plt.subplots(
        figsize=(ONE_HALF_COL, ONE_HALF_COL * 0.75),
        dpi=300,
        constrained_layout=True,
    )

    (_hoz_line,) = ax_c.plot(
        [times[0], times[-1]], [1, 1], label=r"no change", linestyle="--", color="black"
    )

    _vals = sim_div / expected_div
    ax_c.errorbar(
        times,
        _vals.mean(axis=1),
        yerr=_vals.std(axis=1),
        fmt="o",
        markersize=2,
        label="Bottleneck",
        color="C0",
    )
    ax_c.set_xlabel("Bottleneck start (generations ago)")
    ax_c.set_ylabel("Relative change in\ngenetic diversity ($\\pi$)")
    ax_c.set_xlim(0, 100)
    labelLine(_hoz_line, x=times[-21], backgroundcolor="white")

    fig_c.savefig("conceptual_figure_panel_c.pdf")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Panel D: LD at shortest distance bin vs time of change
    """)
    return


@app.cell
def _(baseline_linkage, labelLine, sim_linkage, times):
    fig_d, ax_d = plt.subplots(
        figsize=(ONE_HALF_COL, ONE_HALF_COL * 0.75),
        dpi=300,
        constrained_layout=True,
    )

    _vals = sim_linkage[:, :, -1] / baseline_linkage[:, -1].mean()
    ax_d.errorbar(
        times,
        _vals.mean(axis=1),
        yerr=_vals.std(axis=1),
        fmt="o",
        markersize=2.5,
        label="9.5cM-10.0cM",
        color="C2",
    )

    _vals = sim_linkage[:, :, 0] / baseline_linkage[:, 0].mean()

    ax_d.errorbar(
        times,
        _vals.mean(axis=1),
        yerr=_vals.std(axis=1),
        fmt="o",
        markersize=2.5,
        label="0.5cM-1.0cM",
        color="C1",
    )

    (_hoz_line,) = ax_d.plot(
        [times[0], times[-1]], [1, 1], label=r"no change", linestyle="--", color="black"
    )

    ax_d.set_xlabel("Bottleneck start (generations ago)")
    ax_d.set_ylabel(
        r"Relative change in linkage"
        + "\n"
        + r" disequilibrium ($\mathbb E[X_iX_jY_iY_j]$)"
    )
    ax_d.set_xlim(0, 100)
    ax_d.set_yscale("log")
    ax_d.set_yticks([1.0, 1.5, 3, 6, 9])
    ax_d.yaxis.set_major_formatter(
        plt.matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:g}")
    )
    ax_d.yaxis.set_minor_locator(plt.matplotlib.ticker.NullLocator())
    labelLine(_hoz_line, x=times[75], backgroundcolor="white", drop_label=True)

    ax_d.legend(
        loc="center right", bbox_to_anchor=(1.0, 0.40), title="Genetic distance"
    )

    fig_d.savefig("conceptual_figure_panel_d.pdf")
    return


if __name__ == "__main__":
    app.run()
