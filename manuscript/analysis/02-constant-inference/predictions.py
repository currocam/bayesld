import marimo

__generated_with = "0.17.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import msprime
    import pandas as pd
    import arviz_plots as azp
    import matplotlib.pyplot as plt
    import bayesld.models
    import numpy as np
    return azp, bayesld, mo, np, pd, plt


@app.cell
def _(mo):
    mo.md(r"""
    In this notebook we look at predictions under various models.
    """)
    return


@app.cell
def _(azp, plt):
    azp.style.use('arviz-variat')

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "legend.fontsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "lines.linewidth": 1.5,
        "text.latex.preamble": r"\usepackage{amsmath,amssymb}",
    })
    return


@app.cell
def _(pd):
    data_small_ne = pd.read_pickle("02-constant-inference/steps/simulation_Ne_c500_Ne_a500_t0100_seed468214376.pkl")
    data_large_ne = pd.read_pickle("02-constant-inference/steps/simulation_Ne_c10000_Ne_a10000_t0100_seed468214376.pkl")
    data_ancient_change = pd.read_pickle("02-constant-inference/steps/simulation_Ne_c1000_Ne_a10000_t01000_seed468214376.pkl")
    data_medium_change = pd.read_pickle("02-constant-inference/steps/simulation_Ne_c1000_Ne_a10000_t0200_seed468214376.pkl")
    data_recent_change = pd.read_pickle("02-constant-inference/steps/simulation_Ne_c1000_Ne_a10000_t050_seed468214376.pkl")
    return (
        data_ancient_change,
        data_large_ne,
        data_medium_change,
        data_recent_change,
        data_small_ne,
    )


@app.cell
def _(data_small_ne):
    data_small_ne[0][1]
    return


@app.cell
def _(data_small_ne):
    df = data_small_ne[0][1] # Example for the left and right
    left = df["left"].values[:19]
    right = df["right"].values[:19]
    return left, right


@app.cell
def _(bayesld, left, right):
    def prediction(Ne):
        return bayesld.models.expected_ld_constant(
            Ne=Ne,
            u_i=left,
            u_j=right,
            sample_size=50,
        ).eval()
    return (prediction,)


@app.cell
def _(mo):
    mo.md(r"""
    We expect lower Ne to show a higher error. First, because they will have less number of segregating mutations. Second, because drift can vary allele frequencies in a few generations if Ne is very small.
    """)
    return


@app.cell
def _(np):
    def prepare_datasets(data):
        return np.concatenate(
            [df.pivot(index="window_index", columns="left", values="mean").to_numpy() for df in data],
            axis=0
        )
    return (prepare_datasets,)


@app.cell
def _(
    data_large_ne,
    data_small_ne,
    left,
    plt,
    prediction,
    prepare_datasets,
    right,
):
    fig1, ax1 = plt.subplots(figsize=(4, 3))

    midpoints = (left + right) / 2 * 100

    # Compute mean and standard deviation
    means_500 = prepare_datasets([x[1] for x in data_small_ne]).mean(axis=0)
    means_10000 = prepare_datasets([x[1] for x in data_large_ne]).mean(axis=0)

    line_500, = ax1.plot(midpoints, prediction(500), label=r"$N=500$")
    line_10000, = ax1.plot(midpoints, prediction(10_000), label=r"$N=10000$")


    # Use the same colors for scatter (taken from the plotted lines)
    ax1.scatter(midpoints, means_500, s=30, marker="o",
                color=line_500.get_color(), label="_nolegend_")
    ax1.scatter(midpoints, means_10000, s=30, marker="o",
                color=line_10000.get_color(), label="_nolegend_")

    # Labels and formatting
    ax1.set_xlabel(r"Distance between pairs of loci (cM)")
    ax1.set_ylabel(r"$\mathbb{E}\!\left[X_i X_j Y_i Y_j\right]$")
    ax1.set_yscale("log")

    ax1.margins(x=0.02)
    fig1.savefig("02-constant-inference/constant_predictions.pdf", dpi=600, bbox_inches="tight", transparent=True)
    plt.legend()
    plt.show()
    return fig1, midpoints


@app.cell
def _(
    data_ancient_change,
    data_medium_change,
    data_recent_change,
    fig1,
    midpoints,
    plt,
    prediction,
    prepare_datasets,
):
    fig2, ax2 = plt.subplots(figsize=(4, 3))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # Compute means and stds
    means_recent = prepare_datasets([x[1] for x in data_recent_change]).mean(axis=0)
    stds_recent = prepare_datasets([x[1] for x in data_recent_change]).std(axis=0)

    means_medium = prepare_datasets([x[1] for x in data_medium_change]).mean(axis=0)
    stds_medium = prepare_datasets([x[1] for x in data_medium_change]).std(axis=0)

    means_ancient = prepare_datasets([x[1] for x in data_ancient_change]).mean(axis=0)
    stds_ancient = prepare_datasets([x[1] for x in data_ancient_change]).std(axis=0)

    ref = prediction(1_000)

    # Shift for side-by-side plotting
    width = 0.02 * (max(midpoints) - min(midpoints))  # small horizontal offset
    x_recent = midpoints - width
    x_medium = midpoints
    x_ancient = midpoints + width

    # Plot error bars side by side
    ax2.errorbar(x_recent, (ref - means_recent) / ref, yerr=stds_recent / ref, 
                 fmt='o', markersize=5, label="Recent change ($t=50$)", color=colors[0])
    ax2.errorbar(x_medium, (ref - means_medium) / ref, yerr=stds_medium / ref, 
                 fmt='o', markersize=5, label="Intermediate change ($t=200$)", color=colors[1])
    ax2.errorbar(x_ancient, (ref - means_ancient) / ref, yerr=stds_ancient / ref, 
                 fmt='o', markersize=5, label="Old change ($t=1000$)", color=colors[2])

    # Reference line
    ax2.axhline(0.0, linestyle="--", color=colors[-1])

    # Labels and formatting
    ax2.set_xlabel(r"Distance between pairs of loci (cM)")
    ax2.set_ylabel(r"Relative prediction error")
    ax2.margins(x=0.02)
    fig1.savefig("02-constant-inference/miss_specification_constant.pdf", dpi=600, bbox_inches="tight", transparent=True)

    # Legend at the bottom
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
    plt.show()
    return (colors,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The idea is that the summary statistic we are working with is not informative of anything but very recent Ne changes. Therefore, we expect model misspecification many generations ago not to bias recent Ne results. Perhaps surprisingly, for this example, a model that's more misspecified is better predicted than the less misspecified model ($t=1000$). This is because a higher Ne in the past will make more segregating mutations available, so variance reduces.

    The model that's miss-specified in the range of values where this summary statistics are informative, $<100$ generations, shows a clear deviation. A positive error means we observe less LD than expected in close loci (i.e. Ne was higher at some further point)
    """)
    return


@app.cell
def _(colors, np, plt):
    fig3, ax3 = plt.subplots(figsize=(3, 4))

    # Reference line
    times = np.linspace(0, 1500, 1000)
    ax3.step(times, np.where(times < 50, 1000, 10_000), label="Recent change ($t=50$)", color = colors[0])
    ax3.step(times, np.where(times < 200, 1000, 10_000), label="Intermediate change ($t=200$)", color=colors[1])
    ax3.step(times, np.where(times < 1000, 1000, 10_000), label="Old change ($t=1000$)", color=colors[2])
    ax3.axhline(1000, label="Predicted model", color=colors[-1])

    # Labels and formatting
    ax3.set_xlabel(r"Time (generations ago)")
    ax3.set_ylabel(r"Effective population size")
    ax3.set_xscale("log")

    # Legend at the bottom
    ax3.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15))
    plt.show()
    return


if __name__ == "__main__":
    app.run()
