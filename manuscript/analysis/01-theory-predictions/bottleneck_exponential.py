import marimo

__generated_with = "0.17.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import pymc as pm
    import numpy as np
    import arviz as az
    import bayesld.models
    import matplotlib.pyplot as plt
    import tempfile, subprocess
    import pandas as pd
    return bayesld, np, pd, plt


@app.cell
def _(pd):
    df = pd.read_pickle("simulation_Nef10_seed131562.pkl")[0] # Example for the left and right
    left = df["left"].values
    right = df["right"].values
    return left, right


@app.cell
def _(np):
    Ne_c = 5_000 # Current effective population size
    Ne_a = 6_000 # Ancestral effective population size
    t_inv = 50 # Time of invasions in generations
    #Ne_f = 100 # Founder effective population size
    times = np.arange(70) # Time points to evaluate Ne trajectory
    return Ne_a, Ne_c, t_inv, times


@app.cell
def _(plt):
    from labellines import labelLine, labelLines
    import arviz_plots as azp

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
    return (labelLines,)


@app.cell
def _(Ne_a, Ne_c, np, t_inv, times):
    def ne_trajectory(Ne_f):
        """Generate effective population size trajectory given final Ne."""
        alpha = (np.log(Ne_c) - np.log(Ne_f)) / t_inv
        return np.piecewise(
            times,
            [times < t_inv, times >= t_inv],
            [lambda t: Ne_c * np.exp(-alpha * t),
             Ne_a]
        )
    return (ne_trajectory,)


@app.cell
def _(labelLines, ne_trajectory, plt, times):
    fig, ax = plt.subplots(figsize=(4, 3))

    # Plot the trajectories
    ax.plot(times[:52], ne_trajectory(50)[:52], label=r"$N_f=50$")
    ax.plot(times[:52], ne_trajectory(100)[:52], label=r"$N_f=100$")
    ax.plot(times[:52], ne_trajectory(500)[:52], label=r"$N_f=500$")
    #ax.step(times, ne_trajectory(500), where='post', label=r"$N_e = 500$")

    # Label lines directly
    labelLines(ax.get_lines(), zorder=2, xvals=[15, 25, 35], usetex=True)
    # Axes labels and formatting
    ax.set_xlabel(r"Generations ago")
    ax.set_ylabel(r"Effective population size $N_e(t)$")
    ax.margins(x=0.02)
    fig.savefig("bottleneck_exponential_ne.pdf", dpi=600, bbox_inches="tight", transparent=True)

    plt.show()
    return


@app.cell
def _(Ne_a, Ne_c, bayesld, left, np, right, t_inv):
    def prediction(Ne_f):
        alpha = (np.log(Ne_c) - np.log(Ne_f)) / t_inv
        return bayesld.models.expected_ld_exponential(
            Ne_c=Ne_c,
            Ne_a=Ne_a,
            t0=t_inv,
            alpha=alpha,
            u_i=left,
            u_j=right,
            sample_size=50,
        ).eval()
    return (prediction,)


@app.cell
def _(left, np, plt, prediction, right, vecs):
    fig2, ax2 = plt.subplots(figsize=(4, 3))

    midpoints = (left + right) / 2

    # Compute mean and standard deviation
    means_10 = np.mean(vecs[0], axis=0)
    means_100 = np.mean(vecs[1], axis=0)
    means_500 = np.mean(vecs[2], axis=0)

    # Model predictions (solid lines)
    ax2.plot(midpoints, prediction(50), label=r"$N_f=50$")
    ax2.plot(midpoints, prediction(100), label=r"$N_f=100$")
    ax2.plot(midpoints, prediction(500), label=r"$N_f=500$")


    #ax2.plot(midpoints, prediction(5000), label=r"$N_f=500$")

    # Observed values (dashed lines)
    ax2.scatter(midpoints, means_10, s=30, label=r"$N_f=50$")
    ax2.scatter(midpoints, means_100, s=30, label=r"$N_f=100$")
    ax2.scatter(midpoints, means_500, s=30, label=r"$N_f=500$")

    # Axis labels and formatting
    ax2.set_xlabel(r"Distance between pairs of loci (cM)")
    ax2.set_ylabel(r"$\mathbb{E}\!\left[X_i X_j Y_i Y_j\right]$")
    ax2.set_yscale("log")

    ax2.margins(x=0.02)
    fig2.savefig("bottleneck_exponential.pdf", dpi=600, bbox_inches="tight", transparent=True)
    # Save for publication
    plt.show()

    return


@app.cell
def _(np, pd):
    vecs = [
        np.asarray([x["mean"].values for x in pd.read_pickle(file)])
        for file in [
            "simulation_Nef50_seed131562.pkl",
            "simulation_Nef100_seed131562.pkl",
            "simulation_Nef500_seed131562.pkl"
        ]
    ]
    return (vecs,)


if __name__ == "__main__":
    app.run()
