#!/usr/bin/env -S uv run --script --isolated
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo==0.23.3",
#     "numpy==2.2.6",
#     "matplotlib==3.10.9",
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
    import matplotlib.pyplot as plt
    import numpy as np

    plt.style.use(Path(__file__).parent / "theme.mplstyle")
    plt.rc("figure", autolayout=True)

    ONE_MM = 1 / 25.4
    SINGLE_COL = 85 * ONE_MM
    DOUBLE_COL = SINGLE_COL * 2


@app.function
def bootstrap_ci(data, n_boot=10_000, ci=0.95, seed=1000):
    if data.ndim > 2:
        raise NotImplementedError
    if data.ndim == 2:
        return np.array(
            [bootstrap_ci(x, n_boot=n_boot, ci=ci, seed=seed) for x in data.T]
        )
    rng = np.random.default_rng(seed)
    boots = rng.choice(data, size=(n_boot, len(data)), replace=True).mean(axis=1)
    alpha = (1 - ci) / 2
    lo, hi = np.quantile(boots, [alpha, 1 - alpha])
    m = data.mean()
    return m, lo, hi


@app.cell
def _():
    pkl_path = mo.cli_args().get("pkl", "example_bias_data.pkl")
    with gzip.open(pkl_path, "rb") as f:
        data = pickle.load(f)

    results = data["results"]
    left_bins = data["left_bins"]
    right_bins = data["right_bins"]
    bin_midpoints = (left_bins + right_bins) / 2
    return results, left_bins, right_bins, bin_midpoints


@app.function
def draw_piecewise_skeleton(ax):
    """Manual sketch of a 3-epoch piecewise-constant demography with generic labels."""
    t = [0, 25, 25, 75, 75, 130]
    ne = [1000, 1000, 100, 100, 1000, 1000]

    ax.plot(t, ne, color="black", lw=1.5)
    ax.set_xlim(130, 0)
    ax.set_ylim(0, 1300)
    ax.set_yticks([100, 1000])
    ax.set_yticklabels([r"$100x$", r"$1000x$"])
    ax.set_xticks([0, 25, 75])
    ax.set_xlabel("Time ago (generations)")
    ax.set_ylabel(r"$N_e(t)$")


@app.function
def draw_exponential_skeleton(ax):
    """Manual sketch of a piecewise-exponential demography with generic labels."""
    t_exp = np.linspace(0, 100, 200)
    ne_exp = 5000 * np.exp(-np.log(5000 / 500) / 100 * t_exp)
    t_anc = [100, 160]
    ne_anc = [500, 500]

    ax.plot(t_exp, ne_exp, color="black", lw=1.5)
    ax.plot(t_anc, ne_anc, color="black", lw=1.5)
    ax.set_xlim(160, 0)
    ax.set_ylim(0, 6000)
    ax.set_yticks([500, 5000])
    ax.set_yticklabels([r"$500x$", r"$5000x$"])
    ax.set_xticks([0, 100])
    ax.set_xlabel("Time ago (generations)")
    ax.set_ylabel(r"$N_e(t)$")


@app.function
def plot_ld_comparison(
    ax, bin_midpoints, ld_det, ld_coal, ld_dtwf, title, coal_label="SMC'"
):
    """Plot deterministic vs coalescent vs DTWF LD with bootstrap CI."""
    boots_coal = bootstrap_ci(ld_coal)
    boots_dtwf = bootstrap_ci(ld_dtwf)

    ax.plot(bin_midpoints, ld_det, color="C1", label="Deterministic")

    ax.plot(
        bin_midpoints, boots_coal[:, 0], color="C0", linestyle="--", label=coal_label
    )
    ax.fill_between(
        bin_midpoints,
        boots_coal[:, 1],
        boots_coal[:, 2],
        alpha=0.1,
        color="C0",
        linewidth=0.0,
    )

    ax.plot(bin_midpoints, boots_dtwf[:, 0], color="C2", linestyle=":", label="DTWF")
    ax.fill_between(
        bin_midpoints,
        boots_dtwf[:, 1],
        boots_dtwf[:, 2],
        alpha=0.1,
        color="C2",
        linewidth=0.0,
    )

    ax.set_xscale("log")
    ax.set_xlabel("Genetic distance (Morgan)")
    ax.set_title(title, fontsize=8, fontweight="bold")


@app.function
def save_panel(fig, name):
    if mo.app_meta().mode == "script":
        fig.savefig(f"{name}.pdf")


@app.cell
def _(results, bin_midpoints):
    # --- Piecewise skeleton ---
    fig_pw_skel, _ax = plt.subplots(
        figsize=(SINGLE_COL, SINGLE_COL * 0.75),
        dpi=300,
        constrained_layout=True,
    )
    draw_piecewise_skeleton(_ax)
    _ax.set_title("Demography", fontsize=8, fontweight="bold")
    save_panel(fig_pw_skel, "example_bias_piecewise_skeleton")
    fig_pw_skel
    return


@app.cell
def _(results, bin_midpoints):
    # --- Piecewise small (x=1) ---
    fig_pw_small, _ax = plt.subplots(
        figsize=(SINGLE_COL, SINGLE_COL * 0.75),
        dpi=300,
        constrained_layout=True,
    )
    _r = results["small_piecewise"]
    plot_ld_comparison(
        _ax,
        bin_midpoints,
        _r["ld_det"],
        _r["ld_smc_prime"],
        _r["ld_dtwf"],
        title=r"$x = 1$",
        coal_label="SMC'",
    )
    _ax.set_ylabel(r"$\mathbb{E}[X_iX_jY_iY_j]$")
    _ax.legend(fontsize=6)
    save_panel(fig_pw_small, "example_bias_piecewise_small")
    fig_pw_small
    return


@app.cell
def _(results, bin_midpoints):
    # --- Piecewise large (x=10) ---
    fig_pw_large, _ax = plt.subplots(
        figsize=(SINGLE_COL, SINGLE_COL * 0.75),
        dpi=300,
        constrained_layout=True,
    )
    _r = results["large_piecewise"]
    plot_ld_comparison(
        _ax,
        bin_midpoints,
        _r["ld_det"],
        _r["ld_smc_prime"],
        _r["ld_dtwf"],
        title=r"$x = 10$",
        coal_label="SMC'",
    )
    _ax.set_ylabel(r"$\mathbb{E}[X_iX_jY_iY_j]$")
    save_panel(fig_pw_large, "example_bias_piecewise_large")
    fig_pw_large
    return


@app.cell
def _(results, bin_midpoints):
    # --- Exponential skeleton ---
    fig_exp_skel, _ax = plt.subplots(
        figsize=(SINGLE_COL, SINGLE_COL * 0.75),
        dpi=300,
        constrained_layout=True,
    )
    draw_exponential_skeleton(_ax)
    _ax.set_title("Demography", fontsize=8, fontweight="bold")
    save_panel(fig_exp_skel, "example_bias_exponential_skeleton")
    fig_exp_skel
    return


@app.cell
def _(results, bin_midpoints):
    # --- Exponential small (x=1) ---
    fig_exp_small, _ax = plt.subplots(
        figsize=(SINGLE_COL, SINGLE_COL * 0.75),
        dpi=300,
        constrained_layout=True,
    )
    _r = results["small_exponential"]
    plot_ld_comparison(
        _ax,
        bin_midpoints,
        _r["ld_det"],
        _r["ld_smc"],
        _r["ld_dtwf"],
        title=r"$x = 1$",
        coal_label="SMC",
    )
    _ax.set_ylabel(r"$\mathbb{E}[X_iX_jY_iY_j]$")
    _ax.legend(fontsize=6)
    save_panel(fig_exp_small, "example_bias_exponential_small")
    fig_exp_small
    return


@app.cell
def _(results, bin_midpoints):
    # --- Exponential large (x=10) ---
    fig_exp_large, _ax = plt.subplots(
        figsize=(SINGLE_COL, SINGLE_COL * 0.75),
        dpi=300,
        constrained_layout=True,
    )
    _r = results["large_exponential"]
    plot_ld_comparison(
        _ax,
        bin_midpoints,
        _r["ld_det"],
        _r["ld_smc"],
        _r["ld_dtwf"],
        title=r"$x = 10$",
        coal_label="SMC",
    )
    _ax.set_ylabel(r"$\mathbb{E}[X_iX_jY_iY_j]$")
    save_panel(fig_exp_large, "example_bias_exponential_large")
    fig_exp_large
    return


if __name__ == "__main__":
    app.run()
