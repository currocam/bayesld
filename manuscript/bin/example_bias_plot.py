#!/usr/bin/env -S uv run --script --isolated
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo==0.23.3",
#     "numpy==2.2.6",
#     "matplotlib==3.10.9",
#     "demes==0.2.3",
#     "demesdraw==0.4.0",
# ]
# ///
import marimo

__generated_with = "0.23.3"
app = marimo.App(width="medium")

with app.setup:
    import gzip
    import pickle
    from pathlib import Path

    import demes
    import demesdraw
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    plt.style.use(Path(__file__).parent / "theme.mplstyle")
    plt.rc("figure", autolayout=True)
    plt.rcParams["pgf.texsystem"] = "pdflatex"
    plt.rcParams["pgf.preamble"] = r"\usepackage{amsmath}\usepackage{amssymb}"

    ONE_MM = 1 / 25.4
    SINGLE_COL = 85 * ONE_MM
    PANEL_SIZE = (SINGLE_COL, SINGLE_COL * 0.75)


@app.function
def bootstrap_ci(data, n_boot=10_000, ci=0.95, seed=1000):
    if data.ndim == 1:
        rng = np.random.default_rng(seed)
        boots = rng.choice(data, size=(n_boot, len(data)), replace=True).mean(axis=1)
        alpha = (1 - ci) / 2
        lo, hi = np.quantile(boots, [alpha, 1 - alpha])
        return data.mean(), lo, hi
    if data.ndim == 2:
        return np.array([bootstrap_ci(x, n_boot, ci, seed) for x in data.T])
    raise NotImplementedError


@app.function
def new_panel(title):
    fig, ax = plt.subplots(figsize=PANEL_SIZE, dpi=300, constrained_layout=True)
    ax.set_title(title, fontsize=8, fontweight="bold")
    return fig, ax


@app.function
def legend_below(ax, **kw):
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        frameon=False,
        fontsize=6,
        ncol=2,
        handlelength=1.5,
        columnspacing=1.0,
        title_fontsize=6,
        **kw,
    )


@app.function
def save_panel(fig, name):
    if mo.app_meta().mode == "script":
        for ext in ("pdf", "pgf"):
            fig.savefig(f"{name}.{ext}")


@app.function
def plot_demography(variants, title):
    fig, ax = new_panel(title)
    b = demes.Builder(time_units="generations")
    for i, v in enumerate(variants):
        b.add_deme(f"pop{i}", description=v["label"], epochs=v["epochs"])
    colors = {f"pop{i}": f"C{i}" for i in range(len(variants))}
    demesdraw.size_history(
        b.resolve(), ax=ax, colours=colors, annotate_epochs=False, log_time=False
    )
    handles = [
        plt.Line2D([], [], color=c, label=v["label"])
        for v, c in zip(variants, colors.values())
    ]
    legend_below(ax, handles=handles)
    return fig


@app.function
def plot_ld(variants, bin_midpoints, mc_key, mc_label, title):
    fig, ax = new_panel(f"{title} ({mc_label})")
    for i, v in enumerate(variants):
        color = f"C{i}"
        m, lo, hi = bootstrap_ci(v[mc_key]).T
        ax.plot(bin_midpoints, v["ld_det"], color=color)
        ax.plot(bin_midpoints, m, color=color, linestyle="--", label=v["label"])
        ax.fill_between(bin_midpoints, lo, hi, alpha=0.1, color=color, linewidth=0.0)
    ax.set_xscale("log")
    ax.set_xlabel("Genetic distance (Morgan)")
    ax.set_ylabel(r"$\mathbb{E}[X_iX_jY_iY_j]$")
    legend_below(ax, title="solid: det / dashed: MC")
    return fig


@app.cell
def _():
    pkl_path = mo.cli_args().get("pkl", "example_bias_data.pkl")
    with gzip.open(pkl_path, "rb") as f:
        data = pickle.load(f)
    results = data["results"]
    bin_midpoints = (data["left_bins"] + data["right_bins"]) / 2
    return results, bin_midpoints


@app.cell
def _(results, bin_midpoints):
    blocks = []
    for name, r in results.items():
        smc_label = "SMC'" if r["smc_k"] == 1 else "SMC"
        title = name.replace("_", " ").capitalize()

        fig_demo = plot_demography(r["variants"], title)
        fig_smc = plot_ld(r["variants"], bin_midpoints, "ld_smc", smc_label, title)
        fig_dtwf = plot_ld(r["variants"], bin_midpoints, "ld_dtwf", "DTWF", title)

        save_panel(fig_demo, f"example_bias_{name}_demography")
        save_panel(fig_smc, f"example_bias_{name}_smc")
        save_panel(fig_dtwf, f"example_bias_{name}_dtwf")

        blocks.append(
            mo.vstack([mo.md(f"### {title}"), mo.hstack([fig_demo, fig_smc, fig_dtwf])])
        )
    mo.vstack(blocks)
    return


if __name__ == "__main__":
    app.run()
