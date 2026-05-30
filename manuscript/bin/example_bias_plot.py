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

    LATEX_LABELS = {
        "Nc=5e2": r"$N_c = 5\mathrm{e}2$",
        "Nc=1e3": r"$N_c = 1\mathrm{e}3$",
        "Nc=2e3": r"$N_c = 2\mathrm{e}3$",
        "Nc=1e4": r"$N_c = 1\mathrm{e}4$",
        "Nc=1e3, t0=25": r"$N_c = 1\mathrm{e}3,\ t_0 = 25$",
        "Nc=5e3, t0=25": r"$N_c = 5\mathrm{e}3,\ t_0 = 25$",
        "Nc=1e3, t0=50": r"$N_c = 1\mathrm{e}3,\ t_0 = 50$",
        "Nc=5e3, t0=50": r"$N_c = 5\mathrm{e}3,\ t_0 = 50$",
        "Nc=1e4, t0=25": r"$N_c = 1\mathrm{e}4,\ t_0 = 25$",
        "Nc=1e4, t0=50": r"$N_c = 1\mathrm{e}4,\ t_0 = 50$",
        "Nf=10, t0=25": r"$N_f = 10,\ t_0 = 25$",
        "Nf=100, t0=25": r"$N_f = 100,\ t_0 = 25$",
        "Nf=10, t0=50": r"$N_f = 10,\ t_0 = 50$",
        "Nf=100, t0=50": r"$N_f = 100,\ t_0 = 50$",
    }

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
        bbox_to_anchor=(0.5, -0.38),
        frameon=False,
        fontsize=8,
        ncol=2,
        handlelength=1.5,
        columnspacing=1.0,
        title_fontsize=8,
        **kw,
    )


@app.function
def save_panel(fig, name):
    if mo.app_meta().mode == "script":
        for ext in ("pdf", "pgf"):
            fig.savefig(f"{name}.{ext}")


@app.function
def size_history_curve(epochs, t_max):
    """Return (times_into_past, sizes) clipped to [0, t_max]."""
    epochs = sorted(epochs, key=lambda e: -e["end_time"])
    times, sizes = [], []
    for i, e in enumerate(epochs):
        t_start = epochs[i - 1]["end_time"] if i > 0 else t_max
        t_end = e["end_time"]
        if t_start > t_max:
            t_start = t_max
        if t_end > t_max:
            continue
        n_start = e["start_size"]
        n_end = e.get("end_size", n_start)
        if n_start == n_end:
            times += [t_start, t_end]
            sizes += [n_start, n_end]
        else:
            ts = np.linspace(t_start, t_end, 40)
            ns = n_start * (n_end / n_start) ** ((t_start - ts) / (t_start - t_end))
            times += ts.tolist()
            sizes += ns.tolist()
    return np.array(times), np.array(sizes)


@app.function
def plot_demography(variants, title):
    fig, ax = new_panel(title)
    t_max = 100
    for i, v in enumerate(variants):
        t, n = size_history_curve(v["epochs"], t_max)
        ax.plot(t, n, color=f"C{i}", label=LATEX_LABELS[v["label"]], linewidth=1.0)
        ax.plot(t[0], n[0], marker=">", color=f"C{i}", markersize=4, clip_on=False)
    ax.set_xlim(0, t_max)
    ax.set_xlabel("Time ago (generations)")
    ax.set_ylabel(r"$N_e$")
    ax.legend(fontsize=6, frameon=False, handlelength=1.5, loc="best")
    return fig


@app.function
def plot_ld(variants, bin_midpoints, mc_key, mc_label, title):
    fig, ax = new_panel(f"{title} ({mc_label})")
    for i, v in enumerate(variants):
        color = f"C{i}"
        m, lo, hi = bootstrap_ci(v[mc_key]).T
        ax.plot(bin_midpoints, v["ld_det"], color=color)
        ax.vlines(
            bin_midpoints,
            lo,
            hi,
            capstyle="round",
            linewidth=4,
            alpha=0.3,
            color=color,
        )
        ax.plot(bin_midpoints, m, "o", markersize=3, color=color)
    ax.set_xscale("log")
    ax.set_xlabel("Genetic distance (Morgan)")
    ax.set_ylabel(r"$\mathbb{E}[X_iX_jY_iY_j]$")
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

        save_panel(fig_demo, f"example_bias_{name}_demography")
        save_panel(fig_smc, f"example_bias_{name}_smc")

        blocks.append(
            mo.vstack([mo.md(f"### {title}"), mo.hstack([fig_demo, fig_smc])])
        )
    mo.vstack(blocks)
    return


if __name__ == "__main__":
    app.run()
