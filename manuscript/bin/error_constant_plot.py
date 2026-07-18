#!/usr/bin/env -S uv run --script --isolated
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "numpy",
#     "matplotlib",
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
    plt.rcParams["pgf.texsystem"] = "pdflatex"
    plt.rcParams["pgf.preamble"] = r"\usepackage{amsmath}\usepackage{amssymb}"

    ONE_MM = 1 / 25.4
    SINGLE_COL = 85 * ONE_MM
    DOUBLE_COL = SINGLE_COL * 2
    ONE_HALF_COL = SINGLE_COL * 1.5

    BOOTSTRAP_DRAWS = 500
    BOOTSTRAP_SEED = 0


@app.cell
def _():
    arg = mo.cli_args().get("pkl", "error_constant.pkl")
    pkl_paths = arg if isinstance(arg, list) else [arg]

    rng = np.random.default_rng(BOOTSTRAP_SEED)

    combos = []
    for p in pkl_paths:
        with gzip.open(p, "rb") as f:
            d = pickle.load(f)
        params = d["params"]
        ne_values = np.asarray([r["Ne"] for r in d["results"]])
        ploidy = int(params.get("ploidy", 2))

        ratios = {}
        for det_key in ("ld_det", "ld_det_inf"):
            ratio = np.empty((len(d["results"]), len(d["left_bins"])))
            ratio_lo = np.empty_like(ratio)
            ratio_hi = np.empty_like(ratio)
            for k, r in enumerate(d["results"]):
                ld_mc = np.asarray(r["ld_mc"], dtype=float)  # (n_rep, n_bins)
                ld_mc = np.where(ld_mc == 0, np.nan, ld_mc)
                ld_det = np.asarray(r[det_key])
                ratio[k] = ld_det / np.nanmean(ld_mc, axis=0)
                n_rep = ld_mc.shape[0]
                idx = rng.integers(0, n_rep, size=(BOOTSTRAP_DRAWS, n_rep))
                ld_mc_boot = np.nanmean(ld_mc[idx], axis=1)  # (B, n_bins)
                ratio_boot = ld_det / ld_mc_boot
                ratio_lo[k] = np.nanquantile(ratio_boot, 0.05, axis=0)
                ratio_hi[k] = np.nanquantile(ratio_boot, 0.95, axis=0)
            ratios[det_key] = (ratio, ratio_lo, ratio_hi)

        combos.append(
            {
                "label": f"n={params['num_samples']}",
                "recombination_rate": params["recombination_rate"],
                "num_samples": params["num_samples"],
                "ploidy": ploidy,
                # error_constant.py stores haploid Ne already doubled to match
                # coalescent timescale; halve it so the x-axis is comparable.
                "Ne": ne_values / 2 if ploidy == 1 else ne_values,
                "ratios": ratios,
                "left_bins": np.asarray(d["left_bins"]),
                "right_bins": np.asarray(d["right_bins"]),
            }
        )

    combos.sort(key=lambda c: (c["ploidy"], c["num_samples"], c["recombination_rate"]))
    return (combos,)


@app.cell
def _(combos):
    # Smallest (0.5–1.0 cM) and largest (9.5–10.0 cM) distance bins.
    bin_specs = [
        (0, "C1", r"$u \in [0.5, 1.0]$ cM"),
        (-1, "C0", r"$u \in [9.5, 10.0]$ cM"),
    ]
    ylabel = (
        r"\begin{tabular}{c}Approximation ratio\\"
        r"(closed-form / Monte Carlo)\end{tabular}"
    )

    def _plot_combo(ax, combo, det_key, *, legend=True):
        ratio, ratio_lo, ratio_hi = combo["ratios"][det_key]
        for b, color, label in bin_specs:
            ax.plot(
                combo["Ne"],
                ratio[:, b],
                color=color,
                lw=1.2,
                marker="o",
                markersize=1,
                label=label,
            )
            ax.fill_between(
                combo["Ne"],
                ratio_lo[:, b],
                ratio_hi[:, b],
                color=color,
                alpha=0.15,
                linewidth=0,
            )
        ax.axhline(1, color="k", lw=0.6, ls="-", alpha=0.5)
        ax.axvline(10_000, color="k", lw=0.6, ls=":", alpha=0.5)
        ax.set_xscale("log")
        ax.set_xlabel(r"$N_e$")
        if legend:
            ax.legend(fontsize="x-small")

    haploid = next(c for c in combos if c["ploidy"] == 1)
    diploid = next(c for c in combos if c["ploidy"] == 2)

    # Combined: left haploid uncorrected, right diploid corrected.
    fig_combined, axes = plt.subplots(
        1,
        2,
        figsize=(DOUBLE_COL, SINGLE_COL * 0.75),
        dpi=300,
        sharey=True,
        constrained_layout=True,
    )
    _plot_combo(axes[0], haploid, "ld_det_inf", legend=True)
    _plot_combo(axes[1], diploid, "ld_det", legend=False)
    for ax in axes:
        ax.set_xlim(100, 100_000)
        ax.set_ylim(0.6, 1.1)
    axes[0].set_title("Haploid", fontsize="small")
    axes[1].set_title("Diploid (corrected)", fontsize="small")
    fig_combined.supylabel(ylabel)

    # Diploid uncorrected, single panel.
    fig_uncorr, ax_uncorr = plt.subplots(
        figsize=(SINGLE_COL, SINGLE_COL * 0.75),
        dpi=300,
        constrained_layout=True,
    )
    _plot_combo(ax_uncorr, diploid, "ld_det_inf", legend=True)
    ax_uncorr.set_ylabel(ylabel)

    figs = [fig_combined, fig_uncorr]
    if mo.app_meta().mode == "script":
        fig_combined.savefig("error_constant_haploid_diploid.pdf")
        fig_combined.savefig("error_constant_haploid_diploid.pgf")
        fig_uncorr.savefig("error_constant_diploid_uncorrected.pdf")
        fig_uncorr.savefig("error_constant_diploid_uncorrected.pgf")
    figs
    return


if __name__ == "__main__":
    app.run()
