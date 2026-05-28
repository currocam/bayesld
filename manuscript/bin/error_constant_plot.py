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

        ratio = np.empty((len(d["results"]), len(d["left_bins"])))
        ratio_lo = np.empty_like(ratio)
        ratio_hi = np.empty_like(ratio)
        for k, r in enumerate(d["results"]):
            ld_mc = np.asarray(r["ld_mc"], dtype=float)  # (n_rep, n_bins)
            ld_mc = np.where(ld_mc == 0, np.nan, ld_mc)
            ld_det_inf = np.asarray(r["ld_det_inf"])
            ratio[k] = ld_det_inf / np.nanmean(ld_mc, axis=0)
            n_rep = ld_mc.shape[0]
            idx = rng.integers(0, n_rep, size=(BOOTSTRAP_DRAWS, n_rep))
            ld_mc_boot = np.nanmean(ld_mc[idx], axis=1)  # (B, n_bins)
            ratio_boot = ld_det_inf / ld_mc_boot
            ratio_lo[k] = np.nanquantile(ratio_boot, 0.05, axis=0)
            ratio_hi[k] = np.nanquantile(ratio_boot, 0.95, axis=0)

        ploidy = int(params.get("ploidy", 2))
        combos.append(
            {
                "label": f"{'haploid' if ploidy == 1 else 'diploid'}, n={params['num_samples']}",
                "recombination_rate": params["recombination_rate"],
                "num_samples": params["num_samples"],
                "ploidy": ploidy,
                # error_constant.py stores haploid Ne already doubled to match
                # coalescent timescale; halve it so the x-axis is comparable.
                "Ne": ne_values / 2 if ploidy == 1 else ne_values,
                "ratio": ratio,
                "ratio_lo": ratio_lo,
                "ratio_hi": ratio_hi,
                "left_bins": np.asarray(d["left_bins"]),
                "right_bins": np.asarray(d["right_bins"]),
            }
        )

    combos.sort(key=lambda c: (c["ploidy"], c["num_samples"], c["recombination_rate"]))
    return (combos,)


@app.cell
def _(combos):
    n_bins = combos[0]["ratio"].shape[1]
    left_bins = combos[0]["left_bins"]
    right_bins = combos[0]["right_bins"]

    figs = []
    for b in range(n_bins):
        fig, ax = plt.subplots(
            figsize=(SINGLE_COL, SINGLE_COL * 0.75),
            dpi=300,
            constrained_layout=True,
        )
        for i, c in enumerate(combos):
            color = f"C{i}"
            ax.plot(
                c["Ne"],
                c["ratio"][:, b],
                color=color,
                lw=1.2,
                marker="o",
                markersize=1,
                label=c["label"],
            )
            ax.fill_between(
                c["Ne"],
                c["ratio_lo"][:, b],
                c["ratio_hi"][:, b],
                color=color,
                alpha=0.15,
                linewidth=0,
            )
        ax.axhline(1, color="k", lw=0.6, ls="-", alpha=0.5)
        ax.axvline(10_000, color="k", lw=0.6, ls=":", alpha=0.5)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$N_e$ (effective population size )")
        ax.set_ylabel(
            r"Relative error "
            r"$\widehat{\mathbb{E}}_{\mathrm{F}}[X_iX_jY_iY_j] / "
            r"\widehat{\mathbb{E}}_{\mathrm{MC}}[X_iX_jY_iY_j]$"
        )
        ax.set_title(
            f"r $\\in$ [{left_bins[b]:.2g}, {right_bins[b]:.2g}]", fontsize="small"
        )
        ax.legend(fontsize="x-small")

        if mo.app_meta().mode == "script":
            fig.savefig(f"error_constant_bin{b:02d}.pdf")
            fig.savefig(f"error_constant_bin{b:02d}.pgf")
        figs.append(fig)
    figs
    return


if __name__ == "__main__":
    app.run()
