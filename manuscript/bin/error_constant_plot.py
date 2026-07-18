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
    left_bins = combos[0]["left_bins"]
    right_bins = combos[0]["right_bins"]
    bin_indices = [0, 1, 2]  # first three distance bins

    det_specs = [
        ("ld_det", "corrected"),
        ("ld_det_inf", "uncorrected"),
    ]
    ploidies = [(1, "haploid"), (2, "diploid")]

    figs = []
    for ploidy_val, ploidy_name in ploidies:
        ploidy_combos = [c for c in combos if c["ploidy"] == ploidy_val]
        if not ploidy_combos:
            continue
        # Haploid finite-sample correction is not implemented; only plot uncorrected.
        specs = (
            [("ld_det_inf", "uncorrected")]
            if ploidy_val == 1
            else det_specs
        )
        for _det_key, det_name in specs:
            fig, ax = plt.subplots(
                figsize=(SINGLE_COL, SINGLE_COL * 0.75),
                dpi=300,
                constrained_layout=True,
            )
            for c in ploidy_combos:
                _ratio, _ratio_lo, _ratio_hi = c["ratios"][_det_key]
                for j, b in enumerate(bin_indices):
                    color = f"C{j}"
                    bin_label = f"u $\\in$ [{left_bins[b]:.2g}, {right_bins[b]:.2g}]"
                    ax.plot(
                        c["Ne"],
                        _ratio[:, b],
                        color=color,
                        lw=1.2,
                        marker="o",
                        markersize=1,
                        label=f"{c['label']}, {bin_label}",
                    )
                    ax.fill_between(
                        c["Ne"],
                        _ratio_lo[:, b],
                        _ratio_hi[:, b],
                        color=color,
                        alpha=0.15,
                        linewidth=0,
                    )
            ax.axhline(1, color="k", lw=0.6, ls="-", alpha=0.5)
            ax.axvline(10_000, color="k", lw=0.6, ls=":", alpha=0.5)
            ax.set_xscale("log")
            ax.set_yscale("log")
            # Shared scale for haploid and diploid-corrected panels.
            if ploidy_val == 1 or det_name == "corrected":
                ax.set_xlim(100, 100_000)
                ax.set_ylim(0.3, 1.1)
            ax.set_xlabel(r"$N_e$")
            ax.set_ylabel(r"Approximation ratio (closed-form / Monte Carlo)")
            # ax.set_title(f"{ploidy_name}, {det_name}", fontsize="small")
            ax.legend(fontsize="x-small")

            if mo.app_meta().mode == "script":
                stem = f"error_constant_{ploidy_name}_{det_name}"
                fig.savefig(f"{stem}.pdf")
                fig.savefig(f"{stem}.pgf")
            figs.append(fig)
    figs
    return


if __name__ == "__main__":
    app.run()
