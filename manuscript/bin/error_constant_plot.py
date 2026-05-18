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
    ONE_HALF_COL = SINGLE_COL * 1.5


@app.cell
def _():
    arg = mo.cli_args().get("pkl", "error_constant.pkl")
    pkl_paths = arg if isinstance(arg, list) else [arg]

    combos = []
    for p in pkl_paths:
        with gzip.open(p, "rb") as f:
            d = pickle.load(f)
        params = d["params"]
        ne_values = np.asarray([r["Ne"] for r in d["results"]])
        ld_det = np.stack([np.asarray(r["ld_det"]) for r in d["results"]])
        ld_mc = np.stack([np.asarray(r["ld_mc"]) for r in d["results"]])
        rel_err = np.mean(np.abs(ld_det / ld_mc - 1.0), axis=1)
        combos.append(
            {
                "label": f"r={params['recombination_rate']:.0e}, n={params['num_samples']}",
                "recombination_rate": params["recombination_rate"],
                "num_samples": params["num_samples"],
                "Ne": ne_values,
                "rel_err": rel_err,
            }
        )

    combos.sort(key=lambda c: (c["recombination_rate"], c["num_samples"]))
    return (combos,)


@app.cell
def _(combos):
    fig, ax = plt.subplots(
        figsize=(ONE_HALF_COL, ONE_HALF_COL * 0.75),
        dpi=300,
        constrained_layout=True,
    )

    for c in combos:
        ax.plot(c["Ne"], c["rel_err"], "o-", markersize=3, lw=1.2, label=c["label"])

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Effective population size $N_e$")
    ax.set_ylabel(
        "Mean relative error in LD\n" r"$\langle |\hat{LD}_{det}/LD_{MC} - 1| \rangle$"
    )
    ax.legend(title="", fontsize="small")

    if mo.app_meta().mode == "script":
        fig.savefig("error_constant.pdf")
    fig
    return


if __name__ == "__main__":
    app.run()
