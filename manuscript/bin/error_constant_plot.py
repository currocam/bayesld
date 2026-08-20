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

    def hapne_like_prediction(Ne, ploidy, left_bins, right_bins, sample_size):
        Ne = np.asarray(Ne, dtype=float) / 2 * ploidy
        u_1 = np.asarray(left_bins, dtype=float)
        u_2 = np.asarray(right_bins, dtype=float)
        d = u_2 - u_1

        ld = (
            7 * np.log1p(4 * Ne * d / (1 + 4 * Ne * u_1))
            - 3 * np.log1p(4 * Ne * d / (3 + 4 * Ne * u_1))
        ) / (16 * Ne * d) - 1 / (2 * (1 + 4 * Ne * u_1) * (1 + 4 * Ne * u_2))

        # Check that we get a close value if we use numerical integration
        # with the formula from the supp
        def _smcprime_pointwise(Ne, u):
            gamma = 1 / (2 * Ne)
            return (
                gamma
                * (3 * gamma**2 + 4 * u**2 + 10 * gamma * u)
                / ((gamma + 2 * u) ** 2 * (3 * gamma + 2 * u))
            )

        def _smcprime_binned(Ne, u_1, u_2, n_nodes=200):
            """(1 / (u_2 - u_1)) int_{u_1}^{u_2} S_1(u) du, by Gauss-Legendre."""
            x, w = np.polynomial.legendre.leggauss(n_nodes)
            mid = (u_1 + u_2) / 2
            half = (u_2 - u_1) / 2
            u = mid[..., None] + half[..., None] * x
            ne = Ne[..., None] if np.ndim(Ne) else Ne
            # (1 / (u_2 - u_1)) * half * sum(w f) = sum(w f) / 2
            return (_smcprime_pointwise(ne, u) * w).sum(axis=-1) / 2

        assert np.allclose(ld, _smcprime_binned(Ne, u_1, u_2))

        if sample_size is not None and ploidy == 2:
            n = 2 * sample_size
            b_n = 1 / (n - 1) ** 2
            a_n = ((n**2 - n + 2) ** 2) / ((n**2 - 3 * n + 2) ** 2)
            ld = (a_n - b_n) * ld + 4 * b_n
        return ld


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
        for det_key, sample_size in (
            ("ld_det", params["num_samples"]),
            ("ld_det_inf", None),
        ):
            ratio = np.empty((len(d["results"]), len(d["left_bins"])))
            ratio_lo = np.empty_like(ratio)
            ratio_hi = np.empty_like(ratio)
            for k, r in enumerate(d["results"]):
                ld_mc = np.asarray(r["ld_mc"], dtype=float)  # (n_rep, n_bins)
                ld_mc = np.where(ld_mc == 0, np.nan, ld_mc)
                # Make a fair comparison against the HapNe-LD predictions. 
                # so we compare their SMC' form against SMC' simulations. 
                ld_det = hapne_like_prediction(
                    r["Ne"], ploidy, d["left_bins"], d["right_bins"], sample_size
                )
                ratio[k] = ld_det / np.nanmean(ld_mc, axis=0)
                n_rep = ld_mc.shape[0]
                idx = rng.integers(0, n_rep, size=(BOOTSTRAP_DRAWS, n_rep))
                ld_mc_boot = np.nanmean(ld_mc[idx], axis=1)  # (B, n_bins)
                ratio_boot = ld_det / ld_mc_boot
                ratio_lo[k] = np.nanquantile(ratio_boot, 0.025, axis=0)
                ratio_hi[k] = np.nanquantile(ratio_boot, 0.975, axis=0)
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
    bin_specs = [
        (0, "C1", r"$u \in [0.5, 1.0]$ cM"),
        (2, "C0", r"$u \in [1.5, 2.0]$ cM"),
        (9, "C3", r"$u \in [5.0, 5.5]$ cM"),
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
    bins = [b for b, _, _ in bin_specs]
    bands = np.concatenate(
        [
            c["ratios"]["ld_det_inf"][i][:, bins]
            for c in (haploid, diploid)
            for i in (1, 2)
        ]
    )

    # Combined: left haploid uncorrected, right diploid uncorrected.
    fig_uncorr, axes = plt.subplots(
        1,
        2,
        figsize=(DOUBLE_COL, SINGLE_COL * 0.75),
        dpi=300,
        sharey=True,
        constrained_layout=True,
    )
    _plot_combo(axes[0], haploid, "ld_det_inf", legend=True)
    _plot_combo(axes[1], diploid, "ld_det_inf", legend=False)
    for ax in axes:
        ax.set_xlim(100, 50_000)
    axes[0].set_ylim(np.nanmin(bands), np.nanmax(bands))
    axes[0].set_title("Haploid")
    axes[0].set_xlabel(r"$2N_e$")
    axes[1].set_title("Diploid")
    fig_uncorr.supylabel(ylabel)

    # Diploid corrected, single panel.
    fig_corr, ax_corr = plt.subplots(
        figsize=(SINGLE_COL, SINGLE_COL * 0.75),
        dpi=300,
        constrained_layout=True,
    )
    _plot_combo(ax_corr, diploid, "ld_det", legend=True)
    ax_corr.set_xlim(100, 50_000)
    ax_corr.set_ylabel(ylabel)

    figs = [fig_uncorr, fig_corr]
    if mo.app_meta().mode == "script":
        fig_uncorr.savefig("error_constant_uncorrected.pdf")
        fig_uncorr.savefig("error_constant_uncorrected.pgf")
        fig_corr.savefig("error_constant_diploid_corrected.pdf")
        fig_corr.savefig("error_constant_diploid_corrected.pgf")
    figs
    return


if __name__ == "__main__":
    app.run()
