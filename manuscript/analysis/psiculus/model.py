# /// script
# dependencies = [
#     "arviz==1.1.0",
#     "marimo",
#     "matplotlib==3.10.9",
#     "numpy==2.4.4",
#     "pandas==3.0.2",
#     "arviz-plots==1.1.0",
#     "msprime==1.4.1",
#     "daiquiri==3.4.0",
#     "bayesld==0.1.0",
#     "h5netcdf==1.8.1",
#     "h5py==3.16.0",
#     "netcdf4==1.7.4",
#     "seaborn==0.13.2",
# ]
# requires-python = ">=3.14"
#
# [tool.uv.sources]
# bayesld = { git = "https://github.com/currocam/bayesld.git", rev = "39990f4" }
# ///

import marimo

__generated_with = "0.23.10"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import arviz as az
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import msprime
    from pathlib import Path
    from bayesld.models import PiecewiseExponentialDemography

    plt.style.use("bin/theme.mplstyle")
    plt.rc("figure", autolayout=True)
    return PiecewiseExponentialDemography, az, mo, np, pd, plt


@app.cell
def _(pd):
    data = pd.read_pickle("analysis/psiculus/PM_data.pkl")
    return (data,)


@app.cell
def _(data):
    # From https://academic.oup.com/mbe/article/39/1/msab311/6413643
    generation_time = 2
    mutation_rate_per_year = 1.98e-9
    mutation_rate = mutation_rate_per_year * generation_time
    recombination_rate = 1.59e-8
    window_length = data["right_bins_morgan"][-1] * 2 / recombination_rate
    midpoints = (data["left_bins_morgan"] + data["right_bins_morgan"]) / 2
    return midpoints, mutation_rate, recombination_rate, window_length


@app.cell
def _(midpoints, np, plt):
    def posterior_predictive_stats(idata):
        posterior = idata.posterior

        n_chains, n_draws, D = posterior["mu_y"].shape
        n_total = n_chains * n_draws

        mu_y    = posterior["mu_y"].values.reshape(n_total, D)
        sigma_y = np.exp(posterior["log_sigma_y"].values.reshape(n_total, D))
        L_Omega = posterior["L_Omega"].values.reshape(n_total, D, D)

        # Law of total expectation/variance over the mixture
        # E[y] = E_θ[mu_y]
        pp_mean = mu_y.mean(axis=0)    
        # Var[y] = E_θ[Sigma_diag + mu_y²] - pp_mean²
        sigma_diag = np.einsum("ijk,ik->ij", L_Omega ** 2, sigma_y ** 2)  # diag of L_Sigma L_Sigma'
        pp_var = (sigma_diag + mu_y ** 2).mean(axis=0) - pp_mean ** 2
        return pp_mean, np.sqrt(pp_var)

    def posterior_predictive(idata, data):
        num_windows = data["mean_genetic_diversity"].shape[0]
        pp_mean, pp_std = posterior_predictive_stats(idata)
        pp_std = pp_std / np.sqrt(num_windows)
        fig, (ax_div, ax_ld) = plt.subplots(
          1, 2, figsize=(7, 3),
          gridspec_kw={"width_ratios": [1, 6]},
        )

        # ── Diversity ──────────────────────────────────────────────────────────────
        _mean = data["mean_genetic_diversity"].mean(axis=0)
        _std  = data["mean_genetic_diversity"].std(axis=0) / np.sqrt(num_windows)
        ax_div.vlines(0, _mean - _std, _mean + _std,
                    capstyle="round", linewidth=8, alpha=0.1, color="C0")
        ax_div.plot(0, _mean, "o", markersize=7, color="C0")

        ax_div.vlines(0, pp_mean[0] - pp_std[0], pp_mean[0] + pp_std[0],
                    capstyle="round", linewidth=8, alpha=0.1, color="C1")
        ax_div.plot(0, pp_mean[0], "o", markersize=7, color="C1")

        ax_div.set_xticks([])
        ax_div.set_ylabel("Genetic diversity")
        ax_div.set_xlim(-0.5, 0.5)

        # ── LD ─────────────────────────────────────────────────────────────────────
        _mean = data["mean_linkage_disequilibrium"].mean(axis=0)
        _std  = data["mean_linkage_disequilibrium"].std(axis=0) / np.sqrt(num_windows)
        ax_ld.vlines(midpoints, _mean - _std, _mean + _std,
                   capstyle="round", linewidth=8, alpha=0.1, color="C0")
        ax_ld.plot(midpoints, _mean, "o", markersize=7, label="Observed", color="C0")

        ax_ld.vlines(midpoints, pp_mean[1:] - pp_std[1:], pp_mean[1:] + pp_std[1:],
                   capstyle="round", linewidth=8, alpha=0.1, color="C1")
        ax_ld.plot(midpoints, pp_mean[1:], "o", markersize=7, label="Predicted", color="C1")

        ax_ld.set_xlabel("Distance (Morgan)")
        ax_ld.set_ylabel(r"$\mathbb{E}[X_i X_j Y_i Y_j]$")
        ax_ld.legend()

        return fig, (ax_div, ax_ld)

    return (posterior_predictive,)


@app.cell
def _(az, midpoints, plt):
    def plot_bias(idata):
        fig, ax = plt.subplots()
        _hdi = az.hdi(idata.posterior["gp_bias_ld"]).values
        ax.plot(midpoints, idata.posterior["gp_bias_ld"].mean(dim=["chain", "draw"]))
        ax.fill_between(midpoints, _hdi[:, 0], _hdi[:, 1], alpha=0.1)
        ax.axhline(-1.96 * 0.005, color="black", linestyle="--")
        ax.axhline(+1.96 * 0.005, color="black", linestyle="--")
        ax.set_ylabel("Estimated relative error in LD")
        ax.set_xlabel("Distance (Morgan)")
        return fig

    return


@app.cell
def _(data, mutation_rate):
    data["mean_genetic_diversity"].mean() / 4 / mutation_rate
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Model
    """)
    return


@app.cell
def _(
    PiecewiseExponentialDemography,
    data,
    mo,
    mutation_rate,
    np,
    recombination_rate,
    window_length,
):
    model = PiecewiseExponentialDemography(
        diversity=data["mean_genetic_diversity"],
        ld=data["mean_linkage_disequilibrium"],
        mutation_rate=mutation_rate,
        recombination_rate=recombination_rate,
        num_samples=data["sample_size"],
        left_bins=data["left_bins_morgan"],
        right_bins=data["right_bins_morgan"],
        sequence_length=int(window_length),
        num_workers=8,
        parameters = (
            f"real log_fold_change;\n"
            f"real log_Ne_f;\n"
            f"real<offset=log_ne_offset> log_Ne_a;\n"
            f"real log_t0;\n"
        ),
        transformed_parameters=(
            "real<lower=0> log_Ne_c = log_Ne_f + log_fold_change;\n"
            "real<lower=0> Ne_c = exp(log_Ne_c);\n"
            "real<lower=0> Ne_f = exp(log_Ne_f);\n"
            "real<lower=0> Ne_a = exp(log_Ne_a);\n"
            "real<lower=0> t0 = exp(log_t0);\n"
            "real alpha = log_fold_change / t0;\n"
        ),
        prior=(
            f"    log_Ne_a ~ normal({np.log(1e6):.4f}, 0.25);\n"
            f"    log_Ne_f ~ normal({np.log(1e6):.4f}, 6.0);\n"
            f"    log_fold_change ~ normal(0, 0.5);\n"
            f"    log_t0   ~ normal({np.log(100.0):.4f}, 2.0);"
        ),
    )
    mo.md("Model compiled.")
    return (model,)


@app.cell
def _(model):
    idata = model.sample()
    return


@app.cell
def _(mo, model):
    with mo.status.spinner("Active learning (bias correction)..."):
        model.active_learn_bias(
            n_points_per_iter=20, n_iter=6, strategy="pathfinder",
            min_replicates=20, max_tolerance=0.1
        )
    return


@app.cell
def _(mo, model):
    with mo.status.spinner("NUTS sampling (with bias correction)..."):
        idata_corrected = model.sample(
            iter_warmup=5000,
            iter_sampling=5000,
            seed=2,
        )
    mo.md("Corrected sampling complete.")
    return (idata_corrected,)


@app.cell
def _(idata_corrected):
    idata_corrected
    return


@app.cell
def _(az, idata_corrected):
    az.plot_dist(idata_corrected, var_names=["Ne_c", "Ne_f", "Ne_a", "t0", "log_fold_change"])
    return


@app.cell
def _(data, idata_corrected, plt, posterior_predictive):
    _fig, _axes = posterior_predictive(idata_corrected, data)
    _fig.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Trajectory
    """)
    return


@app.cell
def _():
    import seaborn as sns

    return


if __name__ == "__main__":
    app.run()
