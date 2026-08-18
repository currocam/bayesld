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
#     "matplotlib-label-lines==0.8.1",
# ]
# requires-python = ">=3.12"
#
# [tool.uv.sources]
# bayesld = { git = "https://github.com/currocam/bayesld.git", rev = "94f88fd" }
# ///

import marimo

__generated_with = "0.23.15"
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

    plt.style.use("bin/theme.mplstyle")
    plt.rc("figure", autolayout=True)
    return az, np, pd, plt


@app.cell
def _():
    from bayesld import inference

    return (inference,)


@app.cell
def _(pd):
    data = pd.read_pickle("analysis/psiculus/PM_data.pkl")
    return (data,)


@app.cell
def _(np):
    grid = np.unique(np.round(np.geomspace(1, 1000, num=25)).astype(int))
    dt = np.diff(np.append([0], grid))
    sigmas = 0.05 * np.sqrt(dt)
    grid, sigmas
    return grid, sigmas


@app.cell
def _(grid):
    len(grid)
    return


@app.cell
def _(
    data,
    grid,
    inference,
    mutation_rate,
    np,
    recombination_rate,
    sigmas,
    window_length,
):
    model = inference.RandomWalk(grid=grid).with_data(
        mean_diversity=data["mean_genetic_diversity"],
        mean_ld=data["mean_linkage_disequilibrium"],
        left_bins=data["left_bins_morgan"],
        right_bins=data["right_bins_morgan"],
        recombination_rate=recombination_rate, 
        mutation_rate=mutation_rate, 
        num_samples=data["sample_size"],
        sequence_length=window_length
    ).with_prior(
        mu_log_ne=np.log(100000),
        sigma_log_ne=1,
        sigma_step=sigmas
    )
    return (model,)


@app.cell
def _():
    import seaborn as sns

    return (sns,)


@app.cell
def _():
    # From https://academic.oup.com/mbe/article/39/1/msab311/6413643
    generation_time = 2
    mutation_rate_per_year = 1.98e-9
    mutation_rate = mutation_rate_per_year * generation_time
    recombination_rate = 1.59e-8
    return mutation_rate, recombination_rate


@app.cell
def _(data, recombination_rate):
    window_length = int(data["right_bins_morgan"][-1] * 2 / recombination_rate)
    midpoints = (data["left_bins_morgan"] + data["right_bins_morgan"]) / 2
    window_length
    return (window_length,)


@app.cell
def _(data, mutation_rate, np):
    np.mean(data["mean_genetic_diversity"] / 4 / mutation_rate)
    return


@app.cell
def _(model):
    prior = model.sample_prior()
    prior
    #az.plot_trace(prior, var_names=["Ne_c", "t0", "Ne_a"])
    return (prior,)


@app.cell
def _(az, model, prior):
    model.plot_demography(az.extract(prior, num_samples = 100))
    return


@app.cell
def _(model):
    def train(seed):
        m = model
        for i in range(4):
            m = m.active_learning_round(
                25, rtol=0.1, min_replicates=20,
                seed=seed+i, num_workers=8,
                verbose=True, #method = "nuts"
            )
        return m

    return (train,)


@app.cell
def _(train):
    trained1 = train(9218379)
    return (trained1,)


@app.cell
def _(train):
    trained2 = train(218379)
    return (trained2,)


@app.cell
def _(trained1):
    trace1 = trained1.sample(
        tune=4000, draws=16000,
        chains=4, num_workers=4,
        parallel_chains=4,
        verbose=True,
        seed = 32668
    )
    return (trace1,)


@app.cell
def _(trained2):
    trace2 = trained2.sample(
        tune=4000, draws=16000,
        chains=4, num_workers=4,
        parallel_chains=4,
        verbose=True,
        seed = 32666
    )
    return (trace2,)


@app.function
def concat(idata1, idata2):
    import xarray as xr
    n_chains1 = idata1["posterior"].sizes["chain"]
    groups = {}
    for name, node in idata1.children.items():
        ds1 = node.to_dataset()
        ds2 = idata2[name].to_dataset()

        if "chain" in ds1.dims:
            ds2 = ds2.assign_coords(chain=ds2.coords["chain"].values + n_chains1)
            groups[name] = xr.concat([ds1, ds2], dim="chain")
        else:
            # constant_data, observed_data, etc. don't vary by chain — keep as-is
            groups[name] = ds1

    return xr.DataTree.from_dict(groups)


@app.cell
def _(trace1, trace2):
    idata = concat(trace1, trace2)
    return (idata,)


@app.cell
def _(idata, prior):
    idata["prior"] = prior["posterior"]
    return


@app.cell
def _(az, idata):
    az.plot_trace(idata, var_names=["Ne_values"])
    return


@app.cell
def _(az, idata):
    az.plot_convergence_dist(idata)
    return


@app.cell
def _(az, idata, model, plt):
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    ax = model.plot_demography(az.extract(idata, num_samples=100), ax = ax)
    ax.axvline(
        (2022-1972) / 2,
        linestyle = "--", color = "black",
        label = "Time of introduction"
        )

    plt.xscale("log")
    plt.show()
    return


@app.cell
def _(np, plt, sns):
    def plot_posterior_predictive(idata, colors=("black", "C0")):
        obs = idata["observed_data"]
        ppc = idata["posterior_predictive"]
        midpoints = idata["constant_data"]["midpoint"]
        color_obs, color_pred = colors

        fig, (ax_pi, ax_ld) = plt.subplots(1, 2, figsize=(11, 4))

        post_pi = ppc["observed_pi"].stack(sample=("chain", "draw"))
        pred_ld = ppc["observed_ld"].mean(dim="window").stack(sample=("chain", "draw"))
        n_post = post_pi.sizes["sample"]
        sample_idx = np.linspace(0, n_post - 1, min(50, n_post), dtype=int)

        for i, idx in enumerate(sample_idx):
            sns.kdeplot(
                post_pi.isel(sample=idx),
                color=color_pred,
                alpha=0.3,
                linewidth=1,
                ax=ax_pi,
                cut=0,
                label="Predicted" if i == 0 else None,
            )
        sns.kdeplot(obs["observed_pi"], color=color_obs, ax=ax_pi, cut=0, label="Observed")
        pi_vals = np.concatenate(
            [obs["observed_pi"].values, post_pi.isel(sample=sample_idx).values.ravel()]
        )
        ax_pi.set_xticks([])
        ax_pi.set_xlabel(r"Genetic diversity ($\pi$)")
        ax_pi.legend()

        for i, idx in enumerate(sample_idx):
            ax_ld.plot(
                midpoints,
                pred_ld.isel(sample=idx),
                alpha=0.1,
                color=color_pred,
                label="Predicted" if i == 0 else None,
            )
        ax_ld.plot(midpoints, obs["mean_ld"], color=color_obs, label="Observed")
        ax_ld.set_xlabel("Genetic distance (Morgan)")
        ax_ld.set_ylabel(r"Linkage disequilibrium ($\mathbb{E}[X_i X_j Y_i Y_j]$)")
        ax_ld.legend()

        fig.tight_layout()
        return fig

    return (plot_posterior_predictive,)


@app.cell
def _(idata, plot_posterior_predictive):
    plot_posterior_predictive(idata)
    return


@app.cell
def _(idata):
    idata.to_netcdf("results/psiculus/random_walk.nc")
    return


if __name__ == "__main__":
    app.run()
