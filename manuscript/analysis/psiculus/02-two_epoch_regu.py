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
    from bayesld.models import PiecewiseConstantDemography

    plt.style.use("bin/theme.mplstyle")
    plt.rc("figure", autolayout=True)
    return Path, PiecewiseConstantDemography, az, np, pd, plt


@app.cell
def _(pd):
    data = pd.read_pickle("analysis/psiculus/PM_data.pkl")
    return (data,)


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
    return generation_time, mutation_rate, recombination_rate


@app.cell
def _(data, recombination_rate):
    window_length = int(data["right_bins_morgan"][-1] * 2 / recombination_rate)
    midpoints = (data["left_bins_morgan"] + data["right_bins_morgan"]) / 2
    window_length
    return midpoints, window_length


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

    return (plot_bias,)


@app.cell
def _(data, mutation_rate):
    data["mean_genetic_diversity"].mean() / 4 / mutation_rate
    return


@app.cell
def _():
    return


@app.cell
def _(
    Path,
    PiecewiseConstantDemography,
    data,
    mutation_rate,
    np,
    recombination_rate,
    window_length,
):
    if not Path("analysis/psiculus/steps/two_epoch_regu.nc").is_file():
        _parameters = """\
                real<offset=0> log_change;
                real<offset=log_ne_offset> log_Ne2;
                real<offset=5.7> log_t0;"""

        _transformed_parameters = """\
            real<lower=0> Ne1 = exp(log_Ne2+log_change);
            real<lower=0> Ne2 = exp(log_Ne2);
            real<lower=0> t0  = exp(log_t0);
            vector[2] Ne_values = [Ne1, Ne2]';
            vector[1] t_boundaries = [t0]';"""

        _prior_str = (
            f"    log_change ~ normal(0.0, 0.5);\n"
            f"    log_Ne2 ~ normal({np.log(100_000)}, 1);\n"
            f"    log_t0  ~ normal({np.log(300)}, 1.5);"
        )

        model =  PiecewiseConstantDemography(
                diversity=data["mean_genetic_diversity"],
                ld=data["mean_linkage_disequilibrium"],
                mutation_rate=mutation_rate,
                recombination_rate=recombination_rate,
                num_samples=data["sample_size"],
                left_bins=data["left_bins_morgan"],
                right_bins=data["right_bins_morgan"],
                sequence_length=int(window_length),
                num_workers=8,
                n_epochs=2,
                parameters=_parameters,
                transformed_parameters=_transformed_parameters,
                prior=_prior_str,
            )
        model.active_learn_bias(
                n_points_per_iter=20, n_iter=6, strategy="pathfinder",
                min_replicates=20, max_tolerance=0.1
            )
    return (model,)


@app.cell
def _(Path, az, model):
    if not Path("analysis/psiculus/steps/two_epoch_regu.nc").is_file():
        idata = model.sample(iter_warmup=5000, iter_sampling=5000)
        idata.to_netcdf("analysis/psiculus/steps/two_epoch_regu.nc")
    else:
        idata = az.from_netcdf("analysis/psiculus/steps/two_epoch_regu.nc")
    return (idata,)


@app.cell
def _(az, idata):
    az.summary(idata)
    return


@app.cell
def _(az, idata):
    az.plot_dist(idata, var_names=["Ne1", "Ne2", "t0"])
    return


@app.cell
def _(data, idata, plt, posterior_predictive):
    _fig, _axes = posterior_predictive(idata, data)
    _fig.tight_layout()
    plt.savefig("analysis/psiculus/steps/two_epoch_ppc.pdf")
    plt.show()
    return


@app.cell
def _(idata, plot_bias):
    plot_bias(idata)
    return


@app.cell
def _(np):
    rng = np.random.default_rng(0)
    n_prior = 20_000

    # sample priors in the model's parametrization
    log_change = rng.normal(0.0, 0.5, n_prior)
    log_Ne2    = rng.normal(np.log(100_000), 1.0, n_prior)
    log_t0     = rng.normal(np.log(300), 1.5, n_prior)

    # transform to the same variables that live in the posterior
    prior_dict = {
      "log_change": log_change,
      "log_Ne2":    log_Ne2,
      "log_t0":     log_t0,
      "Ne1": np.exp(log_Ne2 + log_change),
      "Ne2": np.exp(log_Ne2),
      "t0":  np.exp(log_t0),
    }
    return (prior_dict,)


@app.cell
def _():
    return


@app.cell
def _(az, generation_time, idata, plt):
    plt.figure(dpi=400)
    max_t0 = float(az.extract(idata, var_names="t0").max())
    for _i in range(2000):
      _Ne_c = float(az.extract(idata, var_names="Ne1")[_i])
      _Ne_a = float(az.extract(idata, var_names="Ne2")[_i])
      _t0 = float(az.extract(idata, var_names="t0")[_i])
      plt.step([0, _t0, max_t0 * 2], [_Ne_c, _Ne_a, _Ne_a], where="post", color="C0", alpha=0.05)
    plt.plot([], [], color = "C0", label = "Posterior")
    plt.yscale("log")
    #plt.xscale("log")
    plt.xlim(1, 100)
    plt.axvline(
        (2024-1971) / generation_time,
        linestyle = "--", color = "black",
        label = "Time of introduction"
    )
    plt.legend()
    #plt.ylim(2e4, 2e5)
    plt.xlabel("Generations ago")
    plt.ylabel("Effective population size")
    plt.savefig("analysis/psiculus/steps/two_epoch_posterior.pdf")
    plt.show()
    return (max_t0,)


@app.cell
def _(prior_dict):
    prior_dict
    return


@app.cell
def _(max_t0, plt, prior_dict):
    plt.figure(dpi=400)
    for _i in range(4_000):
      _Ne_c = float(prior_dict["Ne1"][_i])
      _Ne_a = float(prior_dict["Ne2"][_i])
      _t0 = float(prior_dict["t0"][_i])
      plt.step([0, _t0, max_t0 * 2], [_Ne_c, _Ne_a, _Ne_a], where="post", color="C7", alpha=0.05)
    plt.plot([], [], color = "C7", label = "Prior")
    plt.yscale("log")
    #plt.xscale("log")
    plt.xlim(1, 500)
    plt.legend()
    #plt.ylim(2e4, 2e5)
    plt.xlabel("Generations ago")
    plt.ylabel("Effective population size")
    plt.savefig("analysis/psiculus/steps/two_epoch_prior.pdf")
    plt.show()
    return


@app.cell
def _():
    return


@app.cell
def _():
    from labellines import labelLine, labelLines

    return


@app.cell
def _():
    return


@app.cell
def _(az, idata, np, plt, prior_dict, sns):
    fig, ax = plt.subplots(figsize=(8, 5), dpi = 300)

    posterior = az.extract(idata, var_names="log_change").values.ravel()
    prior = np.asarray(prior_dict["log_change"]).ravel()

    # Prior
    sns.kdeplot(prior, ax=ax, color="C7", linewidth=1.5, label="Prior")
    # Posterior
    sns.kdeplot(posterior, ax=ax, color="C0", linewidth=2, label="Posterior")

    # Reference line: no change (log-ratio = 0)
    ax.axvline(0, color="0.3", linestyle="--", linewidth=1, zorder=0)
    ax.text(0, ax.get_ylim()[1] * 0.98, " no change",
          rotation=90, va="top", ha="left", color="0.3", fontsize=16)
    ax.set_xlabel(r"Log-ratio  $\log N_c - \log N_a$")
    ax.set_ylabel("Density")
    ax.set_xlim(-8, 8)
    ax.legend(frameon=False)
    sns.despine()
    fig.tight_layout()
    plt.savefig("analysis/psiculus/steps/two_epoch_ratio.pdf")
    plt.show()
    return


if __name__ == "__main__":
    app.run()
