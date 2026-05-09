# /// script
# dependencies = [
#     "arviz==1.1.0",
#     "marimo",
#     "matplotlib==3.10.9",
#     "numpy==2.4.4",
#     "pandas==3.0.2",
#     "seaborn==0.13.2",
#     "statsmodels==0.14.6",
#     "arviz-plots==1.1.0",
#     "msprime==1.4.1",
#     "daiquiri==3.4.0",
#     "bayesld==0.1.0",
#     "h5netcdf==1.8.1",
#     "h5py==3.16.0",
#     "netcdf4==1.7.4",
# ]
# requires-python = ">=3.14"
#
# [tool.uv.sources]
# bayesld = { git = "https://github.com/currocam/bayesld.git", rev = "69d8986a03414fc3eb8f09ff64cac0f2025c7515" }
# ///

import marimo

__generated_with = "0.23.5"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import bayesld
    import arviz as az
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import pandas as pd

    # Theme settings
    plt.style.use("bin/theme.mplstyle")
    plt.rc("figure", autolayout=True)
    return az, mo, np, pd, plt, sns


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Data

    Data consist in large-windowed statistics:
    """)
    return


@app.cell
def _(pd):
    df_seq = pd.read_csv("data/psiculus/sequence_report.tsv", sep = "\t")
    df_seq = df_seq[df_seq["GenBank seq accession"].str.startswith("OZ")]
    df_seq
    return (df_seq,)


@app.cell
def _(pd):
    data = pd.read_pickle("analysis/psiculus/PM_data.pkl")
    data
    return (data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Note: last 3 chromosomes are chromosomes W, X and MIT.
    """)
    return


@app.cell
def _(data, df_seq, np, plt):
    def plot_windows():
        chroms = df_seq["GenBank seq accession"].to_list()
        lengths = df_seq["Seq length"].to_list()

        fig, ax = plt.subplots(figsize=(10, len(df_seq) * 0.5 + 1))
        windows = data["windows"]
        for i, (chrom, length) in enumerate(zip(chroms, lengths)):
            ax.barh(i, length, height=0.6, color="lightgray", edgecolor="black", linewidth=0.5)
            sel = np.where(windows[:, 0]==chrom)[0]
            for j in sel:        
                ax.barh(
                    i, windows[j, 2] - windows[j, 1],
                    left=windows[j, 1], height=0.6,
                    color="steelblue", edgecolor="black",
                    linewidth=0.5
                )
        ax.set_yticks(range(len(chroms)))
        ax.set_yticklabels(chroms)
        ax.set_xlabel("Position (bp)")
        ax.invert_yaxis()
        fig.tight_layout()
        return fig
    plot_windows()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Chromosome 18 (OZ076857.1), which is extremely small, experiences a very high effective population size compared to the rest of the genome. Maybe, this is because of the effect of background selection but, arguably, the chromosome is _too_ small to be used in this analysis.
    """)
    return


@app.cell
def _(data, sns):
    sns.boxplot(
        x = data["mean_genetic_diversity"],
        y = data["windows"][:, 0]
    )
    return


@app.cell
def _(data):
    midpoints = (data["left_bins_morgan"]+data["right_bins_morgan"]) / 2
    return (midpoints,)


@app.cell
def _(data):
    mask = data["windows"][:, 0] != "OZ076857.1"
    return


@app.cell
def _(data, midpoints, plt):
    _mean = data["mean_linkage_disequilibrium"].mean(axis=0)
    _std = data["mean_linkage_disequilibrium"].std(axis=0)

    plt.vlines(
        midpoints, _mean - _std, _mean + _std,
        capstyle="round", linewidth=8, alpha = 0.2
    )
    plt.plot(
        midpoints, _mean, 'o', markersize=7,label = "Observed"
    )
    plt.legend()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Constant model
    """)
    return


@app.cell
def _(data):
    # From https://academic.oup.com/mbe/article/39/1/msab311/6413643
    generation_time = 2
    mutation_rate_per_year = 1.98e-9
    mutation_rate = mutation_rate_per_year * generation_time
    recombination_rate = 1.59e-8
    window_length = data["right_bins_morgan"][-1]*2 / recombination_rate
    return mutation_rate, recombination_rate, window_length


@app.cell
def _():
    import msprime
    from bayesld import ConstantDemography

    return ConstantDemography, msprime


@app.cell
def _():
    from pathlib import Path

    return (Path,)


@app.cell
def _(
    ConstantDemography,
    Path,
    data,
    msprime,
    mutation_rate,
    recombination_rate,
    window_length,
):
    if not Path("analysis/psiculus/steps/constant.nc").is_file():
        constant_model = ConstantDemography(
            diversity = data["mean_genetic_diversity"],
            ld = data["mean_linkage_disequilibrium"],
            mutation_rate = mutation_rate,
            recombination_rate = recombination_rate,
            num_samples = data["sample_size"],
            left_bins = data["left_bins_morgan"],
            right_bins =  data["right_bins_morgan"],
            sequence_length = int(window_length),
            num_workers = 8
        )

    
        constant_model.active_learn_bias(
            n_points_per_iter=5,
            n_iter=1,
            strategy="pathfinder",
            model = msprime.SMCK(k=1)
        )
    return (constant_model,)


@app.cell
def _(Path, az, constant_model):
    if not Path("analysis/psiculus/steps/constant.nc").is_file():
        idata_constant_base = constant_model.sample(iter_warmup=5000, iter_sampling=5000)
        idata_constant_base.to_netcdf(
            "analysis/psiculus/steps/constant.nc",
            )
    else:
        idata_constant_base = az.from_netcdf("analysis/psiculus/steps/constant.nc")
    return (idata_constant_base,)


@app.cell
def _(az, idata_constant_base):
    az.summary(idata_constant_base)
    return


@app.cell
def _(idata_constant_base):
    idata_constant_base.posterior
    return


@app.cell
def _(data, idata_constant_base, midpoints, np, plt):
    def posterior_predictive(idata):
      fig, (ax_div, ax_ld) = plt.subplots(
          1, 2,
          figsize=(7, 3),
          gridspec_kw={"width_ratios": [1, 6]},
      )
      num_windows = data["mean_genetic_diversity"].shape[0]
      _mean = data["mean_genetic_diversity"].mean(axis=0)
      _std = data["mean_genetic_diversity"].std(axis=0) / np.sqrt(num_windows)
      ax_div.vlines(0, _mean - _std, _mean + _std, capstyle="round", linewidth=8, alpha=0.1, color="C0")
      ax_div.plot(0, _mean, 'o', markersize=7, color="C0")
      ax_div.plot(0, idata.posterior["expected_pi"].mean(dim=["chain", "draw"]), 'o', markersize=7, color="C1")
      ax_div.set_xticks([])
      ax_div.set_ylabel("Genetic diversity")
      ax_div.set_xlim(-0.5, 0.5)

      _mean = data["mean_linkage_disequilibrium"].mean(axis=0)
      _std = data["mean_linkage_disequilibrium"].std(axis=0) / np.sqrt(num_windows)
      _postpred = idata.posterior["corrected_ld"].mean(dim=["chain", "draw"])
      ax_ld.vlines(midpoints, _mean - _std, _mean + _std, capstyle="round", linewidth=8, alpha=0.1, color="C0")
      ax_ld.plot(midpoints, _mean, 'o', markersize=7, label="Observed", color="C0")
      ax_ld.vlines(midpoints, _postpred - _std, _postpred + _std, capstyle="round", linewidth=8, alpha=0.1, color="C1")
      ax_ld.plot(midpoints, _postpred, 'o', markersize=7, label="Predicted", color="C1")
      ax_ld.set_xlabel("Distance (Morgan)")
      ax_ld.set_ylabel(r"$\mathbb E[X_iX_jY_iY_j]$")
      ax_ld.legend()
      return fig, (ax_div, ax_ld)

    _fig, _axes = posterior_predictive(idata_constant_base)
    _fig.tight_layout()
    plt.show()
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
def _(idata_constant_base, plot_bias):
    plot_bias(idata_constant_base)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Piecewise constant model
    """)
    return


@app.cell
def _():
    from bayesld.models import PiecewiseConstantDemography

    return (PiecewiseConstantDemography,)


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
    if not Path("analysis/psiculus/steps/two_epoch.nc").is_file():
        _parameters = """\
            real log_Ne1;
            real log_Ne2;
            real log_t0;"""
    
        _transformed_parameters = """\
            real<lower=0> Ne1 = exp(log_Ne1);
            real<lower=0> Ne2 = exp(log_Ne2);
            real<lower=0> t0  = exp(log_t0);
            vector[2] Ne_values = [Ne1, Ne2]';
            vector[1] t_boundaries = [t0]';"""
    
        _prior_str = (
                f"    log_Ne1 ~ normal({np.log(1e5)}, 1);\n"
                f"    log_Ne2 ~ normal({np.log(1e5)}, 1);\n"
                f"    log_t0  ~ normal({np.log(200)}, 1);"
            )
    
        two_epoch_model = PiecewiseConstantDemography(
            diversity = data["mean_genetic_diversity"],
            ld = data["mean_linkage_disequilibrium"],
            mutation_rate = mutation_rate,
            recombination_rate = recombination_rate,
            num_samples = data["sample_size"],
            left_bins = data["left_bins_morgan"],
            right_bins =  data["right_bins_morgan"],
            sequence_length = int(window_length),
            num_workers = 8,
            n_epochs=2,
            parameters=_parameters,
            transformed_parameters=_transformed_parameters,
            prior=_prior_str,
        )
    
        two_epoch_model.active_learn_bias(
            n_points_per_iter=20,
            n_iter=5, strategy="pathfinder",
        )
    return (two_epoch_model,)


@app.cell
def _():
    import pickle

    return (pickle,)


@app.cell
def _(Path, az, pickle, two_epoch_model):
    # Save after active learning
    if not Path("analysis/psiculus/steps/two_epoch.nc").is_file():

        with open("analysis/psiculus/steps/two_epoch_bias.pkl", "wb") as f:
          pickle.dump(two_epoch_model.synthetic_points, f)

        idata_two_epoch_base = two_epoch_model.sample(
            iter_warmup=10_000, iter_sampling=10_000, adapt_delta=0.99
        )

        idata_two_epoch_base.to_netcdf(
        "analysis/psiculus/steps/two_epoch.nc",
        )
    else:
        idata_two_epoch_base = az.from_netcdf("analysis/psiculus/steps/two_epoch.nc")
    return (idata_two_epoch_base,)


@app.cell
def _():
    import arviz_plots as azp

    return (azp,)


@app.cell
def _():
    years_ago = 2024 - 1971
    generations_ago = years_ago / 2
    generations_ago
    return (generations_ago,)


@app.cell
def _(az, idata_two_epoch_base):
    az.plot_dist(idata_two_epoch_base, var_names=["Ne1", "Ne2"])
    return


@app.function
def add_prior(idata, n_draws=10_000):
      import numpy as np
      from arviz import dict_to_dataset

      log_Ne1 = np.random.normal(np.log(1e5), 1, n_draws)
      log_Ne2 = np.random.normal(np.log(1e5), 1, n_draws)
      log_t0 = np.random.normal(np.log(200), 1, n_draws)

      prior_samples = {
          "log_Ne1": log_Ne1[np.newaxis, :],
          "log_Ne2": log_Ne2[np.newaxis, :],
          "log_t0": log_t0[np.newaxis, :],
          "Ne1": np.exp(log_Ne1)[np.newaxis, :],
          "Ne2": np.exp(log_Ne2)[np.newaxis, :],
          "t0": np.exp(log_t0)[np.newaxis, :],
      }

      new = idata.copy()
      new["prior"] = dict_to_dataset(prior_samples)
      return new


@app.cell
def _(idata_two_epoch_base):
    idata_two_epoch = add_prior(idata_two_epoch_base)
    return (idata_two_epoch,)


@app.cell
def _():
    return


@app.cell
def _(idata_two_epoch_base):
    idata_two_epoch_base
    return


@app.cell
def _(azp, idata_two_epoch):
    azp.plot_prior_posterior(
        idata_two_epoch, var_names=["Ne1", "Ne2", "t0"], kind= "ecdf"
    )
    return


@app.cell
def _(az, generations_ago, idata_two_epoch_base, plt):
    az.plot_dist(idata_two_epoch_base, var_names="t0")
    plt.axvline(generations_ago, color = "black", linestyle = "--")
    return


@app.cell
def _(idata_two_epoch_base, plt, posterior_predictive):
    _fig, _axes = posterior_predictive(idata_two_epoch_base)
    _fig.tight_layout()
    plt.show()
    return


@app.cell
def _(idata_two_epoch_base, plot_bias):
    plot_bias(idata_two_epoch_base)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Piecewise exponential
    """)
    return


@app.cell
def _():
    from bayesld.models import PiecewiseExponentialDemography

    return (PiecewiseExponentialDemography,)


@app.cell
def _(
    Path,
    PiecewiseExponentialDemography,
    data,
    mutation_rate,
    np,
    recombination_rate,
    window_length,
):
    if not Path("analysis/psiculus/steps/invasion.nc").is_file():
        _parameters = """\
            real log_Ne_c;
            real log_Ne_a;
            real log_t0;
            real log_fold_change;"""
    
        _transformed_parameters = """\
            real<lower=0> Ne_c = exp(log_Ne_c);
            real<lower=0> Ne_a = exp(log_Ne_a);
            real<lower=0> t0  = exp(log_t0);
            real log_Ne_f  = log(Ne_c) - log_fold_change;
            real<lower=0> Ne_f = exp(log_Ne_f);
            real alpha = log_fold_change / t0;"""
    
        _prior_str = (
                f"    log_Ne_c ~ normal({np.log(1e5)}, 1);\n"
                f"    log_Ne_a ~ normal({np.log(1e5)}, 1);\n"
                f"    log_t0 ~ normal({np.log(200)}, 1);\n"
                f"    log_fold_change  ~ normal(0, 1);"
            )
    
        invasion_model = PiecewiseExponentialDemography(
            diversity = data["mean_genetic_diversity"],
            ld = data["mean_linkage_disequilibrium"],
            mutation_rate = mutation_rate,
            recombination_rate = recombination_rate,
            num_samples = data["sample_size"],
            left_bins = data["left_bins_morgan"],
            right_bins =  data["right_bins_morgan"],
            sequence_length = int(window_length),
            num_workers = 8,
            parameters=_parameters,
            transformed_parameters=_transformed_parameters,
            prior=_prior_str,
        )
    return (invasion_model,)


@app.cell
def _(Path, az, invasion_model, msprime, pickle):
    if not Path("analysis/psiculus/steps/invasion.nc").is_file():
        invasion_model.active_learn_bias(
            n_points_per_iter=20,
            model = msprime.SMCK(k=0),
            seed=7638172,
            n_iter=5, strategy="pathfinder",
        )

        with open("analysis/psiculus/steps/invasion_bias.pkl", "wb") as f3:
          pickle.dump(invasion_model.synthetic_points, f3)

        idata_invasion_base = invasion_model.sample(
            iter_warmup=10_000, iter_sampling=10_000, #adapt_delta=0.99
        )

        idata_invasion_base.to_netcdf(
            "analysis/psiculus/steps/invasion.nc",
        )
    else:
        idata_invasion_base = az.from_netcdf("analysis/psiculus/steps/invasion.nc")
    return (idata_invasion_base,)


@app.cell
def _(az, idata_invasion_base):
    az.plot_dist(idata_invasion_base, var_names=["Ne_c", "Ne_f","Ne_a"])
    return


@app.cell
def _(az, idata_invasion_base):
    az.summary(idata_invasion_base)
    return


@app.cell
def _(az, generations_ago, idata_invasion_base, plt):
    az.plot_dist(idata_invasion_base, var_names="t0")
    plt.axvline(generations_ago, color = "black", linestyle = "--")
    return


@app.cell
def _(idata_invasion_base, plt, posterior_predictive):
    _fig, _axes = posterior_predictive(idata_invasion_base)
    _fig.tight_layout()
    plt.show()
    return


@app.cell
def _(idata_invasion_base, plot_bias):
    plot_bias(idata_invasion_base)
    return


@app.function
def add_prior2(idata, n_draws=10_000):
      import numpy as np
      from arviz import dict_to_dataset

      log_Ne_c = np.random.normal(np.log(1e5), 1, n_draws)
      log_Ne_a = np.random.normal(np.log(1e5), 1, n_draws)
      log_t0 = np.random.normal(np.log(200), 1, n_draws)
      log_fold_change = np.random.normal(0, 1, n_draws)

      prior_samples = {
          "log_Ne_c": log_Ne_c[np.newaxis, :],
          "log_Ne_a": log_Ne_a[np.newaxis, :],
          "log_t0": log_t0[np.newaxis, :],
          "log_fold_change" : log_fold_change[np.newaxis, :],
          "Ne_c": np.exp(log_Ne_c)[np.newaxis, :],
          "Ne_a": np.exp(log_Ne_a)[np.newaxis, :],
          "t0": np.exp(log_t0)[np.newaxis, :],
      }

      new = idata.copy()
      new["prior"] = dict_to_dataset(prior_samples)
      return new


@app.cell
def _(idata_invasion_base):
    idata_invasion = add_prior2(idata_invasion_base)
    return (idata_invasion,)


@app.cell
def _(azp, idata_invasion):
    azp.plot_prior_posterior(
        idata_invasion, var_names="log_fold_change"
    )
    return


@app.cell
def _(azp, idata_invasion):
    azp.plot_prior_posterior(
        idata_invasion, var_names=["Ne_c", "Ne_a", "t0", "log_fold_change"], kind= "ecdf"
    )
    return


@app.cell
def _(idata_invasion):
    def filter_positive(idata, var, groups=("posterior", "prior")):
          import xarray as xr

          filtered = {}
          for group in groups:
              ds = idata[group].ds.stack(sample=("chain", "draw"))
              ds = ds.where(ds[var] > 0, drop=True)
              ds = ds.drop_vars(["chain",
      "draw"]).rename(sample="draw").expand_dims(chain=[0])
              filtered[group] = ds
          return xr.DataTree.from_dict(filtered)

    idata_invasion2= filter_positive(idata_invasion, "log_fold_change")
    idata_invasion2
    return (idata_invasion2,)


@app.cell
def _(idata_invasion2, plt, posterior_predictive):
    _fig, _axes = posterior_predictive(idata_invasion2)
    _fig.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Carrying capacity
    """)
    return


@app.cell
def _():
    from bayesld.models import ExponentialCarryingCapacityDemography

    return (ExponentialCarryingCapacityDemography,)


@app.cell
def _(
    ExponentialCarryingCapacityDemography,
    Path,
    data,
    mutation_rate,
    np,
    recombination_rate,
    window_length,
):
    if not Path("analysis/psiculus/steps/carrying.nc").is_file():
        _parameters = """\
            real log_Ne_c;
            real log_Ne_a;
            ordered[2] log_t_boundaries;
            real<lower=0> log_fold_change;"""
    
        _transformed_parameters = """\
            real<lower=0> Ne_c = exp(log_Ne_c);
            real<lower=0> Ne_a = exp(log_Ne_a);
            real<lower=0> t0   = exp(log_t_boundaries[1]);
            real<lower=0> t1   = exp(log_t_boundaries[2]);
            real alpha = log_fold_change / (t1 - t0);
            real log_Ne_f  = log(Ne_c) - log_fold_change;
            real<lower=0> Ne_f = exp(log_Ne_f);"""
    
        _prior_str = (
                f"    log_Ne_c ~ normal({np.log(1e5)}, 1);\n"
                f"    log_Ne_a ~ normal({np.log(1e5)}, 1);\n"
                f"    log_t_boundaries ~ normal({np.log(200)}, 1);\n"
                f"    log_fold_change  ~ normal(0, 1);"
            )
    
        carrying_model = ExponentialCarryingCapacityDemography(
            diversity = data["mean_genetic_diversity"],
            ld = data["mean_linkage_disequilibrium"],
            mutation_rate = mutation_rate,
            recombination_rate = recombination_rate,
            num_samples = data["sample_size"],
            left_bins = data["left_bins_morgan"],
            right_bins =  data["right_bins_morgan"],
            sequence_length = int(window_length),
            num_workers = 8,
            parameters=_parameters,
            transformed_parameters=_transformed_parameters,
            prior=_prior_str,
        )
    return (carrying_model,)


@app.cell
def _(Path, az, carrying_model, msprime, pickle):
    if not Path("analysis/psiculus/steps/carrying.nc").is_file():
        carrying_model.active_learn_bias(
            n_points_per_iter=5,
            model = msprime.SMCK(k=0),
            seed=7638172,
            n_iter=5, strategy="pathfinder",
        )

        with open("analysis/psiculus/steps/carrying_bias.pkl", "wb") as f4:
          pickle.dump(carrying_model.synthetic_points, f4)

        idata_carrying_base = carrying_model.sample(
            iter_warmup=10_000, iter_sampling=10_000, #adapt_delta=0.99
        )

        idata_carrying_base.to_netcdf(
            "analysis/psiculus/steps/carrying.nc",
        )
    else:
        idata_carrying_base = az.from_netcdf("analysis/psiculus/steps/carrying.nc")
    return


@app.cell
def _(carrying_model):
    idata_carrying_base2 = carrying_model.sample(
        iter_warmup=10_000, iter_sampling=10_000, adapt_delta=0.99
    )
    return


if __name__ == "__main__":
    app.run()
