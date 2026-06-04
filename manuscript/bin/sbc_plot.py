#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "arviz>=0.23.4",
#     "arviz-plots==0.7.0",
#     "marimo",
#     "matplotlib==3.10.9",
#     "numpy==2.4.4",
#     "pandas==3.0.2",
#     "xarray==2026.4.0",
# ]
# ///

import marimo

__generated_with = "0.23.5"
app = marimo.App(width="medium")

with app.setup:
    import pickle
    from pathlib import Path

    import arviz as az
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import xarray as xr
    from arviz_plots import plot_ecdf_pit

    # Theme settings
    plt.style.use(Path(__file__).parent / "theme.mplstyle")
    plt.rc("figure", autolayout=True)

    ONE_MM = 1 / 25.4
    SINGLE_COL = 85 * ONE_MM
    DOUBLE_COL = SINGLE_COL * 2
    ONE_HALF_COL = SINGLE_COL * 1.5


@app.cell
def _():
    pkl_path = mo.cli_args().get("pkl", "results/sbc/constant/constant_high.pkl")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    scenario_name = Path(pkl_path).stem
    return data, scenario_name


@app.cell
def _():
    method = "envelope"
    return (method,)


@app.function
def sbc_base(data, group):
    ne1_draws = data["ne1_draws"]
    ne2_draws = data["ne2_draws"]
    t0_draws = data["t0_draws"]

    prior_draws_dict = {
        "log_Ne_c": np.log(ne1_draws),
        "log_Ne_a": np.log(ne2_draws),
        "log_t0": np.log(t0_draws),
    }

    idatas = data[group]
    ranks_dict = {}

    for var_name, prior_draws in prior_draws_dict.items():
        _ranks = []
        for idata, prior_draw in zip(idatas, prior_draws):
            if not "log_Ne_c" in idata.posterior:
                idata.posterior["log_Ne_c"] = idata.posterior["log_Ne1"]
                idata.posterior["log_Ne_a"] = idata.posterior["log_Ne2"]
            post_draws = idata.posterior[var_name].values.flatten()
            _ranks.append(np.sum(post_draws < prior_draw))
        _arr = np.array(_ranks)[np.newaxis, :]  # (chain=1, draw=n_sim)
        ranks_dict[var_name] = xr.DataArray(_arr, dims=["chain", "draw"])

    return xr.DataTree(
        xr.Dataset(ranks_dict),
        name="prior_sbc",
    )


@app.function
def sbc_constant(data, group):
    ne_draws = data["ne_draws"]

    prior_draws_dict = {
        "log_Ne": np.log(ne_draws),
    }
    idatas = data[group]
    ranks_dict = {}

    for var_name, prior_draws in prior_draws_dict.items():
        _ranks = []
        for idata, prior_draw in zip(idatas, prior_draws):
            post_draws = idata.posterior[var_name].values.flatten()
            _ranks.append(np.sum(post_draws < prior_draw))
        _arr = np.array(_ranks)[np.newaxis, :]  # (chain=1, draw=n_sim)
        ranks_dict[var_name] = xr.DataArray(_arr, dims=["chain", "draw"])

    return xr.DataTree(
        xr.Dataset(ranks_dict),
        name="prior_sbc",
    )


@app.cell
def _(data, method, scenario_name):
    is_constant = "ne_draws" in data
    sbc_fn = sbc_constant if is_constant else sbc_base

    groups = {
        "approximate": "idatas_no_bias",
        "corrected": "idatas_corrected",
    }

    for label, group in groups.items():
        if group not in data:
            continue
        _pc = plot_ecdf_pit(
            sbc_fn(data, group),
            group="prior_sbc",
            method=method,
            visuals={"title": {"text": label}},
        )
        _pc.savefig(f"sbc_{scenario_name}_{label}.pdf")
    return


if __name__ == "__main__":
    app.run()
