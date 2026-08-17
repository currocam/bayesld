#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "arviz-stats==1.1.0",
#     "marimo",
#     "numpy==2.4.4",
#     "xarray==2026.7.0",
# ]
# ///

import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")

with app.setup:
    import csv
    import pickle
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import xarray  # noqa: F401  (needed to unpickle the InferenceData objects)

    GROUPS = {
        "approximate": "idatas_no_bias",
        "corrected": "idatas_corrected",
    }

    # Some runs name the two populations log_Ne1/log_Ne2 in the posterior.
    POSTERIOR_ALIASES = {"log_Ne_c": "log_Ne1", "log_Ne_a": "log_Ne2"}


@app.cell
def _():
    pkl_path = mo.cli_args().get("pkl", "results/sbc/constant/constant_high.pkl")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    scenario_name = Path(pkl_path).stem
    is_constant = "ne_draws" in data
    return data, is_constant, scenario_name


@app.function
def variables(data, is_constant):
    """Map each SBC variable to its ground-truth prior draws."""
    if is_constant:
        return {"log_Ne": np.log(data["ne_draws"])}
    return {
        "log_Ne_c": np.log(data["ne1_draws"]),
        "log_Ne_a": np.log(data["ne2_draws"]),
        "log_t0": np.log(data["t0_draws"]),
    }


@app.function
def posterior_draws(idata, var_name):
    post = idata.posterior
    name = var_name if var_name in post else POSTERIOR_ALIASES[var_name]
    return post[name].values.flatten()


@app.function
def recovery_stats(data, group, is_constant):
    """Correlation and error between the posterior mean and the ground truth."""
    idatas = data[group]
    rows = []
    for var_name, truth in variables(data, is_constant).items():
        truth = np.asarray(truth)
        estimate = np.array(
            [posterior_draws(idata, var_name).mean() for idata in idatas]
        )
        truth = truth[: len(estimate)]
        residual = estimate - truth
        rows.append(
            {
                "variable": var_name,
                "n": len(estimate),
                "correlation": float(np.corrcoef(truth, estimate)[0, 1]),
                "r2": 1 - float(np.sum(residual**2) / np.sum((truth - truth.mean()) ** 2)),
                "rmse": float(np.sqrt(np.mean(residual**2))),
                "mae": float(np.mean(np.abs(residual))),
                "bias": float(np.mean(residual)),
            }
        )
    return rows


@app.cell
def _(data, is_constant, scenario_name):
    table = [
        {"scenario": scenario_name, "group": _label, **_row}
        for _label, _group in GROUPS.items()
        if _group in data
        for _row in recovery_stats(data, _group, is_constant)
    ]
    return (table,)


@app.cell
def _(scenario_name, table):
    _out = Path(f"sbc_stats_{scenario_name}.csv")
    with _out.open("w", newline="") as _f:
        _writer = csv.DictWriter(_f, fieldnames=list(table[0]))
        _writer.writeheader()
        _writer.writerows(table)
    return


@app.cell
def _(table):
    mo.ui.table(table, selection=None)
    return


if __name__ == "__main__":
    app.run()

