#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "arviz-stats==1.1.0",
#     "marimo",
#     "matplotlib==3.10.9",
#     "numpy==2.4.4",
#     "xarray==2026.7.0",
# ]
# ///

import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")

with app.setup:
    import pickle
    from pathlib import Path

    import arviz_stats
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import xarray as xr
    from arviz_stats.ecdf_utils import ecdf_pit

    plt.style.use(Path(__file__).parent / "theme.mplstyle")
    plt.rc("figure", autolayout=True)
    import matplotlib as pl
    pl.rcParams['pgf.texsystem'] = "pdflatex"


    ONE_MM = 1 / 25.4
    SINGLE_COL = 85 * ONE_MM
    DOUBLE_COL = SINGLE_COL * 2

    # Fixed margins (inches) so axes can be sized exactly and match across figures.
    MARGIN = {"left": 0.60, "right": 0.08, "bottom": 0.45, "top": 0.25}
    PANEL_GAP = 0.45
    # Axes width such that a single-panel figure is exactly one column wide.
    PANEL_W = SINGLE_COL - MARGIN["left"] - MARGIN["right"]

    GROUPS = {
        "approximate": "idatas_no_bias",
        "corrected": "idatas_corrected",
    }
    GROUP_COLORS = {"approximate": "C0", "corrected": "C1"}

    VAR_LABELS = {
        "log_Ne_c": r"$\log (N_c)$",
        "log_Ne_a": r"$\log (N_a)$",
        "log_t0": r"$\log (t_0)$",
        "log_Ne": r"$\log (N_e)$",
    }

    # Some runs name the two populations log_Ne1/log_Ne2 in the posterior.
    POSTERIOR_ALIASES = {"log_Ne_c": "log_Ne1", "log_Ne_a": "log_Ne2"}


@app.cell
def _():
    pkl_path = mo.cli_args().get("pkl", "results/sbc/piecewise_exponential/piecewise_exponential_large.pkl")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    scenario_name = Path(pkl_path).stem
    is_constant = "ne_draws" in data
    return data, is_constant, scenario_name


@app.function
def variables(data, is_constant):
    """Map each SBC variable to its (prior draws, prior sigma)."""
    prior = data["prior"]
    if is_constant:
        return {"log_Ne": (np.log(data["ne_draws"]), prior["prior_sigma"])}
    return {
        "log_Ne_c": (np.log(data["ne1_draws"]), prior["prior_sigma_ne"]),
        "log_Ne_a": (np.log(data["ne2_draws"]), prior["prior_sigma_ne"]),
        "log_t0": (np.log(data["t0_draws"]), prior["prior_sigma_t0"]),
    }


@app.function
def posterior_draws(idata, var_name):
    post = idata.posterior
    name = var_name if var_name in post else POSTERIOR_ALIASES[var_name]
    return post[name].values.flatten()


@app.function
def panel_figure(ncols, panel_h, panel_w=PANEL_W):
    """Row of axes with an exact size in inches, so panels match across figures."""
    fig_w = MARGIN["left"] + ncols * panel_w + (ncols - 1) * PANEL_GAP + MARGIN["right"]
    fig_h = MARGIN["bottom"] + panel_h + MARGIN["top"]
    fig, axes = plt.subplots(1, ncols, figsize=(fig_w, fig_h), squeeze=False)
    fig.set_layout_engine("none")  # autolayout would override the fixed margins
    fig.subplots_adjust(
        left=MARGIN["left"] / fig_w,
        right=1 - MARGIN["right"] / fig_w,
        bottom=MARGIN["bottom"] / fig_h,
        top=1 - MARGIN["top"] / fig_h,
        wspace=PANEL_GAP / panel_w,
    )
    return fig, axes


@app.function
def savefig(fig, stem):
    for ext in ("pdf", "pgf"):
        fig.savefig(f"{stem}.{ext}", bbox_inches="tight")


@app.function
def sbc_stats(data, group, is_constant):
    """Per-variable SBC diagnostics across all simulated fits."""
    idatas = data[group]
    stats = {}
    for var_name, (truth, prior_sigma) in variables(data, is_constant).items():
        prior_var = prior_sigma**2
        rank, contraction, zscore, estimate = [], [], [], []
        for idata, prior_draw in zip(idatas, truth):
            post = posterior_draws(idata, var_name)
            mean, var = post.mean(), post.var()
            rank.append(np.sum(post < prior_draw))
            contraction.append(1 - var / prior_var)
            zscore.append((mean - prior_draw) / np.sqrt(var))
            estimate.append(mean)
        stats[var_name] = {
            "rank": np.array(rank),
            "contraction": np.array(contraction),
            "zscore": np.array(zscore),
            "estimate": np.array(estimate),
            "truth": np.asarray(truth),
        }
    return stats


@app.function
def ranks_dataset(stats):
    return xr.Dataset(
        {
            var: xr.DataArray(d["rank"][np.newaxis, :], dims=["chain", "draw"])
            for var, d in stats.items()
        }
    )


@app.function
def ecdf_pit_stats(ranks_ds, envelope_prob=0.99):
    """Δ-ECDF curves and a simultaneous confidence envelope (arviz ``plot_ecdf_pit``)."""
    sample_dims = ["chain", "draw"]
    sample_size = ranks_ds.sizes["draw"]
    distribution = (ranks_ds + 0.5) / (ranks_ds.max() + 1)
    dt_ecdf = distribution.azstats.ecdf(dim=sample_dims, pit=True, npoints=sample_size)

    # Simultaneous confidence band as Δ-ECDF (subtract the reference CDF).
    # I'm not sure why 1000, but it is how it's done in the Arviz source code
    x_ci, _, lower_ci, upper_ci = ecdf_pit(
        np.linspace(0, 1, sample_size), envelope_prob, n_simulations=1000
    )
    envelope = (x_ci, lower_ci - x_ci, upper_ci - x_ci)

    envelope_max = float(np.abs(envelope[1:]).max())
    epsilon = {
        var: max(envelope_max, float(np.abs(dt_ecdf[var].sel(plot_axis="y")).max()))
        for var in ranks_ds.data_vars
    }
    return dt_ecdf, envelope, epsilon


@app.cell
def _(data, is_constant):
    stats = {
        _label: sbc_stats(data, _group, is_constant)
        for _label, _group in GROUPS.items()
        if _group in data
    }
    return (stats,)


@app.cell
def _(scenario_name, stats):
    _ecdf = {_label: ecdf_pit_stats(ranks_dataset(_s)) for _label, _s in stats.items()}
    _var_names = list(next(iter(_ecdf.values()))[0].data_vars)

    _fig, _axes = panel_figure(len(_var_names), PANEL_W * 0.8)
    for _ax, _var in zip(_axes[0], _var_names):
        _epsilon = max(_eps[_var] for _, _, _eps in _ecdf.values())
        # Confidence envelope is identical across groups; draw it once.
        _x_ci, _lower, _upper = next(iter(_ecdf.values()))[1]
        _ax.fill_between(
            _x_ci, _lower, _upper, step="post", color="0.85", lw=0, zorder=0
        )
        for _label, (_dt_ecdf, _, _) in _ecdf.items():
            _ax.plot(
                _dt_ecdf[_var].sel(plot_axis="x").values,
                _dt_ecdf[_var].sel(plot_axis="y").values,
                color=GROUP_COLORS[_label],
                drawstyle="steps-post",
                label=_label,
            )
        _ax.axhline(0, color="k", lw=0.5, alpha=0.5)
        _ax.set_ylim(-0.4, 0.4)
        _ax.set_xlabel("PIT")
        _ax.set_title(VAR_LABELS.get(_var, _var))
    _axes[0, 0].set_ylabel(r"$\Delta$ ECDF")
    _handles, _labels = _axes[0, 0].get_legend_handles_labels()
    _fig.legend(
        _handles,
        _labels,
        # "upper center" makes the legend hang *below* the anchor, clearing the xlabels.
        loc="upper center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=len(_labels),
    )
    savefig(_fig, f"sbc_{scenario_name}")
    return


@app.cell
def _(scenario_name, stats):
    for _label, _s in stats.items():
        _fig, _axes = plt.subplots(
            len(_s),
            2,
            figsize=(DOUBLE_COL, SINGLE_COL * 0.85 * len(_s)),
            constrained_layout=True,
            squeeze=False,
        )
        for (_ax_z, _ax_e), (_var, _d) in zip(_axes, _s.items()):
            _title = VAR_LABELS.get(_var, _var)

            _ax_z.scatter(_d["contraction"], _d["zscore"], s=8)
            _ax_z.axhline(0, color="k", lw=0.5, alpha=0.5)
            _ax_z.set_ylim(-4, 4)
            _ax_z.set_yticks([-4, -2, 0, 2, 4])
            _ax_z.set_xlabel("Posterior contraction")
            _ax_z.set_ylabel("Posterior z-score")
            _ax_z.set_title(_title)

            _ax_e.scatter(_d["truth"], _d["estimate"], s=8)
            _lims = [
                min(_d["truth"].min(), _d["estimate"].min()),
                max(_d["truth"].max(), _d["estimate"].max()),
            ]
            _ax_e.plot(_lims, _lims, "k--", lw=0.5)
            _ax_e.set_xlabel("Ground truth")
            _ax_e.set_ylabel("Posterior mean")
            _ax_e.set_title(_title)

        savefig(_fig, f"sbc_zscore_contraction_{scenario_name}_{_label}")
    return


@app.cell
def _(scenario_name, stats):
    for _var in next(iter(stats.values())):
        _fig, _axes = panel_figure(1, PANEL_W)
        _ax = _axes[0, 0]
        _lo, _hi = np.inf, -np.inf
        for _label, _s in stats.items():
            if _label == "approximate":
                continue
            _d = _s[_var]
            _ax.scatter(
                _d["truth"],
                _d["estimate"],
                s=8,
                color=GROUP_COLORS[_label],
                #label=_label,
            )
            _lo = min(_lo, _d["truth"].min(), _d["estimate"].min())
            _hi = max(_hi, _d["truth"].max(), _d["estimate"].max())
        _ax.plot([_lo, _hi], [_lo, _hi], "k--", lw=0.5)
        _ax.set(
            xlabel="Ground truth",
            ylabel="Posterior mean",
            title=VAR_LABELS.get(_var, _var),
        )
        #_ax.legend(fontsize="small")
        savefig(_fig, f"sbc_ev_{scenario_name}_{_var}")
    return


if __name__ == "__main__":
    app.run()
