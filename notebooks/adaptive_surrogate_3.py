import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    import arviz as az
    import cmdstanpy
    import bayesld
    import bayesld.montecarlo as mc2
    from pathlib import Path
    from bayesld import deterministic as det
    from scipy.stats import norm as sp_norm

    return Path, az, bayesld, cmdstanpy, det, mc2, mo, np, plt, sp_norm


@app.cell
def _(bayesld, np):
    left_bins, right_bins = bayesld.linear_bins()
    mutation_rate = 1e-8
    recombination_rate = 1e-8
    sequence_length = right_bins[-1] * 2 / recombination_rate
    sample_size = 30
    num_replicates = 10
    n_bins = len(left_bins)

    # Piecewise-constant model: 2 epochs
    n_epochs = 2
    Ne_values_true = [1000.0, 10000.0]  # Ne_values[1]=1000, Ne_values[2]=10000
    t_boundaries_true = [20.0]  # single boundary at t=20 generations

    # Shared lognormal prior for all Ne epochs and all log-time boundaries
    log_Ne_prior_mu = np.log(1000.0)
    log_Ne_prior_sigma = 2.0
    log_t_prior_mu = np.log(20.0)
    log_t_prior_sigma = 1.5
    return (
        Ne_values_true,
        left_bins,
        log_Ne_prior_mu,
        log_Ne_prior_sigma,
        log_t_prior_mu,
        log_t_prior_sigma,
        mutation_rate,
        n_bins,
        n_epochs,
        num_replicates,
        recombination_rate,
        right_bins,
        sample_size,
        sequence_length,
        t_boundaries_true,
    )


@app.cell
def _(
    Ne_values_true,
    left_bins,
    mc2,
    mo,
    mutation_rate,
    np,
    num_replicates,
    recombination_rate,
    right_bins,
    sample_size,
    sequence_length,
    t_boundaries_true,
):
    with mo.status.spinner(
        f"Simulating dataset (Ne={Ne_values_true}, t_boundaries={t_boundaries_true})..."
    ):
        pi_data, ld_data = mc2.expected_piecewise_constant(
            Ne_values_true,
            t_boundaries_true,
            left_bins,
            right_bins,
            mutation_rate,
            recombination_rate,
            sequence_length,
            sample_size,
            random_seed=42,
            num_replicates=100,
            ploidy=2,
            model="hudson",
        )

    mean_div = float(np.mean(pi_data))
    est_sigma_div = float(np.std(pi_data, ddof=1) / np.sqrt(num_replicates))
    mean_ld = np.array(np.mean(ld_data, axis=0), dtype=float)
    est_sigma_ld = np.array(
        np.std(ld_data, axis=0, ddof=1) / np.sqrt(num_replicates), dtype=float
    )

    mo.md(f"Dataset simulated. mean π={mean_div:.3e}, mean LD[0]={mean_ld[0]:.4f}")
    return est_sigma_div, est_sigma_ld, mean_div, mean_ld


@app.cell
def _(mean_ld, plt):
    plt.plot(mean_ld)
    return


@app.cell
def _(
    est_sigma_div,
    est_sigma_ld,
    left_bins,
    log_Ne_prior_mu,
    log_Ne_prior_sigma,
    log_t_prior_mu,
    log_t_prior_sigma,
    mean_div,
    mean_ld,
    mutation_rate,
    n_bins,
    n_epochs,
    right_bins,
    sample_size,
):
    base_stan_data = {
        "n_bins": n_bins,
        "left_bins": left_bins.tolist(),
        "right_bins": right_bins.tolist(),
        "mutation_rate": mutation_rate,
        "sample_size": sample_size,
        "mean_div": mean_div,
        "est_sigma_div": est_sigma_div,
        "mean_ld": mean_ld.tolist(),
        "est_sigma_ld": est_sigma_ld.tolist(),
        "n_epochs": n_epochs,
        "log_Ne_prior_mu": log_Ne_prior_mu,
        "log_Ne_prior_sigma": log_Ne_prior_sigma,
        "log_t_prior_mu": log_t_prior_mu,
        "log_t_prior_sigma": log_t_prior_sigma,
        # Hilbert-space GP approximation settings
        "hsgp_c": 1.5,
        "hsgp_m": 20,  # basis functions per dimension (D × 20 total, D = 2*n_epochs-1 = 3)
    }
    return (base_stan_data,)


@app.cell
def _(Path, cmdstanpy, mo):
    _stan_dir = Path(__file__).parent.parent / "stan"
    with mo.status.spinner("Compiling Stan models..."):
        model1 = cmdstanpy.CmdStanModel(
            stan_file=str(_stan_dir / "piecewise_constant_ne.stan")
        )
        model_gp = cmdstanpy.CmdStanModel(
            stan_file=str(_stan_dir / "gp_surrogate_piecewise_constant.stan")
        )
    mo.md("Both Stan models compiled.")
    return model1, model_gp


@app.cell
def _(
    base_stan_data,
    det,
    est_sigma_ld,
    left_bins,
    mc2,
    mean_ld,
    mo,
    model1,
    model_gp,
    mutation_rate,
    n_bins,
    np,
    num_replicates,
    recombination_rate,
    right_bins,
    sample_size,
    sequence_length,
    sp_norm,
):
    _budget = 60
    _points_per_iter = 10
    _B = 200
    _num_workers = -1

    def _ld_loglik(pred_ld):
        return float(np.sum(sp_norm.logpdf(mean_ld, pred_ld, est_sigma_ld)) / n_bins)

    def _mc_eval(ne_values, t_boundaries, seed, rng):
        _, _ld_mc = mc2.expected_piecewise_constant(
            list(ne_values),
            list(t_boundaries),
            left_bins,
            right_bins,
            mutation_rate,
            recombination_rate,
            sequence_length,
            sample_size,
            random_seed=int(seed),
            num_replicates=num_replicates,
            ploidy=2,
        )
        _ld_mc = np.array(_ld_mc)
        _, _ld_det = det.expected_piecewise_constant(
            ne_values,
            t_boundaries,
            left_bins,
            right_bins,
            mutation_rate,
            sample_size=sample_size,
            ploidy=2,
        )
        loglik_det = _ld_loglik(np.array(_ld_det))
        _T_hat = _ld_loglik(_ld_mc.mean(axis=0))
        _boot_lls = np.array(
            [
                _ld_loglik(
                    _ld_mc[rng.integers(0, num_replicates, size=num_replicates)].mean(
                        axis=0
                    )
                )
                for _ in range(_B)
            ]
        )
        _bias = _boot_lls.mean() - _T_hat
        return {
            "ne_values": [float(v) for v in ne_values],
            "t_boundaries": [float(t) for t in t_boundaries],
            "loglik_det": loglik_det,
            "bc_loglik": _T_hat - _bias,
            "epsilon": float(_boot_lls.std(ddof=1)),
        }

    def _make_gp_data(eval_pts):
        # eval_log_params: [log(Ne_1), ..., log(Ne_n), log(t_1), ..., log(t_{n-1})]
        return {
            **base_stan_data,
            "n_eval": len(eval_pts),
            "eval_log_params": [
                [np.log(v) for v in p["ne_values"]]
                + [np.log(t) for t in p["t_boundaries"]]
                for p in eval_pts
            ],
            "eval_loglik_det": [p["loglik_det"] for p in eval_pts],
            "eval_bc_loglik": [p["bc_loglik"] for p in eval_pts],
            "eval_epsilon": [p["epsilon"] for p in eval_pts],
        }

    eval_points = []
    iteration_log = []
    _rng = np.random.default_rng(seed=77)
    _seed = 500

    with mo.status.spinner(f"Adaptive loop — budget={_budget} eval points ..."):
        # Iteration 0: deterministic model
        _pf = model1.pathfinder(
            data=base_stan_data,
            draws=_points_per_iter,
            seed=_seed,
            show_console=False,
        )
        _seed += 1
        _ne_draws = _pf.stan_variable("Ne_values")[
            :_points_per_iter
        ]  # (draws, n_epochs)
        _t_draws = _pf.stan_variable("t_boundaries")[
            :_points_per_iter
        ]  # (draws, n_epochs-1)
        iteration_log.append(
            {
                "iter": 0,
                "model": "det",
                "ne_draws": _ne_draws.tolist(),
                "t_draws": _t_draws.tolist(),
            }
        )
        for _ne, _t in zip(_ne_draws, _t_draws):
            eval_points.append(_mc_eval(_ne, _t, _seed, _rng))
            _seed += 1

        # Iterations 1+: GP surrogate
        _it = 1
        while len(eval_points) < _budget:
            _pf = model_gp.pathfinder(
                data=_make_gp_data(eval_points),
                draws=_points_per_iter,
                seed=_seed,
                show_console=False,
            )
            _seed += 1
            _ne_draws = _pf.stan_variable("Ne_values")[:_points_per_iter]
            _t_draws = _pf.stan_variable("t_boundaries")[:_points_per_iter]
            iteration_log.append(
                {
                    "iter": _it,
                    "model": "gp",
                    "ne_draws": _ne_draws.tolist(),
                    "t_draws": _t_draws.tolist(),
                }
            )
            for _ne, _t in zip(_ne_draws, _t_draws):
                if len(eval_points) < _budget:
                    eval_points.append(_mc_eval(_ne, _t, _seed, _rng))
                    _seed += 1
            _it += 1

    last_pf = _pf

    mo.md(
        f"{len(eval_points)} eval points over {len(iteration_log)} Pathfinder iterations."
    )
    return eval_points, iteration_log, last_pf


@app.cell
def _(base_stan_data, eval_points, last_pf, mo, model_gp, n_epochs, np):
    _final_data = {
        **base_stan_data,
        "n_eval": len(eval_points),
        "eval_log_params": [
            [np.log(v) for v in p["ne_values"]] + [np.log(t) for t in p["t_boundaries"]]
            for p in eval_points
        ],
        "eval_loglik_det": [p["loglik_det"] for p in eval_points],
        "eval_bc_loglik": [p["bc_loglik"] for p in eval_points],
        "eval_epsilon": [p["epsilon"] for p in eval_points],
    }

    # Build one init dict per chain from the last Pathfinder draw pool.
    _D = 2 * n_epochs - 1
    _n_pf = last_pf.stan_variable("log_Ne_values").shape[0]
    _idx = np.random.default_rng(seed=0).choice(_n_pf, size=4, replace=_n_pf < 4)
    _inits = [
        {
            "log_Ne_values": last_pf.stan_variable("log_Ne_values")[i].tolist(),
            "log_t_boundaries": last_pf.stan_variable("log_t_boundaries")[i].tolist(),
            "gp_rho": last_pf.stan_variable("gp_rho")[i].tolist(),
            "gp_alpha": float(last_pf.stan_variable("gp_alpha")[i]),
            "beta": [
                last_pf.stan_variable("beta")[i, d, :].tolist() for d in range(_D)
            ],
        }
        for i in _idx
    ]

    with mo.status.spinner("Running NUTS — GP surrogate model..."):
        fit_nuts = model_gp.sample(
            data=_final_data,
            chains=4,
            iter_warmup=2000,
            iter_sampling=4000,
            seed=42,
            inits=_inits,
            adapt_delta=0.99,
            max_treedepth=12,
            show_progress=True,
            show_console=False,
        )

    _ne_vals = fit_nuts.stan_variable("Ne_values")  # (draws, n_epochs)
    _t_vals = fit_nuts.stan_variable("t_boundaries")  # (draws, n_epochs-1)
    mo.md(f"""
    NUTS complete.

    | Param | Mean | Std | 95% CI |
    |---|---|---|---|
    | Ne\\_1 | {_ne_vals[:, 0].mean():.1f} | {_ne_vals[:, 0].std():.1f} | [{np.percentile(_ne_vals[:, 0], 2.5):.1f}, {np.percentile(_ne_vals[:, 0], 97.5):.1f}] |
    | Ne\\_2 | {_ne_vals[:, 1].mean():.1f} | {_ne_vals[:, 1].std():.1f} | [{np.percentile(_ne_vals[:, 1], 2.5):.1f}, {np.percentile(_ne_vals[:, 1], 97.5):.1f}] |
    | t\\_1  | {_t_vals[:, 0].mean():.2f}  | {_t_vals[:, 0].std():.2f}  | [{np.percentile(_t_vals[:, 0], 2.5):.2f}, {np.percentile(_t_vals[:, 0], 97.5):.2f}] |
    """)
    return (fit_nuts,)


@app.cell
def _(az, fit_nuts, plt):
    idata = az.from_cmdstanpy(fit_nuts)
    az.plot_trace(
        idata,
        var_names=["Ne_values", "t_boundaries", "gp_alpha", "gp_bias"],
        combined=False,
    )
    plt.gcf().suptitle(
        "NUTS trace — GP surrogate (piecewise constant, 2 epochs)", y=1.01, fontsize=12
    )
    plt.tight_layout()
    plt.gcf()
    return (idata,)


@app.cell
def _(az, idata, mo):
    _summary = az.summary(
        idata,
        var_names=["Ne_values", "t_boundaries", "gp_alpha", "gp_bias"],
    )
    mo.md(f"""
    ## Posterior Summary (R-hat, ESS)
    ```
    {_summary.to_string()}
    ```
    """)
    return


@app.cell
def _(
    Ne_values_true,
    eval_points,
    fit_nuts,
    iteration_log,
    np,
    plt,
    t_boundaries_true,
):
    fig_diag, axes = plt.subplots(1, 3, figsize=(15, 4))
    ax_ne1, ax_ne2, ax_t1 = axes

    _colors = {"det": "steelblue", "gp": "darkorange"}
    _param_axes = [
        (ax_ne1, 0, "ne", Ne_values_true[0], "Ne_1"),
        (ax_ne2, 1, "ne", Ne_values_true[1], "Ne_2"),
        (ax_t1, 0, "t", t_boundaries_true[0], "t_1"),
    ]

    for _ax, _idx, _kind, _truth, _label in _param_axes:
        _seen = set()
        for _log in iteration_log:
            _c = _colors[_log["model"]]
            _lbl = _log["model"] if _log["model"] not in _seen else ""
            _seen.add(_log["model"])
            _vals = [
                row[_idx]
                for row in (_log["ne_draws"] if _kind == "ne" else _log["t_draws"])
            ]
            _ax.scatter(
                [_log["iter"]] * len(_vals),
                _vals,
                color=_c,
                alpha=0.75,
                s=50,
                label=_lbl,
            )
        _ax.axhline(
            _truth, color="red", lw=1.5, linestyle="--", label=f"True={_truth:.4g}"
        )
        _ax.set_xlabel("Iteration")
        _ax.set_ylabel(_label)
        _ax.set_title(f"Pathfinder draws — {_label}")
        _ax.legend(fontsize=8)

    # GP bias landscape vs Ne_1
    fig_bias, ax_bias = plt.subplots(figsize=(7, 4))
    _eval_ne1 = np.array([p["ne_values"][0] for p in eval_points])
    _eval_delta = np.array([p["bc_loglik"] - p["loglik_det"] for p in eval_points])
    _eval_eps = np.array([p["epsilon"] for p in eval_points])
    _ne1_samp = fit_nuts.stan_variable("Ne_values")[:, 0]
    _bias_samp = fit_nuts.stan_variable("gp_bias")
    ax_bias.errorbar(
        _eval_ne1,
        _eval_delta,
        yerr=_eval_eps,
        fmt="o",
        color="gray",
        alpha=0.7,
        capsize=3,
        label="Observed δ ± ε",
    )
    ax_bias.scatter(
        _ne1_samp,
        _bias_samp,
        alpha=0.03,
        s=10,
        color="purple",
        label="GP bias at posterior Ne_1",
    )
    ax_bias.axhline(0, color="k", lw=0.8, linestyle=":")
    ax_bias.axvline(
        Ne_values_true[0],
        color="red",
        lw=1.5,
        linestyle="--",
        label=f"True Ne_1={Ne_values_true[0]:.0f}",
    )
    ax_bias.set_xscale("log")
    ax_bias.set_xlabel("Ne_1 (log scale)")
    ax_bias.set_ylabel("Bias  =  bc_loglik − loglik_det")
    ax_bias.set_title("GP bias landscape (Ne_1 dimension)")
    ax_bias.legend(fontsize=8)

    fig_diag.suptitle(
        "Adaptive GP surrogate diagnostics (piecewise constant, 2 epochs)", fontsize=12
    )
    fig_diag.tight_layout()
    fig_diag
    return


if __name__ == "__main__":
    app.run()
