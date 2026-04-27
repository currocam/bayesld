import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path

    import arviz as az
    import bayesld
    import bayesld.montecarlo as mc2
    import cmdstanpy
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from bayesld import deterministic as det
    from scipy.stats import norm as sp_norm

    return Path, az, bayesld, cmdstanpy, det, mc2, mo, np, plt, sp_norm


@app.cell
def _(mo):
    ne_slider = mo.ui.slider(
        start=10,
        stop=5000,
        step=10,
        value=10,
        label="True Ne",
        show_value=True,
    )
    ne_slider
    return (ne_slider,)


@app.cell
def _(bayesld, np):
    left_bins, right_bins = bayesld.linear_bins()
    mutation_rate = 1e-8
    recombination_rate = 1e-8
    sequence_length = right_bins[-1] * 2 / recombination_rate
    sample_size = 30
    num_replicates = 30
    n_bins = len(left_bins)
    log_Ne_prior_mu = np.log(1000.0)
    log_Ne_prior_sigma = 2.0
    return (
        left_bins,
        log_Ne_prior_mu,
        log_Ne_prior_sigma,
        mutation_rate,
        n_bins,
        num_replicates,
        recombination_rate,
        right_bins,
        sample_size,
        sequence_length,
    )


@app.cell
def _(
    left_bins,
    mc2,
    mo,
    mutation_rate,
    ne_slider,
    np,
    num_replicates,
    recombination_rate,
    right_bins,
    sample_size,
    sequence_length,
):
    Ne_truth = ne_slider.value

    with mo.status.spinner(f"Simulating dataset at Ne={Ne_truth}..."):
        pi_data, ld_data = mc2.expected_constant(
            Ne_truth,
            left_bins,
            right_bins,
            mutation_rate,
            recombination_rate,
            sequence_length,
            sample_size,
            random_seed=42,
            num_replicates=num_replicates,
            ploidy=2,
            model="dtwf",
        )

    mean_div = float(np.mean(pi_data))
    est_sigma_div = float(np.std(pi_data, ddof=1) / np.sqrt(num_replicates))
    mean_ld = np.array(np.mean(ld_data, axis=0), dtype=float)
    est_sigma_ld = np.array(
        np.std(ld_data, axis=0, ddof=1) / np.sqrt(num_replicates), dtype=float
    )

    mo.md(f"Dataset: Ne={Ne_truth}, mean π={mean_div:.3e}, mean LD[0]={mean_ld[0]:.4f}")
    return Ne_truth, est_sigma_div, est_sigma_ld, mean_div, mean_ld


@app.cell
def _(
    est_sigma_div,
    est_sigma_ld,
    left_bins,
    log_Ne_prior_mu,
    log_Ne_prior_sigma,
    mean_div,
    mean_ld,
    mutation_rate,
    n_bins,
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
        "log_Ne_prior_mu": log_Ne_prior_mu,
        "log_Ne_prior_sigma": log_Ne_prior_sigma,
        # Hilbert-space GP approximation settings
        "hsgp_c": 1.5,  # boundary factor
        "hsgp_m": 20,  # number of basis functions
    }
    return (base_stan_data,)


@app.cell
def _(Path, cmdstanpy, mo):
    _stan_dir = Path(__file__).parent.parent / "stan"
    with mo.status.spinner("Compiling Stan models..."):
        model1 = cmdstanpy.CmdStanModel(stan_file=str(_stan_dir / "constant_ne.stan"))
        model_gp = cmdstanpy.CmdStanModel(
            stan_file=str(_stan_dir / "gp_surrogate_constant_ne.stan")
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
    _budget = 500
    _points_per_iter = 50
    _B = 200
    _num_workers = -1  # -1 = all cores (joblib convention)

    def _ld_loglik(pred_ld):
        return float(np.sum(sp_norm.logpdf(mean_ld, pred_ld, est_sigma_ld)) / n_bins)

    def _mc_eval(ne, seed, rng):
        _, _ld_mc = mc2.expected_constant(
            float(ne),
            left_bins,
            right_bins,
            mutation_rate,
            recombination_rate,
            sequence_length,
            sample_size,
            random_seed=int(seed),
            num_replicates=num_replicates,
            ploidy=2,
            num_workers=_num_workers,
        )
        _ld_mc = np.array(_ld_mc)
        _, _ld_det = det.expected_constant(
            float(ne),
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
            "ne": float(ne),
            "loglik_det": loglik_det,
            "bc_loglik": _T_hat - _bias,
            "epsilon": float(_boot_lls.std(ddof=1)),
        }

    def _make_gp_data(eval_pts):
        return {
            **base_stan_data,
            "n_eval": len(eval_pts),
            "eval_log_ne": [np.log(p["ne"]) for p in eval_pts],
            "eval_loglik_det": [p["loglik_det"] for p in eval_pts],
            "eval_bc_loglik": [p["bc_loglik"] for p in eval_pts],
            "eval_epsilon": [p["epsilon"] for p in eval_pts],
        }

    eval_points = []
    iteration_log = []
    _rng = np.random.default_rng(seed=77)
    _seed = 500

    with mo.status.spinner(f"Adaptive loop — budget={_budget} eval points …"):
        # Iteration 0: deterministic model, no eval data yet
        _pf = model1.pathfinder(
            data=base_stan_data,
            draws=_points_per_iter,
            seed=_seed,
            show_console=False,
        )
        _seed += 1
        _ne_draws = list(_pf.stan_variable("Ne")[:_points_per_iter])
        iteration_log.append({"iter": 0, "model": "det", "ne_draws": _ne_draws})
        for _ne in _ne_draws:
            eval_points.append(_mc_eval(_ne, _seed, _rng))
            _seed += 1

        # Iterations 1+: GP surrogate, accumulate eval data
        _it = 1
        while len(eval_points) < _budget:
            _pf = model_gp.pathfinder(
                data=_make_gp_data(eval_points),
                draws=_points_per_iter,
                seed=_seed,
                show_console=False,
            )
            _seed += 1
            _ne_draws = list(_pf.stan_variable("Ne")[:_points_per_iter])
            iteration_log.append({"iter": _it, "model": "gp", "ne_draws": _ne_draws})
            for _ne in _ne_draws:
                if len(eval_points) < _budget:
                    eval_points.append(_mc_eval(_ne, _seed, _rng))
                    _seed += 1
            _it += 1

    # _pf is now the last pathfinder fit — keep it for NUTS initialisation
    last_pf = _pf

    mo.md(
        f"{len(eval_points)} eval points over {len(iteration_log)} Pathfinder iterations."
    )
    return eval_points, iteration_log, last_pf


@app.cell
def _(base_stan_data, eval_points, last_pf, mo, model_gp, np):
    _final_data = {
        **base_stan_data,
        "n_eval": len(eval_points),
        "eval_log_ne": [np.log(p["ne"]) for p in eval_points],
        "eval_loglik_det": [p["loglik_det"] for p in eval_points],
        "eval_bc_loglik": [p["bc_loglik"] for p in eval_points],
        "eval_epsilon": [p["epsilon"] for p in eval_points],
    }

    # Build one init dict per chain from the last Pathfinder draw pool.
    # Pathfinder draws are already near the posterior — this skips the cold-start
    # divergence phase that plagues the HSGP funnel geometry.
    _n_pf = last_pf.stan_variable("log_Ne").shape[0]
    _idx = np.random.default_rng(seed=0).choice(_n_pf, size=4, replace=_n_pf < 4)
    _inits = [
        {
            "log_Ne": float(last_pf.stan_variable("log_Ne")[i]),
            "gp_rho": float(last_pf.stan_variable("gp_rho")[i]),
            "gp_alpha": float(last_pf.stan_variable("gp_alpha")[i]),
            "beta": last_pf.stan_variable("beta")[i].tolist(),
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

    _ne = fit_nuts.stan_variable("Ne")
    _rho = fit_nuts.stan_variable("gp_rho")
    _alph = fit_nuts.stan_variable("gp_alpha")
    _bias = fit_nuts.stan_variable("gp_bias")
    mo.md(f"""
    NUTS complete.

    | Param | Mean | Std | 95% CI |
    |---|---|---|---|
    | Ne | {np.mean(_ne):.1f} | {np.std(_ne):.1f} | [{np.percentile(_ne, 2.5):.1f}, {np.percentile(_ne, 97.5):.1f}] |
    | gp\\_rho | {np.mean(_rho):.2f} | {np.std(_rho):.2f} | [{np.percentile(_rho, 2.5):.2f}, {np.percentile(_rho, 97.5):.2f}] |
    | gp\\_alpha | {np.mean(_alph):.3f} | {np.std(_alph):.3f} | [{np.percentile(_alph, 2.5):.3f}, {np.percentile(_alph, 97.5):.3f}] |
    | gp\\_bias | {np.mean(_bias):.3f} | {np.std(_bias):.3f} | [{np.percentile(_bias, 2.5):.3f}, {np.percentile(_bias, 97.5):.3f}] |
    """)
    return (fit_nuts,)


@app.cell
def _(az, fit_nuts, plt):
    idata = az.from_cmdstanpy(fit_nuts)
    az.plot_trace(
        idata, var_names=["Ne", "gp_rho", "gp_alpha", "gp_bias"], combined=False
    )
    plt.gcf().suptitle("NUTS trace — GP surrogate model", y=1.01, fontsize=12)
    plt.tight_layout()
    plt.gcf()
    return (idata,)


@app.cell
def _(az, idata, mo):
    _summary = az.summary(idata, var_names=["Ne", "gp_rho", "gp_alpha", "gp_bias"])
    mo.md(f"""
    ## Posterior Summary (R-hat, ESS)
    ```
    {_summary.to_string()}
    ```
    """)
    return


@app.cell
def _(Ne_truth, eval_points, fit_nuts, iteration_log, np, plt):
    fig_diag, (ax_draws, ax_gp) = plt.subplots(1, 2, figsize=(13, 4))

    # Left: Pathfinder Ne draws per iteration
    _colors = {"det": "steelblue", "gp": "darkorange"}
    _seen = set()
    for _log in iteration_log:
        _c = _colors[_log["model"]]
        _label = _log["model"] if _log["model"] not in _seen else ""
        _seen.add(_log["model"])
        ax_draws.scatter(
            [_log["iter"]] * len(_log["ne_draws"]),
            _log["ne_draws"],
            color=_c,
            alpha=0.75,
            s=50,
            label=_label,
        )
    ax_draws.axhline(
        Ne_truth, color="red", lw=1.5, linestyle="--", label=f"True Ne={Ne_truth}"
    )
    ax_draws.set_xlabel("Iteration")
    ax_draws.set_ylabel("Ne (Pathfinder draw)")
    ax_draws.set_title("Pathfinder Ne draws per iteration")
    ax_draws.legend(fontsize=8)

    # Right: GP bias landscape
    _eval_ne = np.array([p["ne"] for p in eval_points])
    _eval_delta = np.array([p["bc_loglik"] - p["loglik_det"] for p in eval_points])
    _eval_eps = np.array([p["epsilon"] for p in eval_points])
    _ne_samples = fit_nuts.stan_variable("Ne")
    _bias_samp = fit_nuts.stan_variable("gp_bias")

    ax_gp.errorbar(
        _eval_ne,
        _eval_delta,
        yerr=_eval_eps,
        fmt="o",
        color="gray",
        alpha=0.7,
        capsize=3,
        label="Observed δ ± ε",
    )
    ax_gp.scatter(
        _ne_samples,
        _bias_samp,
        alpha=0.03,
        s=10,
        color="purple",
        label="GP bias at posterior Ne",
    )
    ax_gp.axhline(0, color="k", lw=0.8, linestyle=":")
    ax_gp.axvline(
        Ne_truth, color="red", lw=1.5, linestyle="--", label=f"True Ne={Ne_truth}"
    )
    ax_gp.set_xscale("log")
    ax_gp.set_xlabel("Ne (log scale)")
    ax_gp.set_ylabel("Bias  =  bc_loglik − loglik_det")
    ax_gp.set_title("GP bias landscape")
    ax_gp.legend(fontsize=8)

    fig_diag.suptitle("Adaptive GP surrogate diagnostics", fontsize=12)
    fig_diag.tight_layout()
    fig_diag
    return


if __name__ == "__main__":
    app.run()
