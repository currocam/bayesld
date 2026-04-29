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
    sample_size = 50
    num_replicates = 8
    n_bins = len(left_bins)

    # Gauss-Legendre quadrature nodes and weights on [-1, 1]
    # Generated with numpy — change n_quad to increase accuracy.
    n_quad = 20
    gl_nodes, gl_weights = np.polynomial.legendre.leggauss(n_quad)

    # True demographic parameters
    Ne_c_true = 1000.0
    Ne_a_true = 15000.0
    t0_true = 40.0
    # log_fold = alpha * t0 = log(Ne_c / Ne(t0)); here we enforce Ne(t0) = Ne_a
    log_fold_true = np.log(Ne_c_true) - np.log(10)  # ≈ -2.71 (population grew)
    alpha_true = log_fold_true / t0_true  # derived

    # Prior hyperparameters (lognormal priors)
    log_Ne_c_prior_mu = np.log(1000.0)
    log_Ne_c_prior_sigma = 2.0
    log_Ne_a_prior_mu = np.log(1000.0)
    log_Ne_a_prior_sigma = 2.0
    log_t0_prior_mu = np.log(20.0)
    log_t0_prior_sigma = 1.5
    log_fold_prior_sigma = 2.0  # log fold change prior scale: normal(0, sigma)
    return (
        Ne_a_true,
        Ne_c_true,
        alpha_true,
        gl_nodes,
        gl_weights,
        left_bins,
        log_Ne_a_prior_mu,
        log_Ne_a_prior_sigma,
        log_Ne_c_prior_mu,
        log_Ne_c_prior_sigma,
        log_fold_prior_sigma,
        log_fold_true,
        log_t0_prior_mu,
        log_t0_prior_sigma,
        mutation_rate,
        n_bins,
        n_quad,
        num_replicates,
        recombination_rate,
        right_bins,
        sample_size,
        sequence_length,
        t0_true,
    )


@app.cell
def _(
    Ne_a_true,
    Ne_c_true,
    alpha_true,
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
    t0_true,
):
    with mo.status.spinner(
        f"Simulating dataset (Ne_c={Ne_c_true}, Ne_a={Ne_a_true}, "
        f"t0={t0_true}, alpha={alpha_true:.4f})..."
    ):
        pi_data, ld_data = mc2.expected_piecewise_exponential(
            Ne_c_true,
            Ne_a_true,
            t0_true,
            alpha_true,
            left_bins,
            right_bins,
            mutation_rate,
            recombination_rate,
            sequence_length,
            sample_size,
            random_seed=42,
            num_replicates=num_replicates,
            ploidy=2,
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
def _(
    est_sigma_div,
    est_sigma_ld,
    gl_nodes,
    gl_weights,
    left_bins,
    log_Ne_a_prior_mu,
    log_Ne_a_prior_sigma,
    log_Ne_c_prior_mu,
    log_Ne_c_prior_sigma,
    log_fold_prior_sigma,
    log_t0_prior_mu,
    log_t0_prior_sigma,
    mean_div,
    mean_ld,
    mutation_rate,
    n_bins,
    n_quad,
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
        "n_quad": n_quad,
        "gl_nodes": gl_nodes.tolist(),
        "gl_weights": gl_weights.tolist(),
        # Hilbert-space GP approximation settings
        "hsgp_c": 1.5,
        "hsgp_m": 20,  # basis functions per dimension (4 × 20 = 80 total)
        "log_Ne_c_prior_mu": log_Ne_c_prior_mu,
        "log_Ne_c_prior_sigma": log_Ne_c_prior_sigma,
        "log_Ne_a_prior_mu": log_Ne_a_prior_mu,
        "log_Ne_a_prior_sigma": log_Ne_a_prior_sigma,
        "log_t0_prior_mu": log_t0_prior_mu,
        "log_t0_prior_sigma": log_t0_prior_sigma,
        "log_fold_prior_sigma": log_fold_prior_sigma,
    }
    return (base_stan_data,)


@app.cell
def _(Path, cmdstanpy, mo):
    _stan_dir = Path(__file__).parent.parent / "stan"
    with mo.status.spinner("Compiling Stan models..."):
        model1 = cmdstanpy.CmdStanModel(
            stan_file=str(_stan_dir / "piecewise_exponential_ne.stan")
        )
        model_gp = cmdstanpy.CmdStanModel(
            stan_file=str(_stan_dir / "gp_surrogate_piecewise_exponential.stan")
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
    _budget = 4 * 20
    _points_per_iter = 10
    _B = 200
    _num_workers = -1

    def _ld_loglik(pred_ld):
        return float(np.sum(sp_norm.logpdf(mean_ld, pred_ld, est_sigma_ld)) / n_bins)

    def _mc_eval(ne_c, ne_a, t0, log_fold, seed, rng):
        _alpha = log_fold / t0
        _, _ld_mc = mc2.expected_piecewise_exponential(
            float(ne_c),
            float(ne_a),
            float(t0),
            float(_alpha),
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
        _, _ld_det = det.expected_piecewise_exponential(
            float(ne_c),
            float(ne_a),
            float(t0),
            float(_alpha),
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
            "ne_c": float(ne_c),
            "ne_a": float(ne_a),
            "t0": float(t0),
            "log_fold": float(log_fold),
            "loglik_det": loglik_det,
            "bc_loglik": _T_hat - _bias,
            "epsilon": float(_boot_lls.std(ddof=1)),
        }

    def _make_gp_data(eval_pts):
        return {
            **base_stan_data,
            "n_eval": len(eval_pts),
            "eval_log_params": [
                [np.log(p["ne_c"]), np.log(p["ne_a"]), np.log(p["t0"]), p["log_fold"]]
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
        _ne_c_draws = list(_pf.stan_variable("Ne_c")[:_points_per_iter])
        _ne_a_draws = list(_pf.stan_variable("Ne_a")[:_points_per_iter])
        _t0_draws = list(_pf.stan_variable("t0")[:_points_per_iter])
        _log_fold_draws = list(_pf.stan_variable("log_fold")[:_points_per_iter])
        iteration_log.append(
            {
                "iter": 0,
                "model": "det",
                "ne_c_draws": _ne_c_draws,
                "ne_a_draws": _ne_a_draws,
                "t0_draws": _t0_draws,
                "log_fold_draws": _log_fold_draws,
            }
        )
        for _ne_c, _ne_a, _t0, _lf in zip(
            _ne_c_draws, _ne_a_draws, _t0_draws, _log_fold_draws
        ):
            eval_points.append(_mc_eval(_ne_c, _ne_a, _t0, _lf, _seed, _rng))
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
            _ne_c_draws = list(_pf.stan_variable("Ne_c")[:_points_per_iter])
            _ne_a_draws = list(_pf.stan_variable("Ne_a")[:_points_per_iter])
            _t0_draws = list(_pf.stan_variable("t0")[:_points_per_iter])
            _log_fold_draws = list(_pf.stan_variable("log_fold")[:_points_per_iter])
            iteration_log.append(
                {
                    "iter": _it,
                    "model": "gp",
                    "ne_c_draws": _ne_c_draws,
                    "ne_a_draws": _ne_a_draws,
                    "t0_draws": _t0_draws,
                    "log_fold_draws": _log_fold_draws,
                }
            )
            for _ne_c, _ne_a, _t0, _lf in zip(
                _ne_c_draws, _ne_a_draws, _t0_draws, _log_fold_draws
            ):
                if len(eval_points) < _budget:
                    eval_points.append(_mc_eval(_ne_c, _ne_a, _t0, _lf, _seed, _rng))
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
        "eval_log_params": [
            [np.log(p["ne_c"]), np.log(p["ne_a"]), np.log(p["t0"]), p["log_fold"]]
            for p in eval_points
        ],
        "eval_loglik_det": [p["loglik_det"] for p in eval_points],
        "eval_bc_loglik": [p["bc_loglik"] for p in eval_points],
        "eval_epsilon": [p["epsilon"] for p in eval_points],
    }

    # Build one init dict per chain from the last Pathfinder draw pool.
    _n_pf = last_pf.stan_variable("log_Ne_c").shape[0]
    _idx = np.random.default_rng(seed=0).choice(_n_pf, size=4, replace=_n_pf < 4)
    _inits = [
        {
            "log_Ne_c": float(last_pf.stan_variable("log_Ne_c")[i]),
            "log_Ne_a": float(last_pf.stan_variable("log_Ne_a")[i]),
            "log_t0": float(last_pf.stan_variable("log_t0")[i]),
            "log_fold": float(last_pf.stan_variable("log_fold")[i]),
            "gp_rho": last_pf.stan_variable("gp_rho")[i].tolist(),
            "gp_alpha": float(last_pf.stan_variable("gp_alpha")[i]),
            "beta": [last_pf.stan_variable("beta")[i, d, :].tolist() for d in range(4)],
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

    _ne_c = fit_nuts.stan_variable("Ne_c")
    _ne_a = fit_nuts.stan_variable("Ne_a")
    _t0 = fit_nuts.stan_variable("t0")
    _log_fold = fit_nuts.stan_variable("log_fold")
    _alpha = fit_nuts.stan_variable("alpha")
    mo.md(f"""
    NUTS complete.

    | Param | Mean | Std | 95% CI |
    |---|---|---|---|
    | Ne\\_c     | {_ne_c.mean():.1f}      | {_ne_c.std():.1f}      | [{np.percentile(_ne_c, 2.5):.1f}, {np.percentile(_ne_c, 97.5):.1f}] |
    | Ne\\_a     | {_ne_a.mean():.1f}      | {_ne_a.std():.1f}      | [{np.percentile(_ne_a, 2.5):.1f}, {np.percentile(_ne_a, 97.5):.1f}] |
    | t0        | {_t0.mean():.2f}        | {_t0.std():.2f}        | [{np.percentile(_t0, 2.5):.2f}, {np.percentile(_t0, 97.5):.2f}] |
    | log\\_fold | {_log_fold.mean():.3f}  | {_log_fold.std():.3f}  | [{np.percentile(_log_fold, 2.5):.3f}, {np.percentile(_log_fold, 97.5):.3f}] |
    | alpha     | {_alpha.mean():.4f}     | {_alpha.std():.4f}     | [{np.percentile(_alpha, 2.5):.4f}, {np.percentile(_alpha, 97.5):.4f}] |
    """)
    return (fit_nuts,)


@app.cell
def _(alpha_true, t0_true):
    alpha_true, t0_true
    return


@app.cell
def _(az, idata):
    az.plot_pair(
        idata,
        var_names=["Ne_c", "Ne_a", "t0", "alpha"],
    )
    return


@app.cell
def _(az, fit_nuts, plt):
    idata = az.from_cmdstanpy(fit_nuts)
    az.plot_trace(
        idata,
        var_names=[
            "Ne_c",
            "Ne_a",
            "t0",
            "log_fold",
            "alpha",
            "gp_rho",
            "gp_alpha",
            "gp_bias",
        ],
        combined=False,
    )
    plt.gcf().suptitle(
        "NUTS trace — GP surrogate (piecewise exponential)", y=1.01, fontsize=12
    )
    plt.tight_layout()
    plt.gcf()
    return (idata,)


@app.cell
def _(Ne_a_true, Ne_c_true, alpha_true, az, idata):
    az.plot_pair(
        idata,
        var_names=["alpha", "Ne_c", "Ne_a"],
        reference_values=dict(alpha=alpha_true, Ne_c=Ne_c_true, Ne_a=Ne_a_true),
    )
    return


@app.cell
def _(az, idata, mo):
    _summary = az.summary(
        idata,
        var_names=[
            "Ne_c",
            "Ne_a",
            "t0",
            "log_fold",
            "alpha",
            "gp_rho",
            "gp_alpha",
            "gp_bias",
        ],
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
    Ne_a_true,
    Ne_c_true,
    eval_points,
    fit_nuts,
    iteration_log,
    log_fold_true,
    np,
    plt,
    t0_true,
):
    fig_diag, axes = plt.subplots(2, 2, figsize=(13, 8))
    ax_nec, ax_nea, ax_t0, ax_al = axes.flat

    _colors = {"det": "steelblue", "gp": "darkorange"}
    _param_axes = [
        (ax_nec, "ne_c_draws", Ne_c_true, "Ne_c"),
        (ax_nea, "ne_a_draws", Ne_a_true, "Ne_a"),
        (ax_t0, "t0_draws", t0_true, "t0"),
        (ax_al, "log_fold_draws", log_fold_true, "log_fold"),
    ]

    for _ax, _key, _truth, _label in _param_axes:
        _seen = set()
        for _log in iteration_log:
            _c = _colors[_log["model"]]
            _lbl = _log["model"] if _log["model"] not in _seen else ""
            _seen.add(_log["model"])
            _ax.scatter(
                [_log["iter"]] * len(_log[_key]),
                _log[_key],
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

    # GP bias landscape: bc_loglik - loglik_det vs Ne_c (most identifiable dim)
    fig_bias, ax_bias = plt.subplots(figsize=(7, 4))
    _eval_ne_c = np.array([p["ne_c"] for p in eval_points])
    _eval_delta = np.array([p["bc_loglik"] - p["loglik_det"] for p in eval_points])
    _eval_eps = np.array([p["epsilon"] for p in eval_points])
    _ne_c_samp = fit_nuts.stan_variable("Ne_c")
    _bias_samp = fit_nuts.stan_variable("gp_bias")
    ax_bias.errorbar(
        _eval_ne_c,
        _eval_delta,
        yerr=_eval_eps,
        fmt="o",
        color="gray",
        alpha=0.7,
        capsize=3,
        label="Observed δ ± ε",
    )
    ax_bias.scatter(
        _ne_c_samp,
        _bias_samp,
        alpha=0.03,
        s=10,
        color="purple",
        label="GP bias at posterior Ne_c",
    )
    ax_bias.axhline(0, color="k", lw=0.8, linestyle=":")
    ax_bias.axvline(
        Ne_c_true,
        color="red",
        lw=1.5,
        linestyle="--",
        label=f"True Ne_c={Ne_c_true:.0f}",
    )
    ax_bias.set_xscale("log")
    ax_bias.set_xlabel("Ne_c (log scale)")
    ax_bias.set_ylabel("Bias  =  bc_loglik − loglik_det")
    ax_bias.set_title("GP bias landscape (Ne_c dimension)")
    ax_bias.legend(fontsize=8)

    fig_diag.suptitle(
        "Adaptive GP surrogate diagnostics (piecewise exponential)", fontsize=12
    )
    fig_diag.tight_layout()
    fig_diag
    return


if __name__ == "__main__":
    app.run()
