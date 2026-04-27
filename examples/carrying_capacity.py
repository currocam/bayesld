import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import arviz_base as az
    import matplotlib.pyplot as plt
    from bayesld import linear_bins
    from bayesld import montecarlo as mc2
    import jax
    jax.config.update("jax_enable_x64", True)
    from bayesld.models import ExponentialCarryingCapacityDemography

    return ExponentialCarryingCapacityDemography, az, linear_bins, mc2, mo, np, plt


@app.cell
def _(linear_bins):
    mutation_rate = 1e-8
    recombination_rate = 1e-8
    left_bins, right_bins = linear_bins()
    window_length = right_bins[-1] * 2 / recombination_rate
    return (
        left_bins,
        mutation_rate,
        recombination_rate,
        right_bins,
        window_length,
    )


@app.cell
def _(mo):
    ne_c_slider = mo.ui.slider(
        10, 50_000, value=100, step=10,
        label="Ne_c -- contemporary Ne (truth)",
    )
    ne_a_slider = mo.ui.slider(
        10, 50_000, value=200, step=10,
        label="Ne_a -- ancestral Ne (truth)",
    )
    t0_slider = mo.ui.slider(
        1, 200, value=20, step=1,
        label="t0 -- start of exponential phase (gen ago)",
    )
    t1_slider = mo.ui.slider(
        10, 1000, value=200, step=10,
        label="t1 -- end of exponential phase (gen ago)",
    )
    alpha_slider = mo.ui.slider(
        -0.1, 0.1, value=0.01, step=0.001,
        label="alpha -- exponential rate",
    )
    sample_size_slider = mo.ui.slider(
        10, 200, value=50, step=10,
        label="Sample size (diploid individuals)",
    )
    num_windows_slider = mo.ui.slider(
        10, 500, value=50, step=10,
        label="Windows",
    )
    mo.vstack([
        mo.md("## Simulation parameters"),
        ne_c_slider,
        ne_a_slider,
        t0_slider,
        t1_slider,
        alpha_slider,
        sample_size_slider,
        num_windows_slider,
    ])
    return (
        alpha_slider,
        ne_a_slider,
        ne_c_slider,
        num_windows_slider,
        sample_size_slider,
        t0_slider,
        t1_slider,
    )


@app.cell
def _(
    alpha_slider,
    ne_a_slider,
    ne_c_slider,
    num_windows_slider,
    sample_size_slider,
    t0_slider,
    t1_slider,
):
    Ne_c_truth = ne_c_slider.value
    Ne_a_truth = ne_a_slider.value
    t0_truth = t0_slider.value
    t1_truth = t1_slider.value
    alpha_truth = alpha_slider.value
    sample_size = sample_size_slider.value
    num_windows = num_windows_slider.value
    return Ne_a_truth, Ne_c_truth, alpha_truth, num_windows, sample_size, t0_truth, t1_truth


@app.cell
def _(
    Ne_a_truth,
    Ne_c_truth,
    alpha_truth,
    left_bins,
    mc2,
    mo,
    mutation_rate,
    np,
    num_windows,
    recombination_rate,
    right_bins,
    sample_size,
    t0_truth,
    t1_truth,
    window_length,
):
    mo.stop(
        t0_truth >= t1_truth,
        mo.callout(mo.md("t0 must be less than t1."), kind="warn"),
    )
    with mo.status.spinner(
        f"Simulating {num_windows} windows · "
        f"Ne_c={Ne_c_truth:,}, Ne_a={Ne_a_truth:,}, "
        f"t0={t0_truth}, t1={t1_truth}, alpha={alpha_truth:.3f}..."
    ):
        _pi, _ld = mc2.expected_exponential_carrying_capacity(
            float(Ne_c_truth), float(Ne_a_truth),
            float(t0_truth), float(t1_truth), float(alpha_truth),
            left_bins, right_bins,
            mutation_rate, recombination_rate, window_length,
            sample_size,
            random_seed=42,
            num_replicates=num_windows,
            ploidy=2,
        )
    pi_data = np.array(_pi)
    ld_data = np.array(_ld)
    mo.callout(
        mo.md(
            f"Simulated **{len(pi_data)}** windows · "
            f"Ne_c={Ne_c_truth:,}, Ne_a={Ne_a_truth:,}, "
            f"t0={t0_truth}, t1={t1_truth} gen, alpha={alpha_truth:.3f} · "
            f"n={sample_size} · L={window_length/1e6:.0f} Mb"
        ),
        kind="success",
    )
    return ld_data, pi_data


@app.cell
def _(
    ExponentialCarryingCapacityDemography,
    ld_data,
    left_bins,
    mo,
    mutation_rate,
    pi_data,
    recombination_rate,
    right_bins,
    sample_size,
    window_length,
):
    mo.stop(
        pi_data is None,
        mo.callout(mo.md("Simulate data first."), kind="warn"),
    )
    model = ExponentialCarryingCapacityDemography(
        diversity=pi_data,
        ld=ld_data,
        mutation_rate=mutation_rate,
        recombination_rate=recombination_rate,
        num_samples=sample_size,
        left_bins=left_bins,
        right_bins=right_bins,
        sequence_length=window_length,
    )
    mo.md(
        "Model compiled -- approximate and GP-surrogate Stan programs ready."
    )
    return (model,)


@app.cell
def _(Ne_a_truth, Ne_c_truth, mo, model, np, t0_truth, t1_truth):
    mo.stop(model is None)
    with mo.status.spinner("Optimising..."):
        opt_result = model.optimize(
            show_console=False,
            seed=1,
            algorithm="Newton",
            inits={
                "log_Ne_c": np.log(float(Ne_c_truth)),
                "log_Ne_a": np.log(float(Ne_a_truth)),
                "log_t_boundaries": np.log([float(t0_truth), float(t1_truth)]),
                "alpha": 0.0,
            },
        )
    _ne_c = opt_result["Ne_c"]
    _ne_a = opt_result["Ne_a"]
    _t0 = opt_result["t0"]
    _t1 = opt_result["t1"]
    _alpha = opt_result["alpha"]
    _gp_lines = (
        [mo.md(
            f"GP bias LD -- mean={opt_result['gp_bias_ld'].mean():.4f}  "
            f"min={opt_result['gp_bias_ld'].min():.4f}  "
            f"max={opt_result['gp_bias_ld'].max():.4f}  ·  "
            f"rho_r={float(opt_result['gp_rho_r']):.3f}  "
            f"alpha_gp={float(opt_result['gp_alpha']):.3f}"
        )]
        if "gp_bias_ld" in opt_result else []
    )
    mo.vstack([
        mo.md("### MAP result"),
        mo.md(
            f"**Ne_c = {_ne_c:,.0f}** · "
            f"**Ne_a = {_ne_a:,.0f}** · "
            f"**t0 = {_t0:,.0f}** gen · "
            f"**t1 = {_t1:,.0f}** gen · "
            f"**alpha = {_alpha:.4f}** · "
            f"E[pi] = {opt_result['E_pi']:.3e} · "
            f"sum log_lik = {opt_result['log_lik'].sum():.2f}"
        ),
        *_gp_lines,
    ])
    return (opt_result,)


@app.cell
def _(mo, model):
    mo.stop(model is None)
    with mo.status.spinner("Learning surrogate likelihood..."):
        model.learn_surrogate_likelihood(
            n_map_iterations=5,
            n_nuts_samples=5,
            seed=None,
        )
    n_eval_points = len(model.eval_points)
    mo.md(f"Surrogate dataset: **{n_eval_points}** MC evaluation points accumulated.")
    return


@app.cell
def _(az, mo, model):
    mo.stop(model is None)
    with mo.status.spinner("Running NUTS (4 chains x 1000 samples)..."):
        _fit = model.sample(
            chains=4,
            parallel_chains=4,
            threads_per_chain=2,
            iter_warmup=1000,
            iter_sampling=1000,
            show_console=False,
            seed=1,
        )
    idata = az.from_cmdstanpy(_fit)
    return (idata,)


@app.cell
def _(idata, mo, plt):
    mo.stop(
        idata is None,
        mo.md("Run NUTS sampling to see the ArviZ summary."),
    )
    import arviz_plots as azp
    _var_names = ["Ne_c", "Ne_a", "t0", "t1", "alpha"]
    azp.plot_trace(idata, var_names=_var_names)
    plt.gca()
    return


@app.cell
def _(Ne_a_truth, Ne_c_truth, alpha_truth, t0_truth, t1_truth):
    [Ne_c_truth, Ne_a_truth, t0_truth, t1_truth, alpha_truth]
    return


@app.cell
def _(ld_data, left_bins, mo, np, opt_result, plt, right_bins):
    mo.stop(
        opt_result is None,
        mo.md("Run MAP optimisation to see the LD fit."),
    )
    _ne_c = opt_result["Ne_c"]
    _ne_a = opt_result["Ne_a"]
    _t0 = opt_result["t0"]
    _t1 = opt_result["t1"]
    _alpha = opt_result["alpha"]
    _bin_mid = (np.array(left_bins) + np.array(right_bins)) / 2
    _obs_mean_ld = np.mean(ld_data, axis=0)
    _obs_sem_ld = np.std(ld_data, axis=0, ddof=1) / np.sqrt(len(ld_data))

    _fig, _ax = plt.subplots(figsize=(7, 4))
    _ax.errorbar(
        _bin_mid, _obs_mean_ld, yerr=2 * _obs_sem_ld,
        fmt="o", color="steelblue", ms=4, lw=1.2,
        label="Observed mean LD +/- 2 SEM",
    )
    _ax.plot(
        _bin_mid, opt_result["approx_ld"],
        "--", color="tomato", lw=2,
        label="MAP approx LD (deterministic)",
    )
    if "corrected_ld" in opt_result:
        _ax.plot(
            _bin_mid, opt_result["corrected_ld"],
            "-", color="darkorange", lw=2,
            label="MAP corrected LD (approx x (1 + GP bias))",
        )
        _bias_pct = 100 * opt_result["gp_bias_ld"]
        _ax2 = _ax.twinx()
        _ax2.bar(
            _bin_mid, _bias_pct,
            width=(_bin_mid[1] - _bin_mid[0]) * 0.4,
            color="purple", alpha=0.25, label="GP bias (%)",
        )
        _ax2.axhline(0, color="purple", lw=0.8, ls=":")
        _ax2.set_ylabel("GP relative bias (%)", color="purple")
        _ax2.tick_params(axis="y", colors="purple")

    _ax.set_xlabel("Recombination distance (Morgans)")
    _ax.set_ylabel("Mean LD")
    _ax.set_title(
        f"Carrying capacity LD fit  "
        f"(Ne_c={_ne_c:,.0f}, Ne_a={_ne_a:,.0f}, "
        f"t0={_t0:,.0f}, t1={_t1:,.0f}, alpha={_alpha:.4f})"
    )
    _ax.legend(loc="upper right")
    plt.tight_layout()
    _fig
    return


@app.cell
def _(idata, ld_data, left_bins, mo, model, np, plt, right_bins):
    mo.stop(
        idata is None,
        mo.md("Run NUTS sampling to see the posterior predictive LD."),
    )
    _bin_mid = (np.array(left_bins) + np.array(right_bins)) / 2
    _obs_mean_ld = np.mean(ld_data, axis=0)
    _obs_sem_ld = np.std(ld_data, axis=0, ddof=1) / np.sqrt(len(ld_data))

    _post = idata.posterior
    _approx_draws = np.array(_post["approx_ld"]).reshape(-1, len(_bin_mid))

    _has_surrogate = len(model.eval_points) > 0

    _fig, (_ax_ld, _ax_bias) = plt.subplots(
        2, 1, figsize=(7, 6), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    _approx_lo = np.percentile(_approx_draws, 5, axis=0)
    _approx_hi = np.percentile(_approx_draws, 95, axis=0)
    _approx_mid = _approx_draws.mean(axis=0)

    _ax_ld.fill_between(_bin_mid, _approx_lo, _approx_hi, color="tomato", alpha=0.25)
    _ax_ld.plot(_bin_mid, _approx_mid, "-", color="tomato", lw=2, label="Approx LD (posterior mean +/- 90% CI)")

    if _has_surrogate and "corrected_ld_gq" in _post:
        _corr_draws = np.array(_post["corrected_ld_gq"]).reshape(-1, len(_bin_mid))
        _corr_lo = np.percentile(_corr_draws, 5, axis=0)
        _corr_hi = np.percentile(_corr_draws, 95, axis=0)
        _corr_mid = _corr_draws.mean(axis=0)
        _ax_ld.fill_between(_bin_mid, _corr_lo, _corr_hi, color="darkorange", alpha=0.25)
        _ax_ld.plot(_bin_mid, _corr_mid, "-", color="darkorange", lw=2, label="Corrected LD (posterior mean +/- 90% CI)")

    _ax_ld.errorbar(
        _bin_mid, _obs_mean_ld, yerr=2 * _obs_sem_ld,
        fmt="o", color="steelblue", ms=4, lw=1.2, zorder=5,
        label="Observed mean LD +/- 2 SEM",
    )
    _ax_ld.set_ylabel("Mean LD")
    _ax_ld.set_title("Posterior predictive LD")
    _ax_ld.legend(fontsize=8)

    if _has_surrogate and "gp_bias_ld" in _post:
        _bias_draws = np.array(_post["gp_bias_ld"]).reshape(-1, len(_bin_mid))
        _bias_lo = np.percentile(_bias_draws, 5, axis=0)
        _bias_hi = np.percentile(_bias_draws, 95, axis=0)
        _bias_mid = _bias_draws.mean(axis=0)
        _ax_bias.fill_between(_bin_mid, 100 * _bias_lo, 100 * _bias_hi, color="purple", alpha=0.25)
        _ax_bias.plot(_bin_mid, 100 * _bias_mid, "-", color="purple", lw=2)
        _ax_bias.axhline(0, color="gray", lw=0.8, ls="--")
        _ax_bias.set_ylabel("GP bias (%)")
    else:
        _ax_bias.text(0.5, 0.5, "No surrogate active", ha="center", va="center", transform=_ax_bias.transAxes, color="gray")
        _ax_bias.set_ylabel("GP bias (%)")

    _ax_bias.set_xlabel("Recombination distance (Morgans)")
    plt.tight_layout()
    _fig
    return


if __name__ == "__main__":
    app.run()
