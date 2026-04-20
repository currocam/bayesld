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
    from bayesld import montecarlo2 as mc2
    from bayesld.models import ConstantDemography

    return ConstantDemography, az, linear_bins, mc2, mo, np, plt


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
    ne_slider = mo.ui.slider(
        10, 20_000, value=1_000, step=500,
        label="Ne truth",
    )
    sample_size_slider = mo.ui.slider(
        10, 200, value=50, step=10,
        label="Sample size (diploid individuals)",
    )
    num_windows_slider = mo.ui.slider(
        10, 500, value=50, step=10,
        label="Windows",
    )
    num_replicates_slider = mo.ui.slider(
        10, 100, value=10, step=1,
        label="Replicates",
    )
    mo.vstack([
        mo.md("## Simulation parameters"),
        ne_slider,
        sample_size_slider,
        num_windows_slider,
        num_replicates_slider,
    ])
    return (
        ne_slider,
        num_replicates_slider,
        num_windows_slider,
        sample_size_slider,
    )


@app.cell
def _(
    ne_slider,
    num_replicates_slider,
    num_windows_slider,
    sample_size_slider,
):
    Ne_truth = ne_slider.value
    sample_size = sample_size_slider.value
    num_replicates = num_replicates_slider.value
    num_windows = num_windows_slider.value
    return Ne_truth, num_windows, sample_size


@app.cell
def _(
    Ne_truth,
    left_bins,
    mc2,
    mo,
    mutation_rate,
    np,
    num_windows,
    recombination_rate,
    right_bins,
    sample_size,
    window_length,
):
    with mo.status.spinner(f"Simulating {num_windows} windows at Ne={Ne_truth:,}…"):
        _pi, _ld = mc2.expected_constant(
            Ne_truth,
            left_bins,
            right_bins,
            mutation_rate,
            recombination_rate,
            window_length,
            sample_size,
            random_seed=42,
            num_replicates=num_windows,
            ploidy=2,
            model="hudson",
        )
    pi_data = np.array(_pi)
    ld_data = np.array(_ld)
    mo.callout(
        mo.md(
            f"Simulated **{len(pi_data)}** windows · "
            f"Ne={Ne_truth:,} · n={sample_size} · "
            f"L={window_length/1e6:.0f} Mb"
        ),
        kind="success",
    )
    return ld_data, pi_data


@app.cell
def _():
    return


@app.cell
def _(
    ConstantDemography,
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
    model = ConstantDemography(
        diversity=pi_data,
        ld=ld_data,
        mutation_rate=mutation_rate,
        recombination_rate=recombination_rate,
        num_samples=sample_size,
        left_bins=left_bins,
        right_bins=right_bins,
        sequence_length=window_length,
    )
    mo.md("✓ Model compiled — approximate and GP-surrogate Stan programs ready.")
    return (model,)


@app.cell
def _(mo, model):
    mo.stop(model is None)
    with mo.status.spinner("Optimising…"):
        opt_result = model.optimize(show_console=False, seed=1)
    _gp_line = (
        [mo.md(f"GP bias = {opt_result['gp_bias']:.4f}  ·  "
               f"ρ = {opt_result['gp_rho']:.3f}  ·  "
               f"α = {opt_result['gp_alpha']:.3f}")]
        if "gp_bias" in opt_result else []
    )
    mo.vstack([
        mo.md("### MAP result"),
        mo.md(
            f"**Ne = {opt_result['Ne']:,.0f}** &nbsp;·&nbsp; "
            f"E[π] = {opt_result['E_pi']:.3e} &nbsp;·&nbsp; "
            f"Σ log_lik = {opt_result['log_lik'].sum():.2f}"
        ),
        *_gp_line,
    ])
    return (opt_result,)


@app.cell
def _(mo, model):
    mo.stop(model is None)
    n_rounds = 3
    with mo.status.spinner(f"Running {n_rounds} active learning rounds…"):
        for _ in range(n_rounds):
            model.surrogate_active_learning(
                points_per_iter=3,
                max_tolerance=0.001,
                seed=None,
            )
    n_eval_points = len(model.eval_points)
    mo.md(f"Surrogate dataset: **{n_eval_points}** MC evaluation points accumulated.")
    return


@app.cell
def _(az, mo, model):
    mo.stop(model is None)
    with mo.status.spinner("Running NUTS (4 chains × 2000 samples)…"):
        _fit = model.sample(
            chains=4,
            parallel_chains=4,
            threads_per_chain=2,
            iter_warmup=2000,
            iter_sampling=10000,
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
    azp.plot_dist(idata, var_names="Ne")
    plt.gca()
    return (azp,)


@app.cell
def _(azp, idata, mo, model):
    mo.stop(
        len(model.eval_points)==0,
        mo.md("Run active learning"),
    )
    azp.plot_trace(idata, var_names="gp_bias_ld")
    return


@app.cell
def _(ld_data, left_bins, mo, np, opt_result, plt, right_bins):
    mo.stop(
        opt_result is None,
        mo.md("Run MAP optimisation to see the LD fit."),
    )
    _bin_mid = (np.array(left_bins) + np.array(right_bins)) / 2
    _obs_mean_ld = np.mean(ld_data, axis=0)

    _fig, _ax = plt.subplots(figsize=(7, 4))
    _ax.plot(
        _bin_mid, _obs_mean_ld,
        "o", color="steelblue", ms=5, label="Observed mean LD",
    )
    _ax.plot(
        _bin_mid, opt_result["approx_ld"],
        "-", color="tomato", lw=2,
        label=f"MAP approx LD  (Ne = {opt_result['Ne']:,.0f})",
    )
    _ax.set_xlabel("Recombination distance (Morgans)")
    _ax.set_ylabel("Mean LD")
    _ax.set_title("Deterministic LD approximation vs observed")
    _ax.legend()
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

    # Posterior draws: shape (chains * draws, n_bins)
    _post = idata.posterior
    _approx_draws = np.array(_post["approx_ld"]).reshape(-1, len(_bin_mid))

    _has_surrogate = len(model.eval_points) > 0

    _fig, (_ax_ld, _ax_bias) = plt.subplots(
        2, 1, figsize=(7, 6), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    # --- LD panel ---
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

    # --- Bias panel ---
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
