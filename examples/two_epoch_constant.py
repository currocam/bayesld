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
    from bayesld.models import PiecewiseConstantDemography

    return PiecewiseConstantDemography, az, linear_bins, mc2, mo, np, plt


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
    ne1_slider = mo.ui.slider(
        100, 50_000, value=2_000, step=100,
        label="Ne₁ — recent epoch (truth)",
    )
    ne2_slider = mo.ui.slider(
        100, 50_000, value=10_000, step=100,
        label="Ne₂ — ancestral epoch (truth)",
    )
    t_boundary_slider = mo.ui.slider(
        1, 100, value=50, step=1,
        label="t_boundary — change-point (generations ago)",
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
        ne1_slider,
        ne2_slider,
        t_boundary_slider,
        sample_size_slider,
        num_windows_slider,
    ])
    return (
        ne1_slider,
        ne2_slider,
        num_windows_slider,
        sample_size_slider,
        t_boundary_slider,
    )


@app.cell
def _(
    ne1_slider,
    ne2_slider,
    num_windows_slider,
    sample_size_slider,
    t_boundary_slider,
):
    Ne1_truth = ne1_slider.value
    Ne2_truth = ne2_slider.value
    t_boundary_truth = t_boundary_slider.value
    sample_size = sample_size_slider.value
    num_windows = num_windows_slider.value
    return Ne1_truth, Ne2_truth, num_windows, sample_size, t_boundary_truth


@app.cell
def _(
    Ne1_truth,
    Ne2_truth,
    left_bins,
    mc2,
    mo,
    mutation_rate,
    np,
    num_windows,
    recombination_rate,
    right_bins,
    sample_size,
    t_boundary_truth,
    window_length,
):
    with mo.status.spinner(
        f"Simulating {num_windows} windows · "
        f"Ne=({Ne1_truth:,}, {Ne2_truth:,}) · t={t_boundary_truth:,}…"
    ):
        _pi, _ld = mc2.expected_piecewise_constant(
            np.array([float(Ne1_truth), float(Ne2_truth)]),
            np.array([float(t_boundary_truth)]),
            left_bins,
            right_bins,
            mutation_rate,
            recombination_rate,
            window_length,
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
            f"Ne=({Ne1_truth:,}, {Ne2_truth:,}) · "
            f"t={t_boundary_truth:,} gen · n={sample_size} · "
            f"L={window_length/1e6:.0f} Mb"
        ),
        kind="success",
    )
    return ld_data, pi_data


@app.cell
def _(
    PiecewiseConstantDemography,
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
    model = PiecewiseConstantDemography(
        diversity=pi_data,
        ld=ld_data,
        mutation_rate=mutation_rate,
        recombination_rate=recombination_rate,
        num_samples=sample_size,
        left_bins=left_bins,
        right_bins=right_bins,
        n_epochs=2,
        sequence_length=window_length,
        #hsgp_m=5,
    )
    mo.md(
        "✓ Two-epoch model compiled — approximate and GP-surrogate Stan programs ready."
    )
    return (model,)


@app.cell
def _(Ne1_truth, Ne2_truth, mo, model, np):
    mo.stop(model is None)
    with mo.status.spinner("Optimising…"):
        opt_result = model.optimize(
            show_console=False,
            seed=1,
            inits={
                "log_Ne_values": np.log([float(Ne1_truth), float(Ne2_truth)]),
                "log_t_boundaries": np.log([500.0]),
            },
        )
    _ne1, _ne2 = opt_result["Ne_values"]
    _t_map = opt_result["t_boundaries"][0]
    _gp_lines = (
        [mo.md(
            f"GP bias LD — mean={opt_result['gp_bias_ld'].mean():.4f}  "
            f"min={opt_result['gp_bias_ld'].min():.4f}  "
            f"max={opt_result['gp_bias_ld'].max():.4f}  ·  "
            f"ρ_r={float(opt_result['gp_rho_r']):.3f}  "
            f"α={float(opt_result['gp_alpha']):.3f}"
        )]
        if "gp_bias_ld" in opt_result else []
    )
    mo.vstack([
        mo.md("### MAP result"),
        mo.md(
            f"**Ne₁ = {_ne1:,.0f}** (recent) &nbsp;·&nbsp; "
            f"**Ne₂ = {_ne2:,.0f}** (ancestral) &nbsp;·&nbsp; "
            f"**t = {_t_map:,.0f}** gen &nbsp;·&nbsp; "
            f"E[π] = {opt_result['E_pi']:.3e} &nbsp;·&nbsp; "
            f"Σ log_lik = {opt_result['log_lik'].sum():.2f}"
        ),
        *_gp_lines,
    ])
    return (opt_result,)


@app.cell
def _(mo, model):
    mo.stop(model is None)
    with mo.status.spinner("Learning surrogate likelihood…"):
        model.learn_surrogate_likelihood(
            n_map_iterations=20,
            n_nuts_samples=20,
            max_tolerance=0.05,
            seed=None,
        )
    n_eval_points = len(model.eval_points)
    mo.md(f"Surrogate dataset: **{n_eval_points}** MC evaluation points accumulated.")
    return (n_eval_points,)


@app.cell
def _(az, mo, model, n_eval_points):
    mo.stop(model is None and  n_eval_points > 0)
    with mo.status.spinner("Running NUTS (4 chains × 2000 samples)…"):
        _fit = model.sample(
            chains=4,
            parallel_chains=4,
            threads_per_chain=2,
            iter_warmup=2000,
            iter_sampling=4000,
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
    _var_names = ["Ne_values", "t_boundaries"]
    azp.plot_pair(
        idata, var_names=_var_names,
    )
    plt.gca()
    return (azp,)


@app.cell
def _(azp, idata):
    azp.plot_dist(
        idata, var_names=["Ne_values", "t_boundaries"],
    )
    return


@app.cell
def _(Ne1_truth, Ne2_truth, t_boundary_truth):
    [Ne1_truth, Ne2_truth, t_boundary_truth]
    return


@app.cell
def _(ld_data, left_bins, mo, np, opt_result, plt, right_bins):
    mo.stop(
        opt_result is None,
        mo.md("Run MAP optimisation to see the LD fit."),
    )
    _ne1, _ne2 = opt_result["Ne_values"]
    _t_map = opt_result["t_boundaries"][0]
    _bin_mid = (np.array(left_bins) + np.array(right_bins)) / 2
    _obs_mean_ld = np.mean(ld_data, axis=0)
    _obs_sem_ld = np.std(ld_data, axis=0, ddof=1) / np.sqrt(len(ld_data))

    _fig, _ax = plt.subplots(figsize=(7, 4))
    _ax.errorbar(
        _bin_mid, _obs_mean_ld, yerr=2 * _obs_sem_ld,
        fmt="o", color="steelblue", ms=4, lw=1.2,
        label="Observed mean LD ± 2 SEM",
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
            label="MAP corrected LD (approx × (1 + GP bias))",
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
        f"Two-epoch LD fit  (Ne₁={_ne1:,.0f}, Ne₂={_ne2:,.0f}, t={_t_map:,.0f} gen)"
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
    _ax_ld.plot(_bin_mid, _approx_mid, "-", color="tomato", lw=2, label="Approx LD (posterior mean ± 90% CI)")

    if _has_surrogate and "corrected_ld_gq" in _post:
        _corr_draws = np.array(_post["corrected_ld_gq"]).reshape(-1, len(_bin_mid))
        _corr_lo = np.percentile(_corr_draws, 5, axis=0)
        _corr_hi = np.percentile(_corr_draws, 95, axis=0)
        _corr_mid = _corr_draws.mean(axis=0)
        _ax_ld.fill_between(_bin_mid, _corr_lo, _corr_hi, color="darkorange", alpha=0.25)
        _ax_ld.plot(_bin_mid, _corr_mid, "-", color="darkorange", lw=2, label="Corrected LD (posterior mean ± 90% CI)")

    _ax_ld.errorbar(
        _bin_mid, _obs_mean_ld, yerr=2 * _obs_sem_ld,
        fmt="o", color="steelblue", ms=4, lw=1.2, zorder=5,
        label="Observed mean LD ± 2 SEM",
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
