import marimo

__generated_with = "0.23.3"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from bayesld import linear_bins
    from bayesld import montecarlo as mc
    from bayesld.models import PiecewiseExponentialDemography

    return PiecewiseExponentialDemography, linear_bins, mc, mo, np, plt


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
        10, 50_000, value=4000, step=10, label="Ne_c (contemporary, truth)"
    )
    ne_a_slider = mo.ui.slider(
        10, 50_000, value=10000, step=10, label="Ne_a (ancestral, truth)"
    )
    t0_slider = mo.ui.slider(
        1, 500, value=30, step=1, label="t0 (transition time, gen ago)"
    )
    sample_size_slider = mo.ui.slider(
        10, 200, value=50, step=10, label="Sample size (diploid)"
    )
    num_windows_slider = mo.ui.slider(10, 500, value=50, step=10, label="Windows")
    mo.vstack(
        [
            mo.md("## Simulation parameters"),
            ne_c_slider,
            ne_a_slider,
            t0_slider,
            sample_size_slider,
            num_windows_slider,
        ]
    )
    return (
        ne_a_slider,
        ne_c_slider,
        num_windows_slider,
        sample_size_slider,
        t0_slider,
    )


@app.cell
def _(
    ne_a_slider,
    ne_c_slider,
    num_windows_slider,
    sample_size_slider,
    t0_slider,
):
    Ne_c_truth = ne_c_slider.value
    Ne_a_truth = ne_a_slider.value
    t0_truth = t0_slider.value
    sample_size = sample_size_slider.value
    num_windows = num_windows_slider.value
    return Ne_a_truth, Ne_c_truth, num_windows, sample_size, t0_truth


@app.cell
def _(
    Ne_a_truth,
    Ne_c_truth,
    left_bins,
    mc,
    mo,
    mutation_rate,
    np,
    num_windows,
    recombination_rate,
    right_bins,
    sample_size,
    t0_truth,
    window_length,
):
    _lfc = float(np.log(Ne_c_truth / Ne_a_truth)) if Ne_a_truth > 0 else 0.0
    _alpha = _lfc / t0_truth if t0_truth > 0 else 0.0
    with mo.status.spinner(
        f"Simulating {num_windows} windows at "
        f"Ne_c={Ne_c_truth:,}, Ne_a={Ne_a_truth:,}, t0={t0_truth}..."
    ):
        _pi, _ld = mc.expected_piecewise_exponential(
            float(Ne_c_truth),
            float(Ne_a_truth),
            float(t0_truth),
            float(_alpha),
            left_bins,
            right_bins,
            mutation_rate,
            recombination_rate,
            window_length,
            sample_size,
            random_seed=42,
            num_replicates=num_windows,
            ploidy=2,
            num_workers=8,
            model="hudson",
        )
    pi_data = np.array(_pi)
    ld_data = np.array(_ld)
    mo.callout(
        mo.md(
            f"Simulated **{len(pi_data)}** windows - "
            f"Ne_c={Ne_c_truth:,}, Ne_a={Ne_a_truth:,}, "
            f"t0={t0_truth} gen, alpha={_alpha:.4f} - "
            f"n={sample_size} - L={window_length / 1e6:.0f} Mb"
        ),
        kind="success",
    )
    return ld_data, pi_data


@app.cell
def _(
    PiecewiseExponentialDemography,
    ld_data,
    left_bins,
    mo,
    mutation_rate,
    np,
    pi_data,
    recombination_rate,
    right_bins,
    sample_size,
    window_length,
):
    mo.stop(pi_data is None, mo.callout(mo.md("Simulate data first."), kind="warn"))
    # Generic broad priors covering slider ranges (Ne: 10–50,000; t0: 1–500)
    model = PiecewiseExponentialDemography(
        diversity=pi_data,
        ld=ld_data,
        mutation_rate=mutation_rate,
        recombination_rate=recombination_rate,
        num_samples=sample_size,
        left_bins=left_bins,
        right_bins=right_bins,
        sequence_length=window_length,
        num_workers=8,
        prior=(
            f"    log_Ne_c ~ normal({np.log(5000.0):.4f}, 1.5);\n"
            f"    log_Ne_a ~ normal({np.log(5000.0):.4f}, 1.5);\n"
            f"    log_t0   ~ normal({np.log(50.0):.4f}, 1.5);"
        ),
    )
    mo.md("Model compiled.")
    return (model,)


@app.cell
def _(mo, model):
    mo.stop(model is None)
    with mo.status.spinner("NUTS sampling (no bias correction)..."):
        idata_baseline = model.sample(
            iter_warmup=2000,
            iter_sampling=2000,
            seed=1,
        )
    mo.md("Baseline sampling complete.")
    return (idata_baseline,)


@app.cell
def _(mo, model):
    mo.stop(model is None)
    with mo.status.spinner("Active learning (bias correction)..."):
        model.active_learn_bias(
            n_points_per_iter=5,
            n_iter=5,
            max_tolerance=0.1,
            strategy="pathfinder",
            seed=41,
        )
    n_pts = len(model.synthetic_points)
    mo.md(f"Active learning complete: **{n_pts}** synthetic points accumulated.")
    return (n_pts,)


@app.cell
def _(left_bins, mo, model, n_pts, np, plt, right_bins):
    mo.stop(n_pts is None or n_pts == 0)

    points = model.synthetic_points
    _bin_mid = (np.array(left_bins) + np.array(right_bins)) / 2
    _n_bins = len(_bin_mid)

    # Cumulative mean and SE of relative bias after each point
    all_bias = np.array([p["rel_bias"] for p in points])  # (n_pts, n_bins)
    cum_mean = np.cumsum(all_bias, axis=0) / np.arange(1, len(all_bias) + 1)[:, None]
    cum_std = np.zeros_like(cum_mean)
    for k in range(2, len(all_bias) + 1):
        cum_std[k - 1] = all_bias[:k].std(axis=0, ddof=1) / np.sqrt(k)

    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Left: one line per bin showing cumulative mean bias across points
    for j in range(_n_bins):
        _ax1.plot(
            np.arange(1, len(all_bias) + 1),
            100 * cum_mean[:, j],
            alpha=0.6,
            lw=1.2,
        )
    _ax1.axhline(0, color="gray", lw=0.8, ls="--")
    _ax1.set_xlabel("Cumulative synthetic points")
    _ax1.set_ylabel("Cumulative mean bias (%)")
    _ax1.set_title("Bias convergence per bin")

    # Right: final mean bias +/- SE across bins
    final_mean = cum_mean[-1]
    final_se = cum_std[-1]
    _ax2.errorbar(
        _bin_mid,
        100 * final_mean,
        yerr=100 * final_se,
        fmt="o-",
        ms=4,
        lw=1.2,
        color="purple",
    )
    _ax2.axhline(0, color="gray", lw=0.8, ls="--")
    _ax2.set_xlabel("Recombination distance (Morgans)")
    _ax2.set_ylabel("Mean bias +/- SE (%)")
    _ax2.set_title(f"Final bias estimate ({len(all_bias)} points)")
    plt.tight_layout()
    _fig
    return


@app.cell
def _(mo, model, n_pts):
    mo.stop(n_pts is None or n_pts == 0)
    with mo.status.spinner("NUTS sampling (with bias correction)..."):
        idata_corrected = model.sample(
            iter_warmup=2000,
            iter_sampling=2000,
            seed=2,
        )
    mo.md("Corrected sampling complete.")
    return (idata_corrected,)


@app.cell
def _(idata_corrected):
    import arviz_stats as azs

    azs.summary(idata_corrected)
    return


@app.cell
def _(Ne_a_truth, Ne_c_truth, mutation_rate, np, pi_data, plt, t0_truth):
    # Draw from the prior
    _rng = np.random.default_rng(0)
    _n_prior = 10_000
    _log_ne_mu = np.log(np.mean(pi_data) / (4.0 * mutation_rate))
    _log_Ne_a = _rng.normal(_log_ne_mu, 1.0, _n_prior)
    _log_fc = _rng.normal(0, 1.0, _n_prior)
    _prior_Ne_a = np.exp(_log_Ne_a)
    _prior_Ne_c = np.exp(_log_Ne_a + _log_fc)
    _prior_t0 = np.exp(_rng.normal(np.log(100.0), 0.5, _n_prior))

    _fig, (_ax1, _ax2, _ax3) = plt.subplots(1, 3, figsize=(14, 4))
    for _ax, _prior, _truth, _label in [
        (_ax1, _prior_Ne_c, Ne_c_truth, "Ne_c"),
        (_ax2, _prior_Ne_a, Ne_a_truth, "Ne_a"),
        (_ax3, _prior_t0, t0_truth, "t0 (gen)"),
    ]:
        _ax.hist(_prior, bins=50, alpha=0.5, density=True, color="gray")
        _ax.axvline(_truth, color="black", ls="--", lw=2, label=f"Truth ({_truth:,})")
        _ax.set_xlabel(_label)
        _ax.set_ylabel("Density")
        _ax.legend(fontsize=8)
    _fig.suptitle("Prior distributions")
    plt.tight_layout()
    _fig
    return


@app.cell
def _(
    Ne_a_truth,
    Ne_c_truth,
    idata_baseline,
    idata_corrected,
    mo,
    np,
    plt,
    t0_truth,
):
    mo.stop(idata_corrected is None)

    ne_c_base = np.array(idata_baseline["posterior"].ds["Ne_c"]).ravel()
    ne_c_corr = np.array(idata_corrected["posterior"].ds["Ne_c"]).ravel()
    ne_a_base = np.array(idata_baseline["posterior"].ds["Ne_a"]).ravel()
    ne_a_corr = np.array(idata_corrected["posterior"].ds["Ne_a"]).ravel()
    t0_base = np.array(idata_baseline["posterior"].ds["t0"]).ravel()
    t0_corr = np.array(idata_corrected["posterior"].ds["t0"]).ravel()

    _fig, (_ax1, _ax2, _ax3) = plt.subplots(1, 3, figsize=(14, 4))
    for _ax, _base, _corr, _truth, _label in [
        (_ax1, ne_c_base, ne_c_corr, Ne_c_truth, "Ne_c"),
        (_ax2, ne_a_base, ne_a_corr, Ne_a_truth, "Ne_a"),
        (_ax3, t0_base, t0_corr, t0_truth, "t0 (gen)"),
    ]:
        _ax.hist(_base, bins=50, alpha=0.5, density=True, label="Baseline")
        _ax.hist(_corr, bins=50, alpha=0.5, density=True, label="Corrected")
        _ax.axvline(_truth, color="black", ls="--", lw=2, label=f"Truth ({_truth:,})")
        _ax.set_xlabel(_label)
        _ax.set_ylabel("Density")
        _ax.legend(fontsize=8)
    _fig.suptitle("Posterior: baseline vs bias-corrected")
    plt.tight_layout()
    _fig
    return


@app.cell
def _(
    idata_baseline,
    idata_corrected,
    ld_data,
    left_bins,
    mo,
    np,
    plt,
    right_bins,
):
    mo.stop(idata_corrected is None)

    bin_mid = (np.array(left_bins) + np.array(right_bins)) / 2
    obs_mean_ld = np.mean(ld_data, axis=0)
    obs_sem_ld = np.std(ld_data, axis=0, ddof=1) / np.sqrt(len(ld_data))

    _fig, (ax_ld, ax_bias) = plt.subplots(
        2, 1, figsize=(7, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )

    # Baseline
    base_ld = np.array(idata_baseline["posterior"].ds["corrected_expected_ld"]).reshape(
        -1, len(bin_mid)
    )
    base_lo, base_hi = np.percentile(base_ld, [5, 95], axis=0)
    ax_ld.fill_between(bin_mid, base_lo, base_hi, color="tomato", alpha=0.2)
    ax_ld.plot(
        bin_mid, base_ld.mean(axis=0), "-", color="tomato", lw=2, label="Baseline"
    )

    # Corrected
    corr_ld = np.array(idata_corrected["posterior"].ds["corrected_expected_ld"]).reshape(
        -1, len(bin_mid)
    )
    corr_lo, corr_hi = np.percentile(corr_ld, [5, 95], axis=0)
    ax_ld.fill_between(bin_mid, corr_lo, corr_hi, color="darkorange", alpha=0.2)
    ax_ld.plot(
        bin_mid, corr_ld.mean(axis=0), "-", color="darkorange", lw=2, label="Corrected"
    )

    ax_ld.errorbar(
        bin_mid,
        obs_mean_ld,
        yerr=2 * obs_sem_ld,
        fmt="o",
        color="steelblue",
        ms=4,
        lw=1.2,
        zorder=5,
        label="Observed +/- 2 SEM",
    )
    ax_ld.set_ylabel("Mean LD")
    ax_ld.set_title("Posterior predictive LD")
    ax_ld.legend(fontsize=8)

    # GP bias panel
    bias_draws = np.array(idata_corrected["posterior"].ds["gp_bias_ld"]).reshape(
        -1, len(bin_mid)
    )
    bias_lo, bias_hi = np.percentile(bias_draws, [5, 95], axis=0)
    ax_bias.fill_between(
        bin_mid, 100 * bias_lo, 100 * bias_hi, color="purple", alpha=0.25
    )
    ax_bias.plot(bin_mid, 100 * bias_draws.mean(axis=0), "-", color="purple", lw=2)
    ax_bias.axhline(0, color="gray", lw=0.8, ls="--")
    ax_bias.set_ylabel("GP bias (%)")
    ax_bias.set_xlabel("Recombination distance (Morgans)")
    plt.tight_layout()
    _fig
    return


if __name__ == "__main__":
    app.run()
