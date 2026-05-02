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
    from bayesld.models import ConstantDemography

    return ConstantDemography, linear_bins, mc, mo, np, plt


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
    ne_slider = mo.ui.slider(10, 20_000, value=1_000, step=1, label="Ne truth")
    sample_size_slider = mo.ui.slider(
        10, 200, value=50, step=10, label="Sample size (diploid)"
    )
    num_windows_slider = mo.ui.slider(10, 500, value=50, step=10, label="Windows")
    mo.vstack(
        [
            mo.md("## Simulation parameters"),
            ne_slider,
            sample_size_slider,
            num_windows_slider,
        ]
    )
    return ne_slider, num_windows_slider, sample_size_slider


@app.cell
def _(ne_slider, num_windows_slider, sample_size_slider):
    Ne_truth = ne_slider.value
    sample_size = sample_size_slider.value
    num_windows = num_windows_slider.value
    return Ne_truth, num_windows, sample_size


@app.cell
def _(
    Ne_truth,
    left_bins,
    mc,
    mo,
    mutation_rate,
    np,
    num_windows,
    recombination_rate,
    right_bins,
    sample_size,
    window_length,
):
    with mo.status.spinner(f"Simulating {num_windows} windows at Ne={Ne_truth:,}..."):
        _pi, _ld = mc.expected_constant(
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
            num_workers=8,
            model="hudson",
        )
    pi_data = np.array(_pi)
    ld_data = np.array(_ld)
    mo.callout(
        mo.md(
            f"Simulated **{len(pi_data)}** windows - "
            f"Ne={Ne_truth:,} - n={sample_size} - "
            f"L={window_length / 1e6:.0f} Mb"
        ),
        kind="success",
    )
    return ld_data, pi_data


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
    mo.stop(pi_data is None, mo.callout(mo.md("Simulate data first."), kind="warn"))
    model = ConstantDemography(
        diversity=pi_data,
        ld=ld_data,
        mutation_rate=mutation_rate,
        recombination_rate=recombination_rate,
        num_samples=sample_size,
        left_bins=left_bins,
        right_bins=right_bins,
        sequence_length=window_length,
        num_workers=8,
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
def _(Ne_truth, idata_baseline, idata_corrected, mo, np, plt):
    mo.stop(idata_corrected is None)

    ne_base = np.array(idata_baseline["posterior"].ds["Ne"]).ravel()
    ne_corr = np.array(idata_corrected["posterior"].ds["Ne"]).ravel()

    _fig, _ax = plt.subplots(figsize=(7, 4))
    _ax.hist(ne_base, bins=50, alpha=0.5, density=True, label="Baseline (no GP)")
    _ax.hist(ne_corr, bins=50, alpha=0.5, density=True, label="Corrected (with GP)")
    _ax.axvline(
        Ne_truth, color="black", ls="--", lw=2, label=f"Truth (Ne={Ne_truth:,})"
    )
    _ax.set_xlabel("Ne")
    _ax.set_ylabel("Density")
    _ax.set_title("Posterior Ne: baseline vs bias-corrected")
    _ax.legend()
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

    # Baseline corrected_ld (GP near zero)
    base_ld = np.array(idata_baseline["posterior"].ds["corrected_ld"]).reshape(
        -1, len(bin_mid)
    )
    base_lo, base_hi = np.percentile(base_ld, [5, 95], axis=0)
    ax_ld.fill_between(bin_mid, base_lo, base_hi, color="tomato", alpha=0.2)
    ax_ld.plot(
        bin_mid, base_ld.mean(axis=0), "-", color="tomato", lw=2, label="Baseline"
    )

    # Corrected
    corr_ld = np.array(idata_corrected["posterior"].ds["corrected_ld"]).reshape(
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
