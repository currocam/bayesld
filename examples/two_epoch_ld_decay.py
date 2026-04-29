import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from bayesld import linear_bins
    from bayesld import montecarlo as mc

    return linear_bins, mc, mo, np, plt


@app.cell
def _(linear_bins):
    mutation_rate = 1e-8
    recombination_rate = 1e-8
    left_bins, right_bins = linear_bins()
    sequence_length = right_bins[-1] * 2 / recombination_rate
    return left_bins, mutation_rate, recombination_rate, right_bins, sequence_length


@app.cell
def _(
    left_bins,
    mc,
    mutation_rate,
    recombination_rate,
    right_bins,
    sequence_length,
):
    Ne1 = 5_000
    Ne2 = 10_000
    t_boundary = 50  # generations ago

    pi_vec, ld_mat = mc.expected_piecewise_constant(
        Ne_values=[Ne1, Ne2],
        t_boundaries=[t_boundary],
        left_bins=left_bins,
        right_bins=right_bins,
        mutation_rate=mutation_rate,
        recombination_rate=recombination_rate,
        sequence_length=sequence_length,
        sample_size=50,
        random_seed=42,
        num_replicates=50,
    )
    return Ne1, Ne2, ld_mat, pi_vec, t_boundary


@app.cell
def _(Ne1, Ne2, ld_mat, left_bins, np, plt, right_bins, t_boundary):
    bin_midpoints = (left_bins + right_bins) / 2

    fig, ax = plt.subplots(figsize=(8, 5))
    # individual replicates
    for i in range(ld_mat.shape[0]):
        ax.plot(bin_midpoints, ld_mat[i], color="steelblue", alpha=0.1, linewidth=0.5)
    # mean
    ax.plot(
        bin_midpoints,
        np.mean(ld_mat, axis=0),
        color="darkblue",
        linewidth=2,
        label="mean",
    )
    ax.set_xlabel("Genetic distance (Morgan)")
    ax.set_ylabel("LD (r\u00b2)")
    ax.set_title(
        f"LD decay  |  Ne\u2081={Ne1:,}, Ne\u2082={Ne2:,}, "
        f"t={t_boundary} gen  |  SMCK(k=1)"
    )
    ax.legend()
    fig.tight_layout()
    fig
    return


if __name__ == "__main__":
    app.run()
