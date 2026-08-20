# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "arviz==1.1.0",
#     "arviz-plots==1.1.0",
#     "bayesld==0.1.0",
#     "demesdraw==0.4.1",
#     "marimo",
#     "matplotlib==3.10.9",
#     "matplotlib-label-lines==0.8.1",
#     "msprime==1.4.1",
#     "netcdf4==1.7.4",
#     "numba==0.67.0",
#     "numpy==2.4.4",
#     "pandas==3.0.2",
#     "scienceplots==2.2.2",
#     "seaborn==0.13.2",
#     "tqdm==4.70.0",
# ]
# ///

import marimo

__generated_with = "0.23.15"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import arviz as az
    import pandas as pd
    import numpy as np

    return az, mo, np, pd


@app.cell
def _(plt):
    import scienceplots
    plt.style.use("science")    # Theme settings
    plt.style.use("bin/theme.mplstyle")
    plt.rc("figure", autolayout=True)
    plt.rcParams["pgf.texsystem"] = "pdflatex"

    ONE_MM = 1 / 25.4
    SINGLE_COL = 85 * ONE_MM
    DOUBLE_COL = SINGLE_COL * 2
    ONE_HALF_COL = SINGLE_COL * 1.5
    return DOUBLE_COL, ONE_HALF_COL


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Read input data
    """)
    return


@app.cell
def _(pd):
    def _extend_up_to(data, target_time):
        last_row = data.iloc[-1]
        last_time = int(last_row['Generation'])

        new_times = range(last_time + 1, target_time + 1)
        new_rows = pd.DataFrame([last_row.to_dict()] * len(new_times))
        new_rows['Generation'] = list(new_times)

        return pd.concat([data, new_rows], ignore_index=True)
    gone = pd.read_csv("analysis/psiculus/gone/psiculus_GONE2_Ne", sep="\t")
    gone = _extend_up_to(gone, 2000)
    gone
    return (gone,)


@app.cell
def _(pd):
    def _extend_up_to(data, target_time):
        last_row = data.iloc[-1]
        last_time = last_row['TIME']

        new_times = range(last_time + 1, target_time + 1)
        new_rows = pd.DataFrame([last_row.to_dict()] * len(new_times))
        new_rows['TIME'] = list(new_times)

        return pd.concat([data, new_rows], ignore_index=True)
    hapne = pd.read_csv("analysis/psiculus/hapne/results/ld_hapne_estimate.csv")
    hapne = _extend_up_to(hapne, 2000)
    return (hapne,)


@app.cell
def _(np, pd):
    phlash = np.load("analysis/psiculus/phlash/phlash_posteriors.npz")
    phlash = pd.DataFrame(dict(
        Generation=phlash["T"],
        Ne = phlash["Nes"].mean(axis=0),
        lower = np.quantile(phlash["Nes"], 0.025, axis=0),
        median= np.quantile(phlash["Nes"], 0.5, axis=0),
        upper = np.quantile(phlash["Nes"], 0.975, axis=0),
    ))
    return (phlash,)


@app.cell
def _(phlash, plt):
    plt.figure(dpi=300)
    plt.plot("Generation", "median", data=phlash, label = "PHLASH", color = "C1")
    plt.plot("Generation", phlash["lower"], data=phlash, linestyle = "--", color = "C1")
    plt.plot("Generation", phlash["upper"], data=phlash, linestyle = "--", color = "C1")
    plt.fill_between("Generation", "lower", "upper", data=phlash, alpha = 0.1, color = "C1")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Generations ago")
    plt.ylabel("Population size")
    plt.xlim(1, phlash["Generation"].max())
    plt.title("PHLASH (PSMC-like)")
    plt.show()
    return


@app.cell
def _():
    import matplotlib.pyplot as plt

    return (plt,)


@app.cell
def _(az):
    idata_three = az.from_netcdf("results/psiculus/three_epoch.nc")
    return (idata_three,)


@app.cell
def _(az):
    idata_walk = az.from_netcdf("results/psiculus/random_walk.nc")
    return (idata_walk,)


@app.cell
def _(az, np):
    def pred_piecewise(idata):
        t_b = az.extract(idata, var_names="t_boundaries").values   # (n_bounds, n_draws)
        Ne  = az.extract(idata, var_names="Ne_values").values       # (n_epochs, n_draws)

        t = np.arange(0, 2000)
        n_draws = Ne.shape[1]
        matrix = np.empty((n_draws, t.size))

        # start with last epoch as default, then overwrite with earlier ones
        matrix[:] = Ne[-1][:, None]
        for i in reversed(range(t_b.shape[0])):
            mask = t[None, :] < t_b[i][:, None]
            matrix = np.where(mask, Ne[i][:, None], matrix)
        return matrix

    return (pred_piecewise,)


@app.cell
def _(idata_three, idata_walk, pred_piecewise):
    three_mat = pred_piecewise(idata_three)
    walk_mat = pred_piecewise(idata_walk)
    return three_mat, walk_mat


@app.cell
def _(np, pd, three_mat):
    three = pd.DataFrame({
        "TIME" : np.arange(0, 2000), 
        "Q0.025" : np.quantile(three_mat, 0.025, axis=0),
        "Q0.5" : np.quantile(three_mat, 0.5, axis=0),
        "Q0.975" : np.quantile(three_mat, 0.975, axis=0),
    })
    return (three,)


@app.cell
def _(np, pd, walk_mat):
    walk = pd.DataFrame({
        "TIME" : np.arange(0, 2000), 
        "Q0.025" : np.quantile(walk_mat, 0.025, axis=0),
        "Q0.5" : np.quantile(walk_mat, 0.5, axis=0),
        "Q0.975" : np.quantile(walk_mat, 0.975, axis=0),
    })
    return (walk,)


@app.cell
def _():
    first_breeding = 1972
    last_born = 2022
    generations_ago = (2022 - 1972) / 2
    generations_ago
    return (generations_ago,)


@app.cell
def _():
    from labellines import labelLine, labelLines


    return (labelLine,)


@app.cell
def _(
    DOUBLE_COL,
    ONE_HALF_COL,
    bayesld_three_sims,
    bayesld_walk_sims,
    data,
    generations_ago,
    gone,
    gone_sims,
    hapne,
    hapne_sims,
    labelLine,
    midpoints,
    np,
    phlash,
    phlash_sims,
    plt,
    sns,
    three,
    walk,
):
    COLORS = {
        r"bayesld (3-epoch)": "C0",
        r"bayesld (log-random walk)": "C1",
        "GONE": "C2",
        "HapNe-LD": "C3",
        "PHLASH": "C4",
    }


    def _labeled_vline(ax, x, y0, y1, label, label_frac=0.5, x_span=None):
        """A vertical dotted line, labeled mid-line via labellines.

        labelLine interpolates a line's y as a function of x, which breaks for
        a perfectly vertical line (x constant => no x-range to interpolate
        over) — so the line gets an imperceptible x-offset between its
        endpoints, wide enough to have a valid (if tiny) slope to interpolate
        along. label_frac in [0, 1] moves the label from y0 (0) to y1 (1).
        """
        if x_span is None:
            x_span = abs(x) * 1e-6 or 1e-6
        x_lo, x_hi = x - x_span, x + x_span
        line, = ax.plot([x_lo, x_hi], [y0, y1], color="black", lw=1, linestyle=":")
        label_x = x_lo + label_frac * (x_hi - x_lo)
        labelLine(line, x=label_x, label=label, align=True)
        return line


    def plot_ne_panel(ax, generations_ago, three, walk, hapne, phlash, gone):
        def series(x, median, lo, hi, color, label, linestyle="-"):
            ax.fill_between(x, lo, hi, color=color, alpha=0.15, linewidth=0)
            ax.plot(x, median, color=color, lw=2, linestyle=linestyle, label=label)

        series(three["TIME"], three["Q0.5"], three["Q0.025"], three["Q0.975"],
               COLORS[r"bayesld (3-epoch)"], r"bayesld (3-epoch)")
        series(walk["TIME"], walk["Q0.5"], walk["Q0.025"], walk["Q0.975"],
               COLORS[r"bayesld (log-random walk)"], r"bayesld (log-random walk)")
        series(hapne["TIME"], hapne["Q0.5"], hapne["Q0.025"], hapne["Q0.975"],
               COLORS["HapNe-LD"], "HapNe-LD", linestyle="--")
        series(phlash["Generation"], phlash["median"], phlash["lower"], phlash["upper"],
               COLORS["PHLASH"], "PHLASH", linestyle="--")
        ax.plot(gone["Generation"], gone["Ne_diploids"], color=COLORS["GONE"], lw=2,
                linestyle="--", label="GONE")

        ax.set_xlim(1, 2000)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Time ago (generations)")
        ax.set_ylabel(r"Effective population size $N_e$")

        ymin, ymax = ax.get_ylim()
        target_y = 1e4
        label_frac = (target_y - ymin) / (ymax - ymin)
        _labeled_vline(ax, generations_ago, ymin, ymax, "Founder event", label_frac)
        ax.set_ylim(ymin, ymax)


    def plot_diversity_panel(ax, data, bayesld_three_sims, bayesld_walk_sims, phlash_sims):
        # Raw diversity values are tiny decimals (e.g. 0.00083), which makes for
        # long, cluttered tick labels — rescale by 1e3 and fold the factor into
        # the axis label instead (avoids matplotlib's auto sci-notation offset
        # text, which tends to collide with the label in this narrow panel).
        scale = 1e3

        sns.kdeplot(bayesld_three_sims[0] * scale, color=COLORS[r"bayesld (3-epoch)"], fill=True,
                    alpha=0.15, lw=2, label=r"bayesld (3-epoch)", ax=ax)
        sns.kdeplot(bayesld_walk_sims[0] * scale, color=COLORS[r"bayesld (log-random walk)"], fill=True,
                    alpha=0.15, lw=2, label=r"bayesld (log-random walk)", ax=ax)
        sns.kdeplot(phlash_sims[0] * scale, color=COLORS["PHLASH"], fill=True,
                    alpha=0.15, lw=2, linestyle="--", label="PHLASH", ax=ax)

        ax.set_xlabel(r"Nucleotide diversity ($\times 10^{-3}$)")
        ax.set_ylabel("Density")

        empirical_diversity = data["mean_genetic_diversity"].mean() * scale
        ymin, ymax = ax.get_ylim()
        xmin, xmax = ax.get_xlim()
        _labeled_vline(ax, empirical_diversity, ymin, ymax, "Genome-wide mean",
                       label_frac=0.715, x_span=(xmax - xmin) * 1e-6)
        ax.set_ylim(ymin, ymax)


    def plot_ld_panel(ax, data, midpoints, bayesld_three_sims, bayesld_walk_sims,
                       gone_sims, hapne_sims, phlash_sims):
        x = midpoints * 100  # centimorgan, matching convention elsewhere

        def sem(sims):
            return sims.std(axis=0) / np.sqrt(sims.shape[0])

        def band(sims, color, label, linestyle="-"):
            mean = sims.mean(axis=0)
            se = sem(sims)
            ax.fill_between(x, mean - se, mean + se, color=color, alpha=0.15, linewidth=0)
            ax.plot(x, mean, color=color, lw=2, linestyle=linestyle, label=label)

        ax.errorbar(
            x,
            data["mean_linkage_disequilibrium"].mean(axis=0),
            yerr=sem(data["mean_linkage_disequilibrium"]),
            fmt="o",
            markersize=5,
            elinewidth=1.5,
            capsize=2,
            color="black",
            label="Empirical data",
            zorder=5,
        )

        band(bayesld_three_sims[1], COLORS[r"bayesld (3-epoch)"], r"bayesld (3-epoch)")
        band(bayesld_walk_sims[1], COLORS[r"bayesld (log-random walk)"], r"bayesld (log-random walk)")
        band(gone_sims[1], COLORS["GONE"], "GONE", linestyle="--")
        band(hapne_sims[1], COLORS["HapNe-LD"], "HapNe-LD", linestyle="--")
        band(phlash_sims[1], COLORS["PHLASH"], "PHLASH", linestyle="--")

        ax.set_xlabel("Genetic distance (centimorgan)")
        ax.set_ylabel(r"$\overline{X_iX_jY_iY_j}$")


    def shared_legend(fig, axes):
        handles, labels = [], []
        for ax in axes:
            h, l = ax.get_legend_handles_labels()
            for hi, li in zip(h, l):
                if li not in labels:
                    handles.append(hi)
                    labels.append(li)

        fig.legend(
            handles, labels,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=3,
            frameon=False,
        )


    def make_figure(data, midpoints, generations_ago, three, walk, hapne, phlash, gone,
                     bayesld_three_sims, bayesld_walk_sims, gone_sims, hapne_sims, phlash_sims):
        fig = plt.figure(figsize=(DOUBLE_COL, ONE_HALF_COL * 1.3), dpi=300, constrained_layout=True)
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[0.3, 0.7])

        ax_ne = fig.add_subplot(gs[0, :])
        ax_div = fig.add_subplot(gs[1, 0])
        ax_ld = fig.add_subplot(gs[1, 1])

        plot_ne_panel(ax_ne, generations_ago, three, walk, hapne, phlash, gone)
        plot_diversity_panel(ax_div, data, bayesld_three_sims, bayesld_walk_sims, phlash_sims)
        plot_ld_panel(ax_ld, data, midpoints, bayesld_three_sims, bayesld_walk_sims,
                      gone_sims, hapne_sims, phlash_sims)

        shared_legend(fig, (ax_ne, ax_ld, ax_div))
        return fig


    fig = make_figure(data, midpoints, generations_ago, three, walk, hapne, phlash, gone,
                       bayesld_three_sims, bayesld_walk_sims, gone_sims, hapne_sims, phlash_sims)
    fig.savefig("results/psiculus/plot.pdf")
    fig.savefig("results/psiculus/plot.pgf")
    fig
    return


@app.cell
def _(gone, hapne, np):
    def demography_from_columns(time_col, ne_col, population_name="pop"):
        """
        time_col: array-like of times ago (generations), increasing, first entry
                  assumed to be the present-day epoch
        ne_col:   array-like of Ne for each corresponding epoch
        """
        import msprime

        time_col = np.asarray(time_col)
        ne_col = np.asarray(ne_col)
        order = np.argsort(time_col)
        time_col, ne_col = time_col[order], ne_col[order]

        demography = msprime.Demography()
        demography.add_population(name=population_name, initial_size=ne_col[0])

        for t, Ne in zip(time_col[1:], ne_col[1:]):
            demography.add_population_parameters_change(
                time=t, initial_size=Ne, population=population_name
            )

        demography.sort_events()
        return demography


    gone_demo = demography_from_columns(gone["Generation"], gone["Ne_diploids"], "gone")
    hapne_demo = demography_from_columns(hapne["TIME"], hapne["Q0.5"], "hapne")
    return demography_from_columns, gone_demo, hapne_demo


@app.cell
def _(demography_from_columns, np):
    _phlash = np.load("analysis/psiculus/phlash/phlash_posteriors.npz")
    phlash_demo = demography_from_columns(
        _phlash["T"],
        np.quantile(_phlash["Nes"], 0.5, axis=0),
        "phlash"
    )
    return (phlash_demo,)


@app.cell
def _(demography_from_columns, three, walk):
    bayesld3_demo = demography_from_columns(three["TIME"], three["Q0.5"], "three")
    bayesld_walk_demo = demography_from_columns(walk["TIME"], walk["Q0.5"], "walk")
    return bayesld3_demo, bayesld_walk_demo


@app.cell
def _(bayesld, data, mutation_rate, recombination_rate, window_length):
    def simulate(demography, random_seed):
        return bayesld.sim_sufficient_stats(
            samples = data["sample_size"],
            demography = demography,
            mutation_rate = mutation_rate,
            random_seed = random_seed,
            recombination_rate=recombination_rate,
            sequence_length=window_length,
            left_bins = data["left_bins_morgan"],
            right_bins = data["right_bins_morgan"],
            num_workers=10,
            num_replicates=data["mean_genetic_diversity"].shape[0]
        )

    return (simulate,)


@app.cell
def _(gone_demo):
    import demesdraw
    demesdraw.size_history(gone_demo.to_demes())
    return


@app.cell
def _(gone_demo, simulate):
    gone_sims = simulate(gone_demo, 3671)
    return (gone_sims,)


@app.cell
def _(hapne_demo, simulate):
    hapne_sims = simulate(hapne_demo, 3672)
    return (hapne_sims,)


@app.cell
def _(phlash_demo, simulate):
    phlash_sims = simulate(phlash_demo, 3673)
    return (phlash_sims,)


@app.cell
def _(bayesld3_demo, simulate):
    bayesld_three_sims = simulate(bayesld3_demo, 3674)
    return (bayesld_three_sims,)


@app.cell
def _(bayesld_walk_demo, simulate):
    bayesld_walk_sims = simulate(bayesld_walk_demo, 3675)
    return (bayesld_walk_sims,)


@app.cell
def _(data):
    midpoints = (data["left_bins_morgan"] + data["right_bins_morgan"]) / 2
    return (midpoints,)


@app.cell
def _():
    import seaborn as sns

    return (sns,)


@app.cell
def _(pd):
    data = pd.read_pickle("analysis/psiculus/PM_data.pkl")
    return (data,)


@app.function
def posterior_demographies(idata, num_samples=50, random_seed=1234):
    import arviz as az
    import msprime
    posterior = az.extract(
        idata,
        var_names=["Ne_values", "t_boundaries"],
        num_samples=num_samples,
        random_seed=random_seed,
    )
    Ne = posterior["Ne_values"].values
    boundaries = posterior["t_boundaries"].values
    demographies = []
    for sample_index in range(Ne.shape[1]):
        demo = msprime.Demography()
        demo.add_population(name="pop", initial_size=Ne[0, sample_index])
        for epoch_index, time in enumerate(boundaries[:, sample_index]):
            demo.add_population_parameters_change(
                time=time,
                initial_size=Ne[epoch_index + 1, sample_index],
                population="pop",
            )
        demo.sort_events()
        demographies.append(demo)
    return demographies


@app.cell
def _(DOUBLE_COL, ONE_HALF_COL, data, idata_three, plt, simulate, sns):
    def plot_ppc(idata, num_samples=50, random_seed=1234):
        from matplotlib.lines import Line2D

        ld_bins = (0, 4, 12)
        ppc_color = "C7"
        # Sample parameters
        ppc_demographies = posterior_demographies(idata, num_samples=num_samples)
        simulations = []
        for i, x in enumerate(ppc_demographies):
            print(i)
            while True:
                try:
                    simulations.append(simulate(x, random_seed+i))
                except:
                    print("error")
                break
        fig, axes = plt.subplots(
            2, 2,
            figsize=(DOUBLE_COL, ONE_HALF_COL),
            dpi=300,
            constrained_layout=True,
        )
        ax = axes[0, 0]
        for sim in simulations:
            sns.kdeplot(sim[0], ax=ax, color=ppc_color, alpha=0.2, lw=1)
        sns.kdeplot(
            idata["observed_data"]["observed_pi"],
            ax=ax, color="black", lw=1.5,
        )
        ax.set_title("Nucleotide diversity")
        ax.set_xlabel(r"Nucleotide diversity $\pi$")
        ax.set_ylabel("Density")
        for ax, bin_index in zip(axes.flat[1:], ld_bins):
            for sim in simulations:
                sns.kdeplot(
                    sim[1][:, bin_index],
                    ax=ax,
                    color=ppc_color,
                    alpha=0.2,
                    lw=1,
                )
            sns.kdeplot(
                idata["observed_data"]["observed_ld"][:, bin_index],
                ax=ax,
                color="black",
                lw=1.5,
            )
            left_cm = data["left_bins_morgan"][bin_index] * 100
            right_cm = data["right_bins_morgan"][bin_index] * 100
            ax.set_title(rf"$u \in ({left_cm:.3g}, {right_cm:.3g})$ cM")
            ax.set_xlabel(r"$\overline{X_iX_jY_iY_j}$")
            ax.set_ylabel("Density")
        fig.legend(
            handles=[
                Line2D([0], [0], color=ppc_color, lw=2, label="Posterior predictive"),
                Line2D([0], [0], color="black", lw=2, label="Observed data"),
            ],
            loc="outside lower center",
            ncol=2,
            frameon=False,
        )
        return fig, axes

    ppc_fig, ppc_axes = plot_ppc(idata_three, num_samples=30)
    ppc_fig.savefig("results/psiculus/ppc.pdf")
    ppc_fig.savefig("results/psiculus/ppc.pgf")
    plt.show()
    return (ppc_fig,)


@app.cell
def _(ppc_fig):
    ppc_fig
    return


@app.cell
def _(data):
    generation_time = 2
    mutation_rate_per_year = 1.98e-9
    mutation_rate = mutation_rate_per_year * generation_time
    recombination_rate = 1.59e-8
    window_length = int(data["right_bins_morgan"][-1] * 2 / recombination_rate)
    return mutation_rate, recombination_rate, window_length


@app.cell
def _():
    import bayesld


    return (bayesld,)


if __name__ == "__main__":
    app.run()
