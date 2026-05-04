#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "bayesld @ git+https://github.com/currocam/bayesld.git@ae4d14b",
#     "marimo",
#     "matplotlib",
#     "numpy",
#     "demesdraw",
#     "stdpopsim==0.3.0",
# ]
# ///

import marimo

__generated_with = "0.23.3"
app = marimo.App(width="medium")


@app.cell
def _():
    import gzip
    import pickle
    from pathlib import Path

    import demesdraw
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import stdpopsim
    from bayesld import deterministic, linear_bins

    plt.style.use(Path(__file__).parent / "theme.mplstyle")
    left_bins, right_bins = linear_bins()
    bin_midpoints = (left_bins + right_bins) / 2
    return (
        Path,
        bin_midpoints,
        demesdraw,
        deterministic,
        left_bins,
        mo,
        np,
        plt,
        right_bins,
        stdpopsim,
    )


@app.cell
def _(mo):
    mo.md("""
    # LD predictions vs stdpopsim simulations
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    Compare deterministic LD predictions from `bayesld` against
    forward/coalescent simulations from `stdpopsim` for three species.
    """)
    return


@app.function
def bootstrap_ci(data, n_boot=10_000, ci=0.95, seed=1000):
    import numpy as np

    if data.ndim > 2:
        raise NotImplementedError
    if data.ndim == 2:
        return np.array(
            [bootstrap_ci(x, n_boot=n_boot, ci=ci, seed=seed) for x in data.T]
        )
    rng = np.random.default_rng(seed)
    boots = rng.choice(data, size=(n_boot, len(data)), replace=True).mean(axis=1)
    alpha = (1 - ci) / 2
    lo, hi = np.quantile(boots, [alpha, 1 - alpha])
    m = data.mean()
    return m, lo, hi


@app.cell
def _(np):
    def load_data(path):
        import gzip
        import pickle

        with gzip.open(path, "rb") as f:
            return pickle.load(f)

    def piecewise_constant_demography(model):
        """Extract (Ne_values, t_boundaries) from a stdpopsim demographic model."""
        demo = model.model
        assert len(demo.populations), "Only one population only"
        Ne_values = [demo.populations[0].initial_size]
        t_boundaries = []
        for event in demo.events:
            assert event.time > 0
            Ne_values.append(float(event.initial_size))
            t_boundaries.append(float(event.time))
        Ne_values = np.array(Ne_values)
        t_boundaries = np.array(t_boundaries)
        assert np.all(np.sort(t_boundaries) == t_boundaries)
        return np.array(Ne_values), np.array(t_boundaries)

    return load_data, piecewise_constant_demography


@app.cell
def _(
    bin_midpoints,
    demesdraw,
    deterministic,
    left_bins,
    load_data,
    np,
    piecewise_constant_demography,
    plt,
    right_bins,
    stdpopsim,
):
    def plot_species(data_path, species_id, model_id, title):
        """Plot demography and LD decay with 90% CI for one species."""
        data = load_data(data_path)
        results = data["results"]
        params = data["params"]
        num_samples = params["num_samples"]
        num_lineages = num_samples
        mutation_rate = params["mutation_rate"]

        # --- Extract per-window arrays ---
        ld_matrix = np.array([r["mean_linkage_disequilibrium"] for r in results])

        # --- Get demography and predictions ---
        species = stdpopsim.get_species(species_id)
        model = species.get_demographic_model(model_id)
        Ne_values, t_boundaries = piecewise_constant_demography(model)

        _pi_pred, ld_pred = deterministic.expected_piecewise_constant(
            Ne_values=Ne_values,
            t_boundaries=t_boundaries,
            left_bins=left_bins,
            right_bins=right_bins,
            mutation_rate=mutation_rate,
            sample_size=num_samples,
        )

        # --- Figure ---
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        fig.suptitle(
            f"{title} (n={num_lineages} diploids)", fontsize=14, fontweight="bold"
        )

        # Panel A: Demography
        ax = axes[0]
        demes_graph = model.model.to_demes()
        demesdraw.tubes(demes_graph, ax=ax, colours="C0", log_time=True)
        ax.set_title("Demographic history")

        # Panel B: LD decay with 95% CI
        ax = axes[1]
        x = bin_midpoints * 100
        boots = bootstrap_ci(ld_matrix, n_boot=10_000, ci=0.95, seed=1000)
        ld_lo = boots[:, 1]
        ld_hi = boots[:, 2]
        ax.fill_between(x, ld_lo, ld_hi, color="C0", alpha=0.2)
        ax.plot(x, boots[:, 0], color="C0", linewidth=2, label="Observed mean")
        ax.plot(
            x,
            ld_pred,
            color="C1",
            linewidth=2,
            linestyle="--",
            label="Approximate prediction",
        )
        ax.set_xlabel("Genetic distance (centimorgan)")
        ax.set_ylabel(r"Linkage disequilibrium $\mathbb E[X_iX_jY_iY_j]$")
        ax.set_title("Linkage disequilibrium decay")
        ax.legend()

        plt.tight_layout()
        return fig

    return (plot_species,)


@app.cell
def _(mo):
    mo.md("""
    ## Holstein-Friesian cattle (MacLeod et al., 2013)
    """)
    return


@app.cell
def _(Path, plot_species):
    _fig = plot_species(
        data_path=Path(".") / "holsteinfriesian.pkl.gz",
        species_id="BosTau",
        model_id="HolsteinFriesian_1M13",
        title="Holstein-Friesian cattle -- HolsteinFriesian_1M13",
    )
    _fig.savefig("stdpopsim_holsteinfriesian.pdf")
    _fig
    return


@app.cell
def _(mo):
    mo.md("""
    ## Vaquita (Robinson et al., 2022)
    """)
    return


@app.cell
def _(Path, plot_species):
    _fig = plot_species(
        data_path=Path(".") / "vaquita.pkl.gz",
        species_id="PhoSin",
        model_id="Vaquita2Epoch_1R22",
        title="Vaquita -- Vaquita2Epoch_1R22",
    )
    _fig.savefig("stdpopsim_vaquita.pdf")
    _fig
    return


@app.cell
def _(mo):
    mo.md("""
    ## Canis familiaris (constant Ne)
    """)
    return


@app.cell
def _(
    Path,
    deterministic,
    left_bins,
    load_data,
    np,
    plt,
    right_bins,
    stdpopsim,
):
    _data = load_data(Path(".") / "canisfamiliaris.pkl.gz")
    _params = _data["params"]
    _num_samples = _params["num_samples"]

    # Get mutation rate from species genome (constant across chromosomes)
    _species = stdpopsim.get_species("CanFam")
    _first_results = next(iter(_data["results"].values()))
    _mut_rates = set(
        _species.genome.get_chromosome(c).mutation_rate
        for c in set(r["chrom"] for r in _first_results)
    )
    assert len(_mut_rates) == 1
    _mutation_rate = _mut_rates.pop()

    # For each Ne, compute relative error at the smallest bin (index 0)
    _ne_list = []
    _mean_list = []
    _lo_list = []
    _hi_list = []
    for _ne, _results in sorted(_data["results"].items()):
        _ld_matrix = np.array([r["mean_linkage_disequilibrium"] for r in _results])
        _ld_obs_per_window = _ld_matrix[:, 0]  # smallest bin only

        _, _ld_pred = deterministic.expected_constant(
            Ne=_ne,
            left_bins=left_bins,
            right_bins=right_bins,
            mutation_rate=_mutation_rate,
            sample_size=_num_samples,
        )
        _pred_val = _ld_pred[0]

        _rel_err_per_window = (_pred_val - _ld_obs_per_window) / _ld_obs_per_window
        _ci = bootstrap_ci(_rel_err_per_window, n_boot=10_000, ci=0.95, seed=1000)
        _ne_list.append(_ne)
        _mean_list.append(_ci[0])
        _lo_list.append(_ci[1])
        _hi_list.append(_ci[2])

    _ne_arr = np.array(_ne_list)
    _mean_arr = np.array(_mean_list)
    _lo_arr = np.array(_lo_list)
    _hi_arr = np.array(_hi_list)

    _fig, _ax = plt.subplots(1, 1, figsize=(6, 5))
    _fig.suptitle(
        f"Canis familiaris -- constant Ne (n={_num_samples} diploids)",
        fontsize=14,
        fontweight="bold",
    )

    _ax.errorbar(
        _ne_arr,
        _mean_arr,
        yerr=[_mean_arr - _lo_arr, _hi_arr - _mean_arr],
        fmt="o-",
        color="C0",
        linewidth=2,
        markersize=6,
        capsize=4,
    )
    _ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)
    _ax.set_xscale("log")
    _ax.set_xlabel("Effective population size (Ne)")
    _ax.set_ylabel("Relative error (predicted − observed) / observed")
    _ax.set_title(
        f"Prediction error at shortest distance bin "
        f"({left_bins[0] * 100:.2f}–{right_bins[0] * 100:.2f} cM)"
    )

    plt.tight_layout()
    _fig.savefig("stdpopsim_canisfamiliaris.pdf")
    _fig
    return


if __name__ == "__main__":
    app.run()
