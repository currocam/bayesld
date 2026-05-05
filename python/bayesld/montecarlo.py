"""
Monte Carlo approximations for expected LD and genetic diversity.

Uses joblib for parallelisation across replicates and msprime for
coalescent simulation (default model: SMC' via SMCK(k=1)).
"""

import os

import msprime
import numpy as np
from joblib import Parallel, delayed

_DEFAULT_MODEL = msprime.SMCK(k=1)
MAX_REPLICATES = 100_000
_SENTINEL = object()


def _run_replicate(
    seed,
    sample_size,
    demography_demes,
    recombination_rate,
    sequence_length,
    mutation_rate,
    left_bins,
    right_bins,
    ploidy,
    model,
):
    """Run a single Monte Carlo replicate and return (pi, ld_per_bin)."""
    from . import data_from_tree_sequence  # local import for joblib pickling

    demography = msprime.Demography.from_demes(demography_demes)
    ts = msprime.sim_ancestry(
        samples={0: sample_size},
        demography=demography,
        recombination_rate=recombination_rate,
        sequence_length=sequence_length,
        random_seed=seed,
        ploidy=ploidy,
        model=model,
    )
    ts = msprime.sim_mutations(
        ts, rate=mutation_rate, random_seed=seed, model=msprime.BinaryMutationModel()
    )

    stats = data_from_tree_sequence(
        ts=ts,
        recombination_rate=recombination_rate,
        left_bins_morgan=left_bins,
        right_bins_morgan=right_bins,
        ploidy=ploidy,
        progress_bar=False,
    )
    return stats["mean_genetic_diversity"], stats["mean_linkage_disequilibrium"]


def _max_relative_se(values):
    """Max relative standard error of the mean across all components.

    Parameters
    ----------
    values : ndarray, shape (n, ...) with n >= 2

    Returns
    -------
    float — max |SE / mean| across all elements (0 where mean is zero)
    """
    n = values.shape[0]
    mean = values.mean(axis=0)
    se = values.std(axis=0, ddof=1) / np.sqrt(n)
    with np.errstate(divide="ignore", invalid="ignore"):
        rse = np.abs(se / mean)
    return float(np.nanmax(np.where(np.isfinite(rse), rse, 0.0)))


def _parallel_mc(
    build_demography,
    params,
    left_bins,
    right_bins,
    mutation_rate,
    recombination_rate,
    sequence_length,
    sample_size,
    ploidy,
    model,
    random_seed,
    num_replicates,
    num_workers,
    rtol,
):
    """Run MC simulations in parallel via joblib.

    Fixed mode (rtol is None): run exactly num_replicates simulations.
    Adaptive mode (rtol is set): run batches of num_workers until the
    max relative SE of the mean drops below rtol for both pi and LD.

    Returns
    -------
    pi_replicates : ndarray (n,)
    ld_replicates : ndarray (n, num_bins)
    """
    demography = build_demography(*params)
    dem_demes = demography.to_demes()
    rng = np.random.default_rng(random_seed)

    def run_batch(seeds):
        results = Parallel(n_jobs=num_workers)(
            delayed(_run_replicate)(
                int(s),
                sample_size,
                dem_demes,
                recombination_rate,
                sequence_length,
                mutation_rate,
                left_bins,
                right_bins,
                ploidy,
                model,
            )
            for s in seeds
        )
        return (
            [r[0] for r in results],
            [r[1] for r in results],
        )

    if rtol is None:
        seeds = rng.integers(1, 2**32 - 1, size=num_replicates)
        pi_vals, ld_vals = run_batch(seeds)
        return np.asarray(pi_vals), np.asarray(ld_vals)

    # Adaptive mode: run batches until convergence
    batch_size = os.cpu_count() or 1 if num_workers == -1 else max(num_workers, 1)
    pi_all, ld_all = [], []

    while len(pi_all) < MAX_REPLICATES:
        seeds = rng.integers(1, 2**32 - 1, size=batch_size)
        try:
            pi_batch, ld_batch = run_batch(seeds)
        except Exception as _e:
            # `smc(k)` from msprime has right now an unresolved bug
            # https://github.com/tskit-dev/msprime/pull/2524
            continue
        pi_all.extend(pi_batch)
        ld_all.extend(ld_batch)

        if len(pi_all) < 2:
            continue

        pi_arr = np.asarray(pi_all)
        ld_arr = np.asarray(ld_all)
        rse = max(
            _max_relative_se(pi_arr[:, np.newaxis]),
            _max_relative_se(ld_arr),
        )
        if rse < rtol:
            return pi_arr, ld_arr

    raise RuntimeError(
        f"Adaptive MC did not converge after {MAX_REPLICATES} replicates "
        f"(rtol={rtol}, achieved={rse:.4g})"
    )


def _validate_stopping(num_replicates, rtol):
    """Resolve and validate the stopping criterion."""
    if rtol is not None and num_replicates is not _SENTINEL:
        raise ValueError("num_replicates and rtol are mutually exclusive")
    if rtol is None and num_replicates is _SENTINEL:
        num_replicates = 1
    if num_replicates is _SENTINEL:
        num_replicates = None
    return num_replicates, rtol


# ── Public API ────────────────────────────────────────────────────────────────


def expected_constant(
    Ne,
    left_bins,
    right_bins,
    mutation_rate,
    recombination_rate,
    sequence_length,
    sample_size,
    random_seed,
    num_replicates=_SENTINEL,
    ploidy=2,
    model=_DEFAULT_MODEL,
    num_workers=-1,
    rtol=None,
):
    """
    Expected genetic diversity and LD under constant Ne via Monte Carlo.

    Parameters
    ----------
    Ne : float
    left_bins, right_bins : array-like  — bin edges in Morgans
    mutation_rate, recombination_rate, sequence_length : float
    sample_size : int  — diploid individuals
    random_seed : int
    num_replicates : int  — fixed replicate count (mutually exclusive with rtol)
    ploidy : int
    model : msprime ancestry model  — default is SMCK(k=1)
    num_workers : int  — joblib parallel workers (-1 = all cores)
    rtol : float or None  — adaptive: simulate until relative SE < rtol

    Returns
    -------
    pi_replicates : ndarray (n,)
    ld_replicates : ndarray (n, num_bins)
    """
    num_replicates, rtol = _validate_stopping(num_replicates, rtol)
    left_bins = np.asarray(left_bins)
    right_bins = np.asarray(right_bins)

    def build_demography(ne):
        d = msprime.Demography()
        d.add_population(name="pop0", initial_size=ne)
        return d

    return _parallel_mc(
        build_demography,
        (Ne,),
        left_bins,
        right_bins,
        mutation_rate,
        recombination_rate,
        sequence_length,
        sample_size,
        ploidy,
        model,
        random_seed,
        num_replicates,
        num_workers,
        rtol,
    )


def expected_piecewise_exponential(
    Ne_c,
    Ne_a,
    t0,
    alpha,
    left_bins,
    right_bins,
    mutation_rate,
    recombination_rate,
    sequence_length,
    sample_size,
    random_seed,
    num_replicates=_SENTINEL,
    ploidy=2,
    model=_DEFAULT_MODEL,
    num_workers=-1,
    rtol=None,
):
    """
    Expected genetic diversity and LD under a two-phase exponential demography via Monte Carlo.

    Ne(t) = Ne_c * exp(-alpha * t)  for t < t0,  else Ne_a

    Parameters
    ----------
    Ne_c : float  — contemporary Ne
    Ne_a : float  — ancestral Ne
    t0 : float    — transition time (generations)
    alpha : float — exponential rate
    left_bins, right_bins : array-like  — bin edges in Morgans
    mutation_rate, recombination_rate, sequence_length : float
    sample_size : int
    random_seed : int
    num_replicates : int  — fixed replicate count (mutually exclusive with rtol)
    ploidy : int
    model : msprime ancestry model  — default is SMCK(k=1)
    num_workers : int  — joblib parallel workers (-1 = all cores)
    rtol : float or None  — adaptive: simulate until relative SE < rtol

    Returns
    -------
    pi_replicates : ndarray (n,)
    ld_replicates : ndarray (n, num_bins)
    """
    num_replicates, rtol = _validate_stopping(num_replicates, rtol)
    left_bins = np.asarray(left_bins)
    right_bins = np.asarray(right_bins)

    def build_demography(ne_c, ne_a, t0_, alpha_):
        d = msprime.Demography()
        d.add_population(name="pop0", initial_size=ne_c, growth_rate=alpha_)
        d.add_population_parameters_change(time=t0_, initial_size=ne_a, growth_rate=0)
        return d

    return _parallel_mc(
        build_demography,
        (Ne_c, Ne_a, t0, alpha),
        left_bins,
        right_bins,
        mutation_rate,
        recombination_rate,
        sequence_length,
        sample_size,
        ploidy,
        model,
        random_seed,
        num_replicates,
        num_workers,
        rtol,
    )


def expected_exponential_carrying_capacity(
    Ne_c,
    Ne_a,
    t0,
    t1,
    alpha,
    left_bins,
    right_bins,
    mutation_rate,
    recombination_rate,
    sequence_length,
    sample_size,
    random_seed,
    num_replicates=_SENTINEL,
    ploidy=2,
    model=_DEFAULT_MODEL,
    num_workers=-1,
    rtol=None,
):
    """
    Expected genetic diversity and LD under an exponential carrying-capacity demography via MC.

    Ne(t) = Ne_c                          for t < t0
    Ne(t) = Ne_c * exp(-alpha * (t - t0)) for t0 <= t < t1
    Ne(t) = Ne_a                          for t >= t1

    Parameters
    ----------
    Ne_c : float  — contemporary Ne (recent constant phase)
    Ne_a : float  — ancestral Ne
    t0 : float    — start of exponential phase (generations ago)
    t1 : float    — end of exponential phase (generations ago), t1 > t0
    alpha : float — exponential rate
    left_bins, right_bins : array-like  — bin edges in Morgans
    mutation_rate, recombination_rate, sequence_length : float
    sample_size : int
    random_seed : int
    num_replicates : int  — fixed replicate count (mutually exclusive with rtol)
    ploidy : int
    model : msprime ancestry model  — default is SMCK(k=1)
    num_workers : int  — joblib parallel workers (-1 = all cores)
    rtol : float or None  — adaptive: simulate until relative SE < rtol

    Returns
    -------
    pi_replicates : ndarray (n,)
    ld_replicates : ndarray (n, num_bins)
    """
    num_replicates, rtol = _validate_stopping(num_replicates, rtol)
    left_bins = np.asarray(left_bins)
    right_bins = np.asarray(right_bins)

    def build_demography(ne_c, ne_a, t0_, t1_, alpha_):
        d = msprime.Demography()
        d.add_population(name="pop0", initial_size=ne_c, growth_rate=0)
        d.add_population_parameters_change(
            time=t0_, initial_size=ne_c, growth_rate=alpha_
        )
        d.add_population_parameters_change(time=t1_, initial_size=ne_a, growth_rate=0)
        return d

    return _parallel_mc(
        build_demography,
        (Ne_c, Ne_a, t0, t1, alpha),
        left_bins,
        right_bins,
        mutation_rate,
        recombination_rate,
        sequence_length,
        sample_size,
        ploidy,
        model,
        random_seed,
        num_replicates,
        num_workers,
        rtol,
    )


def expected_piecewise_constant(
    Ne_values,
    t_boundaries,
    left_bins,
    right_bins,
    mutation_rate,
    recombination_rate,
    sequence_length,
    sample_size,
    random_seed,
    num_replicates=_SENTINEL,
    ploidy=2,
    model=_DEFAULT_MODEL,
    num_workers=-1,
    rtol=None,
):
    """
    Expected genetic diversity and LD under a piecewise-constant demography via Monte Carlo.

    Ne(t) = Ne_values[i]  for  t in [t_boundaries[i-1], t_boundaries[i])
    Ne(t) = Ne_values[-1] for  t >= t_boundaries[-1]

    Parameters
    ----------
    Ne_values : array-like, shape (n_epochs,)
    t_boundaries : array-like, shape (n_epochs-1,)  — epoch change times in generations
    left_bins, right_bins : array-like  — bin edges in Morgans
    mutation_rate, recombination_rate, sequence_length : float
    sample_size : int
    random_seed : int
    num_replicates : int  — fixed replicate count (mutually exclusive with rtol)
    ploidy : int
    model : msprime ancestry model  — default is SMCK(k=1)
    num_workers : int  — joblib parallel workers (-1 = all cores)
    rtol : float or None  — adaptive: simulate until relative SE < rtol

    Returns
    -------
    pi_replicates : ndarray (n,)
    ld_replicates : ndarray (n, num_bins)
    """
    num_replicates, rtol = _validate_stopping(num_replicates, rtol)
    Ne_values = list(Ne_values)
    t_boundaries = list(t_boundaries)
    left_bins = np.asarray(left_bins)
    right_bins = np.asarray(right_bins)

    def build_demography(ne_vals, t_bounds):
        d = msprime.Demography()
        d.add_population(name="pop0", initial_size=ne_vals[0])
        for t, ne in zip(t_bounds, ne_vals[1:]):
            d.add_population_parameters_change(time=t, initial_size=ne, growth_rate=0)
        return d

    return _parallel_mc(
        build_demography,
        (Ne_values, t_boundaries),
        left_bins,
        right_bins,
        mutation_rate,
        recombination_rate,
        sequence_length,
        sample_size,
        ploidy,
        model,
        random_seed,
        num_replicates,
        num_workers,
        rtol,
    )
