"""
Monte Carlo approximations for expected LD and genetic diversity.

Drop-in replacement for montecarlo.py that uses joblib instead of JAX
shard_map / pure_callback for parallelisation. No JAX dependency.
"""

import msprime
import numpy as np
from joblib import Parallel, delayed


# ── Internal helpers (copied from montecarlo.py, no JAX) ─────────────────────


def discretize_demography(demography, n_steps=20):
    """Convert a single-population msprime Demography to (time, Ne) tuples for smc_prime."""
    pop = demography.populations[0]
    epochs = [(0.0, float(pop.initial_size), float(pop.growth_rate or 0.0))]
    for event in sorted(demography.events, key=lambda e: e.time):
        if not isinstance(event, msprime.demography.PopulationParametersChange):
            continue
        t = float(event.time)
        prev_ne, prev_gr = epochs[-1][1], epochs[-1][2]
        ne = float(event.initial_size) if event.initial_size is not None else prev_ne * np.exp(-prev_gr * (t - epochs[-1][0]))
        gr = float(event.growth_rate) if event.growth_rate is not None else prev_gr
        epochs.append((t, ne, gr))

    tuples = []
    for i, (t_start, ne_start, growth_rate) in enumerate(epochs):
        t_end = float(epochs[i + 1][0]) if i + 1 < len(epochs) else t_start + 10.0 * ne_start
        if abs(growth_rate) < 1e-12:
            tuples.append((t_start, ne_start * 2))
        else:
            ts = np.linspace(t_start, t_end, n_steps + 1)[:-1]
            for t in ts:
                ne = ne_start * np.exp(-growth_rate * (float(t) - t_start))
                tuples.append((float(t), float(ne) * 2))
    return tuples


def _run_replicate(
    seed,
    sample_size,
    demography_arg,
    recombination_rate,
    sequence_length,
    mutation_rate,
    left_bins,
    right_bins,
    ploidy,
    model,
):
    """Run a single Monte Carlo replicate and return (pi, ld_per_bin)."""
    from . import data_from_tree_sequence   # local import for joblib pickling

    if model == "discretized_smc_prime":
        import bayesld as _bayesld
        ts = _bayesld.smc_prime.sim_ancestry(
            population_size=demography_arg,
            num_samples=sample_size * ploidy,
            sequence_length=sequence_length,
            recombination_rate=recombination_rate,
            random_seed=seed,
        )
        ts = msprime.sim_mutations(
            ts, rate=mutation_rate, random_seed=seed, model=msprime.BinaryMutationModel()
        )
    else:
        demography = msprime.Demography.from_demes(demography_arg)
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
):
    """Run `num_replicates` MC simulations in parallel via joblib.

    Parameters
    ----------
    build_demography : callable
        (*float_params) -> msprime.Demography
    params : tuple of float
        Demographic parameters forwarded to build_demography.
    num_workers : int
        Number of parallel workers.  -1 uses all available cores (joblib convention).

    Returns
    -------
    pi_replicates : ndarray (num_replicates,)
    ld_replicates : ndarray (num_replicates, num_bins)
    """
    demography = build_demography(*params)
    dem_arg = (
        discretize_demography(demography)
        if model == "discretized_smc_prime"
        else demography.to_demes()
    )

    rng = np.random.default_rng(random_seed)
    seeds = rng.integers(1, 2**32 - 1, size=num_replicates)

    results = Parallel(n_jobs=num_workers)(
        delayed(_run_replicate)(
            int(s), sample_size, dem_arg, recombination_rate,
            sequence_length, mutation_rate, left_bins, right_bins, ploidy, model,
        )
        for s in seeds
    )

    pi = np.array([r[0] for r in results], dtype=np.float64)
    ld = np.array([r[1] for r in results], dtype=np.float64)
    return pi, ld


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
    num_replicates=1,
    ploidy=2,
    model="hudson",
    num_workers=-1,
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
    num_replicates : int
    ploidy : int
    model : str  — msprime ancestry model
    num_workers : int  — joblib parallel workers (-1 = all cores)

    Returns
    -------
    pi_replicates : ndarray (num_replicates,)
    ld_replicates : ndarray (num_replicates, num_bins)
    """
    left_bins = np.asarray(left_bins)
    right_bins = np.asarray(right_bins)

    def build_demography(ne):
        d = msprime.Demography()
        d.add_population(name="pop0", initial_size=ne)
        return d

    return _parallel_mc(
        build_demography, (Ne,),
        left_bins, right_bins, mutation_rate, recombination_rate,
        sequence_length, sample_size, ploidy, model,
        random_seed, num_replicates, num_workers,
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
    num_replicates=1,
    ploidy=2,
    model="hudson",
    num_workers=-1,
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
    num_replicates : int
    ploidy : int
    model : str
    num_workers : int  — joblib parallel workers (-1 = all cores)

    Returns
    -------
    pi_replicates : ndarray (num_replicates,)
    ld_replicates : ndarray (num_replicates, num_bins)
    """
    left_bins = np.asarray(left_bins)
    right_bins = np.asarray(right_bins)

    def build_demography(ne_c, ne_a, t0_, alpha_):
        d = msprime.Demography()
        d.add_population(name="pop0", initial_size=ne_c, growth_rate=alpha_)
        d.add_population_parameters_change(time=t0_, initial_size=ne_a, growth_rate=0)
        return d

    return _parallel_mc(
        build_demography, (Ne_c, Ne_a, t0, alpha),
        left_bins, right_bins, mutation_rate, recombination_rate,
        sequence_length, sample_size, ploidy, model,
        random_seed, num_replicates, num_workers,
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
    num_replicates=1,
    ploidy=2,
    model="hudson",
    num_workers=-1,
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
    num_replicates : int
    ploidy : int
    model : str
    num_workers : int  — joblib parallel workers (-1 = all cores)

    Returns
    -------
    pi_replicates : ndarray (num_replicates,)
    ld_replicates : ndarray (num_replicates, num_bins)
    """
    left_bins = np.asarray(left_bins)
    right_bins = np.asarray(right_bins)

    def build_demography(ne_c, ne_a, t0_, t1_, alpha_):
        d = msprime.Demography()
        d.add_population(name="pop0", initial_size=ne_c, growth_rate=0)
        d.add_population_parameters_change(time=t0_, initial_size=ne_c, growth_rate=alpha_)
        d.add_population_parameters_change(time=t1_, initial_size=ne_a, growth_rate=0)
        return d

    return _parallel_mc(
        build_demography, (Ne_c, Ne_a, t0, t1, alpha),
        left_bins, right_bins, mutation_rate, recombination_rate,
        sequence_length, sample_size, ploidy, model,
        random_seed, num_replicates, num_workers,
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
    num_replicates=1,
    ploidy=2,
    model="hudson",
    num_workers=-1,
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
    num_replicates : int
    ploidy : int
    model : str
    num_workers : int  — joblib parallel workers (-1 = all cores)

    Returns
    -------
    pi_replicates : ndarray (num_replicates,)
    ld_replicates : ndarray (num_replicates, num_bins)
    """
    Ne_values    = list(Ne_values)
    t_boundaries = list(t_boundaries)
    left_bins    = np.asarray(left_bins)
    right_bins   = np.asarray(right_bins)

    def build_demography(ne_vals, t_bounds):
        d = msprime.Demography()
        d.add_population(name="pop0", initial_size=ne_vals[0])
        for t, ne in zip(t_bounds, ne_vals[1:]):
            d.add_population_parameters_change(time=t, initial_size=ne, growth_rate=0)
        return d

    return _parallel_mc(
        build_demography, (Ne_values, t_boundaries),
        left_bins, right_bins, mutation_rate, recombination_rate,
        sequence_length, sample_size, ploidy, model,
        random_seed, num_replicates, num_workers,
    )
