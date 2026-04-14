"""
Monte Carlo (unbiased) approximations for expected LD and genetic diversity.
"""

import jax
import jax.numpy as jnp
import msprime
import numpy as np
from scipy.stats import norm, truncnorm

from . import deterministic


def discretize_demography(demography, n_steps=20):
    """Convert a single-population msprime Demography to (time, Ne) tuples for smc_prime.

    Parameters
    ----------
    demography : msprime.Demography
        Single-population msprime demography (may include exponential epochs).
    n_steps : int
        Number of steps used to discretize each exponential epoch.

    Returns
    -------
    list of (float, float)
        List of (time_generations_ago, Ne) tuples, sorted by time.
        First tuple always has time=0.0.
    """
    pop = demography.populations[0]
    # Collect epochs as (t_start, initial_size, growth_rate)
    # msprime: N(t) = initial_size * exp(-growth_rate * (t - t_start))
    epochs = [(0.0, float(pop.initial_size), float(pop.growth_rate or 0.0))]
    for event in sorted(demography.events, key=lambda e: e.time):
        if not isinstance(event, msprime.demography.PopulationParametersChange):
            continue
        t = float(event.time)
        prev_ne, prev_gr = epochs[-1][1], epochs[-1][2]
        # If size/growth_rate not specified, inherit from previous epoch
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
    "Run a single Monte Carlo replicate"
    from . import data_from_tree_sequence
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


def _run_replicates(
    demography,
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
):
    "Common logic across different demographic scenarios"
    left_bins = np.asarray(left_bins)
    right_bins = np.asarray(right_bins)
    if len(left_bins) != len(right_bins):
        raise ValueError("left_bins and right_bins must have the same length")
    if np.any(left_bins >= right_bins):
        raise ValueError("left_bins must be less than right_bins")
    rng = np.random.default_rng(int(random_seed))
    seeds = rng.integers(1, 2**32 - 1, size=num_replicates)
    if model == "discretized_smc_prime":
        demography_arg = discretize_demography(demography)
    else:
        demography_arg = demography.to_demes()
    results = [
        _run_replicate(
            seeds[i],
            sample_size,
            demography_arg,
            recombination_rate,
            sequence_length,
            mutation_rate,
            left_bins,
            right_bins,
            ploidy,
            model,
        )
        for i in range(num_replicates)
    ]
    pi_replicates = np.array([r[0] for r in results])
    ld_replicates = np.array([r[1] for r in results])
    return pi_replicates.astype(np.float64), ld_replicates.astype(np.float64)


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
):
    """
    Compute expected genetic diversity and LD under a constant population size model using Monte Carlo simulations via `msprime`.

    Parameters
    ----------
    Ne : float
        Effective population size.
    left_bins : array-like
        Left edges of genomic distance bins in Morgans.
    right_bins : array-like
        Right edges of genomic distance bins in Morgans.
    mutation_rate : float
        Mutation rate per base pair per generation.
    recombination_rate : float
        Recombination rate per base pair per generation.
    sequence_length : float
        Length of the sequence in base pairs.
    sample_size : int
        Number of diploid individuals sampled.
    random_seed : int
        Seed for random number generator.
    num_replicates : int, optional
        Number of replicates to perform. Default is 1.
    ploidy : int, optional
        Ploidy of the individuals. Default is 2.
    model : str, optional
        Model to use for the simulation (as defined in `msprime`). Default is "hudson".

    Returns
    -------
    pi_replicates : array (num_replicates,)
        Genetic diversity for each replicate.
    ld_replicates : array (num_replicates, num_bins)
        LD values for each replicate and distance bin.
    """
    left_bins = np.asarray(left_bins)
    right_bins = np.asarray(right_bins)
    _n_bins = len(left_bins)

    def _mc_callback(ne, seed):
        ne = float(ne)
        demography = msprime.Demography()
        demography.add_population(name="pop0", initial_size=ne)
        return _run_replicates(
            demography=demography,
            left_bins=left_bins,
            right_bins=right_bins,
            mutation_rate=mutation_rate,
            recombination_rate=recombination_rate,
            sequence_length=sequence_length,
            sample_size=sample_size,
            ploidy=ploidy,
            model=model,
            random_seed=seed,
            num_replicates=num_replicates,
        )

    out_type = (
        jax.ShapeDtypeStruct((num_replicates,), jnp.float64),
        jax.ShapeDtypeStruct((num_replicates, _n_bins), jnp.float64),
    )
    return jax.pure_callback(
        _mc_callback, out_type, Ne, random_seed, vmap_method="sequential",
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
):
    """
    Compute expected genetic diversity and LD under a two-phase exponential demography using Monte Carlo simulations via `msprime`.

    Ne(t) = Ne_c * exp(-alpha * t) if t < t0 else Ne_a

    Parameters
    ----------
    Ne_c : float
        Contemporary effective population size.
    Ne_a : float
        Ancestral effective population size.
    t0 : float
        Time of transition from exponential to constant phase.
    alpha : float
        Rate of change of Ne during the exponential phase.
    left_bins : array-like
        Left edges of genomic distance bins in Morgans.
    right_bins : array-like
        Right edges of genomic distance bins in Morgans.
    mutation_rate : float
        Mutation rate per base pair per generation.
    recombination_rate : float
        Recombination rate per base pair per generation.
    sequence_length : float
        Length of the sequence in base pairs.
    sample_size : int
        Number of diploid individuals sampled.
    random_seed : int
        Seed for random number generator.
    num_replicates : int, optional
        Number of replicates to perform. Default is 1.
    ploidy : int, optional
        Ploidy of the individuals. Default is 2.
    model : str, optional
        Model to use for the simulation (as defined in `msprime`). Default is "hudson".

    Returns
    -------
    pi_replicates : array (num_replicates,)
        Genetic diversity for each replicate.
    ld_replicates : array (num_replicates, num_bins)
        LD values for each replicate and distance bin.
    """
    left_bins = np.asarray(left_bins)
    right_bins = np.asarray(right_bins)
    _n_bins = len(left_bins)

    def _mc_callback(ne_c, ne_a, t0_, alpha_, seed):
        ne_c, ne_a, t0_, alpha_ = float(ne_c), float(ne_a), float(t0_), float(alpha_)
        demography = msprime.Demography()
        demography.add_population(name="pop0", initial_size=ne_c, growth_rate=alpha_)
        demography.add_population_parameters_change(
            time=t0_, initial_size=ne_a, growth_rate=0
        )
        return _run_replicates(
            demography=demography,
            left_bins=left_bins,
            right_bins=right_bins,
            mutation_rate=mutation_rate,
            recombination_rate=recombination_rate,
            sequence_length=sequence_length,
            sample_size=sample_size,
            ploidy=ploidy,
            model=model,
            random_seed=seed,
            num_replicates=num_replicates,
        )

    out_type = (
        jax.ShapeDtypeStruct((num_replicates,), jnp.float64),
        jax.ShapeDtypeStruct((num_replicates, _n_bins), jnp.float64),
    )
    return jax.pure_callback(
        _mc_callback, out_type, Ne_c, Ne_a, t0, alpha, random_seed, vmap_method="sequential",
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
):
    """
    Compute expected genetic diversity and LD under a demography where exponential growth stops after reaching a carrying capacity using Monte Carlo simulations via `msprime`.

    Ne(t) = Ne_c if t < t0, Ne_c* exp(-alpha * (t-t0)) if t < t1 else Ne_a

    Parameters
    ----------
    Ne_c : float
        Contemporary effective population size.
    Ne_a : float
        Ancestral effective population size.
    t0 : float
        Time when exponential growth stops.
    t1 : float
        Time when exponential growth starts.
    alpha : float
        Rate of change of Ne during the exponential phase.
    left_bins : array-like
        Left edges of genomic distance bins in Morgans.
    right_bins : array-like
        Right edges of genomic distance bins in Morgans.
    mutation_rate : float
        Mutation rate per base pair per generation.
    recombination_rate : float
        Recombination rate per base pair per generation.
    sequence_length : float
        Length of the sequence in base pairs.
    sample_size : int
        Number of diploid individuals sampled.
    random_seed : int
        Seed for random number generator.
    num_replicates : int, optional
        Number of replicates to perform. Default is 1.
    ploidy : int, optional
        Ploidy of the individuals. Default is 2.
    model : str, optional
        Model to use for the simulation (as defined in `msprime`). Default is "hudson".

    Returns
    -------
    pi_replicates : array (num_replicates,)
        Genetic diversity for each replicate.
    ld_replicates : array (num_replicates, num_bins)
        LD values for each replicate and distance bin.
    """
    left_bins = np.asarray(left_bins)
    right_bins = np.asarray(right_bins)
    _n_bins = len(left_bins)

    def _mc_callback(ne_c, ne_a, t0_, t1_, alpha_, seed):
        ne_c, ne_a = float(ne_c), float(ne_a)
        t0_, t1_, alpha_ = float(t0_), float(t1_), float(alpha_)
        demography = msprime.Demography()
        demography.add_population(name="pop0", initial_size=ne_c, growth_rate=0)
        demography.add_population_parameters_change(
            time=t0_, initial_size=ne_c, growth_rate=alpha_
        )
        demography.add_population_parameters_change(
            time=t1_, initial_size=ne_a, growth_rate=0
        )
        return _run_replicates(
            demography=demography,
            left_bins=left_bins,
            right_bins=right_bins,
            mutation_rate=mutation_rate,
            recombination_rate=recombination_rate,
            sequence_length=sequence_length,
            sample_size=sample_size,
            ploidy=ploidy,
            model=model,
            random_seed=seed,
            num_replicates=num_replicates,
        )

    out_type = (
        jax.ShapeDtypeStruct((num_replicates,), jnp.float64),
        jax.ShapeDtypeStruct((num_replicates, _n_bins), jnp.float64),
    )
    return jax.pure_callback(
        _mc_callback, out_type, Ne_c, Ne_a, t0, t1, alpha, random_seed, vmap_method="sequential",
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
):
    """
    Compute expected genetic diversity and LD under a multi-epoch constant population size model using Monte Carlo simulations via `msprime`.

    Ne(t) = Ne_values[0] for t < t_boundaries[0]
    Ne(t) = Ne_values[i] for t in [t_boundaries[i], t_boundaries[i+1))
    Ne(t) = Ne_values[-1] for t >= t_boundaries[-1]

    Parameters
    ----------
    Ne_values : array-like
        Population sizes for each epoch. Shape: (n_epochs,)
    t_boundaries : array-like
        Time boundaries between epochs. Shape: (n_epochs-1,)
    left_bins : array-like
        Left edges of genomic distance bins in Morgans.
    right_bins : array-like
        Right edges of genomic distance bins in Morgans.
    mutation_rate : float
        Mutation rate per base pair per generation.
    recombination_rate : float
        Recombination rate per base pair per generation.
    sequence_length : float
        Length of the sequence in base pairs.
    sample_size : int
        Number of diploid individuals sampled.
    random_seed : int
        Seed for random number generator.
    num_replicates : int, optional
        Number of replicates to perform. Default is 1.
    ploidy : int, optional
        Ploidy of the individuals. Default is 2.
    model : str, optional
        Model to use for the simulation (as defined in `msprime`). Default is "hudson".

    Returns
    -------
    pi_replicates : array (num_replicates,)
        Genetic diversity for each replicate.
    ld_replicates : array (num_replicates, num_bins)
        LD values for each replicate and distance bin.
    """
    left_bins = np.asarray(left_bins)
    right_bins = np.asarray(right_bins)
    _n_bins = len(left_bins)

    def _mc_callback(ne_values, t_bounds, seed):
        ne_values = np.asarray(ne_values, dtype=float).tolist()
        t_bounds = np.asarray(t_bounds, dtype=float).tolist()
        demography = msprime.Demography.isolated_model(initial_size=[ne_values[0]])
        for t, N in zip(t_bounds, ne_values[1:]):
            demography.add_population_parameters_change(time=t, initial_size=N)
        return _run_replicates(
            demography=demography,
            left_bins=left_bins,
            right_bins=right_bins,
            mutation_rate=mutation_rate,
            recombination_rate=recombination_rate,
            sequence_length=sequence_length,
            sample_size=sample_size,
            ploidy=ploidy,
            model=model,
            random_seed=seed,
            num_replicates=num_replicates,
        )

    out_type = (
        jax.ShapeDtypeStruct((num_replicates,), jnp.float64),
        jax.ShapeDtypeStruct((num_replicates, _n_bins), jnp.float64),
    )
    return jax.pure_callback(
        _mc_callback,
        out_type,
        np.asarray(Ne_values, dtype=float),
        np.asarray(t_boundaries, dtype=float),
        random_seed,
        vmap_method="sequential",
    )


def expected_secondary_introduction(
    Ne_1,
    Ne_2,
    Ne_a,
    t0,
    t1,
    migration_rate,
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
):
    """
    Compute expected genetic diversity and LD under a secondary-introduction invasion scenario using Monte Carlo simulations via `msprime`.

    Ne(t) = Ne_1 if t < t0, Ne_2 if t < t1 else Ne_3
    m(target -> ancestral) = migration_rate if t < t1 else 0
    m(ancestral -> target) = 0

    Parameters
    ----------
    Ne_1 : float
        Contemporary effective population size (migration activated).
    Ne_2 : float
        Intermediate effective population size (migration activated).
    Ne_a : float
        Ancestral effective population size (no migration).
    t0 : float
        Time when population size reaches Ne_2.
    t1 : float
        Time when target population is merged into ancestral population and migration starts.
    migration_rate : float
        Migration rate from target population to ancestral population.
    left_bins : array-like
        Left edges of genomic distance bins in Morgans.
    right_bins : array-like
        Right edges of genomic distance bins in Morgans.
    mutation_rate : float
        Mutation rate per base pair per generation.
    recombination_rate : float
        Recombination rate per base pair per generation.
    sequence_length : float
        Length of the sequence in base pairs.
    sample_size : int
        Number of diploid individuals sampled.
    random_seed : int
        Seed for random number generator.
    num_replicates : int, optional
        Number of replicates to perform. Default is 1.
    ploidy : int, optional
        Ploidy of the individuals. Default is 2.
    model : str, optional
        Model to use for the simulation (as defined in `msprime`). Default is "hudson".

    Returns
    -------
    pi_replicates : array (num_replicates,)
        Genetic diversity for each replicate.
    ld_replicates : array (num_replicates, num_bins)
        LD values for each replicate and distance bin.
    """
    left_bins = np.asarray(left_bins)
    right_bins = np.asarray(right_bins)
    _n_bins = len(left_bins)

    def _mc_callback(ne_1, ne_2, ne_a, t0_, t1_, m, seed):
        ne_1, ne_2, ne_a = float(ne_1), float(ne_2), float(ne_a)
        t0_, t1_, m = float(t0_), float(t1_), float(m)
        demography = msprime.Demography()
        demography.add_population(name="focal", initial_size=ne_1)
        demography.add_population(name="source", initial_size=ne_a)
        demography.add_migration_rate_change(
            time=0, rate=m, source="focal", dest="source"
        )
        demography.add_population_parameters_change(
            population="focal", initial_size=ne_2, time=t0_
        )
        demography.add_migration_rate_change(time=t1_, rate=0, source="focal", dest="source")
        demography.add_mass_migration(
            time=t1_,
            source="focal",
            dest="source",
            proportion=1.0,
        )
        return _run_replicates(
            demography=demography,
            left_bins=left_bins,
            right_bins=right_bins,
            mutation_rate=mutation_rate,
            recombination_rate=recombination_rate,
            sequence_length=sequence_length,
            sample_size=sample_size,
            ploidy=ploidy,
            model=model,
            random_seed=seed,
            num_replicates=num_replicates,
        )

    out_type = (
        jax.ShapeDtypeStruct((num_replicates,), jnp.float64),
        jax.ShapeDtypeStruct((num_replicates, _n_bins), jnp.float64),
    )
    return jax.pure_callback(
        _mc_callback,
        out_type,
        Ne_1, Ne_2, Ne_a, t0, t1, migration_rate, random_seed,
        vmap_method="sequential",
    )


# ══════════════════════════════════════════════════════════════════════════════
# Bootstrap log-likelihood estimation for VBMC
# ══════════════════════════════════════════════════════════════════════════════


def _bootstrap_loglikelihood(
    pi_vec, ld_mat, observed_pi, observed_ld, rng, n_bootstrap=200
):
    """Compute synthetic log-likelihood with bootstrap noise estimation.

    Parameters
    ----------
    pi_vec : array (num_replicates,)
        Per-replicate genetic diversity values.
    ld_mat : array (num_replicates, num_bins)
        Per-replicate LD values.
    observed_pi : array (num_windows,)
        Observed genetic diversity per window.
    observed_ld : array (num_windows, num_bins)
        Observed LD per window and bin.
    rng : np.random.Generator
        Random number generator for bootstrap.
    n_bootstrap : int
        Number of bootstrap resamples.

    Returns
    -------
    corrected_logp : float
        Bias-corrected log-likelihood.
    noise_std : float
        Estimated noise standard deviation.
    """
    num_replicates = len(pi_vec)
    assert num_replicates > 1, "At least two replicates are required."

    def _logp_from_replicates(pi_reps, ld_reps):
        mu_pi = pi_reps.mean()
        sigma_pi = pi_reps.std(ddof=1)
        mu_ld = ld_reps.mean(axis=0)
        sigma_ld = ld_reps.std(axis=0, ddof=1)
        a = -mu_pi / sigma_pi
        logp_pi = truncnorm.logpdf(
            observed_pi, a=a, b=np.inf, loc=mu_pi, scale=sigma_pi
        )
        logp_ld = norm.logpdf(observed_ld, loc=mu_ld, scale=sigma_ld)
        return np.sum(logp_pi) + np.sum(logp_ld)

    full_logp = _logp_from_replicates(pi_vec, ld_mat)

    bootstrap_logps = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        idx = rng.integers(0, num_replicates, size=num_replicates)
        bootstrap_logps[b] = _logp_from_replicates(pi_vec[idx], ld_mat[idx])

    corrected_logp = 2 * full_logp - bootstrap_logps.mean()
    noise_std = bootstrap_logps.std()

    return corrected_logp, noise_std
