"""
Monte Carlo (unbiased) approximations for expected LD and genetic diversity.
"""

import jax
import jax.numpy as jnp
import msprime
import numpy as np
from jax.sharding import Mesh, PartitionSpec as P
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


def _parallel_mc(
    build_demography,
    jax_params,
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
    """Distribute num_replicates simulations across available JAX devices via shmap.

    Each device runs an independent pure_callback executing its share of replicates
    sequentially. The function is compatible with jax.jit when called inside a
    jitted scope — all Python-world constants are captured in closures, while the
    demographic parameters and random_seed flow through as traced JAX values.

    Parameters
    ----------
    build_demography : callable
        Pure-Python function (*float_params) -> msprime.Demography.  Receives the
        concrete float values of jax_params at callback time.
    jax_params : tuple of JAX arrays
        Demographic parameters (scalars) that are traced by JAX.  These are
        forwarded to build_demography inside the pure_callback.
    left_bins, right_bins : np.ndarray
        Bin edges in Morgans — captured as compile-time constants.
    mutation_rate, recombination_rate, sequence_length : float
        Simulation hyperparameters — captured as compile-time constants.
    sample_size, ploidy : int
        Sampling configuration — captured as compile-time constants.
    model : str
        msprime ancestry model — captured as compile-time constant.
    random_seed : JAX scalar (int)
        Base seed; each device derives its own stream via default_rng([seed, device_idx]).
    num_replicates : int
        Total number of replicates.  Padded up to the nearest multiple of n_devices;
        the trailing padding is trimmed before returning.

    Returns
    -------
    pi_replicates : array (num_replicates,)
    ld_replicates : array (num_replicates, num_bins)
    """
    n_bins = len(left_bins)
    n_devices = jax.device_count()
    replicates_per_device = (num_replicates + n_devices - 1) // n_devices

    mesh = Mesh(np.array(jax.devices()), ("devices",))
    # Each device identifies itself via its position in this sharded index array.
    device_indices = jnp.arange(n_devices)          # logical (n_devices,)
    random_seed = jnp.asarray(random_seed, dtype=jnp.int64)

    def _per_device_fn(device_idx, rand_seed, *params):
        """Runs on each device; device_idx and rand_seed are local shards (shape ())."""

        def _callback(d_idx, r_seed, *p):
            demography = build_demography(*[float(x) for x in p])
            if model == "discretized_smc_prime":
                dem_arg = discretize_demography(demography)
            else:
                dem_arg = demography.to_demes()
            # Use a sequence seed so each device gets a statistically independent stream.
            # d_idx arrives as shape-(1,) shard; extract the scalar first.
            rng = np.random.default_rng([int(np.asarray(r_seed).flat[0]),
                                         int(np.asarray(d_idx).flat[0])])
            seeds = rng.integers(1, 2**32 - 1, size=replicates_per_device)
            results = [
                _run_replicate(
                    int(s), sample_size, dem_arg, recombination_rate,
                    sequence_length, mutation_rate, left_bins, right_bins, ploidy, model,
                )
                for s in seeds
            ]
            pi = np.array([r[0] for r in results], dtype=np.float64)
            ld = np.array([r[1] for r in results], dtype=np.float64)
            return pi, ld

        return jax.pure_callback(
            _callback,
            (
                jax.ShapeDtypeStruct((replicates_per_device,), jnp.float64),
                jax.ShapeDtypeStruct((replicates_per_device, n_bins), jnp.float64),
            ),
            device_idx, rand_seed, *params,
            vmap_method="sequential",
        )

    # device_indices is sharded (P('devices')); rand_seed and demographic params
    # are replicated (P()) — every device receives the same value.
    in_specs = (P("devices"), P()) + tuple(P() for _ in jax_params)

    pi_pad, ld_pad = jax.shard_map(
        _per_device_fn,
        mesh=mesh,
        in_specs=in_specs,
        out_specs=(P("devices"), P("devices")),
    )(device_indices, random_seed, *jax_params)

    # Trim padding introduced to reach a multiple of n_devices.
    return pi_pad[:num_replicates], ld_pad[:num_replicates]


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════
# These functions are not decorated with @jax.jit themselves — their Python-world
# constants (left_bins, rates, sample_size, …) are captured in closures, while
# the demographic parameters flow as traced JAX scalars through pure_callback.
# They compose correctly inside a caller's jax.jit without modification.


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

    def build_demography(ne):
        d = msprime.Demography()
        d.add_population(name="pop0", initial_size=ne)
        return d

    return _parallel_mc(
        build_demography, (Ne,),
        left_bins, right_bins, mutation_rate, recombination_rate,
        sequence_length, sample_size, ploidy, model, random_seed, num_replicates,
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

    def build_demography(ne_c, ne_a, t0_, alpha_):
        d = msprime.Demography()
        d.add_population(name="pop0", initial_size=ne_c, growth_rate=alpha_)
        d.add_population_parameters_change(time=t0_, initial_size=ne_a, growth_rate=0)
        return d

    return _parallel_mc(
        build_demography, (Ne_c, Ne_a, t0, alpha),
        left_bins, right_bins, mutation_rate, recombination_rate,
        sequence_length, sample_size, ploidy, model, random_seed, num_replicates,
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

    def build_demography(ne_c, ne_a, t0_, t1_, alpha_):
        d = msprime.Demography()
        d.add_population(name="pop0", initial_size=ne_c, growth_rate=0)
        d.add_population_parameters_change(time=t0_, initial_size=ne_c, growth_rate=alpha_)
        d.add_population_parameters_change(time=t1_, initial_size=ne_a, growth_rate=0)
        return d

    return _parallel_mc(
        build_demography, (Ne_c, Ne_a, t0, t1, alpha),
        left_bins, right_bins, mutation_rate, recombination_rate,
        sequence_length, sample_size, ploidy, model, random_seed, num_replicates,
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
    Ne_values = jnp.asarray(Ne_values, dtype=jnp.float64)
    t_boundaries = jnp.asarray(t_boundaries, dtype=jnp.float64)

    def build_demography(*flat_params):
        n_epochs = len(Ne_values)
        ne_vals = [float(flat_params[i]) for i in range(n_epochs)]
        t_bounds = [float(flat_params[n_epochs + i]) for i in range(n_epochs - 1)]
        d = msprime.Demography.isolated_model(initial_size=[ne_vals[0]])
        for t, N in zip(t_bounds, ne_vals[1:]):
            d.add_population_parameters_change(time=t, initial_size=N)
        return d

    # Flatten Ne_values and t_boundaries into individual JAX scalars so they
    # flow through pure_callback as concrete values.
    jax_params = tuple(Ne_values) + tuple(t_boundaries)

    return _parallel_mc(
        build_demography, jax_params,
        left_bins, right_bins, mutation_rate, recombination_rate,
        sequence_length, sample_size, ploidy, model, random_seed, num_replicates,
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

    def build_demography(ne_1, ne_2, ne_a, t0_, t1_, m):
        d = msprime.Demography()
        d.add_population(name="focal", initial_size=ne_1)
        d.add_population(name="source", initial_size=ne_a)
        d.add_migration_rate_change(time=0, rate=m, source="focal", dest="source")
        d.add_population_parameters_change(population="focal", initial_size=ne_2, time=t0_)
        d.add_migration_rate_change(time=t1_, rate=0, source="focal", dest="source")
        d.add_mass_migration(time=t1_, source="focal", dest="source", proportion=1.0)
        return d

    return _parallel_mc(
        build_demography, (Ne_1, Ne_2, Ne_a, t0, t1, migration_rate),
        left_bins, right_bins, mutation_rate, recombination_rate,
        sequence_length, sample_size, ploidy, model, random_seed, num_replicates,
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
