"""
Likelihood approximations based on LD and genetic diversity.
"""

import math
from functools import lru_cache, partial

import jax
import jax.numpy as jnp
from jax.experimental import mesh_utils
from jax.scipy.optimize import minimize as jax_minimize

from . import deterministic, montecarlo

# ── Parallel MC helper ────────────────────────────────────────────────────────


def _fold_rng_over_axis(rng: jax.Array, axis_name: str) -> jax.Array:
    """Return a device-unique key by folding the device index into `rng`."""
    return jax.random.fold_in(rng, jax.lax.axis_index(axis_name))


@lru_cache(maxsize=None)
def _get_mesh(n_workers: int):
    """Build and cache host-device mesh/sharding for repeated MC evaluations."""
    P = jax.sharding.PartitionSpec
    devices = mesh_utils.create_device_mesh((n_workers,), devices=jax.devices("cpu"))
    mesh = jax.sharding.Mesh(devices, ("x",))
    return mesh, jax.sharding.NamedSharding(mesh, P())


@lru_cache(maxsize=None)
def _get_shard_fn(
    n_workers: int,
    reps_per_worker: int,
    ploidy: int,
    model: str,
    left_bins,
    right_bins,
    mutation_rate: float,
    recombination_rate: float,
    sequence_length: float,
    sample_size: int,
):
    """Create and cache the shard_map kernel for fixed parallelization settings."""
    P = jax.sharding.PartitionSpec
    mesh, _ = _get_mesh(n_workers)

    @partial(
        jax.shard_map,
        mesh=mesh,
        in_specs=(P(), P()),
        out_specs=(P("x"), P("x", None)),
    )
    def _shard_fn(ne, rng):
        device_rng = _fold_rng_over_axis(rng, "x")
        seed = jax.random.randint(
            device_rng, shape=(), minval=1, maxval=2**31 - 1, dtype=jnp.int32
        )
        pi_reps, ld_reps = montecarlo.expected_constant(
            ne,
            left_bins,
            right_bins,
            mutation_rate,
            recombination_rate,
            sequence_length,
            sample_size,
            seed,
            num_replicates=reps_per_worker,
            ploidy=ploidy,
            model=model,
        )
        return pi_reps, ld_reps

    return _shard_fn


def _mc_sims(
    Ne,
    left_bins,
    right_bins,
    mutation_rate,
    recombination_rate,
    sequence_length,
    sample_size,
    random_seed,
    num_replicates,
    ploidy,
    model,
):
    """Run `num_replicates` MC simulations, distributed across all available devices.

    Returns
    -------
    sims_lds : jax array, shape (num_replicates, n_bins)
    """
    n_workers = jax.local_device_count()
    reps_per_worker = math.ceil(num_replicates / n_workers)
    left_bins_key = tuple(float(x) for x in left_bins)
    right_bins_key = tuple(float(x) for x in right_bins)
    _, rng_sharding = _get_mesh(n_workers)
    shard_fn = _get_shard_fn(
        n_workers,
        reps_per_worker,
        ploidy,
        model,
        left_bins_key,
        right_bins_key,
        float(mutation_rate),
        float(recombination_rate),
        float(sequence_length),
        int(sample_size),
    )
    rng = jax.device_put(
        jax.random.PRNGKey(int(random_seed)),
        rng_sharding,
    )
    _, sims_lds = shard_fn(Ne, rng)
    return sims_lds[:num_replicates]


# ── Public API ────────────────────────────────────────────────────────────────


def constant(
    data_pi,
    data_ld,
    log_Ne,
    left_bins,
    right_bins,
    mutation_rate,
    recombination_rate,
    sequence_length,
    sample_size,
    random_seed,
    num_replicates=0,
    ploidy=2,
    model="hudson",
):
    # estimate stderr for pi and ld
    n_windows = data_pi.shape[0]
    mean_pi, std_pi = data_pi.mean(), data_pi.std(ddof=1) / jnp.sqrt(n_windows)
    mean_ld, std_ld = (
        data_ld.mean(axis=0),
        data_ld.std(axis=0, ddof=1) / jnp.sqrt(data_ld.shape[0]),
    )
    # Make predictions
    unbiased_pi, _ = deterministic.expected_constant(
        jnp.exp(log_Ne),
        left_bins,
        right_bins,
        mutation_rate,
        sample_size=sample_size,
        ploidy=ploidy,
    )
    if num_replicates > 0:
        sims_lds = _mc_sims(
            jax.lax.stop_gradient(jnp.exp(log_Ne)),
            left_bins,
            right_bins,
            mutation_rate,
            recombination_rate,
            sequence_length,
            sample_size,
            random_seed,
            num_replicates,
            ploidy,
            model,
        )

        def loss(x):
            _, ld_pred = deterministic.expected_constant(
                jnp.exp(x[0]),
                left_bins,
                right_bins,
                mutation_rate,
                sample_size=sample_size,
                ploidy=ploidy,
            )
            return jnp.sum((sims_lds - ld_pred) ** 2)

        result = jax_minimize(loss, jnp.array([log_Ne]), method="BFGS")
        log_ne_map = result.x[0]
    else:
        log_ne_map = log_Ne

    # Surrogate gradient: shift trick so the gradient is evaluated at Ne_map.
    # stop_grad(log_ne_map) + (log_Ne - stop_grad(log_Ne)) evaluates to log_ne_map
    # but carries d/d(log_Ne) = 1, giving the deterministic gradient at Ne_map.
    log_ne_for_grad = jax.lax.stop_gradient(log_ne_map) + (
        log_Ne - jax.lax.stop_gradient(log_Ne)
    )

    # Bias-corrected LD: value at log_ne_map, gradient at log_ne_map (straight-through).
    _, ld_for_grad = deterministic.expected_constant(
        jnp.exp(log_ne_for_grad),
        left_bins,
        right_bins,
        mutation_rate,
        sample_size=sample_size,
        ploidy=ploidy,
    )
    _, ld_at_map = deterministic.expected_constant(
        jnp.exp(jax.lax.stop_gradient(log_ne_map)),
        left_bins,
        right_bins,
        mutation_rate,
        sample_size=sample_size,
        ploidy=ploidy,
    )
    unbiased_ld = ld_for_grad + jax.lax.stop_gradient(ld_at_map - ld_for_grad)

    log_likelihood = -0.5 * jnp.sum(
        (unbiased_pi - mean_pi) ** 2 / std_pi**2
        + (unbiased_ld - mean_ld) ** 2 / std_ld**2
    )
    return log_likelihood


def piecewise_exponential(
    data_pi,
    data_ld,
    log_Ne_c,
    log_Ne_a,
    log_t0,
    alpha,
    left_bins,
    right_bins,
    mutation_rate,
    recombination_rate,
    sequence_length,
    sample_size,
    random_seed,
    num_replicates=0,
    ploidy=2,
    model="hudson",
):
    # estimate stderr for pi and ld
    n_windows = data_pi.shape[0]
    mean_pi, std_pi = data_pi.mean(), data_pi.std(ddof=1) / jnp.sqrt(n_windows)
    mean_ld, std_ld = (
        data_ld.mean(axis=0),
        data_ld.std(axis=0, ddof=1) / jnp.sqrt(data_ld.shape[0]),
    )
    # Make predictions
    unbiased_pi, _ = deterministic.expected_piecewise_exponential(
        jnp.exp(log_Ne_c),
        jnp.exp(log_Ne_a),
        jnp.exp(log_t0),
        alpha,
        left_bins,
        right_bins,
        mutation_rate,
        sample_size=sample_size,
        ploidy=ploidy,
    )
    # Pack inference parameters as a single vector for MAP optimization.
    # x = [log_Ne_c, log_Ne_a, log_t0, alpha]
    x0 = jnp.array([log_Ne_c, log_Ne_a, log_t0, alpha])
    if num_replicates > 0:
        _, sims_lds = montecarlo.expected_piecewise_exponential(
            jax.lax.stop_gradient(jnp.exp(log_Ne_c)),
            jax.lax.stop_gradient(jnp.exp(log_Ne_a)),
            jax.lax.stop_gradient(jnp.exp(log_t0)),
            jax.lax.stop_gradient(alpha),
            left_bins,
            right_bins,
            mutation_rate,
            recombination_rate,
            sequence_length,
            sample_size,
            random_seed,
            num_replicates,
            ploidy,
            model,
        )

        def loss(x):
            _, ld_pred = deterministic.expected_piecewise_exponential(
                jnp.exp(x[0]),
                jnp.exp(x[1]),
                jnp.exp(x[2]),
                x[3],
                left_bins,
                right_bins,
                mutation_rate,
                sample_size=sample_size,
                ploidy=ploidy,
            )
            return jnp.sum((sims_lds - ld_pred) ** 2)

        x_map = jax_minimize(loss, x0, method="BFGS").x
    else:
        x_map = x0

    # Surrogate gradient: shift trick so the gradient is evaluated at x_map.
    # stop_grad(x_map) + (x0 - stop_grad(x0)) evaluates to x_map but carries
    # d/d(x0) = I, giving the deterministic gradient at x_map.
    x_for_grad = jax.lax.stop_gradient(x_map) + (x0 - jax.lax.stop_gradient(x0))

    # Bias-corrected LD: value = ld_det(x_map), gradient at x_map (straight-through).
    _, unbiased_ld = deterministic.expected_piecewise_exponential(
        jnp.exp(x_for_grad[0]),
        jnp.exp(x_for_grad[1]),
        jnp.exp(x_for_grad[2]),
        x_for_grad[3],
        left_bins,
        right_bins,
        mutation_rate,
        sample_size=sample_size,
        ploidy=ploidy,
    )

    log_likelihood = -0.5 * jnp.sum(
        (unbiased_pi - mean_pi) ** 2 / std_pi**2
        + (unbiased_ld - mean_ld) ** 2 / std_ld**2
    )
    return log_likelihood
