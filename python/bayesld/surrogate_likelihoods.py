"""
Likelihood approximations based on LD and genetic diversity.
"""

import jax
import jax.numpy as jnp
from jax.scipy.optimize import minimize as jax_minimize

from . import deterministic, montecarlo


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
        _, sims_lds = montecarlo.expected_constant(
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
        log_ne_map = jax.lax.stop_gradient(result.x[0])
    else:
        log_ne_map = log_Ne

    # Surrogate gradient: shift trick so the gradient is evaluated at Ne_map.
    # stop_grad(log_ne_map) + (log_Ne - stop_grad(log_Ne)) evaluates to log_ne_map
    # but carries d/d(log_Ne) = 1, giving the deterministic gradient at Ne_map.
    log_ne_for_grad = log_ne_map + (log_Ne - jax.lax.stop_gradient(log_Ne))

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

        x_map = jax.lax.stop_gradient(jax_minimize(loss, x0, method="BFGS").x)
    else:
        x_map = x0

    # Surrogate gradient: shift trick so the gradient is evaluated at x_map.
    x_for_grad = x_map + (x0 - jax.lax.stop_gradient(x0))

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
