import jax
import jax.numpy as jnp
import numpy as np

import bayesld
from bayesld import montecarlo, surrogate_likelihoods

jax.config.update("jax_enable_x64", True)


def test_constant_mc_jit_vmap_smoke():
    left_bins, right_bins = bayesld.linear_bins()
    mutation_rate = 1e-8
    recombination_rate = 1e-8
    sequence_length = right_bins[-1] * 2 / recombination_rate
    sample_size = 10

    pi_data, ld_data = montecarlo.expected_constant(
        Ne=7_000.0,
        left_bins=left_bins,
        right_bins=right_bins,
        mutation_rate=mutation_rate,
        recombination_rate=recombination_rate,
        sequence_length=sequence_length,
        sample_size=sample_size,
        random_seed=318,
        num_replicates=3,
        ploidy=2,
        model="hudson",
    )

    @jax.jit
    def logdensity(log_ne):
        return surrogate_likelihoods.constant(
            pi_data,
            ld_data,
            log_ne,
            left_bins,
            right_bins,
            mutation_rate,
            recombination_rate,
            sequence_length,
            sample_size,
            random_seed=13_867,
            num_replicates=3,
            ploidy=2,
            model="discretized_smc_prime",
        )

    values = jax.vmap(logdensity)(jnp.log(jnp.array([100.0])))
    assert values.shape == (1,)
    assert np.isfinite(np.asarray(values)).all()
