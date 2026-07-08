"""
Tests for the user-friendly ``bayesld.inference`` API.

Fast tests exercise the pure validation paths (no Stan compilation, no simulation).
Slow tests (``-m slow``) compile the static Stan program with cmdstanpy and run a tiny
end-to-end active-learning round; they require cmdstanpy + arviz + msprime.
"""

import msprime
import numpy as np
import pytest

import bayesld
from bayesld.inference import sim_sufficient_stats

LEFT_BINS, RIGHT_BINS = bayesld.linear_bins()
RECOMBINATION_RATE = 1e-8
MUTATION_RATE = 1e-8
# Sequence must span the largest LD bin, else far bins get zero pairs (LD == 0).
SEQUENCE_LENGTH = RIGHT_BINS[-1] * 2 / RECOMBINATION_RATE


def _demography():
    d = msprime.Demography()
    d.add_population(name="pop0", initial_size=200)
    d.add_population_parameters_change(time=100, initial_size=100, growth_rate=0)
    return d


# ── Fast: sim_sufficient_stats argument handling ────────────────────────────────


def test_sim_sufficient_stats_requires_integer_samples():
    with pytest.raises(NotImplementedError):
        sim_sufficient_stats(
            {0: 5},
            demography=_demography(),
            left_bins=LEFT_BINS,
            right_bins=RIGHT_BINS,
            mutation_rate=MUTATION_RATE,
            recombination_rate=RECOMBINATION_RATE,
            sequence_length=SEQUENCE_LENGTH,
            random_seed=1,
        )


def test_sim_sufficient_stats_replicates_and_rtol_mutually_exclusive():
    with pytest.raises(ValueError):
        sim_sufficient_stats(
            5,
            demography=_demography(),
            left_bins=LEFT_BINS,
            right_bins=RIGHT_BINS,
            mutation_rate=MUTATION_RATE,
            recombination_rate=RECOMBINATION_RATE,
            sequence_length=SEQUENCE_LENGTH,
            random_seed=1,
            num_replicates=2,
            rtol=0.1,
        )


# ── Slow: Stan compilation + builder + active learning ──────────────────────────


@pytest.fixture(scope="module")
def data():
    d = _demography()
    pi, ld = sim_sufficient_stats(
        20,
        demography=d,
        left_bins=LEFT_BINS,
        right_bins=RIGHT_BINS,
        mutation_rate=MUTATION_RATE,
        recombination_rate=RECOMBINATION_RATE,
        sequence_length=SEQUENCE_LENGTH,
        random_seed=1,
        num_replicates=50,
        num_workers=-1,
    )
    return pi, ld


@pytest.fixture(scope="module")
def compiled_model():
    from bayesld.inference import PiecewiseConstant

    return PiecewiseConstant(num_epochs=2)


def test_num_epochs_must_be_at_least_one():
    from bayesld.inference import PiecewiseConstant

    with pytest.raises(ValueError):
        PiecewiseConstant(num_epochs=0)


@pytest.mark.slow
def test_stan_code_frozen_at_construction(compiled_model):
    assert "piecewise_constant" in compiled_model.stan_code
    assert compiled_model.num_epochs == 2


@pytest.mark.slow
def test_stan_data_blocks_without_data(compiled_model):
    from bayesld.inference import PiecewiseConstant

    fresh = PiecewiseConstant(num_epochs=2)
    with pytest.raises(RuntimeError):
        fresh._stan_data()


@pytest.mark.slow
def test_with_prior_validates_lengths(compiled_model):
    with pytest.raises(ValueError):
        compiled_model.with_prior(
            [1.0, 2.0], [1.0], [1.0], [1.0]
        )  # sigma_log_ne wrong len
    # Correct lengths: 2 epochs → 2 Ne params, 1 boundary.
    m = compiled_model.with_prior([1.0, 2.0], [1.0, 1.0], [4.0], [1.0])
    assert m._prior["mu_log_ne"].shape == (2,)
    # Builder is immutable: the receiver is untouched.
    assert compiled_model._prior is None


@pytest.mark.slow
def test_with_data_sets_empirical_prior_and_shapes(data):
    from bayesld.inference import PiecewiseConstant

    pi, ld = data
    m = PiecewiseConstant(num_epochs=3).with_data(
        mean_diversity=pi,
        mean_ld=ld,
        left_bins=LEFT_BINS,
        right_bins=RIGHT_BINS,
        recombination_rate=RECOMBINATION_RATE,
        mutation_rate=MUTATION_RATE,
        num_samples=20,
        sequence_length=SEQUENCE_LENGTH,
    )
    sd = m._stan_data()
    assert sd["mu_log_ne"].shape == (3,)
    assert sd["mu_log_t"].shape == (2,)
    assert sd["n_epochs"] == 3
    assert sd["n_bias"] == 0 and sd["n_sigma"] == 0
    # Empirical-Bayes Ne prior centred on Watterson estimate pi / (4 mu).
    expected = np.log(np.mean(pi) / (4.0 * MUTATION_RATE))
    assert np.allclose(sd["mu_log_ne"], expected)


@pytest.mark.slow
def test_sample_returns_posterior(compiled_model, data):
    pi, ld = data
    m = compiled_model.with_data(
        mean_diversity=pi,
        mean_ld=ld,
        left_bins=LEFT_BINS,
        right_bins=RIGHT_BINS,
        recombination_rate=RECOMBINATION_RATE,
        mutation_rate=MUTATION_RATE,
        num_samples=20,
        sequence_length=SEQUENCE_LENGTH,
    )
    idata = m.sample(draws=50, tune=50, chains=1)
    assert idata.posterior["Ne_values"].shape[-1] == 2


@pytest.mark.slow
def test_active_learning_round_accumulates(data):
    from bayesld.inference import PiecewiseConstant

    pi, ld = data
    m = PiecewiseConstant(num_epochs=2).with_data(
        mean_diversity=pi,
        mean_ld=ld,
        left_bins=LEFT_BINS,
        right_bins=RIGHT_BINS,
        recombination_rate=RECOMBINATION_RATE,
        mutation_rate=MUTATION_RATE,
        num_samples=20,
        sequence_length=SEQUENCE_LENGTH,
    )
    m1 = m.active_learning_round(
        num_points=2,
        rtol=0.3,
        min_replicates=3,
        seed=0,
        mc_model=msprime.SMCK(k=1),
        draws=80,
    )
    # A round returns an updated copy; the receiver is left untouched.
    assert m1 is not m
    assert len(m.bias_points) == 0 and len(m.sigma_points) == 0
    assert len(m1.bias_points) == 2
    assert len(m1.sigma_points) == 2

    # Bias points accumulate across rounds; sigma points are last-round only.
    m2 = m1.active_learning_round(
        num_points=2,
        rtol=0.3,
        min_replicates=3,
        seed=1,
        mc_model=msprime.SMCK(k=1),
        draws=80,
    )
    assert len(m2.bias_points) == 4
    assert len(m2.sigma_points) == 2
