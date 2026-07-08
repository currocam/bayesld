"""
Tests for the two-phase ``PiecewiseExponential`` inference engine.

Fast tests exercise the pure prior/param plumbing (no Stan compilation, no
simulation). Slow tests (``-m slow``) compile the static Stan program with
cmdstanpy and run a tiny end-to-end active-learning round; they require
cmdstanpy + arviz + msprime.
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
    # Ne_c = 100 grows back to Ne_a = 200 over t0 = 100 generations.
    d = msprime.Demography()
    alpha = (np.log(100) - np.log(200)) / 100
    d.add_population(name="pop0", initial_size=100, growth_rate=alpha)
    d.add_population_parameters_change(time=100, initial_size=200, growth_rate=0)
    return d


# ── Fast: prior / param plumbing ────────────────────────────────────────────────


def test_default_prior_centres_on_watterson():
    from bayesld.inference import PiecewiseExponential

    pi = np.array([4e-4, 5e-4])
    m = PiecewiseExponential()
    m._data = {"mean_diversity": pi, "mutation_rate": MUTATION_RATE}
    prior = m._default_prior()
    expected = np.log(np.mean(pi) / (4.0 * MUTATION_RATE))
    assert np.isclose(prior["mu_log_ne_a"], expected)
    assert prior["mu_log_alpha_fold"] == 0.0


def test_with_prior_is_immutable():
    from bayesld.inference import PiecewiseExponential

    m = PiecewiseExponential()
    m2 = m.with_prior(
        mu_log_ne_a=np.log(2000),
        sigma_log_ne_a=1.5,
        mu_log_t=np.log(50.0),
        sigma_log_t=1.5,
        mu_log_alpha_fold=0.0,
        sigma_log_alpha_fold=1.5,
    )
    assert m._prior is None
    assert m2._prior["mu_log_ne_a"] == pytest.approx(np.log(2000))


def test_sample_prior_derives_transformed_params():
    from bayesld.inference import PiecewiseExponential

    m = PiecewiseExponential().with_prior(
        mu_log_ne_a=np.log(2000),
        sigma_log_ne_a=1.0,
        mu_log_t=np.log(50.0),
        sigma_log_t=1.0,
        mu_log_alpha_fold=0.5,
        sigma_log_alpha_fold=1.0,
    )
    idata = m.sample_prior(draws=200, seed=0)
    post = idata.posterior
    # Ne_c = Ne_a * exp(log_alpha_fold), alpha = log_alpha_fold / t0.
    assert np.allclose(post["Ne_c"], post["Ne_a"] * np.exp(post["log_alpha_fold"]))
    assert np.allclose(post["alpha"], post["log_alpha_fold"] / post["t0"])


def test_build_demography_matches_parameterization():
    from bayesld.inference import PiecewiseExponential

    m = PiecewiseExponential()
    params = {"ne_c": 100.0, "ne_a": 200.0, "t0": 100.0, "alpha": -0.00693}
    d = m._build_demography(params)
    assert d.populations[0].initial_size == 100.0
    assert d.populations[0].growth_rate == pytest.approx(-0.00693)


# ── Slow: Stan compilation + builder + active learning ──────────────────────────


@pytest.fixture(scope="module")
def data():
    pi, ld = sim_sufficient_stats(
        20,
        demography=_demography(),
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
    from bayesld.inference import PiecewiseExponential

    return PiecewiseExponential()


@pytest.mark.slow
def test_stan_code_frozen_at_construction(compiled_model):
    assert "piecewise_exponential" in compiled_model.stan_code


@pytest.mark.slow
def test_with_data_sets_empirical_prior(data, compiled_model):
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
    sd = m._stan_data()
    expected = np.log(np.mean(pi) / (4.0 * MUTATION_RATE))
    assert np.isclose(sd["mu_log_ne_a"], expected)
    assert sd["n_bias"] == 0 and sd["n_sigma"] == 0


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
    assert "Ne_a" in idata.posterior
    assert "Ne_c" in idata.posterior
    assert "alpha" in idata.posterior


@pytest.mark.slow
def test_active_learning_round_accumulates(compiled_model, data):
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
    m1 = m.active_learning_round(
        num_points=2,
        rtol=0.3,
        min_replicates=3,
        seed=0,
        draws=80,
    )
    assert m1 is not m
    assert len(m.bias_points) == 0
    assert len(m1.bias_points) == 2
    assert len(m1.sigma_points) == 2
