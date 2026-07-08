"""
Tests for the ``RandomWalk`` inference engine (fixed grid, log-space random walk).

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

GRID = np.array([50.0, 200.0])  # 2 boundaries -> 3 epochs


def _demography():
    # Piecewise-constant with the same grid the engine infers over.
    d = msprime.Demography()
    d.add_population(name="pop0", initial_size=100)
    d.add_population_parameters_change(time=50, initial_size=200, growth_rate=0)
    d.add_population_parameters_change(time=200, initial_size=400, growth_rate=0)
    return d


# ── Fast: prior / param plumbing ────────────────────────────────────────────────


def test_grid_validation():
    from bayesld.inference import RandomWalk

    with pytest.raises(ValueError):
        RandomWalk(grid=np.array([]))
    with pytest.raises(ValueError):
        RandomWalk(grid=np.array([100.0, 50.0]))  # not increasing
    with pytest.raises(ValueError):
        RandomWalk(grid=np.array([-10.0]))  # not positive


def test_num_epochs_is_grid_plus_one():
    from bayesld.inference import RandomWalk

    assert RandomWalk(grid=GRID).num_epochs == len(GRID) + 1


def test_default_prior_centres_on_watterson():
    from bayesld.inference import RandomWalk

    pi = np.array([4e-4, 5e-4])
    m = RandomWalk(grid=GRID)
    m._data = {"mean_diversity": pi, "mutation_rate": MUTATION_RATE}
    prior = m._default_prior()
    expected = np.log(np.mean(pi) / (4.0 * MUTATION_RATE))
    assert np.isclose(prior["mu_log_ne"], expected)
    # sigma_step is broadcast to one entry per step (num_epochs - 1).
    assert prior["sigma_step"].shape == (len(GRID),)
    assert np.all(prior["sigma_step"] > 0)


def test_with_prior_is_immutable():
    from bayesld.inference import RandomWalk

    m = RandomWalk(grid=GRID)
    m2 = m.with_prior(mu_log_ne=np.log(2000), sigma_log_ne=1.5, sigma_step=0.5)
    assert m._prior is None
    assert m2._prior["mu_log_ne"] == pytest.approx(np.log(2000))
    # Scalar sigma_step is repeated to one entry per step.
    assert np.allclose(m2._prior["sigma_step"], 0.5)
    assert m2._prior["sigma_step"].shape == (len(GRID),)


def test_sigma_step_accepts_per_step_vector():
    from bayesld.inference import RandomWalk

    m = RandomWalk(grid=GRID)
    per_step = np.array([0.3, 0.7])
    m2 = m.with_prior(mu_log_ne=np.log(2000), sigma_log_ne=1.5, sigma_step=per_step)
    assert np.allclose(m2._prior["sigma_step"], per_step)
    # Wrong length is rejected.
    with pytest.raises(ValueError):
        m.with_prior(mu_log_ne=0.0, sigma_log_ne=1.0, sigma_step=np.array([0.3]))


def test_sample_prior_is_a_reverse_cumsum_random_walk():
    from bayesld.inference import RandomWalk

    m = RandomWalk(grid=GRID).with_prior(
        mu_log_ne=np.log(2000), sigma_log_ne=1.0, sigma_step=0.5
    )
    idata = m.sample_prior(draws=500, seed=0)
    post = idata.posterior
    # Ancient epoch (last) equals the anchor; each epoch is the ancient anchor
    # plus the reverse cumulative sum of steps.
    log_ne = post["log_Ne"].values  # (chain, draw, epoch)
    steps = post["steps"].values  # (chain, draw, step)
    rev_cumsum = np.cumsum(steps[..., ::-1], axis=-1)[..., ::-1]
    recon = log_ne[..., -1:] + rev_cumsum
    assert np.allclose(log_ne[..., :-1], recon)
    # Grid is fixed across draws.
    assert np.allclose(post["t_boundaries"].values, GRID)


def test_build_demography_matches_grid():
    from bayesld.inference import RandomWalk

    m = RandomWalk(grid=GRID)
    params = {"ne": np.array([100.0, 200.0, 400.0]), "t": GRID}
    d = m._build_demography(params)
    assert d.populations[0].initial_size == 100.0
    times = [e.time for e in d.events]
    assert times == [50.0, 200.0]


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
    from bayesld.inference import RandomWalk

    return RandomWalk(grid=GRID)


@pytest.mark.slow
def test_stan_code_frozen_at_construction(compiled_model):
    assert (
        "random walk" in compiled_model.stan_code or "steps" in compiled_model.stan_code
    )


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
    assert np.isclose(sd["mu_log_ne"], expected)
    assert sd["n_epochs"] == len(GRID) + 1
    assert np.allclose(sd["grid"], GRID)
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
    assert "Ne_values" in idata.posterior
    assert idata.posterior["Ne_values"].sizes["epoch"] == len(GRID) + 1
    assert "steps" in idata.posterior


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
