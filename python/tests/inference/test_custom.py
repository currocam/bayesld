"""
Tests for the ``CustomEngine`` bring-your-own-model escape hatch.

The custom engine is exercised by wrapping the *existing* two-phase
piecewise-exponential Stan file with callable seams, so a passing end-to-end run
also confirms the shared/*.stan ``#include`` refactor works for a Stan file
driven purely through the public ``CustomEngine`` API.

Fast tests cover the builder/seam plumbing (a warm Stan cache makes construction
cheap). Slow tests (``-m slow``) run NUTS and a tiny active-learning round.
"""

import pathlib

import msprime
import numpy as np
import pytest

import bayesld
from bayesld import deterministic as det
from bayesld.inference import CustomEngine, sim_sufficient_stats

LEFT_BINS, RIGHT_BINS = bayesld.linear_bins()
RECOMBINATION_RATE = 1e-8
MUTATION_RATE = 1e-8
SEQUENCE_LENGTH = RIGHT_BINS[-1] * 2 / RECOMBINATION_RATE

_STAN = (
    pathlib.Path(bayesld.inference.__file__).resolve().parent
    / "stan"
    / "piecewise_exponential.stan"
)


# ── Seam callables (two-phase piecewise-exponential parameterization) ────────────


def _default_prior(e):
    pi = e._data["mean_diversity"]
    mu = e._data["mutation_rate"]
    return {
        "mu_log_ne_a": float(np.log(np.mean(pi) / (4.0 * mu))),
        "sigma_log_ne_a": 1.5,
        "mu_log_t": float(np.log(50.0)),
        "sigma_log_t": 1.5,
        "mu_log_alpha_fold": 0.0,
        "sigma_log_alpha_fold": 1.5,
    }


def _build_demography(e, p):
    d = msprime.Demography()
    d.add_population(
        name="pop0", initial_size=float(p["ne_c"]), growth_rate=float(p["alpha"])
    )
    d.add_population_parameters_change(
        time=float(p["t0"]), initial_size=float(p["ne_a"]), growth_rate=0
    )
    return d


def _expected(e, p):
    return det.expected_piecewise_exponential(
        p["ne_c"],
        p["ne_a"],
        p["t0"],
        p["alpha"],
        e._left_bins,
        e._right_bins,
        e._data["mutation_rate"],
        sample_size=e._data["num_samples"],
        ploidy=e._data["ploidy"],
    )


def _extract_params(e, fit):
    def draws(name):
        return np.atleast_1d(np.asarray(fit.stan_variable(name))).ravel()

    nc, na, t0, al = draws("Ne_c"), draws("Ne_a"), draws("t0"), draws("alpha")
    return [
        {"ne_c": nc[i], "ne_a": na[i], "t0": t0[i], "alpha": al[i]}
        for i in range(len(nc))
    ]


def _engine(**overrides):
    kwargs = dict(
        default_prior=_default_prior,
        build_demography=_build_demography,
        expected=_expected,
        extract_params=_extract_params,
        title="MyExp",
    )
    kwargs.update(overrides)
    return CustomEngine(_STAN, **kwargs)


def _demography():
    d = msprime.Demography()
    alpha = (np.log(100) - np.log(200)) / 100
    d.add_population(name="pop0", initial_size=100, growth_rate=alpha)
    d.add_population_parameters_change(time=100, initial_size=200, growth_rate=0)
    return d


# ── Fast: seam / builder plumbing ───────────────────────────────────────────────


def test_with_prior_is_immutable():
    m = _engine()
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
    assert m2._prior_stan_data()["mu_log_ne_a"] == pytest.approx(np.log(2000))


def test_default_prior_uses_callback():
    m = _engine()
    m._data = {"mean_diversity": np.array([4e-4, 5e-4]), "mutation_rate": MUTATION_RATE}
    prior = m._default_prior()
    expected = np.log(np.mean(m._data["mean_diversity"]) / (4.0 * MUTATION_RATE))
    assert np.isclose(prior["mu_log_ne_a"], expected)


def test_missing_default_prior_raises():
    m = CustomEngine(_STAN)  # no seams at all
    with pytest.raises(RuntimeError, match="default_prior"):
        m._default_prior()


def test_missing_active_learning_seam_raises():
    m = CustomEngine(_STAN)  # no build_demography provided
    with pytest.raises(NotImplementedError, match="build_demography"):
        m._build_demography({"ne_c": 1.0})


def test_title_surfaces_in_repr_html():
    assert "MyExp" in _engine()._repr_html_()


# ── Slow: compilation via includes + sampling + active learning ─────────────────


@pytest.fixture(scope="module")
def data():
    return sim_sufficient_stats(
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


@pytest.mark.slow
def test_sample_returns_posterior(data):
    pi, ld = data
    m = _engine().with_data(
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
def test_active_learning_round_accumulates(data):
    pi, ld = data
    m = _engine().with_data(
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
        num_points=2, rtol=0.3, min_replicates=3, seed=0, draws=80
    )
    assert m1 is not m
    assert len(m.bias_points) == 0
    assert len(m1.bias_points) == 2
    assert len(m1.sigma_points) == 2
