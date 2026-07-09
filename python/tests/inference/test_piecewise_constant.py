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


# ── Fast: simplified posterior layout ─────────────────────────────────────────


def test_simplify_posterior_drops_surrogate_vars():
    import xarray as xr

    from bayesld.inference._base import _simplify_posterior_dataset

    n_bins = len(LEFT_BINS)
    n_epochs = 2
    coords = {
        "bin": (LEFT_BINS + RIGHT_BINS) / 2.0,
        "epoch": np.arange(n_epochs),
        "boundary": np.arange(n_epochs - 1),
    }
    post = xr.Dataset(
        {
            "Ne_values": (
                ["chain", "draw", "epoch"],
                np.ones((1, 2, n_epochs)),
            ),
            "t_boundaries": (
                ["chain", "draw", "boundary"],
                np.array([[[30.0], [40.0]]]),
            ),
            "gp_bias_ld": (
                ["chain", "draw", "bin"],
                np.zeros((1, 2, n_bins)),
            ),
            "corrected_expected_ld": (
                ["chain", "draw", "bin"],
                np.ones((1, 2, n_bins)),
            ),
            "beta_ld": (
                ["chain", "draw", "beta_ld_dim_0", "bin"],
                np.zeros((1, 2, 6, n_bins)),
            ),
            "approx_expected_ld": (
                ["chain", "draw", "bin"],
                np.zeros((1, 2, n_bins)),
            ),
        },
        coords={
            "chain": [0],
            "draw": [0, 1],
            "epoch": coords["epoch"],
            "boundary": coords["boundary"],
            "bin": coords["bin"],
        },
    )
    simplified = _simplify_posterior_dataset(post, coords=coords)

    assert "beta_ld" not in simplified
    assert "approx_expected_ld" not in simplified
    assert "gp_bias_ld" in simplified
    assert "corrected_expected_ld" in simplified
    assert "t_boundaries" in simplified
    assert simplified["t_boundaries"].sizes["boundary"] == n_epochs - 1
    assert simplified["t_boundaries"].values[0, 0, 0] == 30.0
    assert "boundary" in simplified.dims
    assert not any("_dim_" in d for d in simplified.dims)


def test_simplify_posterior_datatree_assignment_works_with_arviz_summary():
    import arviz as az
    import xarray as xr

    from bayesld.inference._base import _simplify_posterior_dataset

    n_bins = len(LEFT_BINS)
    post = xr.Dataset(
        {
            "Ne_values": (["chain", "draw", "epoch"], np.ones((1, 2, 1))),
            "beta_ld": (
                ["chain", "draw", "beta_ld_dim_0", "bin"],
                np.zeros((1, 2, 6, n_bins)),
            ),
        },
        coords={
            "chain": [0],
            "draw": [0, 1],
            "epoch": [0],
            "bin": np.arange(n_bins),
        },
    )
    idata = xr.DataTree.from_dict({"posterior": post})
    idata["posterior"] = _simplify_posterior_dataset(
        post, coords={"epoch": [0], "bin": np.arange(n_bins)}
    )
    assert isinstance(idata.posterior, xr.DataTree)
    assert "beta_ld" not in idata.posterior
    az.summary(idata, var_names=["Ne_values"])


def test_read_transition_times_cmdstanpy_like_fit():
    """Pathfinder/MCMC fits must not receive ``has_variable`` (cmdstanpy quirk)."""
    from bayesld.inference import PiecewiseConstant

    class _PathfinderLike:
        def stan_variable(self, name: str) -> np.ndarray:
            if name == "Ne_values":
                return np.ones((5, 2))
            if name == "t_boundaries":
                return np.full((5, 1), 30.0)
            raise KeyError(name)

    m = PiecewiseConstant(num_epochs=2)
    t = m._read_transition_times(_PathfinderLike(), n_epochs=2)
    assert t.shape == (5, 1)
    assert np.all(t == 30.0)


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
    import xarray as xr

    assert isinstance(idata, xr.DataTree)
    post = idata.posterior
    assert post["Ne_values"].sizes["epoch"] == 2
    assert "t_boundaries" in post
    assert "gp_bias_ld" in post
    assert "corrected_expected_ld" in post
    assert "beta_ld" not in post
    assert "boundary" in post.dims
    assert "log_likelihood" in idata
    assert "posterior_predictive" in idata
    assert "observed_data" in idata
    ppc = idata.posterior_predictive
    obs = idata.observed_data
    assert set(ppc.data_vars) >= {"observed_pi", "observed_ld", "mean_pi", "mean_ld"}
    assert set(obs.data_vars) >= {"observed_pi", "observed_ld", "mean_pi", "mean_ld"}
    np.testing.assert_allclose(
        ppc["mean_ld"].values,
        ppc["observed_ld"].mean(dim=["window"]).values,
    )
    np.testing.assert_allclose(
        obs["mean_ld"].values,
        obs["observed_ld"].mean(dim="window").values,
    )
    np.testing.assert_allclose(
        ppc["mean_pi"].values,
        ppc["observed_pi"].mean(dim=["window"]).values,
    )
    np.testing.assert_allclose(
        obs["mean_pi"].values,
        obs["observed_pi"].mean(dim="window").values,
    )
    const = idata.constant_data
    assert set(const.data_vars) == {"left", "right", "midpoint"}
    assert list(const["left"].dims) == ["bin"]
    np.testing.assert_allclose(const["left"].values, LEFT_BINS)
    np.testing.assert_allclose(const["right"].values, RIGHT_BINS)
    np.testing.assert_allclose(
        const["midpoint"].values, (LEFT_BINS + RIGHT_BINS) / 2.0, rtol=0, atol=1e-5
    )
    demographies = m.to_msprime_demography(idata)
    assert len(demographies) == 50


@pytest.mark.slow
def test_sample_simplify_false_retains_full_posterior(compiled_model, data):
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
    idata = m.sample(draws=20, tune=20, chains=1, simplify=False)
    post = idata.posterior
    assert "beta_ld" in post
    assert "t_boundaries" in post


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
