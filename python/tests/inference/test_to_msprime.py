"""
Tests for the posterior → msprime round-trip.

Fast tests cover _to_sample_dataset input-shape coercion and error paths.
Slow tests (``-m slow``) compile Stan and verify each engine produces valid
msprime.Demography objects from sampled posteriors.
"""

import numpy as np
import pytest
import xarray as xr

import bayesld

LEFT_BINS, RIGHT_BINS = bayesld.linear_bins()
RECOMBINATION_RATE = 1e-8
MUTATION_RATE = 1e-8
SEQUENCE_LENGTH = RIGHT_BINS[-1] * 2 / RECOMBINATION_RATE


# ── Fast: _to_sample_dataset coercion and error paths ───────────────────────


def _make_post_dataset(n_chain=1, n_draw=3, n_epoch=2):
    """Minimal xarray.Dataset with chain/draw dims for coercion tests."""
    return xr.Dataset(
        {
            "Ne_values": (
                ["chain", "draw", "epoch"],
                np.ones((n_chain, n_draw, n_epoch)) * 500.0,
            ),
            "t_boundaries": (
                ["chain", "draw", "boundary"],
                np.ones((n_chain, n_draw, n_epoch - 1)) * 50.0,
            ),
        },
        coords={
            "chain": np.arange(n_chain),
            "draw": np.arange(n_draw),
            "epoch": np.arange(n_epoch),
            "boundary": np.arange(n_epoch - 1),
        },
    )


def test_to_sample_dataset_accepts_chain_draw():
    from bayesld.inference._base import _BaseEngine

    ds = _make_post_dataset(n_chain=2, n_draw=4)
    result = _BaseEngine._to_sample_dataset(ds)
    assert "sample" in result.dims
    assert result.sizes["sample"] == 2 * 4


def test_to_sample_dataset_accepts_sample_dim():
    from bayesld.inference._base import _BaseEngine

    ds = _make_post_dataset(n_chain=1, n_draw=5)
    stacked = ds.stack(sample=("chain", "draw"))
    result = _BaseEngine._to_sample_dataset(stacked)
    assert "sample" in result.dims
    assert result.sizes["sample"] == 5


def test_to_sample_dataset_accepts_datatree():
    from bayesld.inference._base import _BaseEngine

    ds = _make_post_dataset(n_chain=1, n_draw=3)
    dt = xr.DataTree.from_dict({"posterior": ds})
    result = _BaseEngine._to_sample_dataset(dt)
    assert "sample" in result.dims


def test_to_sample_dataset_no_sample_or_chain_draw_raises():
    from bayesld.inference._base import _BaseEngine

    ds = xr.Dataset({"x": (["time"], np.arange(5))})
    with pytest.raises(ValueError, match="'sample' dim"):
        _BaseEngine._to_sample_dataset(ds)


def test_to_sample_dataset_wrong_type_raises():
    from bayesld.inference._base import _BaseEngine

    with pytest.raises(TypeError, match="xarray"):
        _BaseEngine._to_sample_dataset([1, 2, 3])


# ── Slow: demography round-trip per engine ───────────────────────────────────


def _data_kwargs(n_replicates=20, n_samples=10):
    import msprime

    d = msprime.Demography()
    d.add_population(name="pop0", initial_size=200)
    d.add_population_parameters_change(time=100, initial_size=100, growth_rate=0)

    from bayesld.inference import sim_sufficient_stats

    pi, ld = sim_sufficient_stats(
        n_samples,
        demography=d,
        left_bins=LEFT_BINS,
        right_bins=RIGHT_BINS,
        mutation_rate=MUTATION_RATE,
        recombination_rate=RECOMBINATION_RATE,
        sequence_length=SEQUENCE_LENGTH,
        random_seed=42,
        num_replicates=n_replicates,
    )
    return dict(
        mean_diversity=pi,
        mean_ld=ld,
        left_bins=LEFT_BINS,
        right_bins=RIGHT_BINS,
        recombination_rate=RECOMBINATION_RATE,
        mutation_rate=MUTATION_RATE,
        num_samples=n_samples,
        sequence_length=SEQUENCE_LENGTH,
    )


@pytest.mark.slow
def test_piecewise_constant_to_msprime_round_trip():
    import msprime

    from bayesld.inference import PiecewiseConstant

    m = PiecewiseConstant(num_epochs=2).with_data(**_data_kwargs())
    idata = m.sample(draws=10, tune=20, chains=1, seed=0)
    demographies = m.to_msprime_demography(idata)

    assert len(demographies) == 10
    for d in demographies:
        assert isinstance(d, msprime.Demography)
        d.validate()
        assert len(d.populations) == 1


@pytest.mark.slow
def test_piecewise_exponential_to_msprime_round_trip():
    import msprime

    from bayesld.inference import PiecewiseExponential

    m = PiecewiseExponential().with_data(**_data_kwargs())
    idata = m.sample(draws=10, tune=20, chains=1, seed=0)
    demographies = m.to_msprime_demography(idata)

    assert len(demographies) == 10
    for d in demographies:
        assert isinstance(d, msprime.Demography)
        d.validate()
        assert len(d.populations) == 1


@pytest.mark.slow
def test_random_walk_to_msprime_round_trip():
    import msprime

    from bayesld.inference import RandomWalk

    grid = np.array([50.0, 200.0])
    m = RandomWalk(grid).with_data(**_data_kwargs())
    idata = m.sample(draws=10, tune=20, chains=1, seed=0)
    demographies = m.to_msprime_demography(idata)

    assert len(demographies) == 10
    for d in demographies:
        assert isinstance(d, msprime.Demography)
        d.validate()
        # 2 boundaries + 1 = 3 epochs → 3 size changes (initial + 2 changes)
        assert len(d.populations) == 1


@pytest.mark.slow
def test_to_msprime_all_three_input_shapes():
    """All three accepted input shapes yield identical demographies."""
    import arviz as az

    from bayesld.inference import PiecewiseConstant

    m = PiecewiseConstant(num_epochs=2).with_data(**_data_kwargs())
    idata = m.sample(draws=8, tune=20, chains=1, seed=1)

    # Shape 1: DataTree from sample()
    d1 = m.to_msprime_demography(idata)
    # Shape 2: plain Dataset with chain/draw
    d2 = m.to_msprime_demography(idata.posterior.to_dataset())
    # Shape 3: arviz.extract → sample dim
    d3 = m.to_msprime_demography(az.extract(idata))

    assert len(d1) == len(d2) == len(d3) == 8
    # All input forms must produce the same Ne values.
    for i in range(8):
        np.testing.assert_allclose(
            d1[i].populations[0].initial_size,
            d2[i].populations[0].initial_size,
            rtol=1e-10,
        )
        np.testing.assert_allclose(
            d1[i].populations[0].initial_size,
            d3[i].populations[0].initial_size,
            rtol=1e-10,
        )
