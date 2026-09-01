import numpy as np
import pytest

import bayesld
from bayesld import deterministic as det
from bayesld.inference._base import (
    _MIN_EFFECTIVE_HAPLOIDS,
    _BaseEngine,
    _ld_fs_alpha,
    _ld_fs_beta,
    _rescale_ld_effective_sample,
)

LEFT_BINS, RIGHT_BINS = bayesld.linear_bins()
N_BINS = len(LEFT_BINS)
RECOMBINATION_RATE = 1e-8
MUTATION_RATE = 1e-8
SEQUENCE_LENGTH = RIGHT_BINS[-1] * 2 / RECOMBINATION_RATE


def test_rescale_is_identity_when_s_eff_equals_s():
    mu = np.array([1e-4, 2e-4, 3e-4])
    S = 40.0
    out = _rescale_ld_effective_sample(mu, S, np.full(3, S))
    np.testing.assert_allclose(out, mu, rtol=1e-12)


def test_rescale_increases_as_s_eff_shrinks():
    mu = np.full(4, 1e-4)
    S = 40.0
    s_eff_grid = [40.0, 30.0, 20.0, 10.0]
    values = [
        _rescale_ld_effective_sample(mu, S, np.full(4, s))[0] for s in s_eff_grid
    ]
    assert values == sorted(values), "rescale should be monotone decreasing in S_eff"
    assert values[0] == pytest.approx(mu[0])
    assert all(v > mu[0] for v in values[1:])


def test_alpha_beta_positive_away_from_pole():
    S = np.array([10.0, 50.0, 200.0])
    assert np.all(_ld_fs_alpha(S) > 0)
    assert np.all(_ld_fs_beta(S) > 0)


def _mean_ld(num_windows=3, n_bins=N_BINS):
    return np.full((num_windows, n_bins), 1e-4)


def test_validate_missingness_none_disables():
    assert _BaseEngine._validate_missingness(None, _mean_ld(), 20, 2) is None


def test_validate_missingness_broadcasts_vector():
    ld = _mean_ld(num_windows=3, n_bins=5)
    v = _BaseEngine._validate_missingness(np.full(5, 0.2), ld, 20, 2)
    assert v.shape == (3, 5)
    np.testing.assert_allclose(v, 0.2)


def test_validate_missingness_matrix_passthrough():
    ld = _mean_ld(num_windows=3, n_bins=5)
    m = np.full((3, 5), 0.1)
    v = _BaseEngine._validate_missingness(m, ld, 20, 2)
    np.testing.assert_allclose(v, m)


@pytest.mark.parametrize(
    "bad_shape",
    [(4,), (2, 5), (3, 4), (3,)],
)
def test_validate_missingness_wrong_shape_raises(bad_shape):
    ld = _mean_ld(num_windows=3, n_bins=5)
    with pytest.raises(ValueError):
        _BaseEngine._validate_missingness(np.zeros(bad_shape), ld, 20, 2)


@pytest.mark.parametrize("bad_value", [-0.1, 1.0, 1.5, np.nan])
def test_validate_missingness_out_of_range_raises(bad_value):
    ld = _mean_ld(num_windows=2, n_bins=3)
    m = np.full((2, 3), 0.1)
    m[0, 0] = bad_value
    with pytest.raises(ValueError):
        _BaseEngine._validate_missingness(m, ld, 20, 2)


def test_validate_missingness_ignores_nan_bins_where_ld_is_nan():
    """data_from_* leaves both mean_ld and missingness NaN for empty bins."""
    ld = _mean_ld(num_windows=2, n_bins=3)
    ld[0, 0] = np.nan
    m = np.full((2, 3), 0.1)
    m[0, 0] = np.nan
    v = _BaseEngine._validate_missingness(m, ld, 20, 2)
    assert np.isnan(v[0, 0])


def test_validate_missingness_rejects_near_pole():
    ld = _mean_ld(num_windows=1, n_bins=1)
    # num_samples=2 -> S_full=4; missingness=0.6 -> S_eff=1.6 < _MIN_EFFECTIVE_HAPLOIDS.
    with pytest.raises(ValueError):
        _BaseEngine._validate_missingness(np.array([[0.6]]), ld, 2, 2)
    assert _MIN_EFFECTIVE_HAPLOIDS == 4.0


def test_validate_missingness_ploidy_one_warns_and_ignores():
    ld = _mean_ld(num_windows=2, n_bins=3)
    with pytest.warns(UserWarning, match="ploidy"):
        v = _BaseEngine._validate_missingness(np.full((2, 3), 0.1), ld, 20, 1)
    assert v is None


@pytest.fixture(scope="module")
def compiled_model():
    from bayesld.inference import PiecewiseConstant

    return PiecewiseConstant(num_epochs=2)


@pytest.fixture(scope="module")
def toy_data():
    rng = np.random.default_rng(0)
    num_windows = 3
    pi = rng.uniform(1e-4, 2e-4, size=num_windows)
    ld = rng.uniform(1e-5, 5e-5, size=(num_windows, N_BINS))
    return pi, ld


def _with_toy_data(compiled_model, toy_data, **kwargs):
    pi, ld = toy_data
    return compiled_model.with_data(
        mean_diversity=pi,
        mean_ld=ld,
        left_bins=LEFT_BINS,
        right_bins=RIGHT_BINS,
        recombination_rate=RECOMBINATION_RATE,
        mutation_rate=MUTATION_RATE,
        num_samples=20,
        sequence_length=SEQUENCE_LENGTH,
        **kwargs,
    )


@pytest.mark.slow
def test_stan_data_default_has_no_missingness(compiled_model, toy_data):
    m = _with_toy_data(compiled_model, toy_data)
    sd = m._stan_data()
    assert sd["use_missingness"] == 0
    np.testing.assert_array_equal(sd["missingness"], np.zeros((3, N_BINS)))
    assert sd["missingness"].shape == sd["ld_mat"].shape


@pytest.mark.slow
def test_stan_data_vector_broadcasts_across_windows(compiled_model, toy_data):
    m = _with_toy_data(compiled_model, toy_data, missingness=np.full(N_BINS, 0.15))
    sd = m._stan_data()
    assert sd["use_missingness"] == 1
    assert sd["missingness"].shape == (3, N_BINS)
    np.testing.assert_allclose(sd["missingness"], 0.15)


@pytest.mark.slow
def test_stan_data_matrix_passthrough(compiled_model, toy_data):
    m_matrix = np.linspace(0.0, 0.3, 3 * N_BINS).reshape(3, N_BINS)
    m = _with_toy_data(compiled_model, toy_data, missingness=m_matrix)
    sd = m._stan_data()
    assert sd["use_missingness"] == 1
    np.testing.assert_allclose(sd["missingness"], m_matrix)


@pytest.mark.slow
def test_constant_data_carries_missingness(compiled_model, toy_data):
    m = _with_toy_data(compiled_model, toy_data, missingness=np.full(N_BINS, 0.15))
    idata = m.sample(draws=10, tune=10, chains=1)
    assert "missingness" in idata.constant_data
    assert list(idata.constant_data["missingness"].dims) == ["window", "bin"]


@pytest.mark.slow
def test_constant_data_omits_missingness_by_default(compiled_model, toy_data):
    m = _with_toy_data(compiled_model, toy_data)
    idata = m.sample(draws=10, tune=10, chains=1)
    assert "missingness" not in idata.constant_data


def _fixed_params(hsgp_m_ld: int, n_bins: int) -> dict:
    """An arbitrary, valid, deterministic point in parameter space.

    beta_ld=0 and log_sigma_y=0/L_Omega=I zero out the GP bias and reduce the
    joint covariance to identity, so the Stan log-density's dependence on
    ``use_missingness`` reduces to the plain sum-of-squares residual on
    ``rescale_ld_effective_sample`` — exactly what
    ``_base._rescale_ld_effective_sample`` computes in Python.
    """
    return {
        "log_Ne_indep": [np.log(1000.0), np.log(1200.0)],
        "log_t": [np.log(50.0)],
        "gp_rho": 1.0,
        "gp_alpha": 0.001,
        "beta_ld": np.zeros((hsgp_m_ld, n_bins)).tolist(),
        "log_sigma_y": np.zeros(n_bins + 1).tolist(),
        "L_Omega": np.eye(n_bins + 1).tolist(),
    }


@pytest.mark.slow
def test_log_prob_identity_for_all_zero_missingness(compiled_model, toy_data):
    """use_missingness=1 with an all-zero matrix must reproduce use_missingness=0
    exactly (log_prob is deterministic — no NUTS sampling noise involved)."""
    m = _with_toy_data(compiled_model, toy_data)
    sd_default = m._stan_data()

    sd_zeros = dict(sd_default)
    sd_zeros["use_missingness"] = 1
    sd_zeros["missingness"] = np.zeros_like(sd_default["missingness"])

    params = _fixed_params(sd_default["hsgp_m_ld"], N_BINS)
    lp_default = m._model.log_prob(params=params, data=sd_default)["lp__"].iloc[0]
    lp_zeros = m._model.log_prob(params=params, data=sd_zeros)["lp__"].iloc[0]
    assert lp_default == pytest.approx(lp_zeros, abs=0.0, rel=0.0)


@pytest.mark.slow
def test_log_prob_matches_python_rescale_at_nonzero_missingness(
    compiled_model, toy_data
):
    """Cross-check: the lp__ shift Stan's use_missingness branch introduces equals
    the shift predicted by re-deriving the (chosen) fixed-point demography's LD in
    Python and rescaling it the same way ``_pointwise_log_lik`` does."""
    missingness_value = 0.25
    m_none = _with_toy_data(compiled_model, toy_data)
    m_miss = _with_toy_data(
        compiled_model, toy_data, missingness=np.full(N_BINS, missingness_value)
    )
    sd_none = m_none._stan_data()
    sd_miss = m_miss._stan_data()
    params = _fixed_params(sd_none["hsgp_m_ld"], N_BINS)

    # sig_figs=18: lp__ here is O(1e3) while the quantity under test is the O(1e-3)
    # difference between two evaluations of it — cmdstanpy's default CSV round-trip
    # precision is nowhere near enough to resolve that difference.
    lp_none = m_none._model.log_prob(params=params, data=sd_none, sig_figs=18)[
        "lp__"
    ].iloc[0]
    lp_miss = m_miss._model.log_prob(params=params, data=sd_miss, sig_figs=18)[
        "lp__"
    ].iloc[0]

    # Reconstruct expected_pi / corrected_expected_ld exactly as the Stan
    # transformed parameters block does for this fixed_params point: with
    # beta_ld == 0 the GP bias is zero, so corrected_expected_ld ==
    # approx_expected_ld == det.expected_piecewise_constant's LD output.
    ne_values = np.exp(params["log_Ne_indep"])
    t_boundaries = np.exp(params["log_t"])
    expected_pi, corrected_expected_ld = det.expected_piecewise_constant(
        ne_values,
        t_boundaries,
        LEFT_BINS,
        RIGHT_BINS,
        MUTATION_RATE,
        sample_size=20,
        ploidy=2,
    )
    corrected_expected_ld = np.asarray(corrected_expected_ld)

    y_obs = np.column_stack([toy_data[0], toy_data[1]])  # (window, D)
    D = y_obs.shape[1]

    def _mvn_identity_loglik(mu_ld_per_window: np.ndarray) -> float:
        mu = np.concatenate(
            [np.full((mu_ld_per_window.shape[0], 1), float(expected_pi)), mu_ld_per_window],
            axis=1,
        )
        residual = y_obs - mu
        quad = (residual**2).sum(axis=1)
        log_det = 0.0  # sigma_y=1, L_Omega=I
        return float((-0.5 * D * np.log(2.0 * np.pi) - log_det - 0.5 * quad).sum())

    ld_none = np.broadcast_to(corrected_expected_ld, (3, N_BINS))
    S_full = 2.0 * 20
    S_eff = S_full * (1.0 - missingness_value)
    ld_miss = _rescale_ld_effective_sample(
        corrected_expected_ld, S_full, np.full(N_BINS, S_eff)
    )
    ld_miss = np.broadcast_to(ld_miss, (3, N_BINS))

    expected_lp_diff = _mvn_identity_loglik(ld_miss) - _mvn_identity_loglik(ld_none)
    actual_lp_diff = lp_miss - lp_none
    assert actual_lp_diff == pytest.approx(expected_lp_diff, rel=1e-8, abs=1e-8)
