"""
Tests for bayesld.models.TwoEpochDemography and PiecewiseConstantDemography.

TwoEpoch: two-epoch piecewise constant with default priors.
PiecewiseConstant: generic N-epoch with no defaults (tested with 3 epochs).
"""

import bayesld
import cmdstanpy
import jax
import numpy as np
import pytest
from bayesld import deterministic
from bayesld.models import PiecewiseConstantDemography, TwoEpochDemography

jax.config.update("jax_enable_x64", True)

MUTATION_RATE = 1e-8
RECOMBINATION_RATE = 1e-8
NUM_SAMPLES = 50
NUM_WINDOWS = 200
NOISE_REL_SD = 0.01

LEFT_BINS, RIGHT_BINS = bayesld.linear_bins()

LOG_NE_MU = np.log(500.0)
LOG_NE_SIGMA = 1.0


# ── TwoEpoch simulation ──────────────────────────────────────────────────


def _simulate_two_epoch(seed=42):
    rng = np.random.default_rng(seed)
    ne_a = float(np.exp(rng.normal(LOG_NE_MU, LOG_NE_SIGMA)))
    log_fold_change = float(rng.normal(0, 1.0))
    ne_c = ne_a * np.exp(log_fold_change)
    t0 = float(np.exp(rng.normal(np.log(100), 0.5)))

    pi_exp, ld_exp = deterministic.expected_piecewise_constant(
        np.array([ne_c, ne_a]),
        np.array([t0]),
        LEFT_BINS,
        RIGHT_BINS,
        MUTATION_RATE,
        sample_size=NUM_SAMPLES,
    )
    pi_exp = float(pi_exp)
    ld_exp = np.asarray(ld_exp)

    bias = rng.normal(0, 0.005, size=len(LEFT_BINS))
    ld_true = ld_exp * (1 + bias)

    diversity = rng.normal(pi_exp, abs(pi_exp) * NOISE_REL_SD, size=NUM_WINDOWS)
    ld = rng.normal(
        ld_true,
        np.abs(ld_true)[None, :] * NOISE_REL_SD,
        size=(NUM_WINDOWS, len(LEFT_BINS)),
    )
    assert np.all(np.isfinite(diversity))
    assert np.all(np.isfinite(ld))
    assert np.all(np.std(ld, axis=0) > 0)
    assert np.std(diversity) > 0
    return ne_c, ne_a, t0, log_fold_change, diversity, ld


def _make_two_epoch(**kwargs):
    _, _, _, _, diversity, ld = _simulate_two_epoch()
    defaults = dict(
        diversity=diversity,
        ld=ld,
        mutation_rate=MUTATION_RATE,
        recombination_rate=RECOMBINATION_RATE,
        num_samples=NUM_SAMPLES,
        left_bins=LEFT_BINS,
        right_bins=RIGHT_BINS,
    )
    defaults.update(kwargs)
    return TwoEpochDemography(**defaults)


# ── TwoEpoch non-slow tests ──────────────────────────────────────────────


def test_two_epoch_primitives():
    model = _make_two_epoch()
    code = model.get_stan_code()
    assert isinstance(code, str)
    assert "parameters" in code
    assert "n_synthetic" in code
    assert "n_epochs" in code
    assert isinstance(model.model, cmdstanpy.CmdStanModel)

    data = model.stan_data()
    assert isinstance(data, dict)
    assert data["n_synthetic"] == 0
    assert data["n_epochs"] == 2
    assert data["eval_rel_bias"].shape == (0, len(LEFT_BINS))


def test_two_epoch_stan_data_keys():
    model = _make_two_epoch()
    data = model.stan_data()
    expected_keys = {
        "n_bins",
        "num_windows",
        "left_bins",
        "right_bins",
        "mutation_rate",
        "sample_size",
        "pi_array",
        "ld_mat",
        "n_epochs",
        "n_quad",
        "gl_nodes",
        "gl_weights",
        "n_synthetic",
        "eval_rel_bias",
        "eval_eps_rel",
        "hsgp_c",
        "hsgp_m",
        "gp_alpha_std",
    }
    assert expected_keys == set(data.keys())


def test_two_epoch_update_data():
    model = _make_two_epoch()
    old_model_obj = model.model
    _, _, _, _, new_div, new_ld = _simulate_two_epoch(seed=99)
    model.update_data(diversity=new_div, ld=new_ld)
    assert model.model is old_model_obj
    data = model.stan_data()
    np.testing.assert_array_equal(data["pi_array"], new_div)


def test_two_epoch_update_prior_recompiles():
    model = _make_two_epoch()
    old_model_obj = model.model
    model.update_prior(
        prior="    log_Ne_a ~ normal(5, 2.0);\n    log_t0 ~ normal(4, 1);\n    log_fold_change ~ normal(0, 1);"
    )
    assert model.model is not old_model_obj


def test_two_epoch_update_prior_gp_alpha_std_no_recompile():
    model = _make_two_epoch()
    old_model_obj = model.model
    model.update_prior(gp_alpha_std=0.01)
    assert model.model is old_model_obj
    assert model.stan_data()["gp_alpha_std"] == 0.01


def test_two_epoch_add_synthetic_points():
    model = _make_two_epoch()
    n_bins = len(LEFT_BINS)
    points = [
        {"rel_bias": np.zeros(n_bins), "eps_rel": np.ones(n_bins) * 0.01},
    ]
    model.add_synthetic_points(points)
    assert len(model.synthetic_points) == 1
    data = model.stan_data()
    assert data["n_synthetic"] == 1
    assert data["eval_rel_bias"].shape == (1, n_bins)


def test_two_epoch_add_synthetic_points_validation():
    model = _make_two_epoch()
    with pytest.raises(ValueError, match="keys"):
        model.add_synthetic_points([{"rel_bias": np.zeros(5)}])


def test_two_epoch_generate_stan_default():
    from bayesld.models.piecewise_constant import _generate_stan

    code = _generate_stan()
    assert "log_Ne_a" in code
    assert "log_t0" in code
    assert "log_fold_change" in code
    assert "log_Ne_c = log_Ne_a + log_fold_change" in code
    assert "Ne_values = [Ne_c, Ne_a]'" in code
    assert "t_boundaries = [t0]'" in code
    assert "n_synthetic" in code
    assert "corrected_ld" in code


def test_two_epoch_generate_stan_custom():
    from bayesld.models.piecewise_constant import _generate_stan

    custom_tp = (
        "    real<lower=0> Ne_c = exp(log_Ne_c);\n"
        "    real<lower=0> Ne_a = exp(log_Ne_a);\n"
        "    real<lower=0> t0   = exp(log_t0);\n"
        "    vector[2] Ne_values = [Ne_c, Ne_a]';\n"
        "    vector[1] t_boundaries = [t0]';"
    )
    code = _generate_stan(
        prior="    log_Ne_a ~ normal(10, 0.5);\n    log_t0 ~ normal(4, 0.1);\n    log_fold_change ~ normal(0, 0.5);",
        parameters="    real<offset=log_ne_offset> log_Ne_c;\n    real<offset=log_ne_offset> log_Ne_a;\n    real log_t0;\n    real log_fold_change;",
        transformed_parameters=custom_tp,
    )
    assert "log_Ne_a ~ normal(10, 0.5);" in code
    assert "real<offset=log_ne_offset> log_Ne_c;" in code
    assert "log_Ne_c = log_Ne_a + log_fold_change" not in code


# ── TwoEpoch slow tests ──────────────────────────────────────────────────


@pytest.mark.slow
def test_two_epoch_sample_default_prior():
    model = _make_two_epoch()
    idata = model.sample(chains=2, iter_warmup=10, iter_sampling=10)
    assert "posterior" in idata.children
    assert "Ne_values" in idata["posterior"].ds


@pytest.mark.slow
def test_two_epoch_sample_true_prior():
    ne_c, ne_a, t0, lfc, diversity, ld = _simulate_two_epoch()
    log_ne_c = np.log(ne_c)
    log_ne_a = np.log(ne_a)
    log_t0 = np.log(t0)
    model = TwoEpochDemography(
        diversity=diversity,
        ld=ld,
        mutation_rate=MUTATION_RATE,
        recombination_rate=RECOMBINATION_RATE,
        num_samples=NUM_SAMPLES,
        left_bins=LEFT_BINS,
        right_bins=RIGHT_BINS,
        prior=(
            f"    log_Ne_c ~ normal({log_ne_c:.4f}, 0.1);\n"
            f"    log_Ne_a ~ normal({log_ne_a:.4f}, 0.1);\n"
            f"    log_t0 ~ normal({log_t0:.4f}, 0.1);\n"
            f"    log_fold_change ~ normal(0, 1.0);"
        ),
        parameters=(
            "    real<offset=log_ne_offset> log_Ne_c;\n"
            "    real<offset=log_ne_offset> log_Ne_a;\n"
            "    real log_t0;\n"
            "    real log_fold_change;"
        ),
        transformed_parameters=(
            "    real<lower=0> Ne_c = exp(log_Ne_c);\n"
            "    real<lower=0> Ne_a = exp(log_Ne_a);\n"
            "    real<lower=0> t0   = exp(log_t0);\n"
            "    vector[2] Ne_values = [Ne_c, Ne_a]';\n"
            "    vector[1] t_boundaries = [t0]';"
        ),
    )
    idata = model.sample(chains=2, iter_warmup=10, iter_sampling=10)
    assert "Ne_values" in idata["posterior"].ds


# ── Three-epoch simulation ────────────────────────────────────────────────


def _simulate_three_epoch(seed=42):
    rng = np.random.default_rng(seed)
    ne_values = np.exp(rng.normal(LOG_NE_MU, LOG_NE_SIGMA, size=3))
    t1 = float(np.exp(rng.normal(np.log(50), 0.3)))
    t2 = float(np.exp(rng.normal(np.log(200), 0.3)))
    if t2 <= t1:
        t1, t2 = t2, t1
    t_boundaries = np.array([t1, t2])

    pi_exp, ld_exp = deterministic.expected_piecewise_constant(
        ne_values,
        t_boundaries,
        LEFT_BINS,
        RIGHT_BINS,
        MUTATION_RATE,
        sample_size=NUM_SAMPLES,
    )
    pi_exp = float(pi_exp)
    ld_exp = np.asarray(ld_exp)

    bias = rng.normal(0, 0.005, size=len(LEFT_BINS))
    ld_true = ld_exp * (1 + bias)

    diversity = rng.normal(pi_exp, abs(pi_exp) * NOISE_REL_SD, size=NUM_WINDOWS)
    ld = rng.normal(
        ld_true,
        np.abs(ld_true)[None, :] * NOISE_REL_SD,
        size=(NUM_WINDOWS, len(LEFT_BINS)),
    )
    assert np.all(np.isfinite(diversity))
    assert np.all(np.isfinite(ld))
    return ne_values, t_boundaries, diversity, ld


_THREE_EPOCH_PARAMETERS = """\
    vector<offset=log_ne_offset>[3] log_Ne_values;
    ordered[2] log_t_boundaries;"""

_THREE_EPOCH_TRANSFORMED_PARAMETERS = """\
    vector<lower=0>[3] Ne_values = exp(log_Ne_values);
    vector<lower=0>[2] t_boundaries = exp(log_t_boundaries);"""


def _three_epoch_prior(ne_values, t_boundaries):
    log_ne = np.log(ne_values)
    log_t = np.log(t_boundaries)
    return (
        f"    log_Ne_values ~ normal([{log_ne[0]:.4f}, {log_ne[1]:.4f}, {log_ne[2]:.4f}]', 0.5);\n"
        f"    log_t_boundaries ~ normal([{log_t[0]:.4f}, {log_t[1]:.4f}]', 0.5);"
    )


def _make_three_epoch(**kwargs):
    ne_values, t_boundaries, diversity, ld = _simulate_three_epoch()
    defaults = dict(
        diversity=diversity,
        ld=ld,
        mutation_rate=MUTATION_RATE,
        recombination_rate=RECOMBINATION_RATE,
        num_samples=NUM_SAMPLES,
        left_bins=LEFT_BINS,
        right_bins=RIGHT_BINS,
        n_epochs=3,
        parameters=_THREE_EPOCH_PARAMETERS,
        transformed_parameters=_THREE_EPOCH_TRANSFORMED_PARAMETERS,
        prior=_three_epoch_prior(ne_values, t_boundaries),
    )
    defaults.update(kwargs)
    return PiecewiseConstantDemography(**defaults)


# ── Generic PiecewiseConstant non-slow tests ──────────────────────────────


def test_generic_requires_all_args():
    _, _, diversity, ld = _simulate_three_epoch()
    base = dict(
        diversity=diversity,
        ld=ld,
        mutation_rate=MUTATION_RATE,
        recombination_rate=RECOMBINATION_RATE,
        num_samples=NUM_SAMPLES,
        left_bins=LEFT_BINS,
        right_bins=RIGHT_BINS,
        n_epochs=3,
    )
    with pytest.raises(TypeError):
        PiecewiseConstantDemography(**base)
    with pytest.raises(TypeError):
        PiecewiseConstantDemography(**base, parameters=_THREE_EPOCH_PARAMETERS)
    with pytest.raises(TypeError):
        PiecewiseConstantDemography(
            **base,
            parameters=_THREE_EPOCH_PARAMETERS,
            transformed_parameters=_THREE_EPOCH_TRANSFORMED_PARAMETERS,
        )


def test_generic_primitives():
    model = _make_three_epoch()
    code = model.get_stan_code()
    assert isinstance(code, str)
    assert "n_epochs" in code
    assert isinstance(model.model, cmdstanpy.CmdStanModel)

    data = model.stan_data()
    assert data["n_epochs"] == 3
    assert data["n_synthetic"] == 0


def test_generic_stan_data_keys():
    model = _make_three_epoch()
    data = model.stan_data()
    expected_keys = {
        "n_bins",
        "num_windows",
        "left_bins",
        "right_bins",
        "mutation_rate",
        "sample_size",
        "pi_array",
        "ld_mat",
        "n_epochs",
        "n_quad",
        "gl_nodes",
        "gl_weights",
        "n_synthetic",
        "eval_rel_bias",
        "eval_eps_rel",
        "hsgp_c",
        "hsgp_m",
        "gp_alpha_std",
    }
    assert expected_keys == set(data.keys())


def test_generic_update_data():
    model = _make_three_epoch()
    old_model_obj = model.model
    _, _, new_div, new_ld = _simulate_three_epoch(seed=99)
    model.update_data(diversity=new_div, ld=new_ld)
    assert model.model is old_model_obj
    data = model.stan_data()
    np.testing.assert_array_equal(data["pi_array"], new_div)


def test_generic_generate_stan():
    from bayesld.models.piecewise_constant import _generate_stan

    code = _generate_stan(
        parameters=_THREE_EPOCH_PARAMETERS,
        transformed_parameters=_THREE_EPOCH_TRANSFORMED_PARAMETERS,
        prior="    log_Ne_values ~ normal(6, 1.0);\n    log_t_boundaries ~ normal(4, 1.0);",
    )
    assert "log_Ne_values" in code
    assert "log_t_boundaries" in code
    assert "Ne_values = exp(log_Ne_values)" in code
    assert "t_boundaries = exp(log_t_boundaries)" in code
    assert "mu_div_piecewise_constant" in code


# ── Generic PiecewiseConstant slow tests ──────────────────────────────────


@pytest.mark.slow
def test_generic_sample_three_epoch():
    ne_values, t_boundaries, diversity, ld = _simulate_three_epoch()
    log_ne = np.log(ne_values)
    log_t = np.log(t_boundaries)
    model = PiecewiseConstantDemography(
        diversity=diversity,
        ld=ld,
        mutation_rate=MUTATION_RATE,
        recombination_rate=RECOMBINATION_RATE,
        num_samples=NUM_SAMPLES,
        left_bins=LEFT_BINS,
        right_bins=RIGHT_BINS,
        n_epochs=3,
        parameters=_THREE_EPOCH_PARAMETERS,
        transformed_parameters=_THREE_EPOCH_TRANSFORMED_PARAMETERS,
        prior=(
            f"    log_Ne_values ~ normal([{log_ne[0]:.4f}, {log_ne[1]:.4f}, {log_ne[2]:.4f}]', 0.1);\n"
            f"    log_t_boundaries ~ normal([{log_t[0]:.4f}, {log_t[1]:.4f}]', 0.1);"
        ),
    )
    idata = model.sample(chains=2, iter_warmup=10, iter_sampling=10)
    assert "posterior" in idata.children
    assert "Ne_values" in idata["posterior"].ds
