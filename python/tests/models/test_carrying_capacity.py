"""
Tests for bayesld.models.ExponentialCarryingCapacityDemography.

Strategy: draw Ne_a, log_fold_change, t0, t1 from priors, compute Ne_c and
alpha, compute noiseless expected diversity and LD via the deterministic
module, add small relative noise, and verify the Stan model compiles, runs,
and returns expected types.
"""

import cmdstanpy
import jax
import numpy as np
import pytest

import bayesld
from bayesld import deterministic
from bayesld.models import ExponentialCarryingCapacityDemography

jax.config.update("jax_enable_x64", True)

MUTATION_RATE = 1e-8
RECOMBINATION_RATE = 1e-8
NUM_SAMPLES = 50
NUM_WINDOWS = 200
NOISE_REL_SD = 0.01

LEFT_BINS, RIGHT_BINS = bayesld.linear_bins()

LOG_NE_MU = np.log(500.0)
LOG_NE_SIGMA = 1.0


def _simulate(seed=42):
    rng = np.random.default_rng(seed)
    ne_a = float(np.exp(rng.normal(LOG_NE_MU, LOG_NE_SIGMA)))
    log_fold_change = float(rng.normal(0, 1.0))
    ne_c = ne_a * np.exp(log_fold_change)
    t0 = float(np.exp(rng.normal(np.log(100), 0.5)))
    t1 = float(np.exp(rng.normal(np.log(200), 0.5)))
    if t1 <= t0:
        t1 = t0 + 50.0
    alpha = log_fold_change / (t1 - t0)

    pi_exp, ld_exp = deterministic.expected_exponential_carrying_capacity(
        ne_c,
        ne_a,
        t0,
        t1,
        alpha,
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
    assert np.all(np.isfinite(diversity)), "diversity contains non-finite values"
    assert np.all(np.isfinite(ld)), "ld contains non-finite values"
    assert np.all(np.std(ld, axis=0) > 0), "ld has zero-variance bins"
    assert np.std(diversity) > 0, "diversity has zero variance"
    return ne_c, ne_a, t0, t1, alpha, log_fold_change, diversity, ld


def _make_model(**kwargs):
    _, _, _, _, _, _, diversity, ld = _simulate()
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
    return ExponentialCarryingCapacityDemography(**defaults)


# ── Non-slow tests ──────────────────────────────────────────────────────


def test_primitives():
    model = _make_model()
    code = model.get_stan_code()
    assert isinstance(code, str)
    assert "parameters" in code
    assert "n_synthetic" in code
    assert isinstance(model.model, cmdstanpy.CmdStanModel)

    data = model.stan_data()
    assert isinstance(data, dict)
    assert data["n_synthetic"] == 0
    assert data["eval_rel_bias"].shape == (0, len(LEFT_BINS))


def test_stan_data_keys():
    model = _make_model()
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


def test_update_data():
    model = _make_model()
    old_model_obj = model.model
    _, _, _, _, _, _, new_div, new_ld = _simulate(seed=99)
    model.update_data(diversity=new_div, ld=new_ld)
    assert model.model is old_model_obj
    data = model.stan_data()
    np.testing.assert_array_equal(data["pi_array"], new_div)


def test_update_prior_recompiles():
    model = _make_model()
    old_model_obj = model.model
    model.update_prior(
        prior="    log_Ne_a ~ normal(5, 2.0);\n    log_t_boundaries[1] ~ normal(4, 1);\n    log_t_boundaries[2] ~ normal(5, 1);\n    log_fold_change ~ normal(0, 1);"
    )
    assert model.model is not old_model_obj


def test_update_prior_gp_alpha_std_no_recompile():
    model = _make_model()
    old_model_obj = model.model
    model.update_prior(gp_alpha_std=0.01)
    assert model.model is old_model_obj
    assert model.stan_data()["gp_alpha_std"] == 0.01


def test_add_synthetic_points():
    model = _make_model()
    n_bins = len(LEFT_BINS)
    points = [
        {"rel_bias": np.zeros(n_bins), "eps_rel": np.ones(n_bins) * 0.01},
    ]
    model.add_synthetic_points(points)
    assert len(model.synthetic_points) == 1
    data = model.stan_data()
    assert data["n_synthetic"] == 1
    assert data["eval_rel_bias"].shape == (1, n_bins)


def test_add_synthetic_points_validation():
    model = _make_model()
    with pytest.raises(ValueError, match="keys"):
        model.add_synthetic_points([{"rel_bias": np.zeros(5)}])


def test_generate_stan_default():
    from bayesld.models.carrying_capacity import _generate_stan

    code = _generate_stan()
    assert "log_Ne_a" in code
    assert "log_t_boundaries" in code
    assert "log_fold_change" in code
    assert "log_Ne_c = log_Ne_a + log_fold_change" in code
    assert "n_synthetic" in code
    assert "corrected_ld" in code


def test_generate_stan_custom():
    from bayesld.models.carrying_capacity import _generate_stan

    custom_tp = (
        "    real<lower=0> Ne_c = exp(log_Ne_c);\n"
        "    real<lower=0> Ne_a = exp(log_Ne_a);\n"
        "    real<lower=0> t0   = exp(log_t_boundaries[1]);\n"
        "    real<lower=0> t1   = exp(log_t_boundaries[2]);\n"
        "    real alpha = log_fold_change / (t1 - t0);"
    )
    code = _generate_stan(
        prior="    log_Ne_a ~ normal(10, 0.5);\n    log_t_boundaries[1] ~ normal(4, 0.1);\n    log_t_boundaries[2] ~ normal(5, 0.1);\n    log_fold_change ~ normal(0, 0.5);",
        parameters="    real<offset=log_ne_offset> log_Ne_c;\n    real<offset=log_ne_offset> log_Ne_a;\n    ordered[2] log_t_boundaries;\n    real log_fold_change;",
        transformed_parameters=custom_tp,
    )
    assert "log_Ne_a ~ normal(10, 0.5);" in code
    assert "real<offset=log_ne_offset> log_Ne_c;" in code
    assert "log_Ne_c = log_Ne_a + log_fold_change" not in code


# ── Slow inference tests ────────────────────────────────────────────────


@pytest.mark.slow
def test_sample_default_prior():
    model = _make_model()
    idata = model.sample(chains=2, iter_warmup=10, iter_sampling=10)
    assert "posterior" in idata.children
    assert "Ne_c" in idata["posterior"].ds


@pytest.mark.slow
def test_sample_true_prior():
    ne_c, ne_a, t0, t1, alpha, lfc, diversity, ld = _simulate()
    log_ne_c = np.log(ne_c)
    log_ne_a = np.log(ne_a)
    log_t0 = np.log(t0)
    log_t1 = np.log(t1)
    model = ExponentialCarryingCapacityDemography(
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
            f"    log_t_boundaries[1] ~ normal({log_t0:.4f}, 0.1);\n"
            f"    log_t_boundaries[2] ~ normal({log_t1:.4f}, 0.1);\n"
            f"    log_fold_change ~ normal(0, 1.0);"
        ),
        parameters=(
            "    real<offset=log_ne_offset> log_Ne_c;\n"
            "    real<offset=log_ne_offset> log_Ne_a;\n"
            "    ordered[2] log_t_boundaries;\n"
            "    real log_fold_change;"
        ),
        transformed_parameters=(
            "    real<lower=0> Ne_c = exp(log_Ne_c);\n"
            "    real<lower=0> Ne_a = exp(log_Ne_a);\n"
            "    real<lower=0> t0   = exp(log_t_boundaries[1]);\n"
            "    real<lower=0> t1   = exp(log_t_boundaries[2]);\n"
            "    real alpha = log_fold_change / (t1 - t0);"
        ),
    )
    idata = model.sample(chains=2, iter_warmup=10, iter_sampling=10)
    assert "Ne_c" in idata["posterior"].ds
