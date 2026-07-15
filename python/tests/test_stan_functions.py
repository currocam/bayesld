"""
Tests comparing Stan function implementations against the JAX deterministic approximations.

Known structural differences (documented, not bugs):
  - LD bin integration: Stan uses midpoint evaluation; Python uses Gauss-Legendre quadrature
    averaged over each bin. They agree when bins are narrow.
  - alpha=0 branch: Python has a Taylor-expansion fallback for |alpha| < 1e-5; Stan does not.
    All tests use alpha > 1e-4 to stay away from this singularity.

Test strategy:
  - All assertions use RTOL = 0.1% tolerance.
  - Diversity: same closed-form / same quadrature points passed explicitly.
  - LD: midpoint vs GL quadrature difference documented; bins are the standard linear bins.
"""

from pathlib import Path

import bayesld
import cmdstanpy
import jax
import numpy as np
import pytest
from bayesld import deterministic
from numpy.polynomial.legendre import leggauss
from scipy.stats import norm

jax.config.update("jax_enable_x64", True)

cmdstanpy.utils.get_logger().setLevel("ERROR")

STAN_DIR = Path(bayesld.__file__).resolve().parent / "stan" / "tests"
MUTATION_RATE = 1e-8
SAMPLE_SIZE = 10
RTOL = 1e-3  # 0.1%

# 50-point GL quadrature (same nodes Python uses internally via _LEGENDRE_X_50/_W_50).
GL_X_50, GL_W_50 = leggauss(50)

# Standard bins from bayesld.
LEFT_BINS, RIGHT_BINS = bayesld.linear_bins()

DET_KWARGS = dict(
    mutation_rate=MUTATION_RATE,
    sample_size=SAMPLE_SIZE,
    ploidy=2,
)


# ── helpers ───────────────────────────────────────────────────────────────────


def _run_stan(model, data):
    return model.sample(
        data=data,
        fixed_param=True,
        iter_sampling=1,
        chains=1,
        show_progress=False,
        show_console=False,
        sig_figs=18,
    )


def _div(fit):
    return float(fit.stan_variable("mu_div")[0])


def _ld(fit):
    return np.asarray(fit.stan_variable("mu_ld")[0])


def _lp(fit):
    return float(fit.stan_variable("lp")[0])


# ── session-scoped model fixtures (compile once) ──────────────────────────────


@pytest.fixture(scope="session")
def model_constant():
    return cmdstanpy.CmdStanModel(stan_file=str(STAN_DIR / "test_constant.stan"))


@pytest.fixture(scope="session")
def model_piecewise_constant():
    return cmdstanpy.CmdStanModel(
        stan_file=str(STAN_DIR / "test_piecewise_constant.stan")
    )


@pytest.fixture(scope="session")
def model_piecewise_exponential():
    return cmdstanpy.CmdStanModel(
        stan_file=str(STAN_DIR / "test_piecewise_exponential.stan")
    )


@pytest.fixture(scope="session")
def model_carrying_capacity():
    return cmdstanpy.CmdStanModel(
        stan_file=str(STAN_DIR / "test_carrying_capacity.stan")
    )


# ── Constant ──────────────────────────────────────────────────────────────────


class TestConstant:
    """
    Stan and Python both use the same closed-form bin-integral formula — exact match expected.
    """

    NE = 1000.0

    def _py(self, left_bins, right_bins):
        return deterministic.expected_constant(
            Ne=self.NE, left_bins=left_bins, right_bins=right_bins, **DET_KWARGS
        )

    def _data(self, left_bins, right_bins):
        return dict(
            n_bins=len(left_bins),
            left_bins=left_bins.tolist(),
            right_bins=right_bins.tolist(),
            Ne=self.NE,
            mutation_rate=MUTATION_RATE,
            sample_size=SAMPLE_SIZE,
        )

    def test_diversity(self, model_constant):
        py_div, _ = self._py(LEFT_BINS, RIGHT_BINS)
        stan_div = _div(_run_stan(model_constant, self._data(LEFT_BINS, RIGHT_BINS)))
        np.testing.assert_allclose(float(py_div), stan_div, rtol=RTOL)

    def test_ld(self, model_constant):
        """Closed-form formula — exact match even for wide bins."""
        _, py_ld = self._py(LEFT_BINS, RIGHT_BINS)
        stan_ld = _ld(_run_stan(model_constant, self._data(LEFT_BINS, RIGHT_BINS)))
        np.testing.assert_allclose(np.array(py_ld), stan_ld, rtol=RTOL)


# ── Piecewise Constant ────────────────────────────────────────────────────────

_PC_RNG = np.random.default_rng(42)
_PC_PARAMS = [
    pytest.param(
        _PC_RNG.uniform(200, 5000, n),
        np.sort(_PC_RNG.uniform(20, 400, max(0, n - 1))),
        id=f"{n}_epoch{'s' if n != 1 else ''}",
    )
    for n in [1, 2, 5]
]


@pytest.mark.parametrize("Ne_values,t_boundaries", _PC_PARAMS)
class TestPiecewiseConstant:
    """
    Diversity: same closed-form recurrence — exact match expected.
    LD: both Stan and Python use GL quadrature over bins; Stan uses exact closed-form
        time integral, Python uses GL quadrature in time.
    Tested with 1, 2, and 5 epochs using randomly simulated parameters.
    """

    def _py(self, Ne_values, t_boundaries, left_bins, right_bins):
        return deterministic.expected_piecewise_constant(
            Ne_values=Ne_values,
            t_boundaries=t_boundaries,
            left_bins=left_bins,
            right_bins=right_bins,
            **DET_KWARGS,
        )

    def _data(self, Ne_values, t_boundaries, left_bins, right_bins):
        return dict(
            n_bins=len(left_bins),
            left_bins=left_bins.tolist(),
            right_bins=right_bins.tolist(),
            n_epochs=len(Ne_values),
            Ne_values=Ne_values.tolist(),
            t_boundaries=t_boundaries.tolist(),
            mutation_rate=MUTATION_RATE,
            sample_size=SAMPLE_SIZE,
            n_quad=50,
            gl_nodes=GL_X_50.tolist(),
            gl_weights=GL_W_50.tolist(),
        )

    def test_diversity(self, model_piecewise_constant, Ne_values, t_boundaries):
        py_div, _ = self._py(Ne_values, t_boundaries, LEFT_BINS, RIGHT_BINS)
        stan_div = _div(
            _run_stan(
                model_piecewise_constant,
                self._data(Ne_values, t_boundaries, LEFT_BINS, RIGHT_BINS),
            )
        )
        np.testing.assert_allclose(float(py_div), stan_div, rtol=RTOL)

    def test_ld(self, model_piecewise_constant, Ne_values, t_boundaries):
        _, py_ld = self._py(Ne_values, t_boundaries, LEFT_BINS, RIGHT_BINS)
        stan_ld = _ld(
            _run_stan(
                model_piecewise_constant,
                self._data(Ne_values, t_boundaries, LEFT_BINS, RIGHT_BINS),
            )
        )
        np.testing.assert_allclose(np.array(py_ld), stan_ld, rtol=RTOL)


# ── Piecewise Exponential ─────────────────────────────────────────────────────

_PE_RNG = np.random.default_rng(7)
_PE_PARAMS = [
    pytest.param(
        float(_PE_RNG.uniform(200, 5000)),  # Ne_c
        float(_PE_RNG.uniform(200, 5000)),  # Ne_a
        float(_PE_RNG.uniform(20, 400)),  # t0
        float(_PE_RNG.uniform(1e-4, 0.05)),  # alpha (> 1e-4 to avoid Taylor branch)
        id=f"params_{i}",
    )
    for i in range(5)
] + [
    pytest.param(500.0, 2000.0, 50.0, 0.0, id="alpha_zero"),
    pytest.param(500.0, 2000.0, 50.0, 1e-10, id="alpha_1e-10"),
    pytest.param(500.0, 2000.0, 50.0, 1e-7, id="alpha_1e-7"),
]


@pytest.mark.parametrize("Ne_c,Ne_a,t0,alpha", _PE_PARAMS)
class TestPiecewiseExponential:
    """
    Diversity: same GL quadrature with identical 50-pt nodes passed to Stan — exact match.
    LD: both Stan and Python use GL quadrature over bins — exact match expected.
    alpha > 1e-4 to avoid Stan's missing Taylor branch.
    Tested with 5 randomly sampled parameter sets.
    """

    def _py(self, Ne_c, Ne_a, t0, alpha, left_bins, right_bins):
        return deterministic.expected_piecewise_exponential(
            Ne_c=Ne_c,
            Ne_a=Ne_a,
            t0=t0,
            alpha=alpha,
            left_bins=left_bins,
            right_bins=right_bins,
            **DET_KWARGS,
        )

    def _data(self, Ne_c, Ne_a, t0, alpha, left_bins, right_bins):
        return dict(
            n_bins=len(left_bins),
            left_bins=left_bins.tolist(),
            right_bins=right_bins.tolist(),
            Ne_c=Ne_c,
            Ne_a=Ne_a,
            t0=t0,
            alpha=alpha,
            mutation_rate=MUTATION_RATE,
            sample_size=SAMPLE_SIZE,
            n_quad=50,
            gl_nodes=GL_X_50.tolist(),
            gl_weights=GL_W_50.tolist(),
        )

    def test_diversity(self, model_piecewise_exponential, Ne_c, Ne_a, t0, alpha):
        py_div, _ = self._py(Ne_c, Ne_a, t0, alpha, LEFT_BINS, RIGHT_BINS)
        stan_div = _div(
            _run_stan(
                model_piecewise_exponential,
                self._data(Ne_c, Ne_a, t0, alpha, LEFT_BINS, RIGHT_BINS),
            )
        )
        np.testing.assert_allclose(float(py_div), stan_div, rtol=RTOL)

    def test_ld(self, model_piecewise_exponential, Ne_c, Ne_a, t0, alpha):
        _, py_ld = self._py(Ne_c, Ne_a, t0, alpha, LEFT_BINS, RIGHT_BINS)
        stan_ld = _ld(
            _run_stan(
                model_piecewise_exponential,
                self._data(Ne_c, Ne_a, t0, alpha, LEFT_BINS, RIGHT_BINS),
            )
        )
        np.testing.assert_allclose(np.array(py_ld), stan_ld, rtol=RTOL)


# ── Carrying Capacity ─────────────────────────────────────────────────────────

_CC_RNG = np.random.default_rng(13)
_CC_PARAMS = [
    pytest.param(
        float(_CC_RNG.uniform(200, 5000)),  # Ne_c
        float(_CC_RNG.uniform(200, 5000)),  # Ne_a
        float(_CC_RNG.uniform(10, 100)),  # t0
        float(_CC_RNG.uniform(10, 200)),  # dt (t1 = t0 + dt, always > t0)
        float(_CC_RNG.uniform(1e-4, 0.05)),  # alpha
        id=f"params_{i}",
    )
    for i in range(5)
] + [
    pytest.param(500.0, 2000.0, 20.0, 40.0, 0.0, id="alpha_zero"),
    pytest.param(500.0, 2000.0, 20.0, 40.0, 1e-10, id="alpha_1e-10"),
    pytest.param(500.0, 2000.0, 20.0, 40.0, 1e-7, id="alpha_1e-7"),
]


@pytest.mark.parametrize("Ne_c,Ne_a,t0,dt,alpha", _CC_PARAMS)
class TestCarryingCapacity:
    """
    Diversity: same GL quadrature with identical 50-pt nodes — exact match.
    LD: both Stan and Python use GL quadrature over bins — exact match.
    alpha=0 and near-zero cases covered; expm1_over_x removes the singularity.
    Tested with 5 randomly sampled parameter sets and 3 alpha edge cases.
    """

    def _py(self, Ne_c, Ne_a, t0, dt, alpha, left_bins, right_bins):
        return deterministic.expected_exponential_carrying_capacity(
            Ne_c=Ne_c,
            Ne_a=Ne_a,
            t0=t0,
            t1=t0 + dt,
            alpha=alpha,
            left_bins=left_bins,
            right_bins=right_bins,
            **DET_KWARGS,
        )

    def _data(self, Ne_c, Ne_a, t0, dt, alpha, left_bins, right_bins):
        return dict(
            n_bins=len(left_bins),
            left_bins=left_bins.tolist(),
            right_bins=right_bins.tolist(),
            Ne_c=Ne_c,
            Ne_a=Ne_a,
            t0=t0,
            t1=t0 + dt,
            alpha=alpha,
            mutation_rate=MUTATION_RATE,
            sample_size=SAMPLE_SIZE,
            n_quad=50,
            gl_nodes=GL_X_50.tolist(),
            gl_weights=GL_W_50.tolist(),
        )

    def test_diversity(self, model_carrying_capacity, Ne_c, Ne_a, t0, dt, alpha):
        py_div, _ = self._py(Ne_c, Ne_a, t0, dt, alpha, LEFT_BINS, RIGHT_BINS)
        stan_div = _div(
            _run_stan(
                model_carrying_capacity,
                self._data(Ne_c, Ne_a, t0, dt, alpha, LEFT_BINS, RIGHT_BINS),
            )
        )
        np.testing.assert_allclose(float(py_div), stan_div, rtol=RTOL)

    def test_ld(self, model_carrying_capacity, Ne_c, Ne_a, t0, dt, alpha):
        _, py_ld = self._py(Ne_c, Ne_a, t0, dt, alpha, LEFT_BINS, RIGHT_BINS)
        stan_ld = _ld(
            _run_stan(
                model_carrying_capacity,
                self._data(Ne_c, Ne_a, t0, dt, alpha, LEFT_BINS, RIGHT_BINS),
            )
        )
        np.testing.assert_allclose(np.array(py_ld), stan_ld, rtol=RTOL)


# ── Likelihood functions ───────────────────────────────────────────────────────
# OBS_LD: fixed random observations used across all lp tests.
# mu_ld for each model is obtained from the existing mu_ld Stan models so that
# the Python reference lp = sum_b Normal(obs | mu, SIGMA_LD) can be compared
# directly to the Stan ld_lp_* output.

_LP_RNG = np.random.default_rng(99)
OBS_LD = _LP_RNG.uniform(0.01, 0.1, size=len(LEFT_BINS))
SIGMA_LD = 0.001
SIGMA_LD_ARR = np.full(len(LEFT_BINS), SIGMA_LD)


def _lp_ref(obs_ld, mu_ld):
    """sum_b Normal(obs_ld[b] | mu_ld[b], SIGMA_LD) log-pdf."""
    return float(norm.logpdf(obs_ld, mu_ld, SIGMA_LD).sum())


# ── LP fixtures ───────────────────────────────────────────────────────────────


_THREADS_OPTS = {"cpp_options": {"STAN_THREADS": "true"}}


@pytest.fixture(scope="session")
def model_lp_piecewise_constant():
    return cmdstanpy.CmdStanModel(
        stan_file=str(STAN_DIR / "test_lp_piecewise_constant.stan"),
        **_THREADS_OPTS,
    )


@pytest.fixture(scope="session")
def model_lp_piecewise_exponential():
    return cmdstanpy.CmdStanModel(
        stan_file=str(STAN_DIR / "test_lp_piecewise_exponential.stan"),
        **_THREADS_OPTS,
    )


@pytest.fixture(scope="session")
def model_lp_carrying_capacity():
    return cmdstanpy.CmdStanModel(
        stan_file=str(STAN_DIR / "test_lp_carrying_capacity.stan"),
        **_THREADS_OPTS,
    )


# ── TestLpPiecewiseConstant ───────────────────────────────────────────────────


class TestLpPiecewiseConstant:
    """
    ld_lp_piecewise_constant must equal sum_b Normal(obs | mu_ld[b], sigma).
    mu_ld is obtained from the mu_ld Stan model at the same parameters.
    """

    NE_VALUES = np.array([1000.0, 3000.0])
    T_BOUNDARIES = np.array([100.0])

    def _mu_data(self):
        return dict(
            n_bins=len(LEFT_BINS),
            left_bins=LEFT_BINS.tolist(),
            right_bins=RIGHT_BINS.tolist(),
            n_epochs=len(self.NE_VALUES),
            Ne_values=self.NE_VALUES.tolist(),
            t_boundaries=self.T_BOUNDARIES.tolist(),
            mutation_rate=MUTATION_RATE,
            sample_size=SAMPLE_SIZE,
            n_quad=50,
            gl_nodes=GL_X_50.tolist(),
            gl_weights=GL_W_50.tolist(),
        )

    def _lp_data(self):
        return dict(
            n_bins=len(LEFT_BINS),
            left_bins=LEFT_BINS.tolist(),
            right_bins=RIGHT_BINS.tolist(),
            obs_ld=OBS_LD.tolist(),
            est_sigma_ld=SIGMA_LD_ARR.tolist(),
            n_epochs=len(self.NE_VALUES),
            Ne_values=self.NE_VALUES.tolist(),
            t_boundaries=self.T_BOUNDARIES.tolist(),
            sample_size=SAMPLE_SIZE,
            n_quad=50,
            gl_nodes=GL_X_50.tolist(),
            gl_weights=GL_W_50.tolist(),
        )

    def test_lp(self, model_piecewise_constant, model_lp_piecewise_constant):
        mu_ld = _ld(_run_stan(model_piecewise_constant, self._mu_data()))
        py_lp = _lp_ref(OBS_LD, mu_ld)
        stan_lp = _lp(_run_stan(model_lp_piecewise_constant, self._lp_data()))
        np.testing.assert_allclose(py_lp, stan_lp, rtol=RTOL)


# ── TestLpPiecewiseExponential ────────────────────────────────────────────────


class TestLpPiecewiseExponential:
    """
    ld_lp_piecewise_exponential must equal sum_b Normal(obs | mu_ld[b], sigma).
    """

    NE_C = 1000.0
    NE_A = 3000.0
    T0 = 100.0
    ALPHA = 0.01

    def _mu_data(self):
        return dict(
            n_bins=len(LEFT_BINS),
            left_bins=LEFT_BINS.tolist(),
            right_bins=RIGHT_BINS.tolist(),
            Ne_c=self.NE_C,
            Ne_a=self.NE_A,
            t0=self.T0,
            alpha=self.ALPHA,
            mutation_rate=MUTATION_RATE,
            sample_size=SAMPLE_SIZE,
            n_quad=50,
            gl_nodes=GL_X_50.tolist(),
            gl_weights=GL_W_50.tolist(),
        )

    def _lp_data(self):
        return dict(
            n_bins=len(LEFT_BINS),
            left_bins=LEFT_BINS.tolist(),
            right_bins=RIGHT_BINS.tolist(),
            obs_ld=OBS_LD.tolist(),
            est_sigma_ld=SIGMA_LD_ARR.tolist(),
            Ne_c=self.NE_C,
            Ne_a=self.NE_A,
            t0=self.T0,
            alpha=self.ALPHA,
            sample_size=SAMPLE_SIZE,
            n_quad=50,
            gl_nodes=GL_X_50.tolist(),
            gl_weights=GL_W_50.tolist(),
        )

    def test_lp(self, model_piecewise_exponential, model_lp_piecewise_exponential):
        mu_ld = _ld(_run_stan(model_piecewise_exponential, self._mu_data()))
        py_lp = _lp_ref(OBS_LD, mu_ld)
        stan_lp = _lp(_run_stan(model_lp_piecewise_exponential, self._lp_data()))
        np.testing.assert_allclose(py_lp, stan_lp, rtol=RTOL)


# ── TestLpCarryingCapacity ────────────────────────────────────────────────────


class TestLpCarryingCapacity:
    """
    ld_lp_carrying_capacity must equal sum_b Normal(obs | mu_ld[b], sigma).
    """

    NE_C = 1000.0
    NE_A = 3000.0
    T0 = 50.0
    T1 = 150.0
    ALPHA = 0.01

    def _mu_data(self):
        return dict(
            n_bins=len(LEFT_BINS),
            left_bins=LEFT_BINS.tolist(),
            right_bins=RIGHT_BINS.tolist(),
            Ne_c=self.NE_C,
            Ne_a=self.NE_A,
            t0=self.T0,
            t1=self.T1,
            alpha=self.ALPHA,
            mutation_rate=MUTATION_RATE,
            sample_size=SAMPLE_SIZE,
            n_quad=50,
            gl_nodes=GL_X_50.tolist(),
            gl_weights=GL_W_50.tolist(),
        )

    def _lp_data(self):
        return dict(
            n_bins=len(LEFT_BINS),
            left_bins=LEFT_BINS.tolist(),
            right_bins=RIGHT_BINS.tolist(),
            obs_ld=OBS_LD.tolist(),
            est_sigma_ld=SIGMA_LD_ARR.tolist(),
            Ne_c=self.NE_C,
            Ne_a=self.NE_A,
            t0=self.T0,
            t1=self.T1,
            alpha=self.ALPHA,
            sample_size=SAMPLE_SIZE,
            n_quad=50,
            gl_nodes=GL_X_50.tolist(),
            gl_weights=GL_W_50.tolist(),
        )

    def test_lp(self, model_carrying_capacity, model_lp_carrying_capacity):
        mu_ld = _ld(_run_stan(model_carrying_capacity, self._mu_data()))
        py_lp = _lp_ref(OBS_LD, mu_ld)
        stan_lp = _lp(_run_stan(model_lp_carrying_capacity, self._lp_data()))
        np.testing.assert_allclose(py_lp, stan_lp, rtol=RTOL)
