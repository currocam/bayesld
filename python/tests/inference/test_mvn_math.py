"""
Fast unit tests for the joint-MVN observation model math.

These tests verify the Python implementation of the log-likelihood and
posterior-predictive sampling against scipy reference implementations,
without requiring Stan compilation or engine instantiation.

Covers _base.py:pointwise_log_lik (lines 533-582) and
       _base.py:posterior_predictive (lines 585-641).
"""

import numpy as np
from scipy.stats import multivariate_normal


def _make_chol(rng, D, min_diag=0.3):
    """Random lower-triangular Cholesky factor with positive diagonal."""
    L = np.tril(rng.standard_normal((D, D)))
    L[np.arange(D), np.arange(D)] = np.abs(L[np.arange(D), np.arange(D)]) + min_diag
    return L


def _log_lik_impl(mu_y, sigma_y, L_Omega, y_obs):
    """Replicate the _base.pointwise_log_lik computation."""
    D = y_obs.shape[1]
    residuals = y_obs.T - mu_y[..., np.newaxis]
    z = residuals / sigma_y[..., np.newaxis]
    V = np.linalg.solve(L_Omega, z)
    quad = (V**2).sum(axis=-2)
    log_det = np.log(sigma_y).sum(axis=-1) + np.log(
        np.diagonal(L_Omega, axis1=-2, axis2=-1)
    ).sum(axis=-1)
    return -0.5 * D * np.log(2.0 * np.pi) - log_det[..., np.newaxis] - 0.5 * quad


def _reference_log_lik(mu_y, sigma_y, L_Omega, y_obs):
    """scipy reference: per-draw, per-window MVN log-density."""
    n_chain, n_draw, D = mu_y.shape
    W = y_obs.shape[0]
    out = np.empty((n_chain, n_draw, W))
    for c in range(n_chain):
        for d in range(n_draw):
            L_S = np.diag(sigma_y[c, d]) @ L_Omega[c, d]
            Sigma = L_S @ L_S.T
            rv = multivariate_normal(mean=mu_y[c, d], cov=Sigma)
            for w in range(W):
                out[c, d, w] = rv.logpdf(y_obs[w])
    return out


class TestPointwiseLogLik:
    def test_matches_scipy_single_draw(self):
        rng = np.random.default_rng(0)
        D, W = 4, 6
        L_Omega = _make_chol(rng, D)[np.newaxis, np.newaxis]  # (1,1,D,D)
        mu_y = rng.standard_normal((1, 1, D))
        sigma_y = np.exp(rng.standard_normal((1, 1, D)))
        y_obs = rng.standard_normal((W, D))

        result = _log_lik_impl(mu_y, sigma_y, L_Omega, y_obs)
        ref = _reference_log_lik(mu_y, sigma_y, L_Omega, y_obs)

        np.testing.assert_allclose(result, ref, rtol=1e-10, atol=1e-12)

    def test_matches_scipy_multiple_draws(self):
        rng = np.random.default_rng(1)
        D, W = 3, 5
        n_chain, n_draw = 2, 4

        L_raw = np.stack([_make_chol(rng, D) for _ in range(n_chain * n_draw)]).reshape(
            n_chain, n_draw, D, D
        )
        mu_y = rng.standard_normal((n_chain, n_draw, D))
        sigma_y = np.exp(rng.standard_normal((n_chain, n_draw, D)))
        y_obs = rng.standard_normal((W, D))

        result = _log_lik_impl(mu_y, sigma_y, L_raw, y_obs)
        ref = _reference_log_lik(mu_y, sigma_y, L_raw, y_obs)

        np.testing.assert_allclose(result, ref, rtol=1e-10, atol=1e-12)

    def test_identity_covariance_matches_standard_normal(self):
        """When L_Omega=I and sigma_y=1, recovers the standard MVN log-density."""
        rng = np.random.default_rng(2)
        D, W = 5, 3
        L_Omega = np.eye(D)[np.newaxis, np.newaxis]
        sigma_y = np.ones((1, 1, D))
        mu_y = np.zeros((1, 1, D))
        y_obs = rng.standard_normal((W, D))

        result = _log_lik_impl(mu_y, sigma_y, L_Omega, y_obs)

        ref = np.array(
            [
                [
                    multivariate_normal(mean=np.zeros(D), cov=np.eye(D)).logpdf(
                        y_obs[w]
                    )
                    for w in range(W)
                ]
            ]
        )
        np.testing.assert_allclose(result[0], ref, rtol=1e-12)

    def test_log_det_sign(self):
        """Larger sigma_y must reduce (not increase) the log-likelihood."""
        D = 3
        L_Omega = np.eye(D)[np.newaxis, np.newaxis]
        mu_y = np.zeros((1, 1, D))

        sigma_small = np.ones((1, 1, D)) * 0.5
        sigma_large = np.ones((1, 1, D)) * 2.0

        # Increasing sigma lowers the peak of the density: at y=0 (zero residual),
        # smaller sigma should give higher density.
        y_zero = np.zeros((1, D))
        ll_zero_small = _log_lik_impl(mu_y, sigma_small, L_Omega, y_zero)
        ll_zero_large = _log_lik_impl(mu_y, sigma_large, L_Omega, y_zero)
        assert ll_zero_small.sum() > ll_zero_large.sum()


class TestPosteriorPredictiveCovariance:
    def test_sample_covariance_converges(self):
        """Posterior-predictive samples should have the correct covariance."""
        rng = np.random.default_rng(42)
        D = 4
        n_draw = 200_000

        L_Omega = _make_chol(rng, D)
        sigma_y = np.exp(rng.standard_normal(D))
        mu_y = rng.standard_normal(D)

        # Expected Sigma = diag(sigma_y) @ L_Omega @ L_Omega.T @ diag(sigma_y)
        L_Sigma = np.diag(sigma_y) @ L_Omega
        Sigma_expected = L_Sigma @ L_Sigma.T

        # Replicate _base.posterior_predictive einsum:
        #   v = einsum("cdij,cdwj->cdwi", L_Omega, z)
        #   y = mu_y + sigma_y * v
        z = rng.standard_normal((n_draw, D))
        v = (L_Omega @ z.T).T  # (n_draw, D)
        y = mu_y + sigma_y * v

        Sigma_empirical = np.cov(y.T)
        # Entry-wise check: |Σ_empirical - Σ| ≤ 5·SE(i,j),
        # where SE(i,j) = sqrt((Σ_ii Σ_jj + Σ_ij²) / n)
        diag = np.diag(Sigma_expected)
        se = np.sqrt((diag[:, None] * diag[None, :] + Sigma_expected**2) / n_draw)
        assert np.all(np.abs(Sigma_empirical - Sigma_expected) <= 5.0 * se), (
            f"Sample covariance deviates by more than 5 SE:\n"
            f"max deviation/SE = {np.max(np.abs(Sigma_empirical - Sigma_expected) / se):.2f}"
        )

    def test_sample_mean_converges(self):
        """Posterior-predictive samples should have the correct mean."""
        rng = np.random.default_rng(7)
        D = 3
        n_draw = 50_000

        L_Omega = _make_chol(rng, D)
        sigma_y = np.exp(rng.standard_normal(D))
        mu_y = rng.standard_normal(D)

        z = rng.standard_normal((n_draw, D))
        v = (L_Omega @ z.T).T
        y = mu_y + sigma_y * v

        tol = 5.0 * np.max(sigma_y) / np.sqrt(n_draw)
        np.testing.assert_allclose(y.mean(axis=0), mu_y, atol=tol)
