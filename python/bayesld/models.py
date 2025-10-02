import pytensor.tensor as pt
import numpy as np


# Helper function to re-scale Legendre Gaussian quadrature rules
def gauss(a, b, n=10):
    x, w = np.polynomial.legendre.leggauss(n)
    w = (b - a) / 2 * w
    x = (b - a) / 2 * x + (a + b) / 2
    return x, w


# Correction for finite sample size
def correct_ld(mu, sample_size):
    """
    Apply finite sample size correction proposed by Fournier et al (2023).

    Args:
        mu (float): Computed mean of E[X_iX_jY_iY_j].
        sample_size (int): Number of diploid individuals.

    Returns:
        float: Corrected E[X_iX_jY_iY_j] value.
    """
    S = 2 * sample_size
    beta = 1 / (S - 1) ** 2
    alpha = ((S**2 - S + 2) ** 2) / ((S**2 - 3 * S + 2) ** 2)
    return (alpha - beta) * mu + 4 * beta


def expected_ld_constant(Ne, u_i, u_j, sample_size=None):
    """
    Compute E[X_iX_jY_iY_j] for a constant Ne demography for pairs of SNPs binned across distances that fall within a specified range.

    Args:
        Ne (float): Diploid effective population size.
        u_i (pm.Array): Left distance bin for pairs of SNPs.
        u_j (pm.Array): Right distance bin for pairs of SNPs.


    Returns:
        pm.Array: Expected LD constant for the given parameters.
    """
    u_i = pt.as_tensor_variable(u_i)
    u_j = pt.as_tensor_variable(u_j)
    # Expected LD constant for haploid data
    mu = (-pt.log(4 * Ne * u_i + 1) + pt.log(4 * Ne * u_j + 1)) / (4 * Ne * (u_j - u_i))

    if sample_size is not None:
        return correct_ld(mu, sample_size)
    return mu


def expected_ld_exponential(
    Ne_c,
    Ne_a,
    t0,
    alpha,
    u_i,
    u_j,
    sample_size=None,
    granularity_times=15,
    granularity_bins=10,
):
    """
    Compute E[X_iX_jY_iY_j] for a two piece exponential Ne demography for pairs of SNPs binned across distances that fall within a specified range.

    Args:
        Ne_c (float): Contemporary diploid effective population size.
        Ne_a (float): Ancestral effective population size.
        t0 (float): Time of transition from exponential to constant phase.
        alpha (float): Rate of change of Ne during exponential phase.
        u_i (pm.Array): Left distance bin for pairs of SNPs.
        u_j (pm.Array): Right distance bin for pairs of SNPs.


    Returns:
        pm.Array: Expected LD for the given parameters.
    """

    # From 0 to t0
    def S_ut_piece1(alpha, Ne1, t, u):
        t = pt.as_tensor_variable(t)[:, None]  # Shape (n_quad, 1)
        u = pt.as_tensor_variable(u)[None, :]  # Shape (1, n_points)
        # If alpha is not close to zero
        inner1 = (1 - pt.exp(alpha * t)) / (2 * Ne1 * alpha)
        exponent1 = alpha * t - 2 * t * u + inner1
        res1 = pt.exp(exponent1) / (2 * Ne1)  # Shape (n_quad, n_points)
        # If alpha is close to zero we use Taylor series
        numerator = 4 * Ne1 + alpha * t * (4 * Ne1 - t)
        exponent2 = -t * (4 * Ne1 * u + 1) / (2 * Ne1)
        res2 = numerator * pt.exp(exponent2) / (8 * Ne1**2)
        epsilon = 1e-5
        return pt.switch(pt.abs(alpha) < epsilon, res2, res1)

    # From t0 to infinity
    def S_ut_piece2(alpha, Ne1, Ne2, t0, t, u):
        t = pt.as_tensor_variable(t)[:, None]  # Shape (n_quad, 1)
        u = pt.as_tensor_variable(u)[None, :]  # Shape (1, n_points)
        # If alpha is not close to zero
        inner1 = (Ne1 * alpha * (t0 - t) + Ne2 * (1 - pt.exp(alpha * t0))) / (
            2 * Ne1 * Ne2 * alpha
        )
        exponent1 = -2 * t * u + inner1
        res1 = pt.exp(exponent1) / (2 * Ne2)  # Shape (n_quad, n_points)
        # If alpha is close to zero we use Taylor series
        inner2 = 4 * Ne1 - alpha * t0**2
        exponent2 = (-4 * Ne1 * Ne2 * t * u + Ne1 * (t0 - t) - Ne2 * t0) / (
            2 * Ne1 * Ne2
        )
        res2 = inner2 * pt.exp(exponent2) / (8 * Ne1 * Ne2)
        epsilon = 1e-5
        return pt.switch(pt.abs(alpha) < epsilon, res2, res1)

    # Numerical integration across both time (0->Inf) and bin
    # Per timepoint points and weights
    legendre_x, legendre_w = np.polynomial.legendre.leggauss(granularity_times)
    u_points = np.array([gauss(a, b, granularity_bins)[0] for (a, b) in zip(u_i, u_j)])
    u_weights = np.array(
        [gauss(a, b, granularity_bins)[1] / (b - a) for (a, b) in zip(u_i, u_j)]
    )
    u_col = pt.as_tensor_variable(u_points.flatten())
    # First integral: [0, t0]
    times1 = (t0 - 0) / 2 * legendre_x + (t0 + 0) / 2
    f_t_piece1 = S_ut_piece1(alpha, Ne_c, times1, u_col)  # (n_quad, n_points)
    integral_piece1 = pt.sum(
        f_t_piece1 * legendre_w[:, None] * (t0 - 0) / 2, axis=0
    )  # (n_points,)

    # Second integral: [t0, ∞)
    trans_legendre_x = 0.5 * legendre_x + 0.5
    trans_legendre_w = 0.5 * legendre_w
    times2 = t0 + trans_legendre_x / (1 - trans_legendre_x)
    f_t_piece2 = S_ut_piece2(alpha, Ne_c, Ne_a, t0, times2, u_col)
    integral_piece2 = pt.sum(
        f_t_piece2 * (trans_legendre_w[:, None] / (1 - trans_legendre_x)[:, None] ** 2),
        axis=0,
    )  # (n_points,)

    res_flat = integral_piece1 + integral_piece2  # shape (n_points,)
    res_matrix = res_flat.reshape(u_points.shape)
    res_per_bin = pt.sum(res_matrix * u_weights, axis=1)  # shape (n_bins,)
    if sample_size is not None:
        return correct_ld(res_per_bin, sample_size)  # assume vectorized version
    return res_per_bin
