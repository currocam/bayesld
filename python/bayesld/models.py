import pytensor.tensor as pt
import numpy as np


# Helper function to re-scale Legendre Gaussian quadrature rules
def gauss(a, b, n=10):
    """
    Compute nodes and weights for Gaussian quadrature over [a, b].

    Args:
        a (float): Lower bound of the integration interval.
        b (float): Upper bound of the integration interval.
        n (int, optional): Number of quadrature points. Defaults to 10.

    Returns:
        tuple: Tuple of arrays (nodes, weights) for Gaussian quadrature.
    """
    x, w = np.polynomial.legendre.leggauss(n)
    w = (b - a) / 2 * w
    x = (b - a) / 2 * x + (a + b) / 2
    return x, w


# Correction for finite sample size
def correct_ld(mu, sample_size):
    """
    Apply finite sample size correction proposed by Fournier et al. (2023).

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
    Compute expected LD (E[X_iX_jY_iY_j]) under a constant Ne demography.

    Args:
        Ne (float): Diploid effective population size.
        u_i (array-like): Left distances for SNP pairs.
        u_j (array-like): Right distances for SNP pairs.
        sample_size (int, optional): Number of diploid individuals. If provided, applies finite sample correction.

    Returns:
        pt.TensorVariable: Expected LD values, corrected if sample_size is provided.
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
    Compute expected LD (E[X_iX_jY_iY_j]) under a two-phase exponential demography.

    Args:
        Ne_c (float): Contemporary diploid effective population size.
        Ne_a (float): Ancestral diploid effective population size.
        t0 (float): Time of transition from exponential to constant phase.
        alpha (float): Rate of change of Ne during the exponential phase.
        u_i (array-like): Left distances for SNP pairs.
        u_j (array-like): Right distances for SNP pairs.
        sample_size (int, optional): Number of diploid individuals. If provided, applies finite sample correction.
        granularity_times (int, optional): Number of quadrature points for time integration. Defaults to 15.
        granularity_bins (int, optional): Number of quadrature points for distance bins. Defaults to 10.

    Returns:
        pt.TensorVariable: Expected LD values across SNP distance bins.
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


def expected_ld_piecewise(
    epoch_population_sizes,
    epoch_time_points,
    left,
    right,
    sample_size=None,
    granularity_bins=10,
):
    """
    Compute expected LD (E[X_iX_jY_iY_j]) under an arbitrary piecewise-constant Ne demography.

    Args:
        epoch_population_sizes (list of float): Effective population sizes for each epoch.
        epoch_time_points (list of float): Duration of each epoch (length = len(epoch_population_sizes) - 1).
        left (array-like): Left distances for SNP bins.
        right (array-like): Right distances for SNP bins.
        sample_size (int, optional): Number of diploid individuals. If provided, applies finite sample correction.
        granularity_bins (int, optional): Number of quadrature points per bin. Defaults to 10.

    Returns:
        pt.TensorVariable: Expected LD values for each distance bin.
    """
    n_pieces = len(epoch_population_sizes)
    assert len(epoch_time_points) == n_pieces - 1
    assert n_pieces > 0, "At least one epoch is required."

    # Numerical integration setup
    u_points = []
    w_points = []
    for a, b in zip(left, right):
        x, w = gauss(a, b, granularity_bins)
        u_points.append(x)
        w_points.append(w / (b - a))
    u_points = np.asarray(u_points)
    w_points = np.asarray(w_points)
    u = u_points.flatten()
    w = w_points.flatten()

    # Helper functions for cumulative calculations
    def cumulative_correction(up_to_idx):
        """Calculate cumulative time corrections up to index up_to_idx (exclusive)."""
        corrections = [
            epoch_time_points[i]
            * (
                1 / (2 * epoch_population_sizes[i + 1])
                - 1 / (2 * epoch_population_sizes[i])
            )
            for i in range(up_to_idx)
        ]
        return sum(corrections)

    def time_sum(up_to_idx):
        """Calculate sum of times up to index up_to_idx (exclusive)."""
        return sum(epoch_time_points[:up_to_idx])

    # First term (oldest epoch, i=1)
    t_1 = epoch_time_points[0]
    N_1 = epoch_population_sizes[0]
    exp_term = pt.exp(2 * t_1 * u + t_1 / (2 * N_1))
    first_term = (
        (exp_term - 1) * pt.exp(-2 * t_1 * u - t_1 / (2 * N_1)) / (4 * N_1 * u + 1)
    )

    # Middle terms (epochs i=2 to n-1)
    middle_terms = []
    for i in range(1, n_pieces - 1):
        t_i = epoch_time_points[i - 1]
        t_next = epoch_time_points[i]
        N_i = epoch_population_sizes[i - 1]
        N_next = epoch_population_sizes[i]

        numerator = pt.exp(2 * t_i * u + t_i / (2 * N_i)) - pt.exp(
            2 * t_next * u + t_next / (2 * N_next)
        )

        exponential = pt.exp(
            -2 * time_sum(i + 1) * u - t_next / (2 * N_next) + cumulative_correction(i)
        )

        middle_terms.append(-numerator * exponential / (4 * N_next * u + 1))

    # Last term (newest epoch, i=n)
    N_n = epoch_population_sizes[-1]
    last_term = pt.exp(
        -2 * time_sum(n_pieces - 1) * u + cumulative_correction(n_pieces - 1)
    ) / (4 * N_n * u + 1)

    # Combine all terms
    result = first_term + sum(middle_terms) + last_term

    # Integrate across bins
    res_per_bin = (result * w).reshape(u_points.shape).sum(axis=1)
    if sample_size is not None:
        return correct_ld(res_per_bin, sample_size)  # assume vectorized version
    return res_per_bin
