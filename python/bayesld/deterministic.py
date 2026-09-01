"""
Deterministic approximations for expected LD and genetic diversity.
"""

import jax
import jax.numpy as jnp
import numpy as np

# Pre-compute quadrature rules (numpy for setup, then store as JAX arrays)
_legendre_x_50, _legendre_w_50 = np.polynomial.legendre.leggauss(50)
_LEGENDRE_X_50 = jnp.asarray(_legendre_x_50)
_LEGENDRE_W_50 = jnp.asarray(_legendre_w_50)

# Gauss-Legendre nodes for the per-bin integral over u. `inference._base` imports
# N_QUAD_BINS so the Stan engines default to the same rule as these functions.
N_QUAD_BINS = 10
_legendre_x_bins, _legendre_w_bins = np.polynomial.legendre.leggauss(N_QUAD_BINS)
_LEGENDRE_X_BINS = jnp.asarray(_legendre_x_bins)
_LEGENDRE_W_BINS = jnp.asarray(_legendre_w_bins)


def gauss(a, b, x=_LEGENDRE_X_BINS, w=_LEGENDRE_W_BINS):
    """
    Compute nodes and weights for Gaussian quadrature over [a, b] using Legendre polynomials.

    Args:
        a (float): Lower bound of the integration interval.
        b (float): Upper bound of the integration interval.
        x (array-like): Nodes for Gaussian quadrature (obtained from np.polynomial.legendre.leggauss).
        w (array-like): Weights for Gaussian quadrature (obtained from np.polynomial.legendre.leggauss).

    Returns:
        tuple: Tuple of arrays (nodes, weights) for Gaussian quadrature.
    """
    a = a[..., None]  # Add dimension on the right
    b = b[..., None]
    # Compute transformed weights and nodes
    w_transformed = (b - a) / 2 * w  # broadcasts to (n, len(w))
    x_transformed = (b - a) / 2 * x + (a + b) / 2  # broadcasts to (n, len(x))
    return x_transformed, w_transformed


# There's a singularity at alpha = 0. We do branch here if epsilon is small
ALPHA_EPSILON = 1e-5


def correct_ld_finite_sample(mu, sample_size):
    """
    Adjusts linkage disequilibrium estimates for bias introduced by limited sample sizes using the correction proposed by Fournier et al. (2023).

    Parameters
    ----------
    mu : float or array-like
        Uncorrected mean of E[X_iX_jY_iY_j] (raw LD estimate).
    sample_size : int
        Number of diploid individuals in the sample. Must be positive.

    Returns
    -------
    float or array-like
    """
    # Number of haploid samples
    S = 2 * sample_size

    # Compute correction parameters
    beta = 1 / (S - 1) ** 2
    alpha = ((S**2 - S + 2) ** 2) / ((S**2 - 3 * S + 2) ** 2)

    # Apply correction formula
    return (alpha - beta) * mu + 4 * beta


def expected_constant(
    Ne,
    left_bins,
    right_bins,
    mutation_rate,
    sample_size=None,
    ploidy=2,
):
    """
    Compute expected genetic diversity and LD under a constant population size model using a deterministic approximation.

    Parameters
    ----------
    Ne : float
        Effective population size.
    left_bins : array-like
        Left edges of genomic distance bins in Morgans.
    right_bins : array-like
        Right edges of genomic distance bins in Morgans.
    mutation_rate : float
        Mutation rate per base pair per generation.
    sample_size : int, optional
        Number of individuals sampled. If provided, we correct LD for finite sample size.
    ploidy : int, optional
        Ploidy of the individuals (either 1 or 2). Default is 2.

    Returns
    -------
    expected_pi : float or jax array
    expected_ld : array-like or jax array
    """
    u_i = jnp.asarray(left_bins)
    u_j = jnp.asarray(right_bins)
    Ne = jnp.asarray(Ne) / 2 * ploidy

    # Compute expected LD
    expected_ld = (-jnp.log(4 * Ne * u_i + 1) + jnp.log(4 * Ne * u_j + 1)) / (
        4 * Ne * (u_j - u_i)
    )
    if sample_size is not None and ploidy == 2:
        expected_ld = correct_ld_finite_sample(expected_ld, sample_size)
    # Compute expected genetic diversity (heterozygosity)
    genetic_diversity = 4 * Ne * mutation_rate

    return genetic_diversity, expected_ld


def expected_piecewise_exponential(
    Ne_c,
    Ne_a,
    t0,
    alpha,
    left_bins,
    right_bins,
    mutation_rate,
    sample_size=None,
    ploidy=2,
):
    """
    Compute expected genetic diversity and LD under a two-phase exponential demography using a deterministic approximation.

    Ne(t) = Ne_c * exp(-alpha * t) if t < t0 else Ne_a

    Parameters
    ----------
    Ne_c : float
        Contemporary effective population size.
    Ne_a : float
        Ancestral effective population size.
    t0 : float
        Time of transition from exponential to constant phase.
    alpha : float
        Rate of change of Ne during the exponential phase.
    left_bins : array-like
        Left edges of genomic distance bins in Morgans.
    right_bins : array-like
        Right edges of genomic distance bins in Morgans.
    mutation_rate : float
        Mutation rate per base pair per generation.
    sample_size : int, optional
        Number of individuals sampled. If provided, we correct LD for finite sample size.
    ploidy : int, optional
        Ploidy of the individuals (either 1 or 2). Default is 2.

    Returns
    -------
    expected_pi : float or jax array
    expected_ld : array-like or jax array
    """
    u_i = jnp.asarray(left_bins)
    u_j = jnp.asarray(right_bins)
    Ne_c = jnp.asarray(Ne_c) / 2 * ploidy
    Ne_a = jnp.asarray(Ne_a) / 2 * ploidy
    t0 = jnp.asarray(t0)
    alpha = jnp.asarray(alpha)

    # Compute expected genetic diversity (heterozygosity)
    def compute_diversity():
        w = (t0 - 0) / 2 * _LEGENDRE_W_50
        t = (t0 - 0) / 2 * _LEGENDRE_X_50 + (0 + t0) / 2
        piece1 = jnp.sum(
            w
            * t
            / Ne_c
            * jnp.exp(
                (2 * t * alpha**2 * Ne_c - jnp.exp(t * alpha) + 1) / alpha / Ne_c / 2
            )
            / 2
        )
        piece2 = (2 * Ne_a + t0) * jnp.exp(
            -(-1 + jnp.exp(t0 * alpha)) / alpha / Ne_c / 2
        )
        piece1_taylor = (
            -2 * jnp.exp(-0.1e1 / Ne_c * t0 / 2) * Ne_c
            - jnp.exp(-0.1e1 / Ne_c * t0 / 2) * t0
            + 2 * Ne_c
        )
        piece2_taylor = (2 * Ne_a + t0) * jnp.exp(-1 / Ne_c * t0 / 2)
        e_tmrca_nonzero = piece1 + piece2
        e_tmrca_taylor = piece1_taylor + piece2_taylor
        # If alpha is too close to zero, use Taylor expansion
        e_tmrca = jnp.where(
            jnp.abs(alpha) < ALPHA_EPSILON, e_tmrca_taylor, e_tmrca_nonzero
        )
        return e_tmrca * 2 * mutation_rate

    # Compute expected LD
    def compute_ld():
        def Su_piece1(alpha, Ne_c, t0, u):
            u = u[None, :]
            # If alpha is not close to zero
            t = (t0 - 0) / 2 * _LEGENDRE_X_50 + (t0 + 0) / 2
            t = t[:, None]
            inner1 = jnp.exp(
                (
                    2 * t * alpha**2 * Ne_c
                    - 4 * t * u * Ne_c * alpha
                    - jnp.exp(t * alpha)
                    + 1
                )
                / Ne_c
                / alpha
                / 2
            )
            integral_inner1 = jnp.sum(
                inner1 * _LEGENDRE_W_50[:, None] * (t0 - 0) / 2, axis=0
            )
            expected_nonzero = 1 / Ne_c * integral_inner1 / 2
            # If alpha is close to zero we use Taylor series
            expected_taylor = (-jnp.exp(-t0 * (4 * Ne_c * u + 1) / Ne_c / 2) + 1) / (
                4 * Ne_c * u + 1
            )
            return jnp.where(
                jnp.abs(alpha) < ALPHA_EPSILON, expected_taylor, expected_nonzero
            )

        # There is a closed-form solution for this piece
        def Su_piece2(alpha, Nec, Nea, t0, u):
            # Auto-generated code
            # fmt: off
            expected_nonzero = 1 / (4 * u * Nea + 1) * jnp.exp(-(4 * u * Nec * alpha * t0 + jnp.exp(t0 * alpha) - 1) / Nec / alpha / 2)
            # If alpha is close to zero we use Taylor expansion for alpha=0
            expected_taylor = 1 / (4 * u * Nea + 1) * jnp.exp(-t0 * (4 * Nec * u + 1) / Nec / 2) - 1 / (4 * u * Nea + 1) * jnp.exp(-t0 * (4 * Nec * u + 1) / Nec / 2) * t0 ** 2 / Nec * alpha / 4
            # fmt: on
            return jnp.where(
                jnp.abs(alpha) < ALPHA_EPSILON, expected_taylor, expected_nonzero
            )

        # Numerical integration using pre-computed Legendre quadrature
        u_points, u_weights = gauss(u_i, u_j)
        u_weights = u_weights / (u_j - u_i)[:, None]
        u_col = u_points.flatten()

        # First integral: [0, t0]
        integral_piece1 = Su_piece1(alpha, Ne_c, t0, u_col)
        # Second integral: [t0, ∞)
        integral_piece2 = Su_piece2(alpha, Ne_c, Ne_a, t0, u_col)
        res_flat = integral_piece1 + integral_piece2
        res_matrix = res_flat.reshape(u_points.shape)
        return jnp.sum(res_matrix * u_weights, axis=1)

    expected_ld = compute_ld()
    genetic_diversity = compute_diversity()
    if sample_size is not None and ploidy == 2:
        expected_ld = correct_ld_finite_sample(expected_ld, sample_size)
    return genetic_diversity, expected_ld


def expected_exponential_carrying_capacity(
    Ne_c,
    Ne_a,
    t0,
    t1,
    alpha,
    left_bins,
    right_bins,
    mutation_rate,
    sample_size=None,
    ploidy=2,
):
    """
    Compute expected genetic diversity and LD under a demography where exponential growth stops after reaching a carrying capacity using a deterministic approximation.

    Ne(t) = Ne_c if t < t0, Ne_c* exp(-alpha * (t-t0)) if t < t1 else Ne_a

    Parameters
    ----------
    Ne_c : float
        Contemporary effective population size.
    Ne_a : float
        Ancestral effective population size.
    t0 : float
        Time when exponential growth stops.
    t1 : float
        Time when exponential growth starts.
    alpha : float
        Rate of change of Ne during the exponential phase.
    left_bins : array-like
        Left edges of genomic distance bins in Morgans.
    right_bins : array-like
        Right edges of genomic distance bins in Morgans.
    mutation_rate : float
        Mutation rate per base pair per generation.
    sample_size : int, optional
        Number of individuals sampled. If provided, we correct LD for finite sample size.
    ploidy : int, optional
        Ploidy of the individuals (either 1 or 2). Default is 2.

    Returns
    -------
    expected_pi : float or jax array
    expected_ld : array-like or jax array
    """
    u_i = jnp.asarray(left_bins)
    u_j = jnp.asarray(right_bins)
    Ne_c = jnp.asarray(Ne_c) / 2 * ploidy
    Ne_a = jnp.asarray(Ne_a) / 2 * ploidy
    t0 = jnp.asarray(t0)
    t1 = jnp.asarray(t1)
    alpha = jnp.asarray(alpha)

    # Compute expected genetic diversity (heterozygosity)
    def compute_diversity():
        t = (t1 - t0) / 2 * _LEGENDRE_X_50 + (t0 + t1) / 2
        w = (t1 - t0) / 2 * _LEGENDRE_W_50
        int_piece = jnp.sum(
            w
            * (
                t
                * jnp.exp(
                    (
                        -jnp.exp((t - t0) * alpha)
                        + 1
                        + (2 * t - 2 * t0) * Ne_c * alpha**2
                        - t0 * alpha
                    )
                    / alpha
                    / Ne_c
                    / 2
                )
            )
        )
        expected_tmrca_nonzero = (
            (
                int_piece
                + (2 * t1 + 4 * Ne_a)
                * Ne_c
                * jnp.exp(
                    -(t0 * alpha + jnp.exp(-(t0 - t1) * alpha) - 1) / alpha / Ne_c / 2
                )
                + (-4 * Ne_c**2 - 2 * Ne_c * t0) * jnp.exp(-1 / Ne_c * t0 / 2)
                + 4 * Ne_c**2
            )
            / Ne_c
            / 2
        )
        # Approaching to zero
        expected_tmrca_taylor = (2 * Ne_a - 2 * Ne_c) * jnp.exp(
            -1 / Ne_c * t1 / 2
        ) + 2 * Ne_c
        expected_tmrca = jnp.where(
            jnp.abs(alpha) < ALPHA_EPSILON,
            expected_tmrca_taylor,
            expected_tmrca_nonzero,
        )
        return expected_tmrca * 2 * mutation_rate

    # Compute expected LD
    def compute_ld():
        # Numerical integration using pre-computed Legendre quadrature
        u_points, u_weights = gauss(u_i, u_j)
        u_weights = u_weights / (u_j - u_i)[:, None]
        u = u_points.flatten()
        # Auto-generated code
        # fmt: off
        Nec, Nea = Ne_c, Ne_a
        # Close-form pieces:
        piece1 = (1 - jnp.exp(-t0 * (4 * u * Nec + 1) / Nec / 2)) / (4 * u * Nec + 1)
        # There's a singularity at alpha=0
        piece3_nonzero = jnp.exp((1 - jnp.exp(-(t0 - t1) * alpha) - (4 * Nec * t1 * u + t0) * alpha) / alpha / Nec / 2) / (4 * u * Nea + 1)
        piece3_taylor = jnp.exp(-t1 * (4 * u * Nec + 1) / Nec / 2) / (4 * u * Nea + 1)
        piece3 = jnp.where(jnp.abs(alpha) < ALPHA_EPSILON, piece3_taylor, piece3_nonzero)

        # Numerical integration [t0, t1]
        times2 = (t1 - t0) / 2 * _LEGENDRE_X_50 + (t1 + t0) / 2
        def S_ut_piece2(alpha, Nec, t0, t, u):
            t = t[:, None]
            u = u[None, :]
            res_nonzero = 1 / Nec * jnp.exp((-jnp.exp((t - t0) * alpha) + 1 + (2 * t - 2 * t0) * Nec * alpha ** 2 + (-4 * Nec * t * u - t0) * alpha) / alpha / Nec / 2) / 2
            res_taylor = 1 / Nec * jnp.exp(-t * (4 * u * Nec + 1) / Nec / 2) / 2
            return jnp.where(jnp.abs(alpha) < ALPHA_EPSILON, res_taylor, res_nonzero)
        piece2 = jnp.sum(
            S_ut_piece2(alpha, Ne_c, t0, times2, u) * _LEGENDRE_W_50[:, None] * (t1 - t0) / 2, axis=0
        )
        # fmt: on
        res_flat = piece1 + piece2 + piece3
        res_matrix = res_flat.reshape(u_points.shape)
        return jnp.sum(res_matrix * u_weights, axis=1)

    expected_ld = compute_ld()
    genetic_diversity = compute_diversity()
    if sample_size is not None and ploidy == 2:
        expected_ld = correct_ld_finite_sample(expected_ld, sample_size)
    return genetic_diversity, expected_ld


def expected_piecewise_constant(
    Ne_values,
    t_boundaries,
    left_bins,
    right_bins,
    mutation_rate,
    sample_size=None,
    ploidy=2,
):
    """
    Compute expected genetic diversity and LD under a multi-epoch constant population size model using a deterministic approximation. Notice that this function assumes more than 1 epoch.

    Ne(t) = Ne_values[0] for t < t_boundaries[0]
    Ne(t) = Ne_values[i] for t in [t_boundaries[i], t_boundaries[i+1))
    Ne(t) = Ne_values[-1] for t >= t_boundaries[-1]

    Parameters
    ----------
    Ne_values : (array-like)
        Population sizes for each epoch. Shape: (n_epochs,)
    t_boundaries : (array-like)
        Time boundaries between epochs. Shape: (n_epochs-1,)
    left_bins : array-like
        Left edges of genomic distance bins in Morgans.
    right_bins : array-like
        Right edges of genomic distance bins in Morgans.
    mutation_rate : float
        Mutation rate per base pair per generation.
    sample_size : int, optional
        Number of individuals sampled. If provided, we correct LD for finite sample size.
    ploidy : int, optional
        Ploidy of the individuals (either 1 or 2). Default is 2.

    Returns
    -------
    expected_pi : float or jax array
    expected_ld : array-like or jax array
    """
    u_i = jnp.asarray(left_bins)
    u_j = jnp.asarray(right_bins)
    Ne_values = jnp.asarray(Ne_values) / 2 * ploidy
    t_boundaries = jnp.asarray(t_boundaries)
    # Split into finite epochs and last infinite epoch
    Ne_finite = Ne_values[:-1]
    Ne_last = Ne_values[-1]

    # Compute expected genetic diversity (heterozygosity)
    def compute_diversity():
        def diversity_step(carry, inputs):
            expected_tmrca, Gamma_prev, t_prev = carry
            Ne, t_curr = inputs
            # Interval [t_prev, t_curr]
            integral_finite = jnp.exp(-Gamma_prev) * (
                (-2 * Ne - t_curr) * jnp.exp(-(t_curr - t_prev) / (2 * Ne))
                + 2 * Ne
                + t_prev
            )
            # Update accumulators
            expected_tmrca_new = expected_tmrca + integral_finite
            Gamma_new = Gamma_prev + (t_curr - t_prev) / (2 * Ne)
            return (expected_tmrca_new, Gamma_new, t_curr), None

        # Scan over finite epochs
        init_carry = (
            0.0,  # expected_tmrca
            0.0,  # Gamma_prev
            0.0,  # t_prev
        )
        (tmrca_finite, Gamma_finite, t_start_last), _ = jax.lax.scan(
            diversity_step, init_carry, (Ne_finite, t_boundaries)
        )
        # Add contribution from the last infinite epoch [t_start_last, infinity)
        integral_infinite = jnp.exp(-Gamma_finite) * (2 * Ne_last + t_start_last)
        expected_tmrca = tmrca_finite + integral_infinite
        return expected_tmrca * 2 * mutation_rate

    # Compute expected LD. The time integral has an elementary closed form for
    # piecewise-constant Ne (unlike piecewise_exponential/carrying_capacity below,
    # which keep a numerical time quadrature); this mirrors the Stan model's
    # `S_u_piecewise_constant` exactly rather than approximating the same integral
    # a second, different way.
    def compute_ld():
        u_points, u_weights = gauss(u_i, u_j)
        u_weights = u_weights / (u_j - u_i)[:, None]
        u_col = u_points.flatten()

        def log1mexp(x):
            """log(1 - exp(x)) for x <= 0, stable near x == 0."""
            return jnp.where(
                x > -jnp.log(2.0),
                jnp.log(-jnp.expm1(x)),
                jnp.log1p(-jnp.exp(x)),
            )

        def survival_step(carry, inputs):
            Gamma_prev, t_prev = carry
            Ne, t_curr = inputs
            dt = t_curr - t_prev
            c = 2.0 * u_col + 1.0 / (2.0 * Ne)
            log_term = (
                -Gamma_prev
                - 2.0 * u_col * t_prev
                + log1mexp(-c * dt)
                - jnp.log1p(4.0 * Ne * u_col)
            )
            Gamma_new = Gamma_prev + dt / (2.0 * Ne)
            return (Gamma_new, t_curr), log_term

        init_carry = (0.0, 0.0)  # Gamma_prev, t_prev
        (Gamma_finite, t_start_last), log_terms_finite = jax.lax.scan(
            survival_step, init_carry, (Ne_finite, t_boundaries)
        )

        # Last, infinite epoch [t_start_last, infinity).
        log_term_last = (
            -Gamma_finite
            - 2.0 * u_col * t_start_last
            - jnp.log1p(4.0 * Ne_last * u_col)
        )
        log_terms = jnp.concatenate([log_terms_finite, log_term_last[None, :]], axis=0)
        survival = jnp.exp(jax.scipy.special.logsumexp(log_terms, axis=0))
        res_matrix = survival.reshape(u_points.shape)
        return jnp.sum(res_matrix * u_weights, axis=1)

    expected_ld = compute_ld()
    genetic_diversity = compute_diversity()
    if sample_size is not None and ploidy == 2:
        expected_ld = correct_ld_finite_sample(expected_ld, sample_size)
    return genetic_diversity, expected_ld


# For the secondary introduction, we need `exp1`, the Exponential integral function.
def _inner_secondary_introduction_jax(Nec, Nef, Nea, T1, T2, m, u_i, u_j):
    from jax.scipy.special import exp1

    # Auto-generated code
    # fmt: off
    res =  8 * (-(Nea * Nec * m - Nec / 4 + Nea / 4) * (exp1(T2 * (4 * Nea * u_i + 1) / Nea / 2) - exp1(T2 * (4 * Nea * u_j + 1) / Nea / 2)) * Nef * (Nea * Nef * m + Nef ** 2 * m + Nea / 2 - Nef / 2) * (Nec * m + 0.1e1 / 0.2e1) * Nec * (Nea * m - 0.1e1 / 0.2e1) * jnp.exp((((-4 * Nef * T2 * m + T1 - T2) * Nec - Nef * T1) * Nea + T2 * Nec * Nef) / Nea / Nec / Nef / 2) - 2 * (Nec - Nef) * (Nea * Nec * m - Nec / 4 + Nea / 4) * (exp1(T2 * (4 * Nea * u_i + 1) / Nea / 2) - exp1(T2 * (4 * Nea * u_j + 1) / Nea / 2)) * Nef * (Nea * Nef * m + Nea / 4 - Nef / 4) * m * Nec * jnp.exp(((-2 * m * (T1 + T2) * Nec - T1) * Nea + T2 * Nec) / Nec / Nea / 2) + (Nec - Nef) * (exp1(T1 * (4 * Nea * u_i + 1) / Nea / 2) - exp1(T1 * (4 * Nea * u_j + 1) / Nea / 2)) * Nea ** 3 * (m * Nef + 0.1e1 / 0.2e1) * Nef * m ** 2 * (Nec * m + 0.1e1 / 0.2e1) * Nec * jnp.exp(-T1 * (4 * Nea * Nec * m + Nea - Nec) / Nec / Nea / 2) - 4 * Nea * (Nea * Nec * m - Nec / 4 + Nea / 4) * (exp1(2 * T1 * (0.1e1 / 0.4e1 + (m + u_i) * Nef) / Nef) - exp1(2 * T1 * (0.1e1 / 0.4e1 + (m + u_j) * Nef) / Nef) - exp1(2 * T2 * (0.1e1 / 0.4e1 + (m + u_i) * Nef) / Nef) + exp1(2 * T2 * (0.1e1 / 0.4e1 + (m + u_j) * Nef) / Nef)) * (m * Nef + 0.1e1 / 0.4e1) * (Nea * Nef * m + Nef ** 2 * m + Nea / 2 - Nef / 2) * (Nec * m + 0.1e1 / 0.2e1) * Nec * (Nea * m - 0.1e1 / 0.2e1) * jnp.exp((Nec - Nef) / Nef / Nec * T1 / 2) + 8 * Nef * (Nea * Nef * m + Nea / 4 - Nef / 4) * (Nea * (m * Nef + 0.1e1 / 0.2e1) * (Nec * m + 0.1e1 / 0.4e1) * ((Nec * m + 0.1e1 / 0.2e1) * Nea + Nec ** 2 * m - Nec / 2) * (Nea * m - 0.1e1 / 0.2e1) * exp1(2 * T1 * (0.1e1 / 0.4e1 + (m + u_i) * Nec) / Nec) / 2 - Nea * (m * Nef + 0.1e1 / 0.2e1) * (Nec * m + 0.1e1 / 0.4e1) * ((Nec * m + 0.1e1 / 0.2e1) * Nea + Nec ** 2 * m - Nec / 2) * (Nea * m - 0.1e1 / 0.2e1) * exp1(2 * T1 * (0.1e1 / 0.4e1 + (m + u_j) * Nec) / Nec) / 2 + jnp.exp(-T2 * (2 * Nea * m - 1) / Nea / 2) * (Nea * Nec * m - Nec / 4 + Nea / 4) * (m * Nef + 0.1e1 / 0.2e1) * m * Nec ** 2 * exp1(T2 * (4 * Nea * u_i + 1) / Nea / 2) / 2 - jnp.exp(-T2 * (2 * Nea * m - 1) / Nea / 2) * (Nea * Nec * m - Nec / 4 + Nea / 4) * (m * Nef + 0.1e1 / 0.2e1) * m * Nec ** 2 * exp1(T2 * (4 * Nea * u_j + 1) / Nea / 2) / 2 + (-(Nec - Nef) * (Nea * Nec * m - Nec / 4 + Nea / 4) * (exp1(T1 * (m + 2 * u_i)) - exp1(T1 * (m + 2 * u_j)) - exp1(T2 * (m + 2 * u_i)) + exp1(T2 * (m + 2 * u_j))) * m ** 2 * Nec * jnp.exp(-T1 * (2 * Nec * m + 1) / Nec / 2) / 2 + (m * Nef + 0.1e1 / 0.2e1) * ((Nec * m + 0.1e1 / 0.4e1) * ((Nec * m + 0.1e1 / 0.2e1) * Nea + Nec ** 2 * m - Nec / 2) * (Nea * m - 0.1e1 / 0.2e1) * jnp.log(1 + (4 * m + 4 * u_i) * Nec) / 2 - (Nec * m + 0.1e1 / 0.4e1) * ((Nec * m + 0.1e1 / 0.2e1) * Nea + Nec ** 2 * m - Nec / 2) * (Nea * m - 0.1e1 / 0.2e1) * jnp.log(1 + (4 * m + 4 * u_j) * Nec) / 2 + ((Nec / 4 - Nea * Nec * m - Nea / 4) * exp1(T2 * (m + 2 * u_i)) + (Nea * Nec * m - Nec / 4 + Nea / 4) * exp1(T2 * (m + 2 * u_j)) + Nea * (Nec * m + 0.1e1 / 0.2e1) * jnp.log(4 * Nea * u_i + 1) / 2 + Nea * (-Nec * m - 0.1e1 / 0.2e1) * jnp.log(4 * Nea * u_j + 1) / 2 + (jnp.log(m + 2 * u_j) - jnp.log(m + 2 * u_i)) * (Nea * Nec * m - Nec / 4 + Nea / 4)) * m ** 2 * Nec ** 2)) * Nea)) / Nec / Nea / Nef / (2 * Nea * m - 1) / (4 * Nea * Nec * m + Nea - Nec) / (2 * m * Nef + 1) / (4 * Nea * Nef * m + Nea - Nef) / (2 * Nec * m + 1) / (-u_j + u_i) # noqa
    # fmt: on
    return res


def expected_secondary_introduction(
    Ne_1,
    Ne_2,
    Ne_a,
    t0,
    t1,
    migration_rate,
    left_bins,
    right_bins,
    mutation_rate,
    sample_size=None,
    ploidy=2,
):
    """
    Compute expected genetic diversity and LD under a secondary-introduction invasion scenario using a deterministic approximation.

    Ne(t) = Ne_1 if t < t0, Ne_2 if t < t1 else Ne_3
    m(target -> ancestral) = migration_rate if t < t1 else 0
    m(ancestral -> target) = 0

    Parameters
    ----------
    Ne_1 : float
        Contemporary effective population size (migration activated).
    Ne_2 : float
        Intermediate effective population size (migration activated).
    Ne_a : float
        Ancestral effective population size (no migration).
    t0 : float
        Time when population size reaches Ne_2.
    t1 : float
        Time when target population is merged into ancestral population and migration starts.
    migration_rate : float
        Migration rate from target population to ancestral population.
    left_bins : array-like
        Left edges of genomic distance bins in Morgans.
    right_bins : array-like
        Right edges of genomic distance bins in Morgans.
    mutation_rate : float
        Mutation rate per base pair per generation.
    sample_size : int, optional
        Number of individuals sampled. If provided, we correct LD for finite sample size.
    ploidy : int, optional
        Ploidy of the individuals (either 1 or 2). Default is 2.

    Returns
    -------
    expected_pi : float or jax array
    expected_ld : array-like or jax array
    """
    u_i = jnp.asarray(left_bins)
    u_j = jnp.asarray(right_bins)
    Ne_1 = jnp.asarray(Ne_1) / 2 * ploidy
    Ne_2 = jnp.asarray(Ne_2) / 2 * ploidy
    Ne_a = jnp.asarray(Ne_a) / 2 * ploidy
    t0 = jnp.asarray(t0)
    t1 = jnp.asarray(t1)
    migration_rate = jnp.asarray(migration_rate)

    def compute_ld():
        # Vectorize over u_i and u_j (the bin edges)
        return jax.vmap(
            _inner_secondary_introduction_jax,
            in_axes=(None, None, None, None, None, None, 0, 0),
        )(Ne_1, Ne_2, Ne_a, t0, t1, migration_rate, u_i, u_j)

    def compute_diversity():
        Nec, Nef, Nea, T1, T2, m = Ne_1, Ne_2, Ne_a, t0, t1, migration_rate
        # Auto-generated code
        # fmt: off
        expected_tmrca = (32 * (m * Nec + 0.1e1 / 0.2e1) * (m * Nec + 0.1e1 / 0.4e1) * (Nef * (Nea + Nef) * m + Nea / 2 - Nef / 2) * jnp.exp(((-4 * Nef * T2 * m + T1 - T2) * Nec - Nef * T1) / Nec / Nef / 2) + 64 * (m * Nec + 0.1e1 / 0.4e1) * (m * Nef + 0.1e1 / 0.4e1) * (-Nef + Nec) * jnp.exp((-2 * m * (T1 + T2) * Nec - T1) / Nec / 2) + 128 * (m * Nef + 0.1e1 / 0.2e1) * (-(m * Nec + 0.1e1 / 0.2e1) * (-Nef + Nec) * (Nea * m + 0.3e1 / 0.4e1) * jnp.exp(-T1 * (4 * m * Nec + 1) / Nec / 2) / 4 + ((-m * Nec - 0.1e1 / 0.4e1) * jnp.exp(-T2 * m) + (m * Nec + 0.1e1 / 0.2e1) * (Nea * m + 0.3e1 / 0.4e1)) * (m * Nef + 0.1e1 / 0.4e1) * Nec)) / (4 * m * Nec + 1) / (2 * m * Nef + 1) / (4 * m * Nef + 1) / (2 * m * Nec + 1) # noqa
        # fmt: on
        return expected_tmrca * 2 * mutation_rate

    expected_ld = compute_ld()
    genetic_diversity = compute_diversity()
    if sample_size is not None and ploidy == 2:
        expected_ld = correct_ld_finite_sample(expected_ld, sample_size)
    return genetic_diversity, expected_ld
