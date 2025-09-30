import pytensor.tensor as pt


# Correction for finite sample size
def correct_r2(mu, sample_size):
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
    mu = (-pt.log(4 * Ne * u_i + 1) + pt.log(4 * Ne * u_j + 1)) / (
            4 * Ne * (u_j - u_i)
        )

    if sample_size is not None:
        return correct_r2(mu, sample_size)
    return mu
