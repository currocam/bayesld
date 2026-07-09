"""
Inference primitives for bayesld.

This module exposes a thin, msprime-flavoured entry point for simulating the
sufficient statistics (mean genetic diversity and mean linkage disequilibrium)
that the rest of the library conditions on.
"""

import msprime
import numpy as np

from ..montecarlo import _parallel_mc

__all__ = [
    "sim_sufficient_stats",
    "PiecewiseConstant",
    "PiecewiseExponential",
    "RandomWalk",
    "CustomEngine",
    "plot_demography",
]


def sim_sufficient_stats(
    samples,
    *,
    demography,
    left_bins,
    right_bins,
    mutation_rate,
    recombination_rate,
    sequence_length,
    random_seed,
    num_replicates=None,
    ploidy=2,
    model=None,
    num_workers=1,
    rtol=None,
):
    """
    Simulate the sufficient statistics (pi, LD) under an msprime demography.

    Mirrors the calling convention of :func:`msprime.sim_ancestry`: pass the
    number of sampled individuals and a :class:`msprime.Demography`, and get back
    Monte Carlo replicates of the sufficient statistics.

    Parameters
    ----------
    samples : int
        Number of individuals sampled from the (single) population. Only a single
        sampled population is supported for now; passing anything other than an
        integer raises ``NotImplementedError``.
    demography : msprime.Demography
        Any msprime demography.
    left_bins, right_bins : array-like
        LD distance bin edges in Morgans.
    mutation_rate, recombination_rate, sequence_length : float
    random_seed : int
    num_replicates : int or None
        Fixed number of Monte Carlo replicates (mutually exclusive with ``rtol``).
        If both ``num_replicates`` and ``rtol`` are None, a single replicate is run.
    ploidy : int
        Default 2.
    model : msprime ancestry model or None
        Default is ``SMCK(k=1)``.
    num_workers : int
        joblib parallel workers (default 1; -1 = all cores).
    rtol : float or None
        Adaptive stopping: simulate until the relative SE of the mean < ``rtol``.

    Returns
    -------
    pi_replicates : ndarray, shape (n,)
    ld_replicates : ndarray, shape (n, num_bins)
    """
    if not isinstance(samples, (int, np.integer)):
        raise NotImplementedError(
            "sim_sufficient_stats currently supports only a single sampled "
            "population, so `samples` must be an integer number of individuals. "
            "If you need multiple populations or a richer sampling scheme, please "
            "open an issue: https://github.com/currocam/bayesld/issues"
        )

    if num_replicates is not None and rtol is not None:
        raise ValueError("num_replicates and rtol are mutually exclusive")
    if num_replicates is None and rtol is None:
        num_replicates = 1

    if model is None:
        model = msprime.SMCK(k=1)

    left_bins = np.asarray(left_bins)
    right_bins = np.asarray(right_bins)

    return _parallel_mc(
        lambda: demography,
        (),
        left_bins,
        right_bins,
        mutation_rate,
        recombination_rate,
        sequence_length,
        int(samples),
        ploidy,
        model,
        random_seed,
        num_replicates,
        num_workers,
        rtol,
    )


from .piecewise_constant import PiecewiseConstant  # noqa: E402
from .piecewise_exponential import PiecewiseExponential  # noqa: E402
from .random_walk import RandomWalk  # noqa: E402
from .custom import CustomEngine  # noqa: E402
from ._plotting import plot_demography  # noqa: E402
