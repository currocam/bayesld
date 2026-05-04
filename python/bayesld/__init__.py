import numpy as np
import tskit
from numpy.typing import NDArray
from tqdm.auto import tqdm

# Set module docstring from the bayesld module
from . import bayesld as _bayesld
from . import deterministic, surrogate_likelihoods
from .bayesld import *


def linear_bins(
    min_distance: float = 0.005, max_distance: float = 0.1, n_bins: int = 19
) -> tuple[NDArray, NDArray]:
    """
    Create linearly spaced distance bins for LD analysis.
    By default, the default binning is used.

    Args:
        min_distance: Minimum distance in Morgan (default: 0.005)
        max_distance: Maximum distance in Morgan (default: 0.1)
        n_bins: Number of bins to create (default: 19)
    Returns:
        Tuple of (left_bins_morgan, right_bins_morgan) arrays
    """
    edges = np.linspace(min_distance, max_distance, n_bins + 1)

    left_bins = edges[:-1]
    right_bins = edges[1:]

    return left_bins, right_bins


def _nan_bp_in_range(missing_intervals: np.ndarray, from_bp: float, to_bp: float) -> float:
    """Total NaN base pairs overlapping [from_bp, to_bp]."""
    if len(missing_intervals) == 0:
        return 0.0
    lefts = np.maximum(missing_intervals[:, 0], from_bp)
    rights = np.minimum(missing_intervals[:, 1], to_bp)
    return float(np.sum(np.maximum(rights - lefts, 0.0)))


def data_from_tree_sequence(
    ts: tskit.TreeSequence,
    recombination_rate: "float | msprime.RateMap",
    left_bins_morgan: NDArray,
    right_bins_morgan: NDArray,
    chunk_size: int = 10_000,
    ploidy: int = 2,
    progress_bar: bool = False,
) -> dict:
    """
    Compute linkage disequilibrium and genetic diversity statistics from a tskit tree sequence.

    This function processes the tree sequence in chunks to compute LD statistics
    across distance bins. If memory is limited, adjust the chunk_size parameter.

    Parameters
    ----------
    ts : tskit.TreeSequence
        The tree sequence object to analyze
    recombination_rate : float or msprime.RateMap
        Recombination rate per base pair per generation, or a piecewise-constant RateMap.
    left_bins_morgan : NDArray
        Left endpoints of distance bins in Morgan.
    right_bins_morgan : NDArray
        Right endpoints of distance bins in Morgan.
    chunk_size : int, optional
        Number of loci to process at a time. Default is 10,000
    ploidy : int, optional
        Ploidy of the organism. Default is 2. If ploidy is 2, the samples are assumed to be ordered by individual.
    progress_bar : bool, optional
        Whether to display a progress bar. Default is False

    Returns
    -------
    dict
        Dictionary with LD and diversity statistics.
    """
    # Extract rate map arrays
    if isinstance(recombination_rate, (int, float)):
        map_position_bp = [0.0, float(ts.sequence_length)]
        map_rate = [float(recombination_rate)]
        missing_intervals = np.empty((0, 2))
    else:
        # msprime.RateMap — has .position and .rate attributes
        map_position_bp = list(recombination_rate.position)
        map_rate = list(recombination_rate.rate)
        missing_intervals = recombination_rate.missing_intervals()  # shape (n_missing, 2)

    if ploidy == 2:
        stats = _bayesld.StreamingStatsDiploid(
            left_bins_morgan.tolist(),
            right_bins_morgan.tolist(),
            map_position_bp,
            map_rate,
        )
    elif ploidy == 1:
        stats = _bayesld.StreamingStatsHaploid(
            left_bins_morgan.tolist(),
            right_bins_morgan.tolist(),
            map_position_bp,
            map_rate,
        )
    else:
        raise ValueError("Ploidy must be 1 or 2")

    variant = tskit.Variant(ts)
    num_samples = ts.num_samples
    n_columns = num_samples // ploidy
    num_sites = ts.num_sites
    positions_all = ts.sites_position.astype("int32")
    genotype_buffer = np.empty((chunk_size, n_columns), dtype=np.int32)

    iterator = tqdm(range(0, num_sites, chunk_size), disable=not progress_bar)
    for start_idx in iterator:
        end_idx = min(start_idx + chunk_size, num_sites)
        chunk_len = end_idx - start_idx
        genotype_chunk = genotype_buffer[:chunk_len]

        for i, site_id in enumerate(range(start_idx, end_idx)):
            variant.decode(site_id)
            if ploidy == 2:
                genotype_chunk[i] = variant.genotypes[0::2] + variant.genotypes[1::2]
            else:
                genotype_chunk[i] = variant.genotypes

        positions_chunk = positions_all[start_idx:end_idx]

        # region_span = physical span of this chunk minus any NaN (masked) bp
        chunk_start = float(positions_chunk[0])
        chunk_end = float(positions_chunk[-1])
        nan_bp = _nan_bp_in_range(missing_intervals, chunk_start, chunk_end)
        region_span = (chunk_end - chunk_start) - nan_bp

        stats.add_batch(genotype_chunk, positions_chunk, region_span)

    mat = stats.finalize()
    return {
        "sample_size": n_columns,
        "left_bins_morgan": left_bins_morgan,
        "right_bins_morgan": right_bins_morgan,
        "mean_linkage_disequilibrium": mat[1:, 0],
        "mean_genetic_diversity": mat[0, 0],
        "num_sites_genetic_diversity": mat[0, 1],
        "num_pairs_linkage_disequilibrium": mat[1:, 1],
    }


from . import models, montecarlo
from .models import ConstantDemography

__doc__ = _bayesld.__doc__
if hasattr(_bayesld, "__all__"):
    __all__ = _bayesld.__all__
else:
    __all__ = []
