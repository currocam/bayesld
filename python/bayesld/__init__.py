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


def _nan_bp_in_range(
    missing_intervals: np.ndarray, from_bp: float, to_bp: float
) -> float:
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
    maf_threshold: float = 0.25,
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
        missing_intervals = (
            recombination_rate.missing_intervals()
        )  # shape (n_missing, 2)

    if ploidy == 2:
        stats = _bayesld.StreamingStatsDiploid(
            left_bins_morgan.tolist(),
            right_bins_morgan.tolist(),
            map_position_bp,
            map_rate,
            maf_threshold,
        )
    elif ploidy == 1:
        stats = _bayesld.StreamingStatsHaploid(
            left_bins_morgan.tolist(),
            right_bins_morgan.tolist(),
            map_position_bp,
            map_rate,
            maf_threshold,
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


def _callable_windows(
    recombination_rate,
    start_bp: int,
    end_bp: int,
) -> list[tuple[int, int]]:
    """Return list of callable (non-masked) intervals within [start_bp, end_bp]."""
    if isinstance(recombination_rate, (int, float)):
        return [(start_bp, end_bp)]

    missing = recombination_rate.missing_intervals()  # shape (n, 2), 0-based bp
    # Clip missing intervals to [start_bp, end_bp]
    if len(missing) == 0:
        return [(start_bp, end_bp)]

    # Compute complement of missing within [start_bp, end_bp]
    windows = []
    cursor = start_bp
    for m_start, m_end in missing:
        m_start = int(m_start)
        m_end = int(m_end)
        if m_end <= start_bp or m_start >= end_bp:
            continue
        m_start = max(m_start, start_bp)
        m_end = min(m_end, end_bp)
        if cursor < m_start:
            windows.append((cursor, m_start))
        cursor = max(cursor, m_end)
    if cursor < end_bp:
        windows.append((cursor, end_bp))
    return windows


def data_from_vcf(
    vcf_path: str,
    recombination_rate,
    left_bins_morgan: NDArray,
    right_bins_morgan: NDArray,
    contig: str,
    start_bp: int,
    end_bp: int,
    chunk_size: int = 10_000,
    ploidy: int = 2,
    progress_bar: bool = False,
    samples: list[str] | None = None,
    maf_threshold: float = 0.25,
) -> dict:
    """
    Compute linkage disequilibrium and genetic diversity statistics from a VCF/BCF file.

    The caller is responsible for pre-filtering the VCF/BCF (biallelic SNPs, FILTER=PASS,
    sample subset) using bcftools before calling this function. The file must have a
    tabix (.tbi) or CSI (.csi) index.

    Parameters
    ----------
    vcf_path : str
        Path to the VCF/BCF file (must be indexed).
    recombination_rate : float or msprime.RateMap
        Recombination rate per base pair per generation, or a piecewise-constant RateMap.
        Masked (NaN-rate) intervals in the RateMap are skipped automatically.
    left_bins_morgan : NDArray
        Left endpoints of distance bins in Morgan.
    right_bins_morgan : NDArray
        Right endpoints of distance bins in Morgan.
    contig : str
        Chromosome/contig name as it appears in the VCF header.
    start_bp : int
        Window start, 0-based inclusive. Use 0 for the full chromosome.
    end_bp : int
        Window end, 0-based exclusive. Use sequence length for the full chromosome.
    chunk_size : int, optional
        Number of variants to buffer before flushing to the stats engine. Default 10,000.
    ploidy : int, optional
        1 or 2. Default 2.
    progress_bar : bool, optional
        Whether to display a progress bar over callable windows. Default False.
    samples : list[str] or None, optional
        Subset of sample names to use. If None, all samples are used.

    Returns
    -------
    dict
        Same structure as data_from_tree_sequence.
    """
    try:
        import cyvcf2
    except ImportError:
        raise ImportError("cyvcf2 is required for data_from_vcf: pip install cyvcf2")

    # Build rate map arrays
    if isinstance(recombination_rate, (int, float)):
        if start_bp > 0:
            map_position_bp = [0.0, float(start_bp), float(end_bp)]
            map_rate = [float("nan"), float(recombination_rate)]
        else:
            map_position_bp = [0.0, float(end_bp)]
            map_rate = [float(recombination_rate)]
    else:
        map_position_bp = list(recombination_rate.position)
        map_rate = list(recombination_rate.rate)

    if ploidy == 2:
        stats = _bayesld.StreamingStatsDiploid(
            left_bins_morgan.tolist(),
            right_bins_morgan.tolist(),
            map_position_bp,
            map_rate,
            maf_threshold,
        )
    elif ploidy == 1:
        stats = _bayesld.StreamingStatsHaploid(
            left_bins_morgan.tolist(),
            right_bins_morgan.tolist(),
            map_position_bp,
            map_rate,
            maf_threshold,
        )
    else:
        raise ValueError("Ploidy must be 1 or 2")

    vcf = cyvcf2.VCF(vcf_path, gts012=True, samples=samples)
    n_columns = len(vcf.samples)

    buffer_geno = np.empty((chunk_size, n_columns), dtype=np.int32)
    buffer_pos = np.empty(chunk_size, dtype=np.int32)

    callable_windows = _callable_windows(recombination_rate, start_bp, end_bp)

    for win_start, win_end in tqdm(callable_windows, disable=not progress_bar):
        # cyvcf2 uses 1-based inclusive region strings
        region_str = f"{contig}:{win_start + 1}-{win_end}"
        chunk_len = 0

        for variant in vcf(region_str):
            if ploidy == 2:
                dosage = variant.gt_types.astype(np.int32)
            else:
                dosage = (variant.gt_types > 0).astype(np.int32)

            buffer_geno[chunk_len] = dosage
            buffer_pos[chunk_len] = variant.start  # 0-based
            chunk_len += 1

            if chunk_len == chunk_size:
                geno_chunk = buffer_geno[:chunk_len]
                pos_chunk = buffer_pos[:chunk_len]
                region_span = float(pos_chunk[-1]) - float(pos_chunk[0])
                stats.add_batch(geno_chunk, pos_chunk, region_span)
                chunk_len = 0

        # Flush remainder at end of each callable window
        if chunk_len > 0:
            geno_chunk = buffer_geno[:chunk_len]
            pos_chunk = buffer_pos[:chunk_len]
            region_span = float(pos_chunk[-1]) - float(pos_chunk[0])
            stats.add_batch(geno_chunk, pos_chunk, region_span)

    vcf.close()

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


from . import montecarlo

__doc__ = _bayesld.__doc__
if hasattr(_bayesld, "__all__"):
    __all__ = _bayesld.__all__
else:
    __all__ = []
