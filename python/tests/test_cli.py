"""
Integration tests: compare data_from_vcf against data_from_tree_sequence
using the same data (tree sequence converted to BCF via tskit + bcftools).
"""

import shutil
import subprocess

import bayesld
import numpy as np
import pytest

requires_bcftools = pytest.mark.skipif(
    shutil.which("bcftools") is None, reason="bcftools not found in PATH"
)


# --- CLI argument validation tests (fast, no bcftools needed) ---


def _run_cli(*args):
    """Run vcfbayesld via uv and return the CompletedProcess."""
    return subprocess.run(
        ["uv", "run", "vcfbayesld", *args],
        capture_output=True,
        text=True,
    )


def test_cli_missing_required_args():
    """Exits non-zero when required arguments are omitted."""
    result = _run_cli("--vcf", "x.bcf", "--contig", "chr1")
    assert result.returncode != 0


def test_cli_both_rate_and_hapmap_are_mutually_exclusive():
    """Exits non-zero when both --recombination-rate and --hapmap are given."""
    result = _run_cli(
        "--vcf",
        "x.bcf",
        "--contig",
        "chr1",
        "--start",
        "0",
        "--end",
        "1000",
        "--recombination-rate",
        "1e-8",
        "--hapmap",
        "map.txt",
        "--linear-bins",
        "0.005",
        "0.1",
        "19",
    )
    assert result.returncode != 0


def test_cli_neither_rate_nor_hapmap():
    """Exits non-zero when neither --recombination-rate nor --hapmap is given."""
    result = _run_cli(
        "--vcf",
        "x.bcf",
        "--contig",
        "chr1",
        "--start",
        "0",
        "--end",
        "1000",
        "--linear-bins",
        "0.005",
        "0.1",
        "19",
    )
    assert result.returncode != 0


def test_cli_neither_bins_nor_linear_bins():
    """Exits non-zero when neither --bins nor --linear-bins is given."""
    result = _run_cli(
        "--vcf",
        "x.bcf",
        "--contig",
        "chr1",
        "--start",
        "0",
        "--end",
        "1000",
        "--recombination-rate",
        "1e-8",
    )
    assert result.returncode != 0


def test_cli_invalid_ploidy():
    """Exits non-zero for unsupported ploidy value."""
    result = _run_cli(
        "--vcf",
        "x.bcf",
        "--contig",
        "chr1",
        "--start",
        "0",
        "--end",
        "1000",
        "--recombination-rate",
        "1e-8",
        "--linear-bins",
        "0.005",
        "0.1",
        "19",
        "--ploidy",
        "3",
    )
    assert result.returncode != 0


SEQ_LEN = int(2e7)
N_SAMPLES = 10
NE = 1000
MUTATION_RATE = 1e-8


def _build_bcf(tmp_path, recombination_rate_arg: list[str], seed: int = 42):
    """
    Run msp ancestry + msp mutations, convert to indexed BCF.
    recombination_rate_arg: CLI tokens, e.g. ["--recombination-rate", "1e-8"]
    Returns (mut_trees_path, bcf_path).
    """
    trees_path = tmp_path / "ancestry.trees"
    mut_path = tmp_path / "mutated.trees"
    bcf_path = tmp_path / "example.bcf"

    subprocess.run(
        [
            "uv",
            "run",
            "msp",
            "ancestry",
            str(N_SAMPLES),
            "--population-size",
            str(NE),
            "--length",
            str(SEQ_LEN),
            *recombination_rate_arg,
            "--random-seed",
            str(seed),
            "-o",
            str(trees_path),
        ],
        check=True,
    )
    subprocess.run(
        [
            "uv",
            "run",
            "msp",
            "mutations",
            str(MUTATION_RATE),
            str(trees_path),
            "--random-seed",
            str(seed),
            "-o",
            str(mut_path),
        ],
        check=True,
    )
    tskit_proc = subprocess.Popen(
        ["uv", "run", "tskit", "vcf", str(mut_path)],
        stdout=subprocess.PIPE,
    )
    with open(bcf_path, "wb") as bcf_out:
        subprocess.run(
            ["bcftools", "view", "-O", "b"],
            stdin=tskit_proc.stdout,
            stdout=bcf_out,
            check=True,
        )
    tskit_proc.wait()
    subprocess.run(["bcftools", "index", str(bcf_path)], check=True)

    return mut_path, bcf_path


def _assert_vcf_matches_ts(ts, bcf_path, recombination_rate):
    left_bins, right_bins = bayesld.linear_bins()

    expected = bayesld.data_from_tree_sequence(
        ts,
        recombination_rate=recombination_rate,
        left_bins_morgan=left_bins,
        right_bins_morgan=right_bins,
    )
    result = bayesld.data_from_vcf(
        vcf_path=str(bcf_path),
        recombination_rate=recombination_rate,
        left_bins_morgan=left_bins,
        right_bins_morgan=right_bins,
        contig="1",
        start_bp=0,
        end_bp=SEQ_LEN,
    )

    assert result["sample_size"] == expected["sample_size"]
    assert np.isclose(
        result["mean_genetic_diversity"],
        expected["mean_genetic_diversity"],
        rtol=1e-4,
    ), (
        f"pi mismatch: {result['mean_genetic_diversity']} vs {expected['mean_genetic_diversity']}"
    )
    np.testing.assert_allclose(
        result["num_pairs_linkage_disequilibrium"],
        expected["num_pairs_linkage_disequilibrium"],
    )
    np.testing.assert_allclose(
        result["mean_linkage_disequilibrium"],
        expected["mean_linkage_disequilibrium"],
        rtol=1e-4,
    )


@requires_bcftools
def test_missing_genotypes_do_not_crash(tmp_path):
    """
    Variants with missing genotypes (gt_types==3) are handled by Rust without crashing.
    The site with all-missing calls should still be processed (or silently skipped),
    and the result for a fully-called site should be correct.
    """
    vcf_text = """\
##fileformat=VCFv4.1
##contig=<ID=1,length=100000>
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\tS2\tS3\tS4
1\t1000\t.\tA\tT\t.\tPASS\t.\tGT\t0/0\t0/1\t1/1\t./.
1\t2000\t.\tA\tT\t.\tPASS\t.\tGT\t0/1\t0/1\t0/1\t0/1
"""
    vcf_path = tmp_path / "test.vcf"
    bcf_path = tmp_path / "test.bcf"
    vcf_path.write_text(vcf_text)

    subprocess.run(
        ["bcftools", "view", "-O", "b", "-o", str(bcf_path), str(vcf_path)], check=True
    )
    subprocess.run(["bcftools", "index", str(bcf_path)], check=True)

    left_bins, right_bins = bayesld.linear_bins()
    result = bayesld.data_from_vcf(
        vcf_path=str(bcf_path),
        recombination_rate=1e-8,
        left_bins_morgan=left_bins,
        right_bins_morgan=right_bins,
        contig="1",
        start_bp=0,
        end_bp=100_000,
    )

    assert result["sample_size"] == 4
    assert result["num_sites_genetic_diversity"] > 0


@pytest.mark.slow
@requires_bcftools
def test_vcf_matches_tree_sequence_flat_rate(tmp_path):
    """Flat recombination rate: data_from_vcf matches data_from_tree_sequence."""
    import tskit as tsk

    recombination_rate = 1e-8
    mut_path, bcf_path = _build_bcf(
        tmp_path, ["--recombination-rate", str(recombination_rate)]
    )
    ts = tsk.load(str(mut_path))
    _assert_vcf_matches_ts(ts, bcf_path, recombination_rate)


@pytest.mark.slow
@requires_bcftools
def test_vcf_flat_rate_nonzero_start(tmp_path):
    """Flat rate with start > 0: data_from_vcf matches data_from_tree_sequence on the same interval."""
    import tskit as tsk

    recombination_rate = 1e-8
    start_bp = SEQ_LEN // 4
    end_bp = SEQ_LEN // 2

    mut_path, bcf_path = _build_bcf(
        tmp_path, ["--recombination-rate", str(recombination_rate)]
    )
    ts = tsk.load(str(mut_path))
    ts_trimmed = ts.keep_intervals([[start_bp, end_bp]])

    left_bins, right_bins = bayesld.linear_bins()

    expected = bayesld.data_from_tree_sequence(
        ts_trimmed,
        recombination_rate=recombination_rate,
        left_bins_morgan=left_bins,
        right_bins_morgan=right_bins,
    )
    result = bayesld.data_from_vcf(
        vcf_path=str(bcf_path),
        recombination_rate=recombination_rate,
        left_bins_morgan=left_bins,
        right_bins_morgan=right_bins,
        contig="1",
        start_bp=start_bp,
        end_bp=end_bp,
    )

    assert result["sample_size"] == expected["sample_size"]
    assert np.isclose(
        result["mean_genetic_diversity"],
        expected["mean_genetic_diversity"],
        rtol=1e-4,
    )
    np.testing.assert_allclose(
        result["num_pairs_linkage_disequilibrium"],
        expected["num_pairs_linkage_disequilibrium"],
    )
    np.testing.assert_allclose(
        result["mean_linkage_disequilibrium"],
        expected["mean_linkage_disequilibrium"],
        rtol=1e-4,
    )


@pytest.mark.slow
@requires_bcftools
def test_vcf_matches_tree_sequence_hapmap_rate(tmp_path):
    """
    Semi-complex HapMap rate map: data_from_vcf matches data_from_tree_sequence.

    The rate map has four segments with different rates, including a cold spot
    and a hot spot, covering the full 20 Mb window.
    """
    import msprime
    import tskit as tsk

    # Write a HapMap file with varying rates across 20 Mb
    # Format: Chromosome  Position(bp)  Rate(cM/Mb)  Map(cM)
    hapmap_path = tmp_path / "genetic_map.txt"
    segments = [
        (0, 1.0),  # 0–5 Mb: cold spot (1 cM/Mb)
        (5_000_000, 3.0),  # 5–10 Mb: normal
        (10_000_000, 0.2),  # 10–15 Mb: very cold
        (15_000_000, 5.0),  # 15–20 Mb: hot spot
    ]
    with open(hapmap_path, "w") as f:
        f.write("Chromosome\tPosition(bp)\tRate(cM/Mb)\tMap(cM)\n")
        cumulative_cm = 0.0
        prev_pos = 0
        for i, (pos, rate_cm_mb) in enumerate(segments):
            if i > 0:
                span_mb = (pos - prev_pos) / 1e6
                cumulative_cm += segments[i - 1][1] * span_mb
            f.write(f"chr1\t{pos}\t{rate_cm_mb:.4f}\t{cumulative_cm:.6f}\n")
            prev_pos = pos
        # Terminal entry at sequence end with rate=0
        span_mb = (SEQ_LEN - prev_pos) / 1e6
        cumulative_cm += segments[-1][1] * span_mb
        f.write(f"chr1\t{SEQ_LEN}\t0.0000\t{cumulative_cm:.6f}\n")

    rate_map = msprime.RateMap.read_hapmap(str(hapmap_path), sequence_length=SEQ_LEN)

    mut_path, bcf_path = _build_bcf(tmp_path, ["--recombination-rate", "1e-8"])
    ts = tsk.load(str(mut_path))
    _assert_vcf_matches_ts(ts, bcf_path, rate_map)


@pytest.mark.slow
@requires_bcftools
def test_vcf_panics_on_short_rate_map(tmp_path):
    """
    data_from_vcf raises when the RateMap is shorter than the VCF region,
    so variant positions fall outside the map.
    """
    import msprime

    # BCF spans 20 Mb; rate map only covers the first 1 Mb.
    short_rate_map = msprime.RateMap(
        position=[0.0, 1_000_000.0],
        rate=[1e-8],
    )
    _, bcf_path = _build_bcf(tmp_path, ["--recombination-rate", "1e-8"])

    with pytest.raises(BaseException, match="outside the rate map range"):
        bayesld.data_from_vcf(
            vcf_path=str(bcf_path),
            recombination_rate=short_rate_map,
            left_bins_morgan=bayesld.linear_bins()[0],
            right_bins_morgan=bayesld.linear_bins()[1],
            contig="1",
            start_bp=0,
            end_bp=SEQ_LEN,
        )
