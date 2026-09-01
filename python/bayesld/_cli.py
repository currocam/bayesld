"""Command-line interface for computing LD statistics from a VCF/BCF file."""

import argparse
import json
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="vcfbayesld",
        description="Compute LD and diversity statistics from a VCF/BCF file.",
    )
    parser.add_argument("--vcf", required=True, help="Path to indexed VCF/BCF file.")
    parser.add_argument("--contig", required=True, help="Contig/chromosome name.")
    parser.add_argument(
        "--start", required=True, type=int, help="Window start, 0-based inclusive."
    )
    parser.add_argument(
        "--end", required=True, type=int, help="Window end, 0-based exclusive."
    )

    rate_group = parser.add_mutually_exclusive_group(required=True)
    rate_group.add_argument(
        "--recombination-rate",
        type=float,
        metavar="RATE",
        help="Flat recombination rate per bp per generation.",
    )
    rate_group.add_argument(
        "--hapmap",
        metavar="FILE",
        help="HapMap genetic map file (read with msprime.RateMap.read_hapmap()).",
    )

    bins_group = parser.add_mutually_exclusive_group(required=True)
    bins_group.add_argument(
        "--bins",
        metavar="L1,R1,L2,R2,...",
        help="Comma-separated bin edges: left1,right1,left2,right2,... in Morgan.",
    )
    bins_group.add_argument(
        "--linear-bins",
        nargs=3,
        metavar=("MIN", "MAX", "N"),
        help="Generate N linear bins between MIN and MAX Morgan.",
    )

    parser.add_argument(
        "--samples",
        metavar="S1,S2,...",
        help="Comma-separated list of sample names to include.",
    )
    parser.add_argument("--ploidy", type=int, default=2, choices=[1, 2])
    parser.add_argument("--chunk-size", type=int, default=10_000)
    parser.add_argument(
        "--output", metavar="FILE", help="Write JSON to FILE instead of stdout."
    )
    parser.add_argument("--progress", action="store_true", help="Show progress bar.")

    args = parser.parse_args()

    import numpy as np

    import bayesld

    # Resolve recombination rate
    if args.recombination_rate is not None:
        recombination_rate = args.recombination_rate
    else:
        import msprime

        recombination_rate = msprime.RateMap.read_hapmap(args.hapmap)

    # Resolve bins
    if args.bins is not None:
        values = [float(v) for v in args.bins.split(",")]
        if len(values) % 2 != 0:
            parser.error("--bins must have an even number of comma-separated values.")
        left_bins = np.array(values[0::2])
        right_bins = np.array(values[1::2])
    else:
        min_dist, max_dist, n_bins = (
            float(args.linear_bins[0]),
            float(args.linear_bins[1]),
            int(args.linear_bins[2]),
        )
        left_bins, right_bins = bayesld.linear_bins(min_dist, max_dist, n_bins)

    samples = args.samples.split(",") if args.samples else None

    result = bayesld.data_from_vcf(
        vcf_path=args.vcf,
        recombination_rate=recombination_rate,
        left_bins_morgan=left_bins,
        right_bins_morgan=right_bins,
        contig=args.contig,
        start_bp=args.start,
        end_bp=args.end,
        chunk_size=args.chunk_size,
        ploidy=args.ploidy,
        progress_bar=args.progress,
        samples=samples,
    )

    print(
        f"sample_size={result['sample_size']}  "
        f"n_sites={int(result['num_sites_genetic_diversity'])}  "
        f"n_bins={len(result['left_bins_morgan'])}",
        file=sys.stderr,
    )

    output = {
        "sample_size": result["sample_size"],
        "left_bins_morgan": result["left_bins_morgan"].tolist(),
        "right_bins_morgan": result["right_bins_morgan"].tolist(),
        "mean_linkage_disequilibrium": result["mean_linkage_disequilibrium"].tolist(),
        "mean_genetic_diversity": result["mean_genetic_diversity"],
        "num_sites_genetic_diversity": int(result["num_sites_genetic_diversity"]),
        "num_pairs_linkage_disequilibrium": result[
            "num_pairs_linkage_disequilibrium"
        ].tolist(),
        "missingness": result["missingness"].tolist(),
    }

    json_str = json.dumps(output, indent=2)
    if args.output:
        with open(args.output, "w") as f:
            f.write(json_str)
    else:
        print(json_str)


if __name__ == "__main__":
    main()
