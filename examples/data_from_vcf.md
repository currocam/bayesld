---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.4
kernelspec:
  display_name: bayesld (.venv)
  language: python
  name: bayesld
---

# Extracting summary statistics from a VCF

`bayesld` performs inference over a set of informative summary statistics computed across windows (which are assumed to be independent and identically distributed). These statistics are the mean observed genetic diversity (also known as sample heterozygosity) and a measure of linkage disequilibrium (LD). To measure LD, we compute $\overline{X_i X_j Y_i Y_j}$, the mean product of the (centered and standardized) genotypes across all pairs of individuals, for many pairs of loci $X$ and $Y$ separated by increasing distances. Pairs of genotypes are then aggregated in different bins.

Both summary statistics can be computed from an unphased VCF, provided that estimates of the recombination rate and mutation rate are available. `bayesld` provides functionality to compute this summary statistics in Python (using the `bayesld.data_from_vcf` function) and a command-line tool which is installed together with the Python library called `vcfbayesld`. 

For a full inference workflow on simulated data, see [inference_two_epoch_bottleneck.md](inference_two_epoch_bottleneck.md).

```{code-cell} ipython3
import bayesld
import numpy as np
```

## Input data

An **indexed** VCF or BCF file (tabix `.tbi` or CSI `.csi` index).

```{code-cell} ipython3
%%bash 
ls tutorial_vcf_data/
```

## Split the chromosome into equivalent windows

Inference in `bayesld` works with windowed-summary statistics. This windows are assumed to be independent and identically distributed, which means that, in practice, some amount of bookeeping is neccesary for this to work. 

I recommend using 20cM windows (twice the length of the maximum distance consider between SNPs with default settings). In practice, other window sizes also work provided that they are (1) sufficiently large to contain many SNPs and (2) larger than the maximum distance considered. 

If we assume a flat recombination rate of 1e-8 (more on variable recombination rate maps later): 

```{code-cell} ipython3
recombination_rate = 1e-8
sequence_length = int(1e8)
window_length_cm = 20 
window_length_bp = int(window_length_cm / 100 / recombination_rate)

regions = [
    (start, min(start + window_length_bp, sequence_length))
    for start in range(0, sequence_length, window_length_bp)
]
regions
```

## Command line: `vcfbayesld`

The `vcfbayesld` script (which internally is a wrapper around `data_from_vcf`) and prints JSON file to stdout.

```{code-cell} ipython3
%%bash
vcfbayesld \
  --vcf tutorial_vcf_data/chr1.bcf --progress \
  --contig 1 --start 20000000 --end 40000000 \
  --recombination-rate 1e-8 --linear-bins 0.005 0.1 19 > tutorial_vcf_data/region.json
```

A description of the different flags can be obtained using `--help`

```{code-cell} ipython3
%%bash
vcfbayesld --help
```

## Python API: `bayesld.data_from_vcf`

However, the Python API provides a a more flexible interface. Using default parameters for the first window: 

```{code-cell} ipython3
left_bins_morgan, right_bins_morgan = bayesld.linear_bins()
result = bayesld.data_from_vcf(
    vcf_path="tutorial_vcf_data/chr1.bcf",
    recombination_rate=recombination_rate,
    left_bins_morgan=left_bins_morgan,
    right_bins_morgan=right_bins_morgan,
    contig="1",
    start_bp=0,
    end_bp=20000000,
)
result["mean_genetic_diversity"], result["mean_linkage_disequilibrium"]
```

To obtain the final dataset we can loop through all the genomic windows we define earlier:

```{code-cell} ipython3
obs_pi = []
obs_ld = []
for start, end in windows:
    stats = bayesld.data_from_vcf(
        vcf_path="tutorial_vcf_data/chr1.bcf",
        recombination_rate=recombination_rate,
        left_bins_morgan=left_bins_morgan,
        right_bins_morgan=right_bins_morgan,
        contig="1",
        start_bp=start,
        end_bp=end,
        progress_bar=True
    )
    obs_pi.append(stats["mean_genetic_diversity"])
    obs_ld.append(stats["mean_linkage_disequilibrium"])

obs_pi = np.asarray(obs_pi)
obs_ld = np.asarray(obs_ld)
obs_pi.shape, obs_ld.shape
```

These arrays plug directly into the inference API:

```python
from bayesld.inference import PiecewiseConstant

model = PiecewiseConstant(num_epochs=2).with_data(
    mean_diversity=obs_pi,
    mean_ld=obs_ld,
    left_bins=left_bins_morgan,
    right_bins=right_bins_morgan,
    recombination_rate=recombination_rate,
    mutation_rate=mutation_rate,
    num_samples=num_samples,
    sequence_length=window_length_bp,
)
```

A detail tutorial about inference can be found in [inference_two_epoch_bottleneck.md](inference_two_epoch_bottleneck.md).

+++

# Advanced topics

+++

## Custom recombination rate maps

I obtained good results with empirical data using a chromosome-wide average of recombination rate. However, if you want to use a more detailed recombination rate map you can use it by passing a `msprime.RateMap`. 

https://tskit.dev/msprime/docs/stable/rate_maps.html

```{code-cell} ipython3
import msprime
rate_map = msprime.RateMap(
    position=[0, 10_000_000, sequence_length],
    rate=[1e-8, 2e-8],  # cold spot, then hot spot
)
result = bayesld.data_from_vcf(
    vcf_path="tutorial_vcf_data/chr1.bcf",
    recombination_rate=rate_map,
    left_bins_morgan=left_bins_morgan,
    right_bins_morgan=right_bins_morgan,
    contig="1",
    start_bp=0,
    end_bp=20000000,
)
```

If you have your recombination rate map in HapMap format, you can convert it into a msprime.RateMap using `msprime.RateMap.read_hapmap` or, with the command line, with the `--hapmap` flag. 

+++

## Adjusting the mutation rate for missing data

Genetic diversity scales with the mutation rate: $\pi \approx 4 N_e \mu$ under neutrality. If a fraction of your callable sequence is missing (low coverage, strict filters, or masked regions), this will result in a reduced $\pi$ estimate. 

A simple correction when passing `mutation_rate` to `.with_data()` is to scale it by the fraction of sequence that is actually callable. For example, if roughly 20% of the window is missing or uncallable, use a mutation rate 20% lower than your literature estimate:

```{code-cell} ipython3
mutation_rate = 1e-8
missing_fraction = 0.20
effective_mutation_rate = mutation_rate * (1.0 - missing_fraction)
effective_mutation_rate
```
