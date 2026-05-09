# /// script
# dependencies = [
#     "bayesld==0.1.0",
#     "cyvcf2==0.32.1",
#     "joblib==1.5.3",
#     "marimo",
#     "matplotlib==3.10.9",
#     "numpy==2.4.4",
#     "polars==1.40.1",
#     "pyarrow==24.0.0",
#     "seaborn==0.13.2",
#     "tqdm==4.67.3",
# ]
# requires-python = ">=3.14"
#
# [tool.uv.sources]
# bayesld = { git = "https://github.com/currocam/bayesld.git", rev = "c29b284ece204789126a4528a121aa30bd654c31" }
# ///

import marimo

__generated_with = "0.23.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import bayesld

    return bayesld, mo, pl


@app.cell
def _():
    (0.67+0.59)/2
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Recombination rate

    In the absence of a recombination rate estimate for _Podarcis siculus_, I assume a recombination rate of 0.63 Mbp/cM (the sex-average of _Zootoca vivipara_).

    https://academic.oup.com/gbe/article/12/11/1953/5896527
    """)
    return


@app.cell
def _():
    # Estimated as:
    1e-8 / (0.67+0.59)/2
    return


@app.cell
def _():
    recombination_rate = 1.59e-8
    return (recombination_rate,)


@app.cell
def _(pl):
    df_seq = pl.read_csv("data/psiculus/sequence_report.tsv", separator="\t")
    df_seq = df_seq.filter(pl.col("Sequence name").str.starts_with("SUPER"))
    df_seq
    return (df_seq,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This rough estimate gives on average 1 crossover per meiosis and chromosome, which is reasonable.
    """)
    return


@app.cell
def _(df_seq, recombination_rate):
    (df_seq["Seq length"] * recombination_rate).mean()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Binning scheme

    I will use the default binning scheme which is also a reasonable choice for this model in terms of number of windows.
    """)
    return


@app.cell
def _(bayesld, recombination_rate):
    left_bins, right_bins = bayesld.linear_bins()
    window_length = right_bins[-1]*2 / recombination_rate
    window_length
    return left_bins, right_bins, window_length


@app.cell
def _(df_seq, pl, window_length):
    _rows = []
    for _row in df_seq.iter_rows(named=True):
      chr_name = _row["GenBank seq accession"]
      chr_len = _row["Seq length"]
      n_windows = int(chr_len // window_length)
      offset = (chr_len - n_windows * window_length) / 2
      for i in range(n_windows):
          start = int(offset + i * window_length)
          end = int(offset + (i + 1) * window_length)
          _rows.append({"chromosome": chr_name, "start": start, "end": end})

    df_windows = pl.DataFrame(_rows)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, len(df_seq) * 0.5 + 1))

    chroms = df_seq["GenBank seq accession"].to_list()
    lengths = df_seq["Seq length"].to_list()

    for i, (chrom, length) in enumerate(zip(chroms, lengths)):
      ax.barh(i, length, height=0.6, color="lightgray", edgecolor="black", linewidth=0.5)
      windows = df_windows.filter(pl.col("chromosome") == chrom)
      for _row in windows.iter_rows(named=True):
          ax.barh(i, _row["end"] - _row["start"], left=_row["start"], height=0.6, color="steelblue", edgecolor="black",
    linewidth=0.5)

    ax.set_yticks(range(len(chroms)))
    ax.set_yticklabels(chroms)
    ax.set_xlabel("Position (bp)")
    ax.invert_yaxis()
    fig.tight_layout()
    fig
    return (df_windows,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Data collection

    I measure $\pi$ and LD for samples from the island Pod Mrčaru and Pod Kopište
    """)
    return


@app.cell
def _():
    in_vcf = "data/psiculus/psiculus_inbreeding.no_if1.sf_stringent1.pass.snps.biallelic.autosomes.vcf.gz"
    return (in_vcf,)


@app.cell
def _():
    PM_SAMPLES = [f"24PM{str(i).zfill(2)}" for i in range(1, 32+1)]
    return (PM_SAMPLES,)


@app.cell
def _():
    PK_SAMPLES = "24PK01,24PK02,24PK03,24PK04,24PK05,24PK06,24PK07,24PK08,24PK09,24PK10,24PK11,24PK12,24PK13,24PK14,24PK15,24PK16,24PK17,24PK18,24PK19,24PK20,24PK21,24PK22,24PK23,24PK24,24PK25,24PK26,24PK27,24PK28,24PK29,24PK30,24PK31,24PK32".split(",")
    return (PK_SAMPLES,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## PM samples
    """)
    return


@app.cell
def _():
    import joblib
    from tqdm import tqdm

    num_workers = 8
    return joblib, num_workers, tqdm


@app.cell
def _(
    PM_SAMPLES,
    bayesld,
    df_windows,
    in_vcf,
    joblib,
    left_bins,
    num_workers,
    recombination_rate,
    right_bins,
    tqdm,
):
    def _process_window(row):
        return bayesld.data_from_vcf(
            vcf_path=in_vcf,
            recombination_rate=recombination_rate,
            left_bins_morgan=left_bins,
            right_bins_morgan=right_bins,
            contig=row["chromosome"],
            start_bp=row["start"],
            end_bp=row["end"],
            chunk_size=2000,
            samples=PM_SAMPLES,
            progress_bar=False,
        )

    rows = list(df_windows.iter_rows(named=True))
    results = [
        r for r in tqdm(
            joblib.Parallel(return_as="generator", n_jobs=num_workers)(
                joblib.delayed(_process_window)(row) for row in rows
            ),
            total=len(rows),
        )
    ]
    return (results,)


@app.cell
def _():
    import pickle

    return (pickle,)


@app.cell
def _(df_windows, np, results):
    mask = ~df_windows["chromosome"].is_in(("OZ076856.1", "OZ076852.1"))
    pm_data = {
        "sample_size" : results[0]["sample_size"],
        "left_bins_morgan" : results[0]["left_bins_morgan"],
        "right_bins_morgan" : results[0]["right_bins_morgan"],
        "mean_linkage_disequilibrium" : np.array([_x["mean_linkage_disequilibrium"] for _x in results])[mask],
        "num_pairs_linkage_disequilibrium" : np.array([_x["num_pairs_linkage_disequilibrium"] for _x in results])[mask],
        "mean_genetic_diversity" : np.array([_x["mean_genetic_diversity"] for _x in results])[mask],
        "num_sites_genetic_diversity" : np.array([_x["num_sites_genetic_diversity"] for _x in results])[mask],
        "windows" : df_windows.filter(mask).to_numpy()
    }
    pm_data
    return mask, pm_data


@app.cell
def _(pickle, pm_data):
    with open("analysis/psiculus/PM_data.pkl", "wb") as f:
      pickle.dump(pm_data, f)
    return


@app.cell
def _():
    import numpy as np
    import seaborn as sns

    return np, sns


@app.cell
def _(pm_data, sns):
    sns.boxplot(
        x = pm_data["mean_genetic_diversity"],
        y = pm_data["windows"][:, 0]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## PK samples
    """)
    return


@app.cell
def _(
    PK_SAMPLES,
    bayesld,
    df_windows,
    in_vcf,
    joblib,
    left_bins,
    num_workers,
    recombination_rate,
    right_bins,
    tqdm,
):
    def _process_window(row):
        return bayesld.data_from_vcf(
            vcf_path=in_vcf,
            recombination_rate=recombination_rate,
            left_bins_morgan=left_bins,
            right_bins_morgan=right_bins,
            contig=row["chromosome"],
            start_bp=row["start"],
            end_bp=row["end"],
            chunk_size=2000,
            samples=PK_SAMPLES,
            progress_bar=False,
        )

    rows2 = list(df_windows.iter_rows(named=True))
    results2 = [
        r for r in tqdm(
            joblib.Parallel(return_as="generator", n_jobs=num_workers)(
                joblib.delayed(_process_window)(row2) for row2 in rows2
            ),
            total=len(rows2),
        )
    ]
    return (results2,)


@app.cell
def _(df_windows, mask, np, results2):
    pk_data = {
        "sample_size" : results2[0]["sample_size"],
        "left_bins_morgan" : results2[0]["left_bins_morgan"],
        "right_bins_morgan" : results2[0]["right_bins_morgan"],
        "mean_linkage_disequilibrium" : np.array([_x["mean_linkage_disequilibrium"] for _x in results2])[mask],
        "num_pairs_linkage_disequilibrium" : np.array([_x["num_pairs_linkage_disequilibrium"] for _x in results2])[mask],
        "mean_genetic_diversity" : np.array([_x["mean_genetic_diversity"] for _x in results2])[mask],
        "num_sites_genetic_diversity" : np.array([_x["num_sites_genetic_diversity"] for _x in results2])[mask],
        "windows" : df_windows.filter(mask).to_numpy()
    }
    pk_data
    return (pk_data,)


@app.cell
def _(pk_data, sns):
    sns.boxplot(
        x = pk_data["mean_genetic_diversity"],
        y = pk_data["windows"][:, 0]
    )
    return


@app.cell
def _(pickle, pk_data):
    with open("analysis/psiculus/PK_data.pkl", "wb") as f2:
      pickle.dump(pk_data, f2)
    return


if __name__ == "__main__":
    app.run()
