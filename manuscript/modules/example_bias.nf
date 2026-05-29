process EXAMPLE_BIAS_DATA {
    label 'simulation'

    output:
    path "example_bias_data.pkl"

    script:
    """
    # v3 — 5 scenarios x 4 variants, per-scenario SMC(k) + DTWF
    example_bias_data.py example_bias_data.pkl
    """
}

process EXAMPLE_BIAS_PLOT {
    label 'plotting'

    publishDir "${params.figures_dir}/example_bias", mode: 'copy'

    input:
    path pkl

    output:
    path "*.pdf"
    path "*.pgf"

    script:
    """
    # v3 — demography + SMC + DTWF panels per scenario, PDF + PGF
    example_bias_plot.py --pkl ${pkl}
    """
}
