process EXAMPLE_BIAS_DATA {
    label 'simulation'

    output:
    path "example_bias_data.pkl"

    script:
    """
    # v2 — SMC + SMC' + DTWF, rtol=0.01
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

    script:
    """
    # v2 — per-panel PDFs, coalescent label per row
    example_bias_plot.py --pkl ${pkl}
    """
}
