process CONCEPTUAL_DATA {
    label 'simulation'

    output:
    path "conceptual_data.pkl"

    script:
    """
    conceptual_data.py conceptual_data.pkl
    """
}

process CONCEPTUAL_PLOTS {
    label 'plotting'

    publishDir "${params.figures_dir}/conceptual", mode: 'copy'

    input:
    path pkl

    output:
    path "*.pdf"
    path "*.pgf"

    script:
    """
    conceptual_plot.py --pkl ${pkl}
    """
}
