process HOLSTEINFRIESIAN_DATA {
    label 'simulation'

    output:
    path "holsteinfriesian.pkl.gz"

    script:
    """
    holsteinfriesian_data.py holsteinfriesian
    """
}

process VAQUITA_DATA {
    label 'simulation'

    output:
    path "vaquita.pkl.gz"

    script:
    """
    vaquita_data.py vaquita
    """
}

process CANISFAMILIARIS_DATA {
    label 'simulation'

    output:
    path "canisfamiliaris.pkl.gz"

    script:
    """
    canisfamiliaris_data.py canisfamiliaris
    """
}

process STDPOPSIM_PLOTS {
    label 'plotting'

    publishDir "${params.figures_dir}/stdpopsim", mode: 'copy'

    input:
    path holsteinfriesian_pkl
    path vaquita_pkl
    path canisfamiliaris_pkl

    output:
    path "*.pdf"

    script:
    """
    stdpopsim_predictions_plot.py
    """
}
