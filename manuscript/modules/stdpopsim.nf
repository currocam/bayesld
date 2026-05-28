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

process CAEELE_DATA {
    label 'simulation'

    output:
    path "caeele.pkl.gz"

    script:
    """
    caeele_data.py caeele
    """
}

process ORYSAT_DATA {
    label 'simulation'

    output:
    path "orysat.pkl.gz"

    script:
    """
    orysat_data.py orysat
    """
}

process DROSEC_DATA {
    label 'simulation'

    output:
    path "drosec.pkl.gz"

    script:
    """
    drosec_data.py drosec
    """
}

process STDPOPSIM_PLOTS {
    label 'plotting'

    publishDir "${params.figures_dir}/stdpopsim", mode: 'copy'

    input:
    path holsteinfriesian_pkl
    path vaquita_pkl
    path caeele_pkl
    path orysat_pkl
    path drosec_pkl

    output:
    path "*.pdf"
    path "*.pgf"

    script:
    """
    stdpopsim_predictions_plot.py
    """
}
