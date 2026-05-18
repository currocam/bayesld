process ERROR_CONSTANT_DATA {
    label 'simulation'

    tag "r=${recombination_rate}_n=${num_samples}"

    input:
    tuple val(recombination_rate), val(num_samples)

    output:
    path "error_constant_r${recombination_rate}_n${num_samples}.pkl"

    script:
    """
    error_constant.py ${recombination_rate} ${num_samples} error_constant_r${recombination_rate}_n${num_samples}.pkl
    """
}

process ERROR_CONSTANT_PLOT {
    label 'plotting'

    publishDir "${params.figures_dir}/error_constant", mode: 'copy'

    input:
    path pkls

    output:
    path "*.pdf"

    script:
    def pkl_args = pkls.collect { "--pkl ${it}" }.join(' ')
    """
    error_constant_plot.py ${pkl_args}
    """
}

workflow ERROR_CONSTANT {
    combos = Channel.of(
        [1e-8, 100],
        [1e-8, 10],
        [1e-9, 100],
        [1e-9, 10],
    )
    pkls = ERROR_CONSTANT_DATA(combos)
    ERROR_CONSTANT_PLOT(pkls.collect())
}
