process ERROR_CONSTANT_DATA {
    label 'error_constant'

    tag "r=${recombination_rate}_n=${num_samples}_p=${ploidy}"

    input:
    tuple val(recombination_rate), val(num_samples), val(ploidy)

    output:
    path "error_constant_r${recombination_rate}_n${num_samples}_p${ploidy}.pkl"

    script:
    """
    error_constant.py ${recombination_rate} ${num_samples} ${ploidy} ${task.cpus} error_constant_r${recombination_rate}_n${num_samples}_p${ploidy}.pkl
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
        [1e-8, 20, 1],
        [1e-8, 10, 2],
        [1e-8, 100, 1],
        [1e-8, 50, 2],
    )
    pkls = ERROR_CONSTANT_DATA(combos)
    ERROR_CONSTANT_PLOT(pkls.collect())
}

