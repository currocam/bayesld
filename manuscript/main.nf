nextflow.enable.dsl = 2

include { CONCEPTUAL_DATA; CONCEPTUAL_PLOTS } from './modules/conceptual'
include { EXAMPLE_BIAS_DATA; EXAMPLE_BIAS_PLOT } from './modules/example_bias'

workflow {
    data_ch = CONCEPTUAL_DATA()
    CONCEPTUAL_PLOTS(data_ch)

    bias_ch = EXAMPLE_BIAS_DATA()
    EXAMPLE_BIAS_PLOT(bias_ch)
}
