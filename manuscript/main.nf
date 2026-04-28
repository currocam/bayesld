nextflow.enable.dsl = 2

include { CONCEPTUAL_DATA; CONCEPTUAL_PLOTS } from './modules/conceptual'

workflow {
    data_ch = CONCEPTUAL_DATA()
    CONCEPTUAL_PLOTS(data_ch)
}
