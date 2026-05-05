nextflow.enable.dsl = 2

include { CONCEPTUAL_DATA; CONCEPTUAL_PLOTS } from './modules/conceptual'
include { EXAMPLE_BIAS_DATA; EXAMPLE_BIAS_PLOT } from './modules/example_bias'
include { HOLSTEINFRIESIAN_DATA; VAQUITA_DATA; CANISFAMILIARIS_DATA; STDPOPSIM_PLOTS } from './modules/stdpopsim'
include { SBC_CONSTANT; SBC_PIECEWISE_CONSTANT } from './modules/sbc'

workflow {
    data_ch = CONCEPTUAL_DATA()
    CONCEPTUAL_PLOTS(data_ch)

    bias_ch = EXAMPLE_BIAS_DATA()
    EXAMPLE_BIAS_PLOT(bias_ch)

    holsteinfriesian_ch = HOLSTEINFRIESIAN_DATA()
    vaquita_ch = VAQUITA_DATA()
    canisfamiliaris_ch = CANISFAMILIARIS_DATA()
    STDPOPSIM_PLOTS(holsteinfriesian_ch, vaquita_ch, canisfamiliaris_ch)

    SBC_CONSTANT()
    SBC_PIECEWISE_CONSTANT()
}
