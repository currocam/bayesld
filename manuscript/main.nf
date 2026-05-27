nextflow.enable.dsl = 2

include { CONCEPTUAL_DATA; CONCEPTUAL_PLOTS } from './modules/conceptual'
include { EXAMPLE_BIAS_DATA; EXAMPLE_BIAS_PLOT } from './modules/example_bias'
include { ERROR_CONSTANT_DATA; ERROR_CONSTANT_PLOT; ERROR_CONSTANT } from './modules/error_constant'
include { HOLSTEINFRIESIAN_DATA; VAQUITA_DATA; CANISFAMILIARIS_DATA; STDPOPSIM_PLOTS } from './modules/stdpopsim'
include { SBC_CONSTANT; SBC_PIECEWISE_CONSTANT; SBC_PIECEWISE_EXPONENTIAL; SBC_PLOT } from './modules/sbc'
include { LIZARDS } from './modules/lizards'

workflow {
    data_ch = CONCEPTUAL_DATA()
    CONCEPTUAL_PLOTS(data_ch)

    bias_ch = EXAMPLE_BIAS_DATA()
    EXAMPLE_BIAS_PLOT(bias_ch)

    ERROR_CONSTANT()

    holsteinfriesian_ch = HOLSTEINFRIESIAN_DATA()
    vaquita_ch = VAQUITA_DATA()
    canisfamiliaris_ch = CANISFAMILIARIS_DATA()
    STDPOPSIM_PLOTS(holsteinfriesian_ch, vaquita_ch, canisfamiliaris_ch)

    sbc_constant_ch = SBC_CONSTANT()
    sbc_pc_ch = SBC_PIECEWISE_CONSTANT()
    sbc_pe_ch = SBC_PIECEWISE_EXPONENTIAL()

    SBC_PLOT(
        sbc_constant_ch.results
            .mix(sbc_pc_ch.results)
            .mix(sbc_pe_ch.results)
            .flatten()
    )
}

workflow conceptual {
    CONCEPTUAL_PLOTS(CONCEPTUAL_DATA())
}

workflow example_bias {
    EXAMPLE_BIAS_PLOT(EXAMPLE_BIAS_DATA())
}

workflow error_constant {
    ERROR_CONSTANT()
}

workflow stdpopsim {
    STDPOPSIM_PLOTS(HOLSTEINFRIESIAN_DATA(), VAQUITA_DATA(), CANISFAMILIARIS_DATA())
}

workflow sbc {
    sbc_constant_ch = SBC_CONSTANT()
    sbc_pc_ch = SBC_PIECEWISE_CONSTANT()
    sbc_pe_ch = SBC_PIECEWISE_EXPONENTIAL()
    SBC_PLOT(
        sbc_constant_ch.results
            .mix(sbc_pc_ch.results)
            .mix(sbc_pe_ch.results)
            .flatten()
    )
}

workflow lizards {
    LIZARDS()
}
