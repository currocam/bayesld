nextflow.enable.dsl = 2

include { CONCEPTUAL_DATA; CONCEPTUAL_PLOTS } from './modules/conceptual'
include { EXAMPLE_BIAS_DATA; EXAMPLE_BIAS_PLOT } from './modules/example_bias'
include { ERROR_CONSTANT_DATA; ERROR_CONSTANT_PLOT; ERROR_CONSTANT } from './modules/error_constant'
include { HOLSTEINFRIESIAN_DATA; VAQUITA_DATA; CAEELE_DATA; ORYSAT_DATA; DROSEC_DATA; STDPOPSIM_PLOTS } from './modules/stdpopsim'
include { SBC_CONSTANT; SBC_PIECEWISE_CONSTANT; SBC_PIECEWISE_EXPONENTIAL; SBC_PLOT; SBC_STATS} from './modules/sbc'
include { LIZARDS } from './modules/lizards'
include { BENCHMARK } from './modules/benchmark'

params.only = null

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
    STDPOPSIM_PLOTS(HOLSTEINFRIESIAN_DATA(), VAQUITA_DATA(), CAEELE_DATA(), ORYSAT_DATA(), DROSEC_DATA())
}

workflow sbc {
    sbc_constant_ch = SBC_CONSTANT()
    sbc_pc_ch = SBC_PIECEWISE_CONSTANT()
    sbc_pe_ch = SBC_PIECEWISE_EXPONENTIAL()
    sbc_pkl_ch = sbc_constant_ch.results
        .mix(sbc_pc_ch.results)
        .mix(sbc_pe_ch.results)
        .flatten()
    SBC_PLOT(sbc_pkl_ch)
    SBC_STATS(sbc_pkl_ch)
}

workflow lizards {
    LIZARDS()
}

workflow benchmark {
    BENCHMARK()
}

workflow {
    if (params.only == 'conceptual') {
        conceptual()
    } else if (params.only == 'example_bias') {
        example_bias()
    } else if (params.only == 'error_constant') {
        error_constant()
    } else if (params.only == 'stdpopsim') {
        stdpopsim()
    } else if (params.only == 'sbc') {
        sbc()
    } else if (params.only == 'lizards') {
        lizards()
    } else if (params.only == 'benchmark') {
        benchmark()
    } else {
        conceptual()
        example_bias()
        error_constant()
        stdpopsim()
        sbc()
        lizards()
        benchmark()
    }
}
