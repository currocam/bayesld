// ── Lizards (P. siculus) ─────────────────────────────────────────────────────

include { GONE2_COMPILE; GONE2_VCF; GONE2_RUN } from './gone'

params.lizards_vcf     = "${projectDir}/data/psiculus/psiculus_inbreeding.no_if1.sf_stringent1.pass.snps.biallelic.autosomes.vcf.gz"
params.PM_SAMPLES       = "24PM01,24PM02,24PM03,24PM04,24PM05,24PM06,24PM07,24PM08,24PM09,24PM10,24PM11,24PM12,24PM13,24PM14,24PM15,24PM16,24PM17,24PM18,24PM19,24PM20,24PM21,24PM22,24PM23,24PM24,24PM25,24PM26,24PM27,24PM28,24PM29,24PM30,24PM31,24PM32"
params.lizards_rec_rate = 0.63 // cM/Mb
params.lizards_dir      = "${projectDir}/analysis/psiculus"

workflow LIZARDS {
    main:
    gone2 = GONE2_COMPILE()

    gone_input = Channel.of(
        ["psiculus", file(params.lizards_vcf), params.lizards_rec_rate, params.PM_SAMPLES, true]
    )

    vcf_ch = GONE2_VCF(gone_input)
    GONE2_RUN(gone2, vcf_ch)

    emit:
    gone_results = GONE2_RUN.out
}
