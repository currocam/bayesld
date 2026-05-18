// ── Lizards (P. siculus) ─────────────────────────────────────────────────────

include { GONE2_COMPILE; GONE2_VCF; GONE2_RUN } from './gone'
include { HAPNE_MAPS; HAPNE_VCF; HAPNE_RUN } from './hapne'
include { PHLASH_RUN } from './phlash'

params.lizards_vcf     = "${projectDir}/data/psiculus/psiculus_inbreeding.no_if1.sf_stringent1.pass.snps.biallelic.autosomes.vcf.gz"
params.PM_SAMPLES       = "24PM01,24PM02,24PM03,24PM04,24PM05,24PM06,24PM07,24PM08,24PM09,24PM10,24PM11,24PM12,24PM13,24PM14,24PM15,24PM16,24PM17,24PM18,24PM19,24PM20,24PM21,24PM22,24PM23,24PM24,24PM25,24PM26,24PM27,24PM28,24PM29,24PM30,24PM31,24PM32"
params.lizards_rec_rate = 0.63 // cM/Mb
params.lizards_sequence_report = "${projectDir}/data/psiculus/sequence_report.tsv"

workflow LIZARDS {
    main:
    def vcf_file = file(params.lizards_vcf)
    def seq_report = file(params.lizards_sequence_report)

    // ── GONE2 ──
    gone2 = GONE2_COMPILE()

    gone_input = Channel.of(
        ["psiculus", vcf_file, params.lizards_rec_rate, params.PM_SAMPLES, true]
    )

    vcf_ch = GONE2_VCF(gone_input)
    GONE2_RUN(gone2, vcf_ch)

    // ── HapNe-LD ──
    maps_ch = HAPNE_MAPS(Channel.of(["psiculus", vcf_file, params.lizards_rec_rate]))

    // Fan out chroms.txt into parallel per-chromosome VCF extraction jobs
    hapne_vcf_input = maps_ch.flatMap { name, maps, rec_rate, chroms_file ->
        chroms_file.text.readLines().collect { line ->
            def chrom = line.split(/\s+/)[0]
            tuple(name, chrom, vcf_file, params.PM_SAMPLES)
        }
    }
    per_chrom_vcfs = HAPNE_VCF(hapne_vcf_input)

    // Collect all per-chrom VCFs, combine with maps, and run HapNe
    hapne_run_input = per_chrom_vcfs
        .map { name, vcf, tbi -> tuple(name, vcf, tbi) }
        .groupTuple()
        .combine(maps_ch.map { name, maps, rec_rate, chroms -> tuple(name, maps, rec_rate) }, by: 0)
        .map { name, vcf_list, tbi_list, maps, rec_rate ->
            tuple(name, maps, vcf_list + tbi_list, rec_rate)
        }

    HAPNE_RUN(hapne_run_input)

    // ── phlash ──
    def vcf_tbi = file("${params.lizards_vcf}.tbi")
    def vcf_csi = file("${params.lizards_vcf}.csi")
    PHLASH_RUN(
        Channel.of(["psiculus", vcf_file, vcf_tbi, vcf_csi, seq_report, params.PM_SAMPLES])
    )

    emit:
    gone_results   = GONE2_RUN.out
    hapne_results  = HAPNE_RUN.out
    phlash_results = PHLASH_RUN.out
}
