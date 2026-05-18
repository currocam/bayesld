// ── phlash ──────────────────────────────────────────────────────────────────

// Parse VCF and fit phlash model, extract posterior Ne trajectories (GPU).
process PHLASH_RUN {
    label 'phlash_run'

    publishDir "${params.lizards_dir}/phlash", mode: 'copy'

    input:
    tuple val(name), path(vcf_gz), path(vcf_tbi), path(vcf_csi), path(sequence_report), val(samples)

    output:
    tuple val(name), path("phlash_fit.pkl"), path("phlash_posteriors.npz")

    script:
    """
    lizards_phlash.py ${vcf_gz} ${sequence_report} ${samples} phlash_fit.pkl phlash_posteriors.npz
    """
}
