// ── phlash ──────────────────────────────────────────────────────────────────

// Parse VCF into phlash contig objects (CPU-only).
process PHLASH_DATA {
    label 'phlash_data'

    input:
    tuple val(name), path(vcf_gz), path(sequence_report), val(samples)

    output:
    tuple val(name), path("contigs.pkl")

    script:
    """
    lizards_phlash_data.py ${vcf_gz} ${sequence_report} ${samples} contigs.pkl
    """
}

// Fit phlash model and extract posterior Ne trajectories (GPU).
process PHLASH_RUN {
    label 'phlash_run'

    publishDir "${params.lizards_dir}/phlash", mode: 'copy'

    input:
    tuple val(name), path(contigs_pkl)

    output:
    tuple val(name), path("phlash_fit.pkl"), path("phlash_posteriors.npz")

    script:
    """
    lizards_phlash.py ${contigs_pkl} phlash_fit.pkl phlash_posteriors.npz
    """
}
