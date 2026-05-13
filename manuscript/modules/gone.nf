// ── GONE2 ────────────────────────────────────────────────────────────────────

process GONE2_COMPILE {
    label 'compilation'

    output:
    path "gone2"

    script:
    """
    git clone https://github.com/esrud/GONE2
    cd GONE2
    make gone
    cp gone2 ../
    """
}

process GONE2_VCF {
    label 'gone_vcf'

    input:
    tuple val(name), path(vcf_gz), val(rec_rate), val(samples), val(subsample)

    output:
    tuple val(name), path("${name}.vcf"), val(rec_rate)

    script:
    def samples_arg = samples ? "--samples ${samples}" : ""
    def max_loci = 1999999
    """
    if [ "${subsample}" = "true" ]; then
        # Count total SNPs
        n_total=\$(bcftools view ${samples_arg} -H ${vcf_gz} | wc -l)
        if [ "\${n_total}" -le "${max_loci}" ]; then
            # No subsampling needed
            bcftools view ${samples_arg} ${vcf_gz} -O v -o ${name}.vcf
        else
            # Retain each SNP with probability max_loci/n_total (single pass, already sorted)
            bcftools view ${samples_arg} ${vcf_gz} \
                | awk -v n_total="\${n_total}" -v max_loci="${max_loci}" \
                    'BEGIN{srand(25376); p=max_loci/n_total; k=0} /^#/{print; next} k < max_loci && rand() < p {print; k++} END{print k" SNPs retained out of "n_total > "/dev/stderr"}' \
                > ${name}.vcf
        fi
    else
        bcftools view ${samples_arg} ${vcf_gz} -O v -o ${name}.vcf
    fi
    """
}

process GONE2_RUN {
    label 'gone'

    publishDir "${params.lizards_dir}/gone", mode: 'copy'

    input:
    path gone2
    tuple val(name), path(vcf), val(rec_rate)

    output:
    tuple val(name), path("${name}_GONE_Ne"), path("${name}_GONE_d2"), path("${name}_GONE_STATS")

    script:
    """
    chmod +x ${gone2}
    ./${gone2} -S 25376 -g 0 -r ${rec_rate} -t ${task.cpus} -o ${name} ${vcf}
    """
}
