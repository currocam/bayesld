// ── GONE2 ────────────────────────────────────────────────────────────────────

process GONE2_COMPILE {
    label 'compilation'

    output:
    path "gone2"

    script:
    """
    git clone https://github.com/esrud/GONE2
    cd GONE2
    git checkout d26797e
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
    # Remove chromosomes shorter than 20 cM (GONE2 requirement)
    min_bp=\$(awk -v r=${rec_rate} 'BEGIN{printf "%d", 20 / r * 1e6}')
    bcftools view -h ${vcf_gz} | grep '^##contig' \
        | sed 's/.*ID=\\([^,]*\\),length=\\([0-9]*\\).*/\\1 \\2/' \
        | awk -v min="\${min_bp}" '\$2 >= min {print \$1}' \
        > keep_chroms.txt
    echo "Kept \$(wc -l < keep_chroms.txt) chromosomes >= 20 cM (\${min_bp} bp at ${rec_rate} cM/Mb)" >&2
    REGIONS=\$(paste -sd, keep_chroms.txt)

    if [ "${subsample}" = "true" ]; then
        # Count total SNPs (after chromosome filter)
        n_total=\$(bcftools view ${samples_arg} -t \${REGIONS} -H ${vcf_gz} | wc -l)
        if [ "\${n_total}" -le "${max_loci}" ]; then
            bcftools view ${samples_arg} -t \${REGIONS} ${vcf_gz} -O v -o ${name}.vcf
        else
            # Retain each SNP with probability max_loci/n_total (single pass, already sorted)
            bcftools view ${samples_arg} -t \${REGIONS} ${vcf_gz} \
                | awk -v n_total="\${n_total}" -v max_loci="${max_loci}" \
                    'BEGIN{srand(25376); p=max_loci/n_total; k=0} /^#/{print; next} k < max_loci && rand() < p {print; k++} END{print k" SNPs retained out of "n_total > "/dev/stderr"}' \
                > ${name}.vcf
        fi
    else
        bcftools view ${samples_arg} -t \${REGIONS} ${vcf_gz} -O v -o ${name}.vcf
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
    tuple val(name), path("${name}_GONE2_Ne"), path("${name}_GONE2_d2"), path("${name}_GONE2_STATS")

    script:
    """
    chmod +x ${gone2}
    ./${gone2} -S 25376 -g 0 -r ${rec_rate} -t ${task.cpus} -o ${name} ${vcf}
    """
}
