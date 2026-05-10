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

process GONE2_RUN {
    label 'gone'

    publishDir "${params.lizards_dir}/gone", mode: 'copy'

    input:
    path gone2
    tuple val(name), path(vcf_gz), val(rec_rate), val(samples), val(subsample)

    output:
    tuple val(name), path("${name}_GONE_Ne"), path("${name}_GONE_d2"), path("${name}_GONE_STATS")

    script:
    def samples_arg = samples ? "--samples ${samples}" : ""
    def max_loci = 1999999
    """
    if [ "${subsample}" = "true" ]; then
        # Header
        bcftools view ${samples_arg} -h ${vcf_gz} > ${name}.unsorted.vcf
        # Random subsample of ${max_loci} SNPs (https://www.biostars.org/p/9550551/)
        bcftools view ${samples_arg} -H ${vcf_gz} \
            | awk 'BEGIN{srand(25376)} {printf("%f\\t%s\\n",rand(),\$0)}' \
            | (sort -t \$'\\t' -T . -k1,1g || true) \
            | head -n ${max_loci} \
            | cut -f 2- >> ${name}.unsorted.vcf
        bcftools sort -o ${name}.vcf ${name}.unsorted.vcf
        rm ${name}.unsorted.vcf
    else
        bcftools view ${samples_arg} ${vcf_gz} -O v -o ${name}.vcf
    fi

    # Run GONE2
    chmod +x ${gone2}
    ./${gone2} -S 25376 -g 0 -r ${rec_rate} -t ${task.cpus} -o ${name} ${name}.vcf

    # Clean up temporary VCF
    rm ${name}.vcf
    """
}
