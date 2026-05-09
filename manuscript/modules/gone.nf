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
    tuple val(name), path(vcf_gz), val(rec_rate), val(samples)

    output:
    tuple val(name), path("${name}_GONE_Ne"), path("${name}_GONE_d2"), path("${name}_GONE_STATS")

    script:
    def samples_arg = samples ? "--samples ${samples}" : ""
    """
    # Decompress VCF with optional sample subsetting
    bcftools view ${samples_arg} ${vcf_gz} -O v -o ${name}.vcf

    # Run GONE2
    chmod +x ${gone2}
    ./${gone2} -S 25376 -g 0 -r ${rec_rate} -t ${task.cpus} -o ${name} ${name}.vcf

    # Clean up temporary VCF
    rm ${name}.vcf
    """
}
