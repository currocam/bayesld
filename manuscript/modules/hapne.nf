// ── HapNe-LD ────────────────────────────────────────────────────────────────

// Generate SHAPEIT maps from VCF header (lightweight, no data scan).
// Emits per-chromosome map files + chroms list for parallel VCF extraction.
process HAPNE_MAPS {
    label 'hapne_prepare'

    input:
    tuple val(name), path(vcf_gz), val(rec_rate)

    output:
    tuple val(name), path("maps/*.shapeit.map"), val(rec_rate), path("chroms.txt")

    script:
    """
    mkdir maps

    # Filter chromosomes >= 20 cM
    min_bp=\$(awk -v r=${rec_rate} 'BEGIN{printf "%d", 20 / r * 1e6}')
    bcftools view -h ${vcf_gz} | grep '^##contig' \
        | sed 's/.*ID=\\([^,]*\\),length=\\([0-9]*\\).*/\\1 \\2/' \
        | awk -v min="\${min_bp}" '\$2 >= min {print \$1, \$2}' \
        > chroms.txt
    echo "Kept \$(wc -l < chroms.txt) chromosomes >= 20 cM (\${min_bp} bp at ${rec_rate} cM/Mb)" >&2

    while read chrom length; do
        genetic_length=\$(awk -v l="\${length}" -v r="${rec_rate}" 'BEGIN{printf "%.6f", l * r / 1e6}')
        printf "position COMBINED_rate(cM/Mb) Genetic_Map(cM)\\n" > "maps/\${chrom}.shapeit.map"
        printf "0 ${rec_rate} 0\\n" >> "maps/\${chrom}.shapeit.map"
        printf "%s ${rec_rate} %s\\n" "\${length}" "\${genetic_length}" >> "maps/\${chrom}.shapeit.map"
    done < chroms.txt
    """
}

// Extract a single chromosome from the multi-chrom VCF (parallelized by Nextflow).
process HAPNE_VCF {
    label 'hapne_prepare'

    input:
    tuple val(name), val(chrom), path(vcf_gz), val(samples)

    output:
    tuple val(name), path("${chrom}.vcf.gz"), path("${chrom}.vcf.gz.tbi")

    script:
    def samples_arg = samples ? "--samples ${samples}" : ""
    """
    bcftools view ${samples_arg} -t ${chrom} ${vcf_gz} -O z -o ${chrom}.vcf.gz
    bcftools index ${chrom}.vcf.gz
    """
}

// Run HapNe-LD via snakemake (all deps from modules + uvx).
process HAPNE_RUN {
    label 'hapne_run'

    publishDir "${params.lizards_dir}/hapne", mode: 'copy'

    input:
    tuple val(name), path(maps), path(vcfs), val(rec_rate)

    output:
    tuple val(name), path("results/ld_hapne_estimate.csv"), path("results/ld_hapne_pop_trajectory.png")

    script:
    def python_deps = "pandas>=2.2.3,pandas-plink>=2.3.1,numba>=0.61.0,scipy>=1.15.2,matplotlib>=3.10.1,scikit-learn>=1.6.1,pyyaml>=6.0.2"
    def shadow_dir = "/tmp/vsc21003"
    """
    git clone https://github.com/currocam/hapne-snakemake
    cd hapne-snakemake

    mkdir data
    ln -s \$(readlink -f ${maps}) data/
    ln -s \$(readlink -f ${vcfs}) data/

    cat > config.yaml <<EOF
data_dir: "data/"
out_dir: "results/"
map_file_suffix: ".shapeit.map"
method: "ld"
maf_threshold: 0.25
nb_points: 1000000
apply_filter: false
EOF

    uvx --with "${python_deps}" snakemake -c ${task.cpus} --configfile config.yaml --shadow-prefix ${shadow_dir}
    cp -r results ../results
    """
}
