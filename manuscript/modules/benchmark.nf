include { GONE2_COMPILE } from './gone'

process BENCHMARK_SIMULATE {
    label 'simulation'

    input:
    tuple val(name), val(script)

    output:
    tuple val(name), path("${name}.vcf.gz"), path("${name}.vcf.gz.csi"), path("${name}.demes.yaml")

    script:
    def g = params.benchmark_genome
    """
    ${projectDir}/bin/benchmark/${script} ${g.num_chroms} ${g.chrom_length_bp} ${g.recombination_rate} ${g.mutation_rate} ${g.num_individuals} ${g.seed} ${name}
    """
}

process BENCHMARK_GONE2_VCF {
    label 'gone_vcf'

    input:
    tuple val(name), path(vcf_gz), path(vcf_csi), val(rec_rate), val(samples)

    output:
    tuple val(name), path("${name}.vcf"), val(rec_rate)

    script:
    def samples_arg = samples ? "--samples ${samples}" : ""
    // This might fail if chromosomes are too short or have too many variants.
    """
    bcftools view ${samples_arg} ${vcf_gz} -O v -o ${name}.vcf
    """
}

process BENCHMARK_GONE2_RUN {
    label 'gone'

    publishDir { "${params.benchmark_dir}/${name}/gone" }, mode: 'copy'

    input:
    path gone2
    tuple val(name), path(vcf), val(rec_rate)

    output:
    tuple val(name), path("${name}_GONE2_Ne"), path("${name}_GONE2_d2"), path("${name}_GONE2_STATS")

    script:
    """
    chmod +x ${gone2}
    # The simulated VCF is phased, but we pretend it's not.
    ./${gone2} -S 12345 -g 0 -r ${rec_rate} -t ${task.cpus} -o ${name} ${vcf}
    """
}

process BENCHMARK_HAPNE_MAPS {
    label 'hapne_prepare'

    input:
    tuple val(name), path(vcf_gz), path(vcf_csi), val(rec_rate)

    output:
    tuple val(name), path("maps/*.shapeit.map"), val(rec_rate), path("chroms.txt")

    script:
    """
    mkdir maps

    bcftools view -h ${vcf_gz} | grep '^##contig' \
        | sed 's/.*ID=\\([^,]*\\),length=\\([0-9]*\\).*/\\1 \\2/' \
        > chroms.txt

    while read chrom length; do
        genetic_length=\$(awk -v l="\${length}" -v r="${rec_rate}" 'BEGIN{printf "%.6f", l * r / 1e6}')
        printf "position COMBINED_rate(cM/Mb) Genetic_Map(cM)\\n" > "maps/\${chrom}.shapeit.map"
        printf "0 ${rec_rate} 0\\n" >> "maps/\${chrom}.shapeit.map"
        printf "%s ${rec_rate} %s\\n" "\${length}" "\${genetic_length}" >> "maps/\${chrom}.shapeit.map"
    done < chroms.txt
    """
}

process BENCHMARK_HAPNE_VCF {
    label 'hapne_prepare'

    input:
    tuple val(name), val(chrom), path(vcf_gz), path(vcf_csi), val(samples)

    output:
    tuple val(name), path("${chrom}.vcf.gz"), path("${chrom}.vcf.gz.tbi")

    script:
    def samples_arg = samples ? "--samples ${samples}" : ""
    """
    bcftools view ${samples_arg} -t ${chrom} ${vcf_gz} -O z -o ${chrom}.vcf.gz
    bcftools index --tbi ${chrom}.vcf.gz
    """
}

process BENCHMARK_HAPNE_RUN {
    label 'hapne_run'

    publishDir { "${params.benchmark_dir}/${name}/hapne" }, mode: 'copy'

    input:
    tuple val(name), path(maps), path(vcfs), val(rec_rate)

    output:
    tuple val(name), path("results/ld_hapne_estimate.csv"), path("results/ld_hapne_pop_trajectory.png")

    script:
    def python_deps = "pandas>=2.2.3,pandas-plink>=2.3.1,numba>=0.61.0,scipy>=1.15.2,matplotlib>=3.10.1,scikit-learn>=1.6.1,pyyaml>=6.0.2"
    """
    WORKDIR=\$(pwd)

    git clone https://github.com/currocam/hapne-snakemake
    cd hapne-snakemake
    mkdir data
    for f in ${vcfs}; do
        case "\${f}" in *.tbi) continue ;; esac
        chrom=\$(basename "\${f}" .vcf.gz)
        ln -s "\${WORKDIR}/\${f}" data/
        ln -s "\${WORKDIR}/\${f}.tbi" data/
        ln -s "\${WORKDIR}/\${chrom}.shapeit.map" data/ 2>/dev/null || true
    done

    cat > config.yaml <<EOF
data_dir: "data/"
out_dir: "results/"
map_file_suffix: ".shapeit.map"
method: "ld"
maf_threshold: 0.25
nb_points: 1000000
apply_filter: false
EOF

    uvx --with "${python_deps}" snakemake -c ${task.cpus} --configfile config.yaml --shadow-prefix ${params.shadow_dir}
    cp -r results \${WORKDIR}/results
    """
}

// ── bayesld ──

process BENCHMARK_BAYESLD_DATA {
    label 'simulation'

    input:
    tuple val(name), path(vcf_gz), path(vcf_csi)

    output:
    tuple val(name), path("${name}_data.pkl")

    script:
    """
    ${projectDir}/bin/benchmark/bayesld_data.py ${vcf_gz} ${params.benchmark_genome.recombination_rate} ${task.cpus} ${name}_data.pkl
    """
}

process BENCHMARK_FIT {
    label 'inference'

    input:
    tuple val(name), val(model), path(data_pkl), val(fit_script)

    output:
    tuple val(name), val(model), path("${name}_${model}.nc")

    script:
    """
    ${projectDir}/bin/benchmark/${fit_script} ${data_pkl} ${params.benchmark_genome.recombination_rate} ${params.benchmark_genome.mutation_rate} ${task.cpus} ${name}_${model}.nc
    """
}

// ── plot ──

process BENCHMARK_PLOT {
    label 'plotting'

    publishDir { "${params.benchmark_dir}/${name}" }, mode: 'copy'

    input:
    tuple val(name), path(demes_yaml), val(models), path(ncs), path(gone_ne), path(hapne_csv)

    output:
    tuple val(name), path("${name}.pdf"), path("${name}.pgf"), path("${name}_prior.pdf"), path("${name}_prior.pgf")

    script:
    def model_ncs = [models, ncs].transpose().collect { m, nc -> "${m}=${nc}" }.join(' ')
    """
    ${projectDir}/bin/benchmark/plot.py ${name} ${demes_yaml} ${gone_ne} ${hapne_csv} ${name} ${model_ncs}
    """
}

workflow BENCHMARK {
    main:
    def rec_rate_cm_mb = params.benchmark_genome.recombination_rate * 1e8

    scenarios = Channel.fromList(params.benchmark_scenarios)
    // One (name, model, fit_script) tuple per fit_script in the scenario, so
    // each scenario can be fit against several bayesld models.
    fit_scripts = scenarios.flatMap { s ->
        s.fit_scripts.collect { fs ->
            def model = fs.replaceFirst(/^fit_/, '').replaceFirst(/\.py$/, '')
            tuple(s.name, model, fs)
        }
    }

    sim = BENCHMARK_SIMULATE(scenarios.map { s -> tuple(s.name, s.simulate_script) }) // (name, vcf, csi, demes)

    // ── GONE2 ──
    gone2 = GONE2_COMPILE().first()

    gone_vcf_input = sim.map { name, vcf, csi, demes -> tuple(name, vcf, csi, rec_rate_cm_mb, null) }
    gone_vcf_ch = BENCHMARK_GONE2_VCF(gone_vcf_input)
    gone_ch = BENCHMARK_GONE2_RUN(gone2, gone_vcf_ch) // (name, Ne, d2, STATS)

    // ── HapNe-LD ──
    maps_ch = BENCHMARK_HAPNE_MAPS(
        sim.map { name, vcf, csi, demes -> tuple(name, vcf, csi, rec_rate_cm_mb) }
    )

    hapne_vcf_input = maps_ch
        .combine(sim.map { name, vcf, csi, demes -> tuple(name, vcf, csi) }, by: 0)
        .flatMap { name, maps, rec_rate, chroms_file, vcf, csi ->
            chroms_file.text.readLines().collect { line ->
                def chrom = line.split(/\s+/)[0]
                tuple(name, chrom, vcf, csi, null)
            }
        }
    per_chrom_vcfs = BENCHMARK_HAPNE_VCF(hapne_vcf_input)

    hapne_run_input = per_chrom_vcfs
        .map { name, vcf, tbi -> tuple(name, vcf, tbi) }
        .groupTuple()
        .combine(maps_ch.map { name, maps, rec_rate, chroms -> tuple(name, maps, rec_rate) }, by: 0)
        .map { name, vcf_list, tbi_list, maps, rec_rate -> tuple(name, maps, vcf_list + tbi_list, rec_rate) }

    hapne_ch = BENCHMARK_HAPNE_RUN(hapne_run_input) // (name, csv, png)

    // ── bayesld ──
    data_ch = BENCHMARK_BAYESLD_DATA(sim.map { name, vcf, csi, demes -> tuple(name, vcf, csi) })
    fit_input = data_ch
        .combine(fit_scripts, by: 0) // (name, data_pkl, model, fit_script)
        .map { name, data_pkl, model, fit_script -> tuple(name, model, data_pkl, fit_script) }
    bayesld_ch = BENCHMARK_FIT(fit_input) // (name, model, nc)

    // Regroup the per-model fits back into one (name, [model,...], [nc,...])
    // tuple per scenario, so the plot overlays every model fit to it.
    bayesld_grouped = bayesld_ch.groupTuple(by: 0)

    // ── join everything back by scenario name for the comparison plot ──
    plot_input = sim.map { name, vcf, csi, demes -> tuple(name, demes) }
        .combine(bayesld_grouped, by: 0)
        .combine(gone_ch.map { name, ne, d2, stats -> tuple(name, ne) }, by: 0)
        .combine(hapne_ch.map { name, csv, png -> tuple(name, csv) }, by: 0)

    BENCHMARK_PLOT(plot_input)

    emit:
    gone_results     = gone_ch
    hapne_results    = hapne_ch
    bayesld_results  = bayesld_ch
    plots            = BENCHMARK_PLOT.out
}
