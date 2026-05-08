// ── Constant Ne ──────────────────────────────────────────────────────────────

process SBC_CONSTANT_SIMULATE {
    label 'simulation'

    input:
    tuple val(experiment), val(batch_idx)

    output:
    tuple val(experiment), val(batch_idx), path("${experiment.name}_${batch_idx}.pkl")

    script:
    """
    ${projectDir}/bin/sbc/constant/simulate.py \
        ${experiment.name}_${batch_idx}.pkl \
        --prior-ne    ${experiment.prior_ne} \
        --prior-sigma ${experiment.prior_sigma} \
        --batch-size  ${experiment.batch_size} \
        --sample-size ${experiment.sample_size} \
        --num-windows ${experiment.num_windows} \
        --seed        ${321736 + batch_idx}
    """
}

process SBC_CONSTANT_INFER_UNCORRECTED {
    label 'inference'

    input:
    tuple val(experiment), val(batch_idx), path(batch_pkl)

    output:
    tuple val(experiment.name), path("uncorrected_${experiment.name}_${batch_idx}.pkl")

    script:
    """
    ${projectDir}/bin/sbc/constant/infer_uncorrected.py \
        ${batch_pkl} \
        uncorrected_${experiment.name}_${batch_idx}.pkl
    """
}

process SBC_CONSTANT_INFER_CORRECTED {
    label 'inference'

    input:
    tuple val(experiment), val(batch_idx), path(batch_pkl)

    output:
    tuple val(experiment.name), path("corrected_${experiment.name}_${batch_idx}.pkl")

    script:
    """
    ${projectDir}/bin/sbc/constant/infer_corrected.py \
        ${batch_pkl} \
        corrected_${experiment.name}_${batch_idx}.pkl \
        --n-points-per-iter ${experiment.n_points_per_iter} \
        --n-iter            ${experiment.n_iter}
    """
}

process SBC_CONSTANT_INFER_NO_BIAS {
    label 'inference'

    input:
    tuple val(experiment), val(batch_idx), path(batch_pkl)

    output:
    tuple val(experiment.name), path("nobias_${experiment.name}_${batch_idx}.pkl")

    script:
    """
    ${projectDir}/bin/sbc/constant/infer_assume_no_bias.py \
        ${projectDir}/bin/sbc/constant/nobias_model.stan \
        ${batch_pkl} \
        nobias_${experiment.name}_${batch_idx}.pkl
    """
}

process SBC_CONSTANT_COLLECT {
    label 'simulation'

    publishDir "${params.sbc_results_dir}/constant", mode: 'copy'

    input:
    tuple val(name), path(batch_pkls), path(uncorrected_pkls), path(corrected_pkls), path(nobias_pkls)

    output:
    path "${name}.pkl"

    script:
    """
    ${projectDir}/bin/sbc/collect.py \
        ${name}.pkl \
        --batches     ${batch_pkls} \
        --uncorrected ${uncorrected_pkls} \
        --corrected   ${corrected_pkls} \
        --no-bias     ${nobias_pkls}
    """
}

workflow SBC_CONSTANT {
    main:
    SBC_CONSTANT_SIMULATE(
        Channel.fromList(params.sbc_constant)
            .combine(Channel.of(0..<params.sbc_n_batches))
    )

    SBC_CONSTANT_INFER_UNCORRECTED(SBC_CONSTANT_SIMULATE.out)
    SBC_CONSTANT_INFER_CORRECTED(SBC_CONSTANT_SIMULATE.out)
    SBC_CONSTANT_INFER_NO_BIAS(SBC_CONSTANT_SIMULATE.out)

    SBC_CONSTANT_COLLECT(
        SBC_CONSTANT_SIMULATE.out
            .map { experiment, idx, pkl -> [experiment.name, pkl] }.groupTuple()
            .join(SBC_CONSTANT_INFER_UNCORRECTED.out.groupTuple())
            .join(SBC_CONSTANT_INFER_CORRECTED.out.groupTuple())
            .join(SBC_CONSTANT_INFER_NO_BIAS.out.groupTuple())
    )

    emit:
    results = SBC_CONSTANT_COLLECT.out
}

process SBC_PLOT {
    label 'plotting'

    publishDir "${params.figures_dir}/sbc", mode: 'copy'

    input:
    path pkl

    output:
    path "*.pdf"

    script:
    """
    sbc_plot.py --pkl ${pkl}
    """
}

// ── Piecewise Constant (2-epoch) ────────────────────────────────────────────

process SBC_PIECEWISE_CONSTANT_SIMULATE {
    label 'simulation'

    input:
    tuple val(experiment), val(batch_idx)

    output:
    tuple val(experiment), val(batch_idx), path("${experiment.name}_${batch_idx}.pkl")

    script:
    """
    ${projectDir}/bin/sbc/piecewise_constant/simulate.py \
        ${experiment.name}_${batch_idx}.pkl \
        --prior-ne1      ${experiment.prior_ne1} \
        --prior-sigma-ne ${experiment.prior_sigma_ne} \
        --prior-ne2      ${experiment.prior_ne2} \
        --prior-t0       ${experiment.prior_t0} \
        --prior-sigma-t0 ${experiment.prior_sigma_t0} \
        --batch-size     ${experiment.batch_size} \
        --sample-size    ${experiment.sample_size} \
        --num-windows    ${experiment.num_windows} \
        --seed           ${321736 + batch_idx}
    """
}

process SBC_PIECEWISE_CONSTANT_INFER_UNCORRECTED {
    label 'inference'

    input:
    tuple val(experiment), val(batch_idx), path(batch_pkl)

    output:
    tuple val(experiment.name), path("uncorrected_${experiment.name}_${batch_idx}.pkl")

    script:
    """
    ${projectDir}/bin/sbc/piecewise_constant/infer_uncorrected.py \
        ${batch_pkl} \
        uncorrected_${experiment.name}_${batch_idx}.pkl
    """
}

process SBC_PIECEWISE_CONSTANT_INFER_CORRECTED {
    label 'inference'

    input:
    tuple val(experiment), val(batch_idx), path(batch_pkl)

    output:
    tuple val(experiment.name), path("corrected_${experiment.name}_${batch_idx}.pkl")

    script:
    """
    ${projectDir}/bin/sbc/piecewise_constant/infer_corrected.py \
        ${batch_pkl} \
        corrected_${experiment.name}_${batch_idx}.pkl \
        --n-points-per-iter ${experiment.n_points_per_iter} \
        --n-iter            ${experiment.n_iter}
    """
}

process SBC_PIECEWISE_CONSTANT_INFER_NO_BIAS {
    label 'inference'

    input:
    tuple val(experiment), val(batch_idx), path(batch_pkl)

    output:
    tuple val(experiment.name), path("nobias_${experiment.name}_${batch_idx}.pkl")

    script:
    """
    ${projectDir}/bin/sbc/piecewise_constant/infer_assume_no_bias.py \
        ${projectDir}/bin/sbc/piecewise_constant/nobias_model.stan \
        ${batch_pkl} \
        nobias_${experiment.name}_${batch_idx}.pkl
    """
}

process SBC_PIECEWISE_CONSTANT_COLLECT {
    label 'simulation'

    publishDir "${params.sbc_results_dir}/piecewise_constant", mode: 'copy'

    input:
    tuple val(name), path(batch_pkls), path(uncorrected_pkls), path(corrected_pkls), path(nobias_pkls)

    output:
    path "${name}.pkl"

    script:
    """
    ${projectDir}/bin/sbc/collect.py \
        ${name}.pkl \
        --batches     ${batch_pkls} \
        --uncorrected ${uncorrected_pkls} \
        --corrected   ${corrected_pkls} \
        --no-bias     ${nobias_pkls}
    """
}

workflow SBC_PIECEWISE_CONSTANT {
    main:
    SBC_PIECEWISE_CONSTANT_SIMULATE(
        Channel.fromList(params.sbc_piecewise_constant)
            .combine(Channel.of(0..<params.sbc_n_batches))
    )

    SBC_PIECEWISE_CONSTANT_INFER_UNCORRECTED(SBC_PIECEWISE_CONSTANT_SIMULATE.out)
    SBC_PIECEWISE_CONSTANT_INFER_CORRECTED(SBC_PIECEWISE_CONSTANT_SIMULATE.out)
    SBC_PIECEWISE_CONSTANT_INFER_NO_BIAS(SBC_PIECEWISE_CONSTANT_SIMULATE.out)

    SBC_PIECEWISE_CONSTANT_COLLECT(
        SBC_PIECEWISE_CONSTANT_SIMULATE.out
            .map { experiment, idx, pkl -> [experiment.name, pkl] }.groupTuple()
            .join(SBC_PIECEWISE_CONSTANT_INFER_UNCORRECTED.out.groupTuple())
            .join(SBC_PIECEWISE_CONSTANT_INFER_CORRECTED.out.groupTuple())
            .join(SBC_PIECEWISE_CONSTANT_INFER_NO_BIAS.out.groupTuple())
    )

    emit:
    results = SBC_PIECEWISE_CONSTANT_COLLECT.out
}

// ── Piecewise Exponential (2-epoch) ────────────────────────────────────────────

process SBC_PIECEWISE_EXPONENTIAL_SIMULATE {
    label 'simulation'

    input:
    tuple val(experiment), val(batch_idx)

    output:
    tuple val(experiment), val(batch_idx), path("${experiment.name}_${batch_idx}.pkl")

    script:
    """
    ${projectDir}/bin/sbc/piecewise_exponential/simulate.py \
        ${experiment.name}_${batch_idx}.pkl \
        --prior-ne1      ${experiment.prior_ne1} \
        --prior-sigma-ne ${experiment.prior_sigma_ne} \
        --prior-ne2      ${experiment.prior_ne2} \
        --prior-t0       ${experiment.prior_t0} \
        --prior-sigma-t0 ${experiment.prior_sigma_t0} \
        --batch-size     ${experiment.batch_size} \
        --sample-size    ${experiment.sample_size} \
        --num-windows    ${experiment.num_windows} \
        --seed           ${785874 + batch_idx}
    """
}

process SBC_PIECEWISE_EXPONENTIAL_INFER_UNCORRECTED {
    label 'inference'

    input:
    tuple val(experiment), val(batch_idx), path(batch_pkl)

    output:
    tuple val(experiment.name), path("uncorrected_${experiment.name}_${batch_idx}.pkl")

    script:
    """
    ${projectDir}/bin/sbc/piecewise_exponential/infer_uncorrected.py \
        ${batch_pkl} \
        uncorrected_${experiment.name}_${batch_idx}.pkl
    """
}

process SBC_PIECEWISE_EXPONENTIAL_INFER_CORRECTED {
    label 'inference'

    input:
    tuple val(experiment), val(batch_idx), path(batch_pkl)

    output:
    tuple val(experiment.name), path("corrected_${experiment.name}_${batch_idx}.pkl")

    script:
    """
    ${projectDir}/bin/sbc/piecewise_exponential/infer_corrected.py \
        ${batch_pkl} \
        corrected_${experiment.name}_${batch_idx}.pkl \
        --n-points-per-iter ${experiment.n_points_per_iter} \
        --n-iter            ${experiment.n_iter}
    """
}

process SBC_PIECEWISE_EXPONENTIAL_INFER_NO_BIAS {
    label 'inference'

    input:
    tuple val(experiment), val(batch_idx), path(batch_pkl)

    output:
    tuple val(experiment.name), path("nobias_${experiment.name}_${batch_idx}.pkl")

    script:
    """
    ${projectDir}/bin/sbc/piecewise_exponential/infer_assume_no_bias.py \
        ${projectDir}/bin/sbc/piecewise_exponential/nobias_model.stan \
        ${batch_pkl} \
        nobias_${experiment.name}_${batch_idx}.pkl
    """
}

process SBC_PIECEWISE_EXPONENTIAL_COLLECT {
    label 'simulation'

    publishDir "${params.sbc_results_dir}/piecewise_exponential", mode: 'copy'

    input:
    tuple val(name), path(batch_pkls), path(uncorrected_pkls), path(corrected_pkls), path(nobias_pkls)

    output:
    path "${name}.pkl"

    script:
    """
    ${projectDir}/bin/sbc/collect.py \
        ${name}.pkl \
        --batches     ${batch_pkls} \
        --uncorrected ${uncorrected_pkls} \
        --corrected   ${corrected_pkls} \
        --no-bias     ${nobias_pkls}
    """
}

workflow SBC_PIECEWISE_EXPONENTIAL {
    main:
    SBC_PIECEWISE_EXPONENTIAL_SIMULATE(
        Channel.fromList(params.sbc_piecewise_exponential)
            .combine(Channel.of(0..<params.sbc_n_batches))
    )

    SBC_PIECEWISE_EXPONENTIAL_INFER_UNCORRECTED(SBC_PIECEWISE_EXPONENTIAL_SIMULATE.out)
    SBC_PIECEWISE_EXPONENTIAL_INFER_CORRECTED(SBC_PIECEWISE_EXPONENTIAL_SIMULATE.out)
    SBC_PIECEWISE_EXPONENTIAL_INFER_NO_BIAS(SBC_PIECEWISE_EXPONENTIAL_SIMULATE.out)

    SBC_PIECEWISE_EXPONENTIAL_COLLECT(
        SBC_PIECEWISE_EXPONENTIAL_SIMULATE.out
            .map { experiment, idx, pkl -> [experiment.name, pkl] }.groupTuple()
            .join(SBC_PIECEWISE_EXPONENTIAL_INFER_UNCORRECTED.out.groupTuple())
            .join(SBC_PIECEWISE_EXPONENTIAL_INFER_CORRECTED.out.groupTuple())
            .join(SBC_PIECEWISE_EXPONENTIAL_INFER_NO_BIAS.out.groupTuple())
    )

    emit:
    results = SBC_PIECEWISE_EXPONENTIAL_COLLECT.out
}
