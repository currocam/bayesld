use bayesld::{Ploidy, SiteStatistics, linkage_disequilibrium};
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use rand::Rng;

const SAMPLE_SIZES: [usize; 6] = [10, 20, 50, 100, 200, 500];

/// Simulate random diploid genotypes (0, 1, or 2) of given length
fn simulate_diploid_genotypes(n: usize) -> Vec<i32> {
    let mut rng = rand::thread_rng();
    (0..n).map(|_| rng.gen_range(0..=2)).collect()
}

/// Scalar (non-SIMD) implementation of linkage disequilibrium for benchmarking
fn linkage_disequilibrium_scalar(x: &[f64], y: &[f64]) -> f64 {
    assert_eq!(x.len(), y.len());
    let s = x.len() as f64;

    let mut ld = 0.0;
    let mut ld_square = 0.0;

    for i in 0..x.len() {
        let prod = x[i] * y[i];
        ld += prod;
        ld_square += prod * prod;
    }

    (ld * ld - ld_square) / (s * (s - 1.0))
}

fn bench_linkage_disequilibrium(c: &mut Criterion) {
    let mut group = c.benchmark_group("LinkageDisequilibrium");

    for size in SAMPLE_SIZES.iter() {
        let genotypes1 = simulate_diploid_genotypes(*size);
        let genotypes2 = simulate_diploid_genotypes(*size);
        let stats1 = SiteStatistics::from_diploid(&genotypes1);
        let stats2 = SiteStatistics::from_diploid(&genotypes2);
        let normalized1 = Ploidy::Diploid.standardize(&genotypes1, stats1.allele_frequency);
        let normalized2 = Ploidy::Diploid.standardize(&genotypes2, stats2.allele_frequency);

        group.bench_with_input(BenchmarkId::new("SIMD", size), size, |b, _| {
            b.iter(|| linkage_disequilibrium(black_box(&normalized1), black_box(&normalized2)));
        });

        group.bench_with_input(BenchmarkId::new("Scalar", size), size, |b, _| {
            b.iter(|| {
                linkage_disequilibrium_scalar(black_box(&normalized1), black_box(&normalized2))
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_linkage_disequilibrium,);
criterion_main!(benches);
