#![feature(portable_simd)]

mod ratemap;
pub use ratemap::{RateMap, RateMapError};

use numpy::PyArray2;
use numpy::PyReadonlyArray1;
use numpy::PyReadonlyArray2;
use pyo3::prelude::*;
use std::collections::BTreeMap;
use std::simd::f64x8;
use std::simd::prelude::SimdFloat;
pub const LANES: usize = 8;

#[derive(Debug, Clone, Copy)]
pub struct OnlineAverage {
    count: f64,
    mean: f64,
}
impl OnlineAverage {
    pub fn new() -> Self {
        Self {
            count: 0.0,
            mean: 0.0,
        }
    }
    /// Update the statistics with a new value
    pub fn update(&mut self, value: f64) {
        self.count += 1.0;
        let delta = value - self.mean;
        self.mean += delta / self.count;
    }

    /// Update the statistics with a value observed `n_observations` times
    /// This is useful when you need to account for positions with a certain value
    /// that were observed multiple times (e.g., zero diversity for unobserved positions)
    pub fn update_with_count(&mut self, value: f64, n_observations: f64) {
        let new_count = self.count + n_observations;
        let delta = value - self.mean;
        self.mean += delta * (n_observations as f64) / new_count;
        self.count = new_count;
    }
}

pub struct SiteStatistics {
    pub genetic_diversity: f64,
    pub minor_allele_frequency: f64,
    pub allele_frequency: f64,
}

impl SiteStatistics {
    pub fn from_diploid(genotypes: &[i32]) -> Self {
        assert!(genotypes.iter().all(|&gt| gt >= 0 && gt <= 2));
        let (n_ref, n_alt): (i32, i32) = genotypes
            .iter()
            .filter(|&&gt| gt >= 0 && gt <= 2)
            .fold((0, 0), |(ref_acc, alt_acc), &gt| {
                (ref_acc + 2 - gt, alt_acc + gt)
            });

        let n_total = n_ref + n_alt;

        if n_total < 2 {
            return Self {
                genetic_diversity: 0.0,
                minor_allele_frequency: 0.0,
                allele_frequency: 0.0,
            };
        }

        let pi = (2.0 * n_ref as f64 * n_alt as f64) / (n_total as f64 * (n_total - 1) as f64);
        let af = n_alt as f64 / n_total as f64;
        let maf = (n_ref.min(n_alt) as f64) / (n_total as f64);

        Self {
            genetic_diversity: pi,
            minor_allele_frequency: maf,
            allele_frequency: af,
        }
    }
    pub fn from_haploid(genotypes: &[i32]) -> Self {
        assert!(genotypes.iter().all(|&gt| gt == 0 || gt == 1));
        let n_alt: i32 = genotypes.iter().sum();
        let n_ref: i32 = genotypes.len() as i32 - n_alt;
        let n_total = n_ref + n_alt;
        if n_total < 2 {
            return Self {
                genetic_diversity: 0.0,
                minor_allele_frequency: 0.0,
                allele_frequency: 0.0,
            };
        }
        let pi = (2.0 * n_ref as f64 * n_alt as f64) / (n_total as f64 * (n_total - 1) as f64);
        let af = n_alt as f64 / n_total as f64;
        let maf = (n_ref.min(n_alt) as f64) / (n_total as f64);
        Self {
            genetic_diversity: pi,
            minor_allele_frequency: maf,
            allele_frequency: af,
        }
    }
}

// Measured as E[X_iY_iX_jY_j]
pub fn linkage_disequilibrium(x: &[f64], y: &[f64]) -> f64 {
    assert_eq!(x.len(), y.len());
    let s = x.len() as f64;

    let chunks = x.len() / LANES;

    // Process SIMD chunks
    let mut ld_simd = f64x8::splat(0.0);
    let mut ld_square_simd = f64x8::splat(0.0);

    for i in 0..chunks {
        let offset = i * LANES;
        let x_vec = f64x8::from_slice(&x[offset..offset + LANES]);
        let y_vec = f64x8::from_slice(&y[offset..offset + LANES]);
        let prod = x_vec * y_vec;
        ld_simd += prod;
        ld_square_simd += prod * prod;
    }

    // Sum up SIMD lanes
    let mut ld = ld_simd.reduce_sum();
    let mut ld_square = ld_square_simd.reduce_sum();

    // Process remainder elements
    let remainder_start = chunks * LANES;
    for i in remainder_start..x.len() {
        let prod = x[i] * y[i];
        ld += prod;
        ld_square += prod * prod;
    }

    (ld * ld - ld_square) / (s * (s - 1.0))
}

pub enum Ploidy {
    Haploid,
    Diploid,
}

impl Ploidy {
    fn are_valid_genotypes(&self, genotypes: &[i32]) -> bool {
        genotypes.iter().all(|&x| match self {
            Ploidy::Haploid => x == 0 || x == 1,
            Ploidy::Diploid => x == 0 || x == 1 || x == 2,
        })
    }
    // Standardize genotypes assuming Hardy-Weinberg equilibrium
    pub fn standardize(&self, genotypes: &[i32], allele_frequency: f64) -> Vec<f64> {
        let mean = match self {
            Ploidy::Haploid => allele_frequency,
            Ploidy::Diploid => 2.0 * allele_frequency,
        };
        let std_dev = match self {
            Ploidy::Haploid => (allele_frequency * (1.0 - allele_frequency)).sqrt(),
            Ploidy::Diploid => (2.0 * allele_frequency * (1.0 - allele_frequency)).sqrt(),
        };
        genotypes
            .iter()
            .map(|&x| ((x as f64) - mean) / std_dev)
            .collect()
    }
}

pub struct StreamingStats {
    left_bins: Vec<f64>,  // Morgan
    right_bins: Vec<f64>, // Morgan
    rate_map: RateMap,
    // pos_bp → (genetic_pos_morgan, interval_idx, standardized_gt)
    rolling_map: BTreeMap<u64, (f64, usize, Vec<f64>)>,
    maf_threshold: f64,
    genetic_diversity: OnlineAverage,
    linkage_disequilibrium: Vec<OnlineAverage>,
    ploidy: Ploidy,
}

impl StreamingStats {
    fn new(left_bins: Vec<f64>, right_bins: Vec<f64>, ploidy: Ploidy, rate_map: RateMap) -> Self {
        assert_eq!(left_bins.len(), right_bins.len());
        let n = left_bins.len();
        StreamingStats {
            left_bins,
            right_bins,
            rate_map,
            rolling_map: BTreeMap::new(),
            maf_threshold: 0.25,
            genetic_diversity: OnlineAverage::new(),
            linkage_disequilibrium: vec![OnlineAverage::new(); n],
            ploidy,
        }
    }

    fn add_site(&mut self, position: i32, genotypes: &[i32]) {
        let position_bp = position as u64;
        if !self.ploidy.are_valid_genotypes(genotypes) {
            return;
        }

        // Sites in NaN (masked) regions are skipped entirely — no diversity, no LD
        let genetic_pos = self.rate_map.genetic_position_morgan(position_bp as f64);
        if genetic_pos.is_nan() {
            return;
        }

        let site = match self.ploidy {
            Ploidy::Haploid => SiteStatistics::from_haploid(genotypes),
            Ploidy::Diploid => SiteStatistics::from_diploid(genotypes),
        };
        self.genetic_diversity.update(site.genetic_diversity);

        if site.minor_allele_frequency < self.maf_threshold {
            return;
        }

        let interval_idx = self.rate_map.interval_of(position_bp as f64);
        let standardized = self.ploidy.standardize(genotypes, site.allele_frequency);
        self.rolling_map
            .insert(position_bp, (genetic_pos, interval_idx, standardized));
    }

    fn add_batch(
        &mut self,
        genotypes: PyReadonlyArray2<i32>,
        positions: PyReadonlyArray1<i32>,
        region_span: f64, // non-NaN bp in this chunk (Python computes via missing_intervals())
    ) -> PyResult<()> {
        let genotypes = genotypes.as_array();
        let positions = positions.as_array();
        let shape = genotypes.shape();
        if positions.len() != shape[0] {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Expected {} rows in genotypes, got {}",
                positions.len(),
                shape[0]
            )));
        }

        let max_bin = *self.right_bins.last().expect("bins are empty");
        let min_bin = *self.left_bins.first().expect("bins are empty");

        // Add sites from this batch, counting only those outside NaN regions
        let mut n_valid_sites = 0u64;
        for i in 0..shape[0] {
            let position = positions[i];
            let gp = self.rate_map.genetic_position_morgan(position as f64);
            if !gp.is_nan() {
                n_valid_sites += 1;
            }
            let row: Vec<i32> = genotypes.row(i).to_vec();
            self.add_site(position, &row);
        }

        // Account for HOMREF positions: region_span is non-NaN bp, subtract observed non-NaN sites
        let num_homref = region_span - n_valid_sites as f64 + 1.0;
        assert!(
            num_homref.is_finite() && num_homref >= 0.0,
            "num_homref: {}",
            num_homref
        );
        self.genetic_diversity.update_with_count(0.0, num_homref);

        // Process pairs from the front of the rolling map.
        // We only remove the front when the genetic span from front to back exceeds max_bin,
        // meaning no future site can pair with the front within any bin.
        while self.rolling_map.len() > 1 {
            let first_gp = self.rolling_map.first_key_value().expect("len > 1").1 .0;
            let last_gp = self.rolling_map.last_key_value().expect("len > 1").1 .0;
            if last_gp - first_gp <= max_bin {
                // Future sites could still pair with the front (within max_bin),
                // so we can't remove it yet.
                break;
            }

            let (_, (genetic_pos1, interval_idx1, genotypes1)) = self
                .rolling_map
                .pop_first()
                .expect("checked non-empty in while condition");

            // bin_index is monotone: distances strictly increase as we iterate,
            // so bin_index never needs to go backwards.
            let mut bin_index = 0;
            for (_, (genetic_pos2, interval_idx2, genotypes2)) in self.rolling_map.iter() {
                let distance = genetic_pos2 - genetic_pos1;
                if distance > max_bin {
                    break;
                }
                if distance < min_bin {
                    continue;
                }

                // NaN span check: any NaN interval between the two sites?
                // nan_prefix is non-decreasing, so once a NaN gap appears all
                // subsequent (farther) sites will also have one — break, not continue.
                if self.rate_map.nan_prefix[interval_idx2 + 1]
                    - self.rate_map.nan_prefix[interval_idx1]
                    > 0
                {
                    break;
                }

                // Advance past bins whose right edge is below this distance.
                while bin_index < self.left_bins.len()
                    && distance > self.right_bins[bin_index]
                {
                    bin_index += 1;
                }
                if bin_index >= self.left_bins.len() {
                    break;
                }
                if distance >= self.left_bins[bin_index] {
                    let ld = linkage_disequilibrium(&genotypes1, genotypes2);
                    self.linkage_disequilibrium[bin_index].update(ld);
                }
            }
        }
        Ok(())
    }

    fn finalize<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let max_bin = *self.right_bins.last().expect("bins are empty");
        let min_bin = *self.left_bins.first().expect("bins are empty");

        // Drain all remaining entries (no span guard — process everything left)
        while self.rolling_map.len() > 1 {
            let (_, (genetic_pos1, interval_idx1, genotypes1)) = self
                .rolling_map
                .pop_first()
                .expect("checked len > 1 in while condition");

            let mut bin_index = 0;
            for (_, (genetic_pos2, interval_idx2, genotypes2)) in self.rolling_map.iter() {
                let distance = genetic_pos2 - genetic_pos1;
                if distance > max_bin {
                    break;
                }
                if distance < min_bin {
                    continue;
                }

                if self.rate_map.nan_prefix[interval_idx2 + 1]
                    - self.rate_map.nan_prefix[interval_idx1]
                    > 0
                {
                    continue;
                }

                while bin_index < self.left_bins.len()
                    && distance > self.right_bins[bin_index]
                {
                    bin_index += 1;
                }
                if bin_index >= self.left_bins.len() {
                    break;
                }
                if distance >= self.left_bins[bin_index] {
                    let ld = linkage_disequilibrium(&genotypes1, genotypes2);
                    self.linkage_disequilibrium[bin_index].update(ld);
                }
            }
        }

        let mut mean = vec![0.0; self.linkage_disequilibrium.len() + 1];
        let mut count = vec![0.0; self.linkage_disequilibrium.len() + 1];
        mean[0] = self.genetic_diversity.mean;
        count[0] = self.genetic_diversity.count as f64;
        for i in 0..self.linkage_disequilibrium.len() {
            mean[i + 1] = self.linkage_disequilibrium[i].mean;
            count[i + 1] = self.linkage_disequilibrium[i].count as f64;
        }
        let matrix: Vec<Vec<f64>> = (0..mean.len()).map(|i| vec![mean[i], count[i]]).collect();
        Ok(PyArray2::from_vec2(py, &matrix)?)
    }
}

#[pyclass]
pub struct StreamingStatsDiploid {
    _inner: StreamingStats,
}

#[pymethods]
impl StreamingStatsDiploid {
    #[new]
    fn new(
        left_bins_morgan: Vec<f64>,
        right_bins_morgan: Vec<f64>,
        map_position_bp: Vec<f64>,
        map_rate: Vec<f64>,
    ) -> PyResult<Self> {
        let rate_map = RateMap::build(map_position_bp, map_rate)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        let stats = StreamingStats::new(left_bins_morgan, right_bins_morgan, Ploidy::Diploid, rate_map);
        Ok(Self { _inner: stats })
    }

    fn add_batch(
        &mut self,
        genotypes: PyReadonlyArray2<i32>,
        positions: PyReadonlyArray1<i32>,
        region_span: f64,
    ) -> PyResult<()> {
        self._inner.add_batch(genotypes, positions, region_span)
    }

    fn finalize<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self._inner.finalize(py)
    }
}

#[pyclass]
pub struct StreamingStatsHaploid {
    _inner: StreamingStats,
}

#[pymethods]
impl StreamingStatsHaploid {
    #[new]
    fn new(
        left_bins_morgan: Vec<f64>,
        right_bins_morgan: Vec<f64>,
        map_position_bp: Vec<f64>,
        map_rate: Vec<f64>,
    ) -> PyResult<Self> {
        let rate_map = RateMap::build(map_position_bp, map_rate)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        let stats = StreamingStats::new(left_bins_morgan, right_bins_morgan, Ploidy::Haploid, rate_map);
        Ok(Self { _inner: stats })
    }

    fn add_batch(
        &mut self,
        genotypes: PyReadonlyArray2<i32>,
        positions: PyReadonlyArray1<i32>,
        region_span: f64,
    ) -> PyResult<()> {
        self._inner.add_batch(genotypes, positions, region_span)
    }

    fn finalize<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self._inner.finalize(py)
    }
}

#[pymodule]
fn bayesld(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<StreamingStatsDiploid>()?;
    m.add_class::<StreamingStatsHaploid>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    impl Ploidy {
        fn randn(&self, n: usize) -> Vec<i32> {
            let mut rng = rand::thread_rng();
            match self {
                Ploidy::Haploid => (0..n).map(|_| rng.gen_range(0..=1)).collect(),
                Ploidy::Diploid => (0..n).map(|_| rng.gen_range(0..=1)).collect(),
            }
        }
    }

    #[test]
    fn test_online_average() {
        // Test regular update
        let mut avg = OnlineAverage::new();
        avg.update(10.0);
        avg.update(20.0);
        avg.update(30.0);
        assert_eq!(avg.mean, 20.0);
        assert_eq!(avg.count, 3.0);

        // Test update_with_count
        let mut avg2 = OnlineAverage::new();
        avg2.update_with_count(5.0, 3.0); // Three observations of 5.0
        avg2.update_with_count(10.0, 2.0); // Two observations of 10.0
        // Expected: (5*3 + 10*2) / 5 = 35 / 5 = 7.0
        assert_eq!(avg2.mean, 35.0 / 5.0);
        assert_eq!(avg2.count, 5.0);

        // Test accounting for unobserved positions (zero diversity)
        let genotypes = Ploidy::Diploid.randn(200);
        let mut avg3 = OnlineAverage::new();
        for genotype in genotypes.clone() {
            avg3.update(genotype as f64);
        }
        let mut avg3_unobserved = OnlineAverage::new();
        let mut seen = 0;
        for genotype in genotypes.clone().into_iter().filter(|x| *x != 0) {
            avg3_unobserved.update(genotype as f64);
            seen += 1;
        }
        let n_unobserved = genotypes.len() - seen;
        avg3_unobserved.update_with_count(0.0, n_unobserved as f64);

        assert!((avg3_unobserved.mean - avg3.mean).abs() < 1e-6);
        assert_eq!(avg3_unobserved.count, avg3.count);
    }

    #[test]
    fn test_site_statistics_cases() {
        // Case 1: All homozygous reference (genotype = 0)
        let genotypes = vec![0, 0, 0, 0];
        let stats = SiteStatistics::from_diploid(&genotypes);
        assert_eq!(stats.genetic_diversity, 0.0);
        assert_eq!(stats.minor_allele_frequency, 0.0);
        assert_eq!(stats.allele_frequency, 0.0);
        // Case 2: All homozygous derived (genotype = 2)
        let genotypes = vec![2, 2, 2, 2];
        let stats = SiteStatistics::from_diploid(&genotypes);
        assert_eq!(stats.genetic_diversity, 0.0);
        assert_eq!(stats.minor_allele_frequency, 0.0);
        assert_eq!(stats.allele_frequency, 1.0);
        // Case 3: Mixed genotypes with heterozygotes
        // genotypes: [0, 1, 2, 1]
        // n_ref = 4, n_alt = 4, n_total = 8
        // pi = (2 * 4 * 4) / (8 * 7) = 32 / 56
        // maf = min(4, 4) / 8 = 0.5
        let genotypes = vec![0, 1, 2, 1];
        let stats = SiteStatistics::from_diploid(&genotypes);
        assert!((stats.genetic_diversity - 32.0 / 56.0).abs() < 1e-10);
        assert_eq!(stats.minor_allele_frequency, 0.5);
        assert_eq!(stats.allele_frequency, 0.5);
    }
    #[test]
    fn test_site_statistics_invariants() {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        for _ in 0..100 {
            // Generate random sample size between 1 and 100
            let sample_size = rng.gen_range(1..100);
            // Use the helper function to simulate genotypes
            let genotypes = Ploidy::Diploid.randn(sample_size);
            let stats = SiteStatistics::from_diploid(&genotypes);
            // Genetic diversity should be non-negative
            assert!(
                stats.genetic_diversity >= 0.0,
                "Genetic diversity should be non-negative, got {} for genotypes {:?}",
                stats.genetic_diversity,
                genotypes
            );
            // Minor allele frequency should be between 0 and 0.5 (by definition of "minor")
            assert!(
                stats.minor_allele_frequency >= 0.0 && stats.minor_allele_frequency <= 0.5,
                "Minor allele frequency should be between 0 and 0.5, got {} for genotypes {:?}",
                stats.minor_allele_frequency,
                genotypes
            );
            // Allele frequency should be between 0 and 1 (by definition of "frequency")
            assert!(
                stats.allele_frequency >= 0.0 && stats.allele_frequency <= 1.0,
                "Allele frequency should be between 0 and 1, got {} for genotypes {:?}",
                stats.allele_frequency,
                genotypes
            );
        }
    }
    #[test]
    fn test_linkage_disequilibrium_cases() {
        // Case 1: [0, 0, 0], [0, 0, 0]
        let genotypes1 = vec![0.0, 0.0, 0.0];
        let genotypes2 = vec![0.0, 0.0, 0.0];
        let stats = linkage_disequilibrium(&genotypes1, &genotypes2);
        assert_eq!(stats, 0.0);
        // Case 2: [1, 1, 1], [1, 1, 1]
        let genotypes1 = vec![1.0, 1.0, 1.0];
        let genotypes2 = vec![1.0, 1.0, 1.0];
        let stats = linkage_disequilibrium(&genotypes1, &genotypes2);
        assert_eq!(stats, 1.0);
        // Case 3: [1, 1, 1], [-1, -1, 1]
        let genotypes1 = vec![1.0, 1.0, 1.0];
        let genotypes2 = vec![1.0, 1.0, 1.0];
        let stats = linkage_disequilibrium(&genotypes1, &genotypes2);
        assert_eq!(stats, 1.0);
        // Case 4: [0, 1, 1], [1, 0, 1]
        let genotypes1 = vec![0.0, 1.0, 1.0];
        let genotypes2 = vec![1.0, 0.0, 1.0];
        let stats = linkage_disequilibrium(&genotypes1, &genotypes2);
        assert_eq!(stats, 0.0);
        // Case 5: [1, 1, 1], [1, 0, 1]
        let genotypes1 = vec![1.0, 1.0, 1.0];
        let genotypes2 = vec![1.0, 0.0, 1.0];
        let stats = linkage_disequilibrium(&genotypes1, &genotypes2);
        assert!((stats - 1.0 / 3.0).abs() < 1e-10);
    }
    // Naive implementation of linkage disequilibrium
    fn naive_linkage_disequilibrium(genotypes1: &[f64], genotypes2: &[f64]) -> f64 {
        let n = genotypes1.len();
        let mut acc = 0.0;
        for i in 0..n {
            for j in i + 1..n {
                acc += genotypes1[i] * genotypes2[i] * genotypes1[j] * genotypes2[j];
            }
        }
        acc / (n as f64 * (n - 1) as f64 / 2.0)
    }
    #[test]
    fn test_linkage_disequilibrium_random() {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        for _ in 0..1000 {
            // Generate random sample size between 2 and 100
            let sample_size = rng.gen_range(2..100);
            // Use the helper function to simulate genotypes
            let genotypes1 = Ploidy::Diploid.randn(sample_size);
            let stats1 = SiteStatistics::from_diploid(&genotypes1);
            let genotypes2 = Ploidy::Diploid.randn(sample_size);
            let stats2 = SiteStatistics::from_diploid(&genotypes2);
            let normalized1 = Ploidy::Diploid.standardize(&genotypes1, stats1.allele_frequency);
            let normalized2 = Ploidy::Diploid.standardize(&genotypes2, stats2.allele_frequency);
            let ld = linkage_disequilibrium(&normalized1, &normalized2);
            let naive_ld = naive_linkage_disequilibrium(&normalized1, &normalized2);
            assert!((ld - naive_ld).abs() < 1e-10 || (ld.is_nan() && naive_ld.is_nan()));
        }
        for _ in 0..1000 {
            // Generate random sample size between 2 and 100
            let sample_size = rng.gen_range(2..100);
            // Use the helper function to simulate genotypes
            let genotypes1 = Ploidy::Haploid.randn(sample_size);
            let stats1 = SiteStatistics::from_haploid(&genotypes1);
            let genotypes2 = Ploidy::Haploid.randn(sample_size);
            let stats2 = SiteStatistics::from_haploid(&genotypes2);
            let normalized1 = Ploidy::Haploid.standardize(&genotypes1, stats1.allele_frequency);
            let normalized2 = Ploidy::Haploid.standardize(&genotypes2, stats2.allele_frequency);
            let ld = linkage_disequilibrium(&normalized1, &normalized2);
            let naive_ld = naive_linkage_disequilibrium(&normalized1, &normalized2);
            assert!((ld - naive_ld).abs() < 1e-10 || (ld.is_nan() && naive_ld.is_nan()));
        }
    }
}
