#[derive(Debug, Clone)]
pub struct RateMap {
    pub(crate) position_bp: Vec<f64>, // n+1 breakpoints (starts at 0)
    pub(crate) rate: Vec<f64>,        // n rates (NaN = masked)
    pub(crate) cumulative_morgan: Vec<f64>, // n+1 prefix sums of genetic distance
    pub(crate) nan_prefix: Vec<u32>,  // n+1 prefix count of NaN intervals; needed to detect
                                      // NaN gaps *between* two valid sites in O(1)
}

#[derive(Debug, thiserror::Error)]
pub enum RateMapError {
    #[error("position_bp.len() ({positions}) != rate.len() + 1 ({rates_plus_one})")]
    LengthMismatch {
        positions: usize,
        rates_plus_one: usize,
    },
    #[error("position_bp[0] must be 0.0, got {value}")]
    InvalidStart { value: f64 },
    #[error("position_bp not strictly increasing at index {index}")]
    NotStrictlyIncreasing { index: usize },
    #[error("negative rate at index {index}: {value}")]
    NegativeRate { index: usize, value: f64 },
}

impl RateMap {
    pub fn build(position_bp: Vec<f64>, rate: Vec<f64>) -> Result<Self, RateMapError> {
        if position_bp.len() != rate.len() + 1 {
            return Err(RateMapError::LengthMismatch {
                positions: position_bp.len(),
                rates_plus_one: rate.len() + 1,
            });
        }
        if position_bp[0] != 0.0 {
            return Err(RateMapError::InvalidStart {
                value: position_bp[0],
            });
        }
        for i in 1..position_bp.len() {
            if position_bp[i] <= position_bp[i - 1] {
                return Err(RateMapError::NotStrictlyIncreasing { index: i });
            }
        }
        for (i, &r) in rate.iter().enumerate() {
            if !r.is_nan() && r < 0.0 {
                return Err(RateMapError::NegativeRate { index: i, value: r });
            }
        }

        let n = rate.len();
        let mut cumulative_morgan = Vec::with_capacity(n + 1);
        let mut nan_prefix = Vec::with_capacity(n + 1);
        cumulative_morgan.push(0.0);
        nan_prefix.push(0);

        for i in 0..n {
            let span = position_bp[i + 1] - position_bp[i];
            let is_nan = rate[i].is_nan();
            let mass = if is_nan { 0.0 } else { rate[i] * span };
            cumulative_morgan.push(cumulative_morgan[i] + mass);
            nan_prefix.push(nan_prefix[i] + is_nan as u32);
        }

        Ok(Self {
            position_bp,
            rate,
            cumulative_morgan,
            nan_prefix,
        })
    }

    pub fn constant(rate: f64, sequence_length: f64) -> Self {
        Self::build(vec![0.0, sequence_length], vec![rate])
            .expect("constant rate map is always valid")
    }

    /// Binary search for the interval containing x_bp.
    /// Returns i such that position_bp[i] <= x_bp < position_bp[i+1].
    /// The last interval is closed on the right: position_bp[n-1] <= x_bp <= position_bp[n].
    /// Panics if x_bp is out of [position_bp[0], position_bp[n]].
    pub(crate) fn interval_of(&self, x_bp: f64) -> usize {
        assert!(
            x_bp >= self.position_bp[0]
                && x_bp <= *self.position_bp.last().expect("position_bp is non-empty"),
            "x_bp={} out of range [{}, {}]",
            x_bp,
            self.position_bp[0],
            self.position_bp.last().expect("position_bp is non-empty")
        );
        // partition_point returns first index where position_bp[i] > x_bp
        let idx = self.position_bp.partition_point(|&p| p <= x_bp);
        // idx is at least 1 (since position_bp[0] = 0 <= x_bp), subtract 1 for interval index
        // but clamp to n-1 (last interval) for the right endpoint
        (idx - 1).min(self.rate.len() - 1)
    }

    /// Returns the cumulative genetic map value (in Morgan) at physical position x_bp.
    /// Returns NaN if x_bp falls within a NaN-rate interval.
    pub fn genetic_position_morgan(&self, x_bp: f64) -> f64 {
        let i = self.interval_of(x_bp);
        if self.rate[i].is_nan() {
            return f64::NAN;
        }
        self.cumulative_morgan[i] + self.rate[i] * (x_bp - self.position_bp[i])
    }

    /// Genetic distance between two positions in Morgan.
    /// Returns NaN if any interval in [from_bp, to_bp) has a NaN rate.
    /// Requires from_bp <= to_bp.
    pub fn genetic_distance_morgan(&self, from_bp: f64, to_bp: f64) -> f64 {
        debug_assert!(from_bp <= to_bp);
        let i = self.interval_of(from_bp);
        let j = self.interval_of(to_bp);
        // Check if any NaN interval lies between (inclusive of both endpoint intervals)
        if self.nan_prefix[j + 1] - self.nan_prefix[i] > 0 {
            return f64::NAN;
        }
        self.genetic_position_morgan(to_bp) - self.genetic_position_morgan(from_bp)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn arb_ratemap() -> impl Strategy<Value = RateMap> {
        // 1-20 intervals, spans in [1, 1e5] bp, rates in [0, 1e-6] or NaN (~10%)
        (1usize..20)
            .prop_flat_map(|n| {
                let spans = proptest::collection::vec(1.0f64..1e5, n);
                let rates = proptest::collection::vec(
                    prop_oneof![
                        9 => 0.0f64..1e-6,
                        1 => Just(f64::NAN),
                    ],
                    n,
                );
                (spans, rates)
            })
            .prop_map(|(spans, rates)| {
                let mut position_bp = vec![0.0];
                for s in &spans {
                    position_bp
                        .push(position_bp.last().expect("position_bp is non-empty") + s);
                }
                RateMap::build(position_bp, rates).expect("generated rate map must be valid")
            })
    }

    proptest! {
        #[test]
        fn prop_additivity(map in arb_ratemap(), a in 0.0f64..1.0, b in 0.0f64..1.0, c in 0.0f64..1.0) {
            let seq_len = *map.position_bp.last().expect("position_bp is non-empty");
            let mut pts = [a * seq_len, b * seq_len, c * seq_len];
            pts.sort_by(|x, y| x.partial_cmp(y).expect("no NaN in test positions"));
            let [a, b, c] = pts;

            let d_ac = map.genetic_distance_morgan(a, c);
            let d_ab = map.genetic_distance_morgan(a, b);
            let d_bc = map.genetic_distance_morgan(b, c);

            if !d_ac.is_nan() && !d_ab.is_nan() && !d_bc.is_nan() {
                prop_assert!((d_ac - (d_ab + d_bc)).abs() < 1e-10);
            }
        }

        #[test]
        fn prop_monotonicity(map in arb_ratemap(), a in 0.0f64..1.0, b in 0.0f64..1.0) {
            let seq_len = *map.position_bp.last().expect("position_bp is non-empty");
            let (a, b) = if a < b { (a * seq_len, b * seq_len) } else { (b * seq_len, a * seq_len) };
            let ga = map.genetic_position_morgan(a);
            let gb = map.genetic_position_morgan(b);
            if !ga.is_nan() && !gb.is_nan() {
                prop_assert!(gb >= ga);
            }
        }

        #[test]
        fn prop_non_negative(map in arb_ratemap(), a in 0.0f64..1.0, b in 0.0f64..1.0) {
            let seq_len = *map.position_bp.last().expect("position_bp is non-empty");
            let (a, b) = if a < b { (a * seq_len, b * seq_len) } else { (b * seq_len, a * seq_len) };
            let d = map.genetic_distance_morgan(a, b);
            if !d.is_nan() {
                prop_assert!(d >= 0.0);
            }
        }

        #[test]
        fn prop_constant_scale(rate in 0.0f64..1e-6, len in 1e3f64..1e8, a in 0.0f64..1.0, b in 0.0f64..1.0) {
            let map = RateMap::constant(rate, len);
            let (a, b) = if a < b { (a * len, b * len) } else { (b * len, a * len) };
            let d = map.genetic_distance_morgan(a, b);
            let expected = rate * (b - a);
            prop_assert!((d - expected).abs() < 1e-12);
        }

        #[test]
        fn prop_nan_propagation(map in arb_ratemap(), a in 0.0f64..1.0, b in 0.0f64..1.0) {
            let seq_len = *map.position_bp.last().expect("position_bp is non-empty");
            let (a, b) = if a < b { (a * seq_len, b * seq_len) } else { (b * seq_len, a * seq_len) };
            let i = map.interval_of(a);
            let j = map.interval_of(b);
            let has_nan = map.nan_prefix[j + 1] - map.nan_prefix[i] > 0;
            let d = map.genetic_distance_morgan(a, b);
            prop_assert_eq!(d.is_nan(), has_nan);
        }
    }

    // --- Validation error tests ---

    #[test]
    fn test_build_length_mismatch() {
        let err = RateMap::build(vec![0.0, 100.0, 200.0], vec![1e-8]).expect_err("should fail");
        assert!(matches!(err, RateMapError::LengthMismatch { .. }));
    }

    #[test]
    fn test_build_invalid_start() {
        let err = RateMap::build(vec![5.0, 100.0], vec![1e-8]).expect_err("should fail");
        assert!(matches!(err, RateMapError::InvalidStart { .. }));
    }

    #[test]
    fn test_build_not_increasing() {
        let err =
            RateMap::build(vec![0.0, 100.0, 50.0], vec![1e-8, 2e-8]).expect_err("should fail");
        assert!(matches!(err, RateMapError::NotStrictlyIncreasing { .. }));
    }

    #[test]
    fn test_build_negative_rate() {
        let err = RateMap::build(vec![0.0, 100.0], vec![-1e-8]).expect_err("should fail");
        assert!(matches!(err, RateMapError::NegativeRate { .. }));
    }

    // --- Hand-computed tests ---

    #[test]
    fn test_two_intervals() {
        // [0, 100) rate=1e-8, [100, 200] rate=2e-8
        let map = RateMap::build(vec![0.0, 100.0, 200.0], vec![1e-8, 2e-8]).expect("valid rate map");

        // Within first interval
        assert!((map.genetic_distance_morgan(0.0, 50.0) - 50.0 * 1e-8).abs() < 1e-15);
        // Within second interval
        assert!((map.genetic_distance_morgan(100.0, 150.0) - 50.0 * 2e-8).abs() < 1e-15);
        // Spanning both
        let expected = 100.0 * 1e-8 + 100.0 * 2e-8;
        assert!((map.genetic_distance_morgan(0.0, 200.0) - expected).abs() < 1e-15);
        // Partial span across boundary
        let expected = 40.0 * 1e-8 + 30.0 * 2e-8;
        assert!((map.genetic_distance_morgan(60.0, 130.0) - expected).abs() < 1e-15);
    }

    #[test]
    fn test_nan_middle() {
        // [0, 100) rate=1e-8, [100, 200) rate=NaN, [200, 300] rate=2e-8
        let map = RateMap::build(
            vec![0.0, 100.0, 200.0, 300.0],
            vec![1e-8, f64::NAN, 2e-8],
        )
        .expect("valid rate map");

        // Within first interval: valid
        assert!(!map.genetic_distance_morgan(10.0, 50.0).is_nan());
        // Within third interval: valid
        assert!(!map.genetic_distance_morgan(210.0, 280.0).is_nan());
        // Crossing NaN: returns NaN
        assert!(map.genetic_distance_morgan(50.0, 250.0).is_nan());
        // Genetic position in NaN interval: NaN
        assert!(map.genetic_position_morgan(150.0).is_nan());
    }

    #[test]
    fn test_at_breakpoints() {
        let map =
            RateMap::build(vec![0.0, 100.0, 200.0], vec![1e-8, 2e-8]).expect("valid rate map");
        // At breakpoint: uses the interval *starting* at that breakpoint (right-closed clamping)
        let gp = map.genetic_position_morgan(100.0);
        assert!((gp - 100.0 * 1e-8).abs() < 1e-15);
        // Distance from 0 to breakpoint
        assert!((map.genetic_distance_morgan(0.0, 100.0) - 100.0 * 1e-8).abs() < 1e-15);
    }

    #[test]
    fn test_constant_equivalent_to_build() {
        let rate = 1e-8;
        let len = 1_000_000.0;
        let c = RateMap::constant(rate, len);
        let b = RateMap::build(vec![0.0, len], vec![rate]).expect("valid rate map");
        assert_eq!(c.cumulative_morgan, b.cumulative_morgan);
        assert_eq!(c.nan_prefix, b.nan_prefix);
    }
}
