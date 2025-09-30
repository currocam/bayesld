use anyhow::{bail, Context, Result};
use rust_htslib::bcf::header::HeaderRecord;
use rust_htslib::bcf::{IndexedReader, Read, Record};
use std::cmp;
use std::collections::BTreeMap;
use std::ops::Deref;

// This function computes E[X_iY_iX_jY_j]
pub fn linkage_disequilibrium(genotypes1: &[f64], genotypes2: &[f64], n_samples: usize) -> f64 {
    let s = n_samples as f64;
    let (ld, ld_square) = genotypes1.iter().zip(genotypes2.iter()).fold(
        (0.0, 0.0),
        |(ld_acc, ld_square_acc), (&a, &b)| {
            let prod = a * b;
            (ld_acc + prod, ld_square_acc + prod * prod)
        },
    );
    (ld * ld - ld_square) / (s * (s - 1.0))
}

pub struct Bins {
    pub nbins: usize,
    pub left_edges_in_cm: Vec<f64>,
    pub right_edges_in_cm: Vec<f64>,
    pub left_edges_in_bp: Vec<f64>,
    pub right_edges_in_bp: Vec<f64>,
    pub minimum: i64,
    pub maximum: i64,
}

impl Bins {
    // From HapNe supplementary material
    pub fn hapne_default(recombination_rate: f64) -> Self {
        let nbins = 19;
        let mut left_edges_in_cm = Vec::with_capacity(nbins);
        let mut right_edges_in_cm = Vec::with_capacity(nbins);

        for i in 0..nbins {
            let i = i as f64;
            left_edges_in_cm.push(0.5 + 0.5 * i);
            right_edges_in_cm.push(1.0 + 0.5 * i);
        }
        // Transform to base pairs using x / 100 / recombination_rate
        let left_edges_in_bp = left_edges_in_cm
            .iter()
            .map(|&x| x / 100.0 / recombination_rate)
            .collect::<Vec<f64>>();
        let right_edges_in_bp = right_edges_in_cm
            .iter()
            .map(|&x| x / 100.0 / recombination_rate)
            .collect::<Vec<f64>>();
        let minimum = left_edges_in_bp[0].round() as i64;
        let maximum = right_edges_in_bp[nbins - 1].round() as i64;
        Self {
            nbins,
            left_edges_in_cm,
            right_edges_in_cm,
            left_edges_in_bp,
            right_edges_in_bp,
            minimum,
            maximum,
        }
    }
}

// The sufficient summary statistics for computing the log-likelihood later
#[derive(Debug)]
pub struct SufficientSummaryStats {
    pub mean: Vec<f64>,
    pub variance: Vec<f64>,
    pub n: Vec<usize>,
}

// We use a classic streaming algorithm to do only one pass over the data
// and have a constant memory footprint.
#[derive(Debug, Clone)]
pub struct StreamingStats {
    pub counts: Vec<usize>,
    pub ld: Vec<f64>,
    pub ld_square: Vec<f64>,
}
impl StreamingStats {
    pub fn new(n_bins: usize) -> Self {
        Self {
            counts: vec![0; n_bins],
            ld: vec![0.0; n_bins],
            ld_square: vec![0.0; n_bins],
        }
    }
    pub fn update(
        &mut self,
        index: usize,
        genotypes1: &[f64],
        genotypes2: &[f64],
        n_samples: usize,
    ) {
        // Compute the sufficient statistics
        let new_value = linkage_disequilibrium(genotypes1, genotypes2, n_samples);
        self.counts[index] += 1;
        let delta = new_value - self.ld[index];
        self.ld[index] += delta / self.counts[index] as f64;
        let delta2 = new_value - self.ld[index];
        self.ld_square[index] += delta * delta2;
    }
    pub fn finalize(&mut self) -> SufficientSummaryStats {
        let mut mean = self.ld.clone();
        let mut var = self.ld_square.clone();
        for i in 0..self.counts.len() {
            if self.counts[i] > 1 {
                var[i] /= self.counts[i] as f64;
            } else {
                var[i] = f64::NAN;
                mean[i] = f64::NAN;
            }
        }
        SufficientSummaryStats {
            mean,
            variance: var,
            n: self.counts.clone(),
        }
    }
}

// The key part of this algorithm is how to tradeoff memory and computation time.
// Here, I propose a solution that does a one pass over the data with two pointers
// but avoids recomputing some intermediate results with a B-tree
pub struct RollingMap {
    // Maps a given position to a vector of standarized genotypes
    pub map: BTreeMap<u64, Box<[f64]>>,
    last_position: u64,
    // Map samples to their indices
    sample_indices: Vec<usize>,
    use_precomputed_maf: bool,
    contig: Contig,
    maf_threshold: f64,
    pub(crate) n_samples: usize,
}

impl RollingMap {
    pub fn build(
        header: &rust_htslib::bcf::header::HeaderView,
        contig: Contig,
        sample_names: Option<Vec<String>>,
        maf_threshold: f64,
        use_precomputed_maf: bool,
    ) -> Result<Self> {
        // Create a bijection for the samples (or return an informative error)
        let sample_indices = match &sample_names {
            Some(sample_names) => {
                let mut sample_indices = Vec::with_capacity(sample_names.len());
                for name in sample_names {
                    match header.sample_id(name.as_bytes()) {
                        Some(idx) => sample_indices.push(idx),
                        None => bail!("Sample name '{}' not found in VCF header", name),
                    }
                }
                sample_indices.sort();
                if sample_indices.len() < 2 {
                    bail!("No enough samples (at least 2 required)");
                }
                if sample_indices.windows(2).any(|w| w[0] == w[1]) {
                    bail!("Duplicate sample names found in VCF header");
                }
                sample_indices
            }
            None => (0..header.sample_count() as usize).collect::<Vec<_>>(),
        };
        let n_samples = sample_indices.len();
        // Check whether a MAF column is present
        let has_maf = header.name_to_id(b"MAF").is_ok();
        if has_maf && use_precomputed_maf {
            // If MAF is present and user wants to use precomputed, skip (do nothing)
        } else if has_maf && !use_precomputed_maf {
            eprintln!(
                "Warning: MAF INFO field present in VCF, but --use-precomputed-maf not set. Using on-the-fly MAF calculation."
            );
        } else if !has_maf && use_precomputed_maf {
            bail!(
                "Requested to use precomputed MAF (--use-precomputed-maf), but no MAF INFO field found in VCF header."
            );
        }

        Ok(Self {
            map: BTreeMap::new(),
            last_position: 0,
            sample_indices,
            use_precomputed_maf: use_precomputed_maf,
            n_samples,
            maf_threshold: maf_threshold,
            contig,
        })
    }
    #[allow(clippy::borrowed_box)]
    pub fn lookup(
        &mut self,
        record: &Record,
        genotypes_buffer: &mut [f64],
    ) -> Result<(u64, Option<&Box<[f64]>>)> {
        let pos = record.pos() as u64;

        if self.map.contains_key(&pos) {
            return Ok((pos, self.map.get(&pos)));
        }
        // Try to process and insert
        if self.process(record, genotypes_buffer)?.is_some() {
            let genotypes = genotypes_buffer.to_vec().into_boxed_slice();
            self.map.insert(pos, genotypes);
            Ok((pos, self.map.get(&pos)))
        } else {
            Ok((pos, None))
        }
    }

    pub fn roll_window(
        &mut self,
        reader: &mut IndexedReader,
        genotypes_buffer: &mut [f64],
        start: u64,
        end: u64,
    ) -> Result<()> {
        let start = cmp::max(start, self.contig.start);
        let end = cmp::min(end, self.contig.end);
        // This function rolls the window to the start-end region
        // First, we split off the part of the map that is before the new window start
        self.map = self.map.split_off(&start);
        // Then, we have to fetch next records
        let _ = reader.fetch(self.contig.rid, self.last_position, Some(end));
        // Now iterate over the collected records
        for record in reader.records() {
            let record = record.context("Error while reading record")?;
            // We call lookup to process and potentially insert the record.
            let _ = self.lookup(&record, genotypes_buffer)?;
        }
        self.last_position = end;
        Ok(())
    }

    fn process(&self, record: &Record, genotypes_buffer: &mut [f64]) -> Result<Option<()>> {
        // First, we handle the case where we're using precomputed MAF
        if self.use_precomputed_maf {
            let maf = record.info(b"MAF").float().context("Error getting MAF")?;
            let maf = match maf {
                Some(maf) => *maf.deref().first().unwrap_or(&0.0) as f64,
                None => return Ok(None),
            };
            if maf.is_nan() {
                eprintln!("MAF is NaN");
                return Ok(None);
            }
            if maf < self.maf_threshold {
                return Ok(None);
            }
        }
        let n_samples = self.n_samples;
        let raw_genotypes = record.genotypes().context("Error getting genotypes")?;
        let mut total = 0;
        genotypes_buffer.fill(0.0);
        for (index, val) in genotypes_buffer.iter_mut().enumerate() {
            let i = self.sample_indices[index];
            let sample = raw_genotypes.get(i);
            for j in 0..2 {
                if let Some(gt) = sample[j].index() {
                    *val += gt as f64;
                    total += gt;
                } else {
                    return Ok(None);
                }
            }
        }
        let allele_freq = total as f64 / (2 * n_samples) as f64;
        if !self.use_precomputed_maf
            && (allele_freq < self.maf_threshold || allele_freq > (1.0 - self.maf_threshold))
        {
            return Ok(None);
        }
        // Standardize the genotypes
        let denom = (2.0 * allele_freq * (1.0 - allele_freq)).sqrt();
        for val in genotypes_buffer.iter_mut() {
            *val = (*val - 2.0 * allele_freq) / denom;
        }
        Ok(Some(()))
    }
}
#[derive(Debug, Clone)]
pub struct Contig {
    pub rid: u32,
    pub length: u64,
    pub start: u64,
    pub end: u64,
}

impl Contig {
    fn new(rid: u32, length: u64, start: u64, end: u64) -> Self {
        Contig {
            rid,
            length,
            start,
            end,
        }
    }
    pub fn build(
        header_records: &Vec<rust_htslib::bcf::HeaderRecord>,
        contig_name: &str,
    ) -> Result<Self> {
        // Contig names might be in the format chr1:1-1000000
        let mut split = contig_name.split(':');
        let name = split.next().unwrap_or(contig_name);
        let mut remainder = split.next().unwrap_or("");
        if remainder.is_empty() {
            remainder = "1-";
        }
        let mut remainder = remainder.split('-');
        // We take start if present, otherwise default to 1 (but zero-indexed)
        let start = remainder
            .next()
            .expect("Start position not found")
            .parse::<u64>()
            .context("Invalid start position")?;
        let start = start - 1;
        let mut rid = 0;
        for record in header_records {
            if let HeaderRecord::Contig { values, .. } = record {
                if values.get("ID").unwrap_or(&"".to_string()) == name {
                    let contig_length = values.get("length").context("Contig length not found")?;
                    if let Ok(contig_length) = contig_length.parse::<u64>() {
                        let end = remainder.next();
                        let end = match end {
                            Some(e) if !e.trim().is_empty() => {
                                e.parse::<u64>().context("Invalid end position")? - 1
                            }
                            _ => contig_length,
                        };
                        return Ok(Contig::new(rid, end - start, start, end));
                    }
                }
                rid += 1;
            }
        }
        bail!("Contig not found")
    }
}
