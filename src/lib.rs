use ldbucket::{Bins, Contig, RollingMap, StreamingStats};
use pyo3::prelude::*;
use rust_htslib::bcf::{IndexedReader, Read};
mod ldbucket;
/// Computes data from BCF/VCF files.
/// Computes LD statistics from a BCF/VCF file for a given contig.
///
/// Args:
///     infile (str): Path to the input BCF/VCF file (must be indexed).
///     contig_name (str): Name of the contig/chromosome to analyze.
///     recombination_rate (float): Recombination rate to use for binning.
///
/// Returns:
///     Tuple of numpy arrays: (mean, variance, n) for each bin.
///
/// Raises:
///     FileNotFoundError: If the input file cannot be opened.
///     ValueError: If the contig is not found, or its length is too short, or other input errors.
///     RuntimeError: For errors encountered during processing.
#[pyfunction]
fn compute_ld<'py>(
    py: Python<'py>,
    infile: &str,
    contig_name: &str,
    recombination_rate: f64,
) -> PyResult<(
    &'py numpy::PyArray1<f64>,
    &'py numpy::PyArray1<f64>,
    &'py numpy::PyArray1<u64>,
    &'py numpy::PyArray1<f64>,
    &'py numpy::PyArray1<f64>,
)> {
    // Open the BCF/VCF file
    let mut file = IndexedReader::from_path(infile).map_err(|e| {
        pyo3::exceptions::PyFileNotFoundError::new_err(format!(
            "Failed to open input file '{}': {}",
            infile, e
        ))
    })?;

    // Get contig information
    let header = file.header();
    let header_records = header.header_records();
    let contig = Contig::build(&header_records, contig_name).map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!(
            "Failed to find contig '{}': {}",
            contig_name, e
        ))
    })?;

    // Check if the length of the contig is greater than bins.minimum
    let bins = Bins::hapne_default(recombination_rate);
    if contig.length < (bins.minimum as u64) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Contig length is less than minimum bin size",
        ));
    }

    // Initialize data structures
    let mut streaming = StreamingStats::new(bins.nbins);

    // We need a second pointer to iterate across all pairs
    let mut reader = IndexedReader::from_path(infile).map_err(|e| {
        pyo3::exceptions::PyFileNotFoundError::new_err(format!(
            "Failed to open input file for rolling window '{}': {}",
            infile, e
        ))
    })?;
    let _ = reader.header();

    // Initialize rolling window
    let mut rolling_window =
        RollingMap::build(header, contig.clone(), None, 0.25, false).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!(
                "Failed to initialize rolling window: {}",
                e
            ))
        })?;
    let n_samples = rolling_window.n_samples;
    let mut genotypes_buffer = vec![0.0; n_samples];

    // Fetch the entire chromosome of interest
    file.fetch(contig.rid, contig.start, Some(contig.end))
        .map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!(
                "Failed to fetch contig region: {}",
                e
            ))
        })?;

    // Iterate across the records
    for record1_result in file.records() {
        let record1 = match record1_result {
            Ok(r) => r,
            Err(e) => {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "Error reading record: {}",
                    e
                )))
            }
        };
        // Keep processing records in ascending order
        let (pos1, genotypes1) = {
            let lookup_result = rolling_window.lookup(&record1, &mut genotypes_buffer);
            let (p, maybe_ref) = match lookup_result {
                Ok(val) => val,
                Err(e) => {
                    return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                        "Error in rolling_window.lookup: {}",
                        e
                    )))
                }
            };
            match maybe_ref {
                None => continue,
                Some(g_ref) => (p, g_ref.clone()),
            }
        };
        let (start, end) = (pos1 + bins.minimum as u64, pos1 + bins.maximum as u64);

        // Roll the window to the next position
        if let Err(e) = rolling_window.roll_window(&mut reader, &mut genotypes_buffer, pos1, end) {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "Error in rolling_window.roll_window: {}",
                e
            )));
        }

        // Most of the time, the second record will be in the first bin
        // and we know the bin index is monotonic increasing.
        let mut index = 0;
        for (pos2, genotypes2) in rolling_window.map.range(start..end) {
            let distance = (pos2 - pos1) as f64;
            if !(distance >= bins.minimum as f64 && distance <= bins.maximum as f64) {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "Distance {} out of bin range ({}-{})",
                    distance, bins.minimum, bins.maximum
                )));
            }
            // Find current bin index
            while distance > bins.right_edges_in_bp[index] {
                index += 1;
                if index >= bins.nbins {
                    return Err(pyo3::exceptions::PyRuntimeError::new_err(
                        "Bin index out of range while searching for appropriate bin.",
                    ));
                }
            }
            if bins.left_edges_in_bp[index] <= distance && distance <= bins.right_edges_in_bp[index]
            {
                // Update the sufficient statistics
                streaming.update(index, &genotypes1, genotypes2, n_samples);
            }
        }
    }
    let result = streaming.finalize();

    // Convert results to numpy arrays
    use numpy::IntoPyArray;
    let mean = result.mean.into_pyarray(py);
    let variance = result.variance.into_pyarray(py);
    let n = result
        .n
        .iter()
        .map(|&x| x as u64)
        .collect::<Vec<u64>>()
        .into_pyarray(py);
    let left = bins.left_edges_in_cm.into_pyarray(py);
    let right = bins.right_edges_in_cm.into_pyarray(py);
    Ok((mean, variance, n, left, right))
}

/// A Python module implemented in Rust.
#[pymodule]
fn bayesld(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_ld, m)?)?;
    Ok(())
}
