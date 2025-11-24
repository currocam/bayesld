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
///     num_windows (Option<usize>): Number of genomic windows to split the contig into. If None, analyzes the entire contig.
///
/// Returns:
///     If num_windows is None: Tuple of numpy arrays: (mean, variance, n, left, right) for each bin.
///     If num_windows is Some: Tuple of numpy arrays: (mean, variance, n, left, right, window_id) with one row per bin and window.
///
/// Raises:
///     FileNotFoundError: If the input file cannot be opened.
///     ValueError: If the contig is not found, or its length is too short, or other input errors.
///     RuntimeError: For errors encountered during processing.
#[pyfunction]
#[pyo3(signature = (infile, contig_name, recombination_rate, num_windows=None))]
fn compute_ld<'py>(
    py: Python<'py>,
    infile: &str,
    contig_name: &str,
    recombination_rate: f64,
    num_windows: Option<usize>,
) -> PyResult<(
    &'py numpy::PyArray1<f64>,
    &'py numpy::PyArray1<f64>,
    &'py numpy::PyArray1<u64>,
    &'py numpy::PyArray1<f64>,
    &'py numpy::PyArray1<f64>,
    &'py numpy::PyArray1<usize>,
)> {
    // Open the BCF/VCF file
    let file = IndexedReader::from_path(infile).map_err(|e| {
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

    // Determine the genomic windows to process
    let windows = match num_windows {
        None => vec![(contig.start, contig.end, 0)],
        Some(n) if n == 0 => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "num_windows must be at least 1",
            ));
        }
        Some(n) => {
            let window_size = contig.length / n as u64;
            if window_size < bins.minimum as u64 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Window size is less than minimum bin size. Reduce num_windows.",
                ));
            }
            (0..n)
                .map(|i| {
                    let start = contig.start + i as u64 * window_size;
                    let end = if i == n - 1 {
                        contig.end
                    } else {
                        contig.start + (i + 1) as u64 * window_size
                    };
                    (start, end, i)
                })
                .collect::<Vec<_>>()
        }
    };

    // Collect results for all windows
    let mut all_means = Vec::new();
    let mut all_variances = Vec::new();
    let mut all_counts = Vec::new();
    let mut all_left_edges = Vec::new();
    let mut all_right_edges = Vec::new();
    let mut all_window_ids = Vec::new();

    for (window_start, window_end, window_id) in windows {
        // Create a subcontig for this window
        let window_contig = Contig {
            rid: contig.rid,
            length: window_end - window_start,
            start: window_start,
            end: window_end,
        };

        // Initialize data structures for this window
        let mut streaming = StreamingStats::new(bins.nbins);

        // We need a second pointer to iterate across all pairs
        let mut reader = IndexedReader::from_path(infile).map_err(|e| {
            pyo3::exceptions::PyFileNotFoundError::new_err(format!(
                "Failed to open input file for rolling window '{}': {}",
                infile, e
            ))
        })?;
        let _ = reader.header();

        // Initialize rolling window for this window
        let mut rolling_window =
            RollingMap::build(header, window_contig.clone(), None, 0.25, false).map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "Failed to initialize rolling window: {}",
                    e
                ))
            })?;
        let n_samples = rolling_window.n_samples;
        let mut genotypes_buffer = vec![0.0; n_samples];

        // Reopen file for this window
        let mut file_window = IndexedReader::from_path(infile).map_err(|e| {
            pyo3::exceptions::PyFileNotFoundError::new_err(format!(
                "Failed to open input file '{}': {}",
                infile, e
            ))
        })?;

        // Fetch this window region
        file_window
            .fetch(window_contig.rid, window_start, Some(window_end))
            .map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "Failed to fetch window region: {}",
                    e
                ))
            })?;

        // Iterate across the records in this window
        for record1_result in file_window.records() {
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
            if let Err(e) = rolling_window.roll_window(&mut reader, &mut genotypes_buffer, pos1, end)
            {
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
                if bins.left_edges_in_bp[index] <= distance
                    && distance <= bins.right_edges_in_bp[index]
                {
                    // Update the sufficient statistics
                    streaming.update(index, &genotypes1, genotypes2, n_samples);
                }
            }
        }

        // Finalize results for this window
        let result = streaming.finalize();

        // Append results
        all_means.extend_from_slice(&result.mean);
        all_variances.extend_from_slice(&result.variance);
        all_counts.extend(result.n.iter().map(|&x| x as u64));
        all_left_edges.extend_from_slice(&bins.left_edges_in_cm);
        all_right_edges.extend_from_slice(&bins.right_edges_in_cm);
        all_window_ids.extend(vec![window_id; bins.nbins]);
    }

    // Convert results to numpy arrays
    use numpy::IntoPyArray;
    let mean = all_means.into_pyarray(py);
    let variance = all_variances.into_pyarray(py);
    let n = all_counts.into_pyarray(py);
    let left = all_left_edges.into_pyarray(py);
    let right = all_right_edges.into_pyarray(py);
    let window_id = all_window_ids.into_pyarray(py);
    Ok((mean, variance, n, left, right, window_id))
}

/// A Python module implemented in Rust.
#[pymodule]
fn bayesld(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_ld, m)?)?;
    Ok(())
}
