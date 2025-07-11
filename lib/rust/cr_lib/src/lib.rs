//! cr_lib
#![deny(missing_docs)]

mod align_and_count_metrics;
pub mod align_metrics;

/// Align homopolymer sequences.
pub mod align_homopolymer;

/// Align reads, annotate, deduplicate UMI and count genes for
/// one group of reads from a single barcode
pub mod aligner;

/// Barcode correction stage (and metrics)
mod barcode_correction_metrics;

mod barcode_overlap;

/// Barcode sorting workflow used by MAKE_SHARD
pub mod barcode_sort;

/// Struct containing cell annotation metrics for all figures
mod cell_annotation_ws_parameters;

pub mod detect_chemistry;

/// Read environment variables.
mod env;

/// Functions to do piecewise linear fitting for gDNA
mod fit_piecewise_linear_model;

/// gDNA utils
mod gdna_utils;

/// Miscellaneous macros
mod macros;

/// Metrics for MAKE_SHARD
pub mod make_shard_metrics;

/// Parquet file IO
pub mod parquet_file;

/// Preflight checks and utilities
pub mod preflight;

/// Probe barcode matrix I/O
mod probe_barcode_matrix;

/// Shared code for handling read-level multiplexing.
pub mod read_level_multiplexing;

/// Martian stages.
pub mod stages;

/// Testing utilities
pub mod testing;

/// Datatypes and martian file declarations
mod types;
pub use types::*;

/// Utility code: should be refactored to other places gradually
mod utils;

// initialize insta test harness
#[cfg(test)]
#[ctor::ctor]
fn init() {
    // when we need things like samtools or bgzip, put them on path
    dui_tests::bazel_utils::set_env_vars(&[("PATH", "lib/bin")], &[]).unwrap_or_default();
    // this ensures insta knows where to find its snap tests
    let cwd = std::env::current_dir().unwrap();
    let workspace_root = cwd.parent().unwrap();
    std::env::set_var("INSTA_WORKSPACE_ROOT", workspace_root);
}
