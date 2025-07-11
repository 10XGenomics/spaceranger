//! spatial
#![deny(missing_docs)]

use anyhow::Result;
use docopt::Docopt;
use martian::prelude::*;
use serde::Deserialize;

const HEADER: &str = "# Copyright 2023 10x Genomics, Inc. All rights reserved.";

const USAGE: &str = "
Test Rust stage
Usage:
  spatial martian <adapter>...
  spatial mro [--file=<filename>] [--rewrite]
  spatial --help
Options:
     --help            Show this screen.
";

#[derive(Deserialize)]
struct Args {
    // Martian interface
    cmd_martian: bool,
    cmd_mro: bool,
    arg_adapter: Vec<String>,
    flag_file: Option<String>,
    flag_rewrite: bool,
}

fn main() -> Result<()> {
    let args: Args = Docopt::new(USAGE)
        .and_then(|d| d.deserialize())
        .unwrap_or_else(|e| e.exit());

    let (stage_registry, mro_registry) = martian_stages![
        spatial::stages::assign_nuclei::AssignNuclei,
        spatial::stages::bin_count_matrix::BinCountMatrix,
        spatial::stages::bin_spots_to_cells::BinSpotsToCells,
        spatial::stages::compute_bin_metrics::ComputeBinMetrics,
        spatial::stages::compute_subsampled_bin_metrics::ComputeSubsampledBinMetrics,
        spatial::stages::create_hd_feature_slice::CreateHdFeatureSlice,
        spatial::stages::generate_hd_websummary_cs::GenerateHdWebsummaryCs,
        spatial::stages::generate_segment_websummary::GenerateSegmentWebsummary,
        spatial::stages::merge_bin_metrics::MergeBinMetrics,
        spatial::stages::preprocess_instance_mask::PreprocessInstanceMask,
        spatial::stages::preprocess_nucleus_segmentation_geojson::PreprocessNucleusSegmentationGeojson,
        spatial::stages::setup_binning::SetupBinning,
        spatial::stages::write_binned_h5_matrix::WriteBinnedH5Matrix,
        spatial::stages::write_spatial_barcode_index::WriteSpatialBarcodeIndex,
    ];

    if args.cmd_martian {
        // Call the martian adapter
        let adapter = MartianAdapter::new(stage_registry);

        // Suppress any logging that would be emitted via crate log.
        let adapter = adapter.log_level(LevelFilter::Warn);

        let retcode = adapter.run(args.arg_adapter);
        std::process::exit(retcode);
    } else if args.cmd_mro {
        // Create the mro for all the stages in this adapter
        martian_make_mro(HEADER, args.flag_file, args.flag_rewrite, mro_registry)?;
    } else {
        // If you need custom commands, implement them here
        unimplemented!()
    }
    Ok(())
}
