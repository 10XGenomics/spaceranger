//! cr_lib
#![deny(missing_docs)]

use anyhow::Result;
use docopt::Docopt;
use martian::prelude::*;
use serde::Deserialize;

const HEADER: &str = "# Copyright 2023 10x Genomics, Inc. All rights reserved.";

const USAGE: &str = "
Test Rust stage
Usage:
  cr_lib martian <adapter>...
  cr_lib mro [--file=<filename>] [--rewrite]
  cr_lib --help
Options:
  matrix-computer      Run the MatrixComputer stage for a specific correctness test
     --out=<path>      Output directory.
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
        cr_lib::stages::align_and_count::AlignAndCount,
        cr_lib::stages::barcode_correction::BarcodeCorrection,
        cr_lib::stages::build_per_sample_vdj_ws_contents::BuildPerSampleVdjWsContents,
        cr_lib::stages::call_tags_overhang::CallTagsOH,
        cr_lib::stages::call_tags_rtl::CallTagsRTL,
        cr_lib::stages::check_barcodes_compatibility::CheckBarcodesCompatibility,
        cr_lib::stages::check_barcodes_compatibility_vdj::CheckBarcodesCompatibilityVdj,
        cr_lib::stages::check_single_beam_mode::CheckSingleBeamMode,
        cr_lib::stages::collate_metrics::CollateMetrics,
        cr_lib::stages::collate_probe_metrics::CollateProbeMetrics,
        cr_lib::stages::compute_antigen_vdj_metrics::ComputeAntigenVdjMetrics,
        cr_lib::stages::copy_chemistry_spec::CopyChemistrySpec,
        cr_lib::stages::create_multi_graph::CreateMultiGraph,
        cr_lib::stages::demux_probe_bc_matrix::DemuxProbeBcMatrix,
        cr_lib::stages::detect_chemistry::DetectChemistry,
        cr_lib::stages::detect_vdj_receptor::DetectVdjReceptor,
        cr_lib::stages::expect_single_barcode_whitelist::ExpectSingleBarcodeWhitelist,
        cr_lib::stages::extract_single_chemistry::ExtractSingleChemistry,
        cr_lib::stages::generate_cas_websummary::GenerateCasWebsummary,
        cr_lib::stages::get_chemistry_def::GetChemistryDef,
        cr_lib::stages::get_gdna_metrics::GetGdnaMetrics,
        cr_lib::stages::logic_not::LogicNot,
        cr_lib::stages::make_correction_map::MakeCorrectionMap,
        cr_lib::stages::make_shard::MakeShard,
        cr_lib::stages::merge_gem_well_files::MergeGemWellFiles,
        cr_lib::stages::merge_metrics::MergeMetrics,
        cr_lib::stages::multi_preflight::MultiPreflight,
        cr_lib::stages::multi_setup_chunks::MultiSetupChunks,
        cr_lib::stages::parse_multi_config::ParseMultiConfig,
        cr_lib::stages::pick_beam_analyzer::PickBeamAnalyzer,
        cr_lib::stages::rust_bridge::RustBridge,
        cr_lib::stages::setup_reference_info::SetupReferenceInfo,
        cr_lib::stages::setup_vdj_analysis::SetupVdjAnalysis,
        cr_lib::stages::setup_vdj_demux::SetupVDJDemux,
        cr_lib::stages::write_barcode_index::WriteBarcodeIndex,
        cr_lib::stages::write_barcode_summary::WriteBarcodeSummary,
        cr_lib::stages::write_gene_index::WriteGeneIndex,
        cr_lib::stages::write_h5_matrix::WriteH5Matrix,
        cr_lib::stages::write_matrix_market::WriteMatrixMarket,
        cr_lib::stages::write_molecule_info::WriteMoleculeInfo,
        cr_lib::stages::write_multi_web_summary_json::WriteMultiWebSummaryJson,
        cr_lib::stages::write_pos_bam::WritePosBam,
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
