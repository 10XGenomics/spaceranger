//! cr_lib::stages
#![deny(missing_docs)]

pub mod align_and_count;
pub mod barcode_correction;
pub mod build_per_sample_vdj_ws_contents;
pub mod call_tags_overhang;
pub mod call_tags_rtl;
pub mod check_barcodes_compatibility;
pub mod check_barcodes_compatibility_vdj;
pub mod check_single_beam_mode;
pub mod collate_metrics;
pub mod collate_probe_metrics;
pub mod compute_antigen_vdj_metrics;
pub mod copy_chemistry_spec;
pub mod create_multi_graph;
pub mod demux_probe_bc_matrix;
pub mod detect_chemistry;
pub mod detect_chemistry_test;
pub mod detect_vdj_receptor;
pub mod expect_single_barcode_whitelist;
pub mod extract_single_chemistry;
pub mod generate_cas_websummary;
pub mod get_chemistry_def;
pub mod get_gdna_metrics;
pub mod logic_not;
pub mod make_correction_map;
pub mod make_shard;
pub mod merge_gem_well_files;
pub mod merge_metrics;
pub mod multi_preflight;
pub mod multi_setup_chunks;
pub mod parse_multi_config;
pub mod pick_beam_analyzer;
pub mod rust_bridge;
pub mod setup_reference_info;
pub mod setup_vdj_analysis;
pub mod setup_vdj_demux;
pub mod write_barcode_index;
pub mod write_barcode_summary;
pub mod write_gene_index;
pub mod write_h5_matrix;
pub mod write_matrix_market;
pub mod write_molecule_info;
pub mod write_multi_web_summary_json;
pub mod write_pos_bam;

#[cfg(feature = "tenx_internal")]
mod internal;
#[cfg(feature = "tenx_source_available")]
mod stubs;
