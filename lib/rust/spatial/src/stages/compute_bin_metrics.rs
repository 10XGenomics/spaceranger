//! Martian stage COMPUTE_BIN_METRICS

use barcode::binned::SquareBinIndex;
use cr_h5::count_matrix::{CountMatrix, CountMatrixFile, RawCount};
use cr_types::reference::feature_reference::FeatureType;
use cr_types::H5File;
use hd_feature_slice::FeatureSliceH5;
use martian::prelude::*;
use martian_derive::{make_mro, MartianStruct};
use martian_filetypes::json_file::JsonFile;
use martian_filetypes::{FileTypeRead, FileTypeWrite};
use metric::{PercentMetric, TxHashSet};
use serde::{Deserialize, Serialize};
use statrs::statistics::{Data, Distribution, Max, Median, Min};

#[derive(Serialize, Deserialize, MartianStruct)]
pub struct ComputeBinMetricsStageInputs {
    filtered_matrix_h5: CountMatrixFile,
    hd_feature_slice: H5File,
    bin_scale: usize,
    metrics_json: JsonFile<MetricsSubset>,
}

#[derive(Deserialize)]
struct MetricsSubset {
    // Total number of GEX reads
    total_read_pairs: i64,
}

#[derive(Serialize, MartianStruct)]
pub struct ComputeBinMetricsStageOutputs {
    summary: JsonFile<BinMetrics>,
}

pub struct ComputeBinMetrics;

#[derive(Serialize, Deserialize)]
pub struct BinMetrics {
    pub bin_scale: i64,
    pub bin_size_um: i64,
    pub bins_under_tissue: i64,
    pub bins_under_tissue_frac: f64,
    pub mean_reads_per_bin: f64,
    pub mean_umis_per_bin: f64,
    pub median_umis_per_bin: f64,
    pub max_umis_per_bin: f64,
    pub min_umis_per_bin: f64,
    pub median_genes_per_bin: f64,
    pub mean_genes_per_bin: f64,
    pub total_genes_detected_under_tissue: i64,
    pub mean_reads_under_tissue_per_bin: f64,
}

fn mean_reads_under_tissue_per_bin(
    filt_matrix: &CountMatrix,
    feat_slice: &FeatureSliceH5,
    bin_scale: usize,
) -> Result<f64, Error> {
    let read_slice = feat_slice.load_total_reads(bin_scale)?;
    let mut total_reads_under_tissue = 0;
    for barcode in filt_matrix.barcodes() {
        let barcode = SquareBinIndex::from_bytes(barcode.as_bytes())?;
        total_reads_under_tissue += read_slice[[barcode.row, barcode.col]];
    }
    Ok(PercentMetric::from((
        total_reads_under_tissue as i64,
        filt_matrix.num_barcodes() as i64,
    ))
    .fraction()
    .unwrap_or(0.0))
}

#[make_mro(volatile = strict)]
impl MartianStage for ComputeBinMetrics {
    type StageInputs = ComputeBinMetricsStageInputs;
    type StageOutputs = ComputeBinMetricsStageOutputs;
    type ChunkInputs = MartianVoid;
    type ChunkOutputs = MartianVoid;

    fn split(
        &self,
        args: Self::StageInputs,
        _rover: MartianRover,
    ) -> Result<StageDef<Self::ChunkInputs>, Error> {
        let matrix_dim = args.filtered_matrix_h5.load_dimensions()?;
        let mem_gib = matrix_dim.estimate_mem_gib() // Matrix
          + 20e-9 * (matrix_dim.num_barcodes as f64) // UMIs/Genes per BC
          + 0.5; // Misc
        Ok(StageDef::with_join_resource(Resource::with_mem_gb(
            mem_gib.ceil() as isize,
        )))
    }

    fn main(
        &self,
        _args: Self::StageInputs,
        _chunk_args: Self::ChunkInputs,
        _rover: MartianRover,
    ) -> Result<Self::ChunkOutputs, Error> {
        unreachable!()
    }

    fn join(
        &self,
        args: Self::StageInputs,
        _chunk_defs: Vec<Self::ChunkInputs>,
        _chunk_outs: Vec<Self::ChunkOutputs>,
        rover: MartianRover,
    ) -> Result<Self::StageOutputs, Error> {
        let matrix = args.filtered_matrix_h5.read()?;
        assert!(
            matrix
                .feature_reference()
                .feature_defs
                .iter()
                .all(|fdef| fdef.feature_type == FeatureType::Gene),
            "Stage is not equipped to support multiple feature types yet!"
        );
        let feat_slice = FeatureSliceH5::open(&args.hd_feature_slice)?;
        let slide = feat_slice.slide()?;

        let mut umis_per_bc = vec![0.0f64; matrix.num_barcodes()];
        let mut genes_per_bc = vec![0.0f64; matrix.num_barcodes()];
        let mut genes_detected = TxHashSet::default();

        for RawCount {
            count,
            barcode_idx,
            feature_idx,
        } in matrix.raw_counts()
        {
            umis_per_bc[barcode_idx] += count as f64;
            genes_per_bc[barcode_idx] += 1.0;
            genes_detected.insert(feature_idx);
        }
        let umis_per_bc = Data::new(umis_per_bc);
        let genes_per_bc = Data::new(genes_per_bc);
        let metrics = BinMetrics {
            bin_scale: args.bin_scale as i64,
            bin_size_um: slide.spot_pitch() as i64 * args.bin_scale as i64,
            bins_under_tissue: matrix.num_barcodes() as i64,
            bins_under_tissue_frac: PercentMetric::from((
                matrix.num_barcodes() as i64,
                slide.num_spots(Some(args.bin_scale as u32)) as i64,
            ))
            .fraction()
            .unwrap_or(0.0),
            mean_reads_per_bin: args.metrics_json.read()?.total_read_pairs as f64
                / matrix.num_barcodes() as f64,
            mean_umis_per_bin: umis_per_bc.mean().unwrap_or(0.0),
            median_umis_per_bin: umis_per_bc.median(),
            max_umis_per_bin: umis_per_bc.max(),
            min_umis_per_bin: umis_per_bc.min(),
            median_genes_per_bin: genes_per_bc.median(),
            mean_genes_per_bin: genes_per_bc.mean().unwrap_or(0.0),
            total_genes_detected_under_tissue: genes_detected.len() as i64,
            mean_reads_under_tissue_per_bin: mean_reads_under_tissue_per_bin(
                &matrix,
                &feat_slice,
                args.bin_scale,
            )?,
        };
        let summary: JsonFile<BinMetrics> = rover.make_path("summary");
        summary.write(&metrics)?;
        Ok(ComputeBinMetricsStageOutputs { summary })
    }
}
