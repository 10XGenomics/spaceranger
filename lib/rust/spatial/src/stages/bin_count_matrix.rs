//! Martian stage BIN_COUNT_MATRIX_PD

use anyhow::Result;
use barcode::binned::SquareBinIndex;
use barcode::Barcode;
use cr_bam::constants::{ALN_BC_DISK_CHUNK_SZ, ALN_BC_ITEM_BUFFER_SZ, ALN_BC_SEND_BUFFER_SZ};
use cr_types::{
    BarcodeIndex, BarcodeIndexFormat, BarcodeThenFeatureOrder, CountShardFile, FeatureBarcodeCount,
    H5File,
};
use hd_feature_slice::FeatureSliceH5;
use itertools::Itertools;
use martian::prelude::*;
use martian_derive::{make_mro, martian_filetype, MartianStruct};
use martian_filetypes::json_file::JsonFile;
use martian_filetypes::tabular_file::CsvFile;
use martian_filetypes::{FileTypeRead, FileTypeWrite};
use metric::{TxHashMap, TxHashSet};
use parquet::basic::{Compression, ZstdLevel};
use parquet::file::properties::WriterProperties;
use parquet::file::writer::SerializedFileWriter;
use parquet::record::RecordWriter;
use parquet_derive::ParquetRecordWriter;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use shardio::{ShardReader, ShardWriter};
use slide_design::{GridIndex2D, SpotPacking, Transform, VisiumHdSlide};

#[derive(Debug, Clone, Deserialize, MartianStruct)]
pub struct BinCountMatrixStageInputs {
    bin_scale: u32,
    hd_feature_slice: H5File,
    counts: Vec<CountShardFile>,
    barcodes_under_tissue: JsonFile<Vec<String>>,
    barcode_index: BarcodeIndexFormat,
    scalefactors: JsonFile<ScaleFactors>,
}

#[derive(Serialize, Deserialize)]
struct ScaleFactors {
    spot_diameter_fullres: f64,
    bin_size_um: Option<f64>,
    microns_per_pixel: Option<f64>,
    #[serde(flatten)]
    _remaining_keys: TxHashMap<String, Value>,
}

#[derive(Serialize, Deserialize, ParquetRecordWriter, Debug)]
struct TissuePositionRow {
    barcode: String,
    in_tissue: u8,
    array_row: u32,
    array_col: u32,
    pxl_row_in_fullres: f64,
    pxl_col_in_fullres: f64,
}

#[derive(Debug, PartialEq, Eq, Hash, Copy, Clone)]
struct Index<T> {
    row: T,
    col: T,
}

type IndexI32 = Index<i32>;
type IndexF64 = Index<f64>;

struct PixelIndex {
    pitch_um: u32,
    // Map from array coordinates to pixel coordinates
    transform: Transform,
}

impl PixelIndex {
    fn new(feature_slide: &FeatureSliceH5, pitch_um: u32) -> Result<Self, Error> {
        let transform = feature_slide
            .transform_spot_to_microscope()
            .cloned()
            .unwrap_or_else(|| {
                feature_slide
                    .transform_spot_to_cytassist()
                    .cloned()
                    .unwrap_or_else(Transform::identity)
            });

        Ok(PixelIndex {
            pitch_um,
            transform,
        })
    }
    fn pixel_index(&self, Index { row, col }: IndexI32) -> IndexF64 {
        let (pixel_col, pixel_row) = self.transform.apply((col as f64, row as f64));
        IndexF64 {
            row: pixel_row,
            col: pixel_col,
        }
    }
    fn _corner_offset(&self, scale: u32) -> IndexF64 {
        let start = self.pixel_index(Index { row: 0, col: 0 });
        let end = self.pixel_index(Index {
            row: scale as i32 - 1,
            col: scale as i32 - 1,
        });
        IndexF64 {
            row: (end.row - start.row) / 2.0,
            col: (end.col - start.col) / 2.0,
        }
    }
    fn binned_pixel_index(&self, bin_index: &SquareBinIndex) -> IndexF64 {
        let bottom_corner = self.pixel_index(Index {
            row: bin_index.row as i32 * bin_index.scale(self.pitch_um) as i32,
            col: bin_index.col as i32 * bin_index.scale(self.pitch_um) as i32,
        });
        let offset = self._corner_offset(bin_index.scale(self.pitch_um));
        Index {
            row: bottom_corner.row + offset.row,
            col: bottom_corner.col + offset.col,
        }
    }
}

martian_filetype! {ParquetFile, "parquet"}

#[derive(Debug, Clone, Serialize, Deserialize, MartianStruct)]
pub struct BinCountMatrixStageOutputs {
    binned_counts: CountShardFile,
    binned_barcode_index: BarcodeIndexFormat,
    filtered_bin_barcodes: JsonFile<TxHashSet<String>>,
    binned_scalefactors: JsonFile<ScaleFactors>,
    binned_tissue_positions: CsvFile<TissuePositionRow>,
    binned_tissue_positions_parquet: ParquetFile,
}

// This is our stage struct
pub struct BinCountMatrix;

#[make_mro(mem_gb = 12, volatile = strict, stage_name = BIN_COUNT_MATRIX)]
impl MartianMain for BinCountMatrix {
    type StageInputs = BinCountMatrixStageInputs;
    type StageOutputs = BinCountMatrixStageOutputs;
    fn main(
        &self,
        args: Self::StageInputs,
        rover: MartianRover,
    ) -> Result<Self::StageOutputs, Error> {
        let feature_slice = FeatureSliceH5::open(&args.hd_feature_slice)?;
        let metadata = feature_slice.metadata();

        let slide = VisiumHdSlide::from_name_and_layout(&metadata.slide_name, None)?;
        assert!(slide.spot_packing() == SpotPacking::Square);
        assert!(args.bin_scale >= 1);

        let pitch = slide.spot_pitch();
        assert!(pitch.fract() == 0.0);
        let pitch = pitch as u32;

        let binned_barcode_of_barcode: TxHashMap<_, _> = slide
            .spots()
            .map(|spot| {
                let GridIndex2D { row, col } = spot.grid_index();
                (
                    Barcode::with_square_bin_index(
                        1,
                        SquareBinIndex {
                            row: (row as usize),
                            col: (col as usize),
                            size_um: pitch,
                        },
                    ),
                    Barcode::with_square_bin_index(
                        1,
                        SquareBinIndex {
                            row: (row as usize) / (args.bin_scale as usize),
                            col: (col as usize) / (args.bin_scale as usize),
                            size_um: pitch * args.bin_scale,
                        },
                    ),
                )
            })
            .collect();

        let barcodes_under_tissue: TxHashSet<Barcode> = {
            let bc_str = args.barcodes_under_tissue.read()?;
            bc_str.into_iter().map(|bc| bc.parse().unwrap()).collect()
        };

        // We call a binned barcode as under the tissue if
        // any of the spots inside this bin is under the tissue.
        let filtered_bin_barcodes: TxHashSet<Barcode> = barcodes_under_tissue
            .iter()
            .map(|bc| binned_barcode_of_barcode[bc])
            .collect();

        let reader: ShardReader<FeatureBarcodeCount, BarcodeThenFeatureOrder> =
            ShardReader::open_set(&args.counts)?;

        let binned_counts_shard: CountShardFile = rover.make_path("binned_counts");
        // Binned count data ordered by barcode.
        let mut binned_bc_counts: ShardWriter<FeatureBarcodeCount, BarcodeThenFeatureOrder> =
            ShardWriter::new(
                &binned_counts_shard,
                ALN_BC_SEND_BUFFER_SZ,
                ALN_BC_DISK_CHUNK_SZ,
                ALN_BC_ITEM_BUFFER_SZ,
            )?;

        let mut sender = binned_bc_counts.get_sender();

        for fb_count in reader.iter()? {
            let FeatureBarcodeCount {
                barcode,
                feature_idx,
                umi_count,
            } = fb_count?;
            sender.send(FeatureBarcodeCount {
                barcode: binned_barcode_of_barcode[&barcode],
                feature_idx,
                umi_count,
            })?;
        }

        sender.finished()?;
        binned_bc_counts.finish()?;

        let binned_barcode_index_file = {
            // The matrix produced in `MATRIX_COMPUTER` only includes barcodes that
            // have at least one read. Reading the barcode index ensures that the
            // binned raw matrices follow the same convention.
            let binned_barcode_index: BarcodeIndex = args
                .barcode_index
                .read()?
                .sorted_barcodes()
                .iter()
                .map(|bc| binned_barcode_of_barcode[bc])
                .unique()
                .collect();
            rover
                .make_path::<BarcodeIndexFormat>("binned_barcode_index")
                .with_content(&binned_barcode_index)?
        };

        let filtered_bin_barcodes_file: JsonFile<_> = rover.make_path("filtered_bin_barcodes");
        filtered_bin_barcodes_file.write(
            &filtered_bin_barcodes
                .iter()
                .map(ToString::to_string)
                .collect(),
        )?;

        // Write the new tissue positions file
        let pixel_index = PixelIndex::new(&feature_slice, pitch)?;

        // Collect all rows so that we can write it into a single row group in parquet.
        // From https://parquet.apache.org/docs/file-format/configurations/#row-group-size
        // Larger row groups allow for larger column chunks which makes it possible to do larger
        // sequential IO. Larger groups also require more buffering in the write path
        // (or a two pass write). We recommend large row groups (512MB - 1GB).
        //
        // NOTE: tissues positions file includes all the barcodes unlike the raw matrix
        let tissue_position_rows = binned_barcode_of_barcode
            .values()
            .unique()
            .sorted()
            .map(|bin_barcode| {
                let bin_index = bin_barcode.spatial_index();
                let this_pixel_index = pixel_index.binned_pixel_index(&bin_index);
                TissuePositionRow {
                    in_tissue: filtered_bin_barcodes.contains(bin_barcode) as u8,
                    barcode: bin_barcode.to_string(),
                    array_row: bin_index.row as u32,
                    array_col: bin_index.col as u32,
                    pxl_row_in_fullres: this_pixel_index.row,
                    pxl_col_in_fullres: this_pixel_index.col,
                }
            })
            .collect_vec();

        let binned_tissue_positions = rover
            .make_path::<CsvFile<_>>("binned_tissue_positions")
            .with_content(&tissue_position_rows)?;

        let binned_tissue_positions_parquet: ParquetFile = {
            let parquet_file: ParquetFile = rover.make_path("binned_tissue_positions_parquet");
            let mut writer = SerializedFileWriter::new(
                parquet_file.buf_writer()?,
                tissue_position_rows.as_slice().schema()?,
                WriterProperties::builder()
                    .set_compression(Compression::ZSTD(ZstdLevel::try_new(3)?))
                    .build()
                    .into(),
            )?;
            let mut row_group = writer.next_row_group()?;
            tissue_position_rows
                .as_slice()
                .write_to_row_group(&mut row_group)?;
            row_group.close()?;
            writer.close()?;

            parquet_file
        };

        Ok(BinCountMatrixStageOutputs {
            binned_counts: binned_counts_shard,
            binned_barcode_index: binned_barcode_index_file,
            filtered_bin_barcodes: filtered_bin_barcodes_file,
            binned_scalefactors: {
                let mut scalefactors = args.scalefactors.read()?;
                scalefactors.spot_diameter_fullres *= args.bin_scale as f64;
                let bin_size_um = slide.spot_pitch() as f64 * args.bin_scale as f64;
                scalefactors.bin_size_um = Some(bin_size_um);
                scalefactors.microns_per_pixel =
                    Some(bin_size_um / scalefactors.spot_diameter_fullres);

                let output_file: JsonFile<_> = rover.make_path("binned_scalefactors");
                output_file.write(&scalefactors)?;
                output_file
            },
            binned_tissue_positions,
            binned_tissue_positions_parquet,
        })
    }
}
