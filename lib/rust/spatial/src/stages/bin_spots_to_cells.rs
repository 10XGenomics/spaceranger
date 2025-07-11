//! BinSpotsToCells stage code
//! Martian stage BIN_SPOTS_TO_CELLS
#![allow(missing_docs)]

use crate::NpyFile;
use anyhow::Result;
use barcode::binned::SquareBinIndex;
use barcode::{Barcode, BarcodeContent};
use cr_bam::constants::{ALN_BC_DISK_CHUNK_SZ, ALN_BC_ITEM_BUFFER_SZ, ALN_BC_SEND_BUFFER_SZ};
use cr_types::{
    BarcodeIndexFormat, BarcodeIndexOutput, BarcodeThenFeatureOrder, CountShardFile,
    FeatureBarcodeCount,
};
use itertools::Itertools;
use martian::prelude::*;
use martian_derive::{make_mro, MartianStruct};
use martian_filetypes::json_file::JsonFile;
use martian_filetypes::FileTypeWrite;
use ndarray::Array2;
use ndarray_npy::ReadNpyExt;
use serde::{Deserialize, Serialize};
use shardio::{ShardReader, ShardWriter};
use std::collections::HashSet;
use std::fs::File;
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize, MartianStruct)]
pub struct BinSpotsToCellsStageInputs {
    pub segmentation_spot_mask: Option<NpyFile>,
    pub counts: Option<Vec<CountShardFile>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, MartianStruct)]
pub struct BinSpotsToCellsStageOutputs {
    pub cell_counts: Option<CountShardFile>,
    pub cell_barcode_index: Option<BarcodeIndexFormat>,
    pub filtered_cell_barcodes: Option<JsonFile<Vec<String>>>,
    pub disable_writing_matrices: bool,
}

// This is our stage struct
pub struct BinSpotsToCells;

#[make_mro(mem_gb = 4, volatile = strict, stage_name = BIN_SPOTS_TO_CELLS)]
impl MartianMain for BinSpotsToCells {
    type StageInputs = BinSpotsToCellsStageInputs;
    type StageOutputs = BinSpotsToCellsStageOutputs;

    fn main(
        &self,
        args: Self::StageInputs,
        rover: MartianRover,
    ) -> Result<Self::StageOutputs, Error> {
        if let (Some(counts), Some(cell_segmentation_mask)) =
            (args.counts, args.segmentation_spot_mask)
        {
            let polygon_mask = Array2::<u64>::read_npy(File::open(cell_segmentation_mask)?)?;
            let num_rows = polygon_mask.nrows();
            let num_cols = polygon_mask.ncols();
            println!("Mask shape is: {num_rows} x {num_cols}");

            let segmented_counts_shard: CountShardFile = rover.make_path("binned_counts");
            let mut segmented_bc_counts: ShardWriter<FeatureBarcodeCount, BarcodeThenFeatureOrder> =
                ShardWriter::new(
                    &segmented_counts_shard,
                    ALN_BC_SEND_BUFFER_SZ,
                    ALN_BC_DISK_CHUNK_SZ,
                    ALN_BC_ITEM_BUFFER_SZ,
                )?;
            let mut sender = segmented_bc_counts.get_sender();

            let reader: ShardReader<FeatureBarcodeCount, BarcodeThenFeatureOrder> =
                ShardReader::open_set(&counts)?;
            let useful_bc_iter = reader.iter()?.filter_map(|fb_count| {
                if let Ok(FeatureBarcodeCount {
                    barcode:
                        Barcode {
                            gem_group: _,
                            valid: _,
                            content:
                                BarcodeContent::SpatialIndex(SquareBinIndex {
                                    row,
                                    col,
                                    size_um: _,
                                }),
                        },
                    feature_idx,
                    umi_count,
                }) = fb_count
                {
                    // polygon_mask[[row, col]]=0 means barcode (row, col) was assigned to no polygon
                    if polygon_mask[[row, col]] > 0 {
                        Some(FeatureBarcodeCount {
                            barcode: Barcode::with_cell_id(1, polygon_mask[[row, col]] as u32),
                            feature_idx,
                            umi_count,
                        })
                    } else {
                        None
                    }
                } else {
                    None
                }
            });

            let mut all_cells_with_some_umis = HashSet::new();
            for fb_count in useful_bc_iter {
                all_cells_with_some_umis.insert(fb_count.barcode);
                sender.send(fb_count)?;
            }
            sender.finished()?;
            segmented_bc_counts.finish()?;

            println!("Num unique bcs {}", all_cells_with_some_umis.len());
            let sorted_filtered_cells: Vec<_> = all_cells_with_some_umis
                .into_iter()
                .map(|bc| bc.to_string())
                .sorted()
                .collect();

            let filtered_cells_file: JsonFile<_> = rover.make_path("filtered_cells");
            filtered_cells_file.write(&sorted_filtered_cells)?;

            let BarcodeIndexOutput {
                index: cell_barcode_index,
                num_barcodes: cell_barcode_count,
            } = BarcodeIndexFormat::write(
                &rover.make_path::<PathBuf>("cell_barcode_index"),
                polygon_mask
                    .into_iter()
                    .filter_map(|cell_id| {
                        // cell_id=0 means barcode was assigned to no polygon
                        if cell_id > 0 {
                            Some(Barcode::with_cell_id(1, cell_id as u32))
                        } else {
                            None
                        }
                    })
                    .sorted()
                    .dedup(),
            )?;

            Ok(BinSpotsToCellsStageOutputs {
                cell_counts: Some(segmented_counts_shard),
                cell_barcode_index: Some(cell_barcode_index),
                filtered_cell_barcodes: Some(filtered_cells_file),
                disable_writing_matrices: cell_barcode_count == 0,
            })
        } else {
            Ok(BinSpotsToCellsStageOutputs {
                cell_counts: None,
                cell_barcode_index: None,
                filtered_cell_barcodes: None,
                disable_writing_matrices: true,
            })
        }
    }
}
