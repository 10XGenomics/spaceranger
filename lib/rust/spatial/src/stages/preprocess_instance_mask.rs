//! PreprocessInstanceMask stage code
//! Martian stage PREPROCESS_INSTANCE_MASK

use crate::{H5File, NpyFile, TiffFile};
use anyhow::{bail, Context, Result};
use barcode::binned::SquareBinIndex;
use barcode::cell_name::CellId;
use hd_feature_slice::FeatureSliceH5;
use itertools::Itertools;
use martian::{MartianFileType, MartianRover, MartianStage, MartianVoid, Resource, StageDef};
use martian_derive::{make_mro, MartianStruct};
use martian_filetypes::json_file::JsonFile;
use martian_filetypes::tabular_file::CsvFile;
use martian_filetypes::FileTypeRead;
use ndarray::Array2;
use ndarray_npy::{ReadNpyExt, WriteNpyExt};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::BufReader;
use std::str::FromStr;
use tiff::decoder::{Decoder, DecodingResult, Limits};

#[derive(Debug, Deserialize)]
#[serde(untagged)]
/// Representation of a cell
pub enum Cell {
    /// id
    Id(usize),
    /// barcode
    Barcode(String),
}

impl Cell {
    fn to_id(&self) -> Result<usize> {
        match self {
            Cell::Id(id) => Ok(*id),
            Cell::Barcode(bc) => Ok(CellId::from_str(bc)?.id as usize),
        }
    }
}

#[derive(Debug, Deserialize)]
/// Struct to read barcode to cell map into
pub struct BarcodeAndCell {
    #[serde(alias = "Square_002um", alias = "square_002um", alias = "Barcode")]
    /// Barcode
    pub barcode: String,
    #[serde(alias = "Cell_id", alias = "cell_id", alias = "id")]
    /// cell ID
    pub id: Option<Cell>,
}

#[derive(Debug, Deserialize)]
/// Tissue image stats
pub struct TissueImageShape {
    /// height
    pub height: u64,
    /// width
    pub width: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, MartianStruct)]
/// Stage  Inputs
pub struct PreprocessInstanceMaskStageInputs {
    /// instance mask TIFF file in microscope image coordinates
    pub instance_mask_tiff: Option<TiffFile>,
    /// instance mask NPY file in microscope image coordinates
    pub instance_mask_npy: Option<NpyFile>,
    /// Barcode to cell map
    pub square_barcode_to_cell_map: Option<CsvFile<BarcodeAndCell>>,
    ///  HD feature slice
    pub hd_feature_slice: H5File,
    /// tissue image stats
    pub tissue_image_shape: Option<JsonFile<TissueImageShape>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, MartianStruct)]
/// Stage outputs
pub struct PreprocessInstanceMaskStageOutputs {
    /// Image with segmentation number a spot is in. Segmentations are numbered
    /// from 1 onwards and 0 indicates that the spot is in no segment.
    spot_mask_from_user: Option<NpyFile>,
}

/// This is our stage struct
pub struct PreprocessInstanceMask;

#[make_mro(volatile = strict, stage_name = PREPROCESS_INSTANCE_MASK)]
impl MartianStage for PreprocessInstanceMask {
    type StageInputs = PreprocessInstanceMaskStageInputs;
    type StageOutputs = PreprocessInstanceMaskStageOutputs;
    type ChunkInputs = MartianVoid;
    type ChunkOutputs = MartianVoid;

    fn split(
        &self,
        args: Self::StageInputs,
        _rover: MartianRover,
    ) -> Result<StageDef<Self::ChunkInputs>> {
        match (
            args.instance_mask_tiff.is_some(),
            args.instance_mask_npy.is_some(),
            args.square_barcode_to_cell_map.is_some(),
        ) {
            (false, false, false) => Ok(StageDef::new()),
            (true, _, _) | (_, true, _) => {
                if let Some(tissue_image_stats_json) = args.tissue_image_shape {
                    let tissue_image_stats = tissue_image_stats_json.read()?;
                    // we only store u32 which take 4 bytes. One gigabyte has 2^30 bytes
                    // and thus can hold 2^28 = 268_435_456 u32s
                    let estimated_mem_gb = ((tissue_image_stats.height * tissue_image_stats.width)
                        as f64
                        / 268_000_000.0)
                        .ceil() as isize
                        + 2;
                    Ok(StageDef::with_join_resource(Resource::with_mem_gb(
                        estimated_mem_gb,
                    )))
                } else {
                    bail!("Something unexpected happened, and we have a tissue mask without image dimensions!");
                }
            }
            (_, _, true) => Ok(StageDef::with_join_resource(Resource::with_mem_gb(8))),
        }
    }

    fn main(
        &self,
        _args: Self::StageInputs,
        _chunk_args: Self::ChunkInputs,
        _rover: MartianRover,
    ) -> Result<Self::ChunkOutputs> {
        unreachable!()
    }

    fn join(
        &self,
        args: Self::StageInputs,
        _chunk_defs: Vec<Self::ChunkInputs>,
        _chunk_outs: Vec<Self::ChunkOutputs>,
        rover: MartianRover,
    ) -> Result<Self::StageOutputs> {
        match (
            args.instance_mask_tiff.is_some(),
            args.instance_mask_npy.is_some(),
            args.square_barcode_to_cell_map.is_some(),
        ) {
            (false, false, false) => Ok(PreprocessInstanceMaskStageOutputs {
                spot_mask_from_user: None,
            }),
            (true, _, false) | ( _, true, false) => {
                println!("Reading an instance mask.");
                let mut image_arr = match (args.instance_mask_tiff, args.instance_mask_npy) {
                    (Some(_), Some(_)) => bail!("Specify either an NPY or TIFF-based instance mask, not both."),
                    (Some(instance_mask_tiff), None) => {
                        println!("Reading an instance mask TIFF.");
                        let mut decoder = Decoder::new(BufReader::new(
                            File::open(instance_mask_tiff)
                                .with_context(|| "Error opening TIFF image.")?,
                        ))?
                        .with_limits(Limits::unlimited());
                        let image_data = decoder.read_image()?;
                        let dims = decoder.dimensions()?;
                        match image_data {
                            DecodingResult::U32(data) => {
                                Array2::from_shape_vec((dims.1 as usize, dims.0 as usize), data)?
                            }
                            DecodingResult::U16(data) => Array2::from_shape_vec(
                                (dims.1 as usize, dims.0 as usize),
                                data.into_iter().map(|x| x as u32).collect_vec(),
                            )?,
                            _ => bail!("Only accepting Tiffs of datatype Uint32 and Uint16."),
                        }
                    }
                    (None, Some(instance_mask_npy)) => {
                        println!("Reading an instance mask NPY.");
                        Array2::<u32>::read_npy(File::open(instance_mask_npy)?)?
                    }
                    _ => unreachable!(
                        "Both instance mask tiff and instance mask npy can not be null"
                    ),
                };
                image_arr.swap_axes(0, 1);
                let mask_dims = image_arr.raw_dim();

                let feature_slice = FeatureSliceH5::open(&args.hd_feature_slice)?;
                let metadata = feature_slice.metadata();
                println!("num rows {}; num cols {}", metadata.nrows, metadata.ncols);

                let transform_spot_to_hires = feature_slice
                    .transform_spot_to_microscope()
                    .expect("No transform matrix in feature slice");

                let mut polygon_mask = Array2::<u64>::zeros((metadata.nrows, metadata.ncols));

                for ((col_idx, row_idx), value) in polygon_mask.indexed_iter_mut() {
                    let (transformed_row, transformed_col) =
                        transform_spot_to_hires.apply((row_idx as f64, col_idx as f64));
                    let (transformed_row_idx, transformed_col_idx) = (
                        transformed_row.floor() as isize,
                        transformed_col.floor() as isize,
                    );

                    if (0 <= transformed_row_idx)
                        && (transformed_row_idx < mask_dims[0] as isize)
                        && (0 <= transformed_col_idx)
                        && (transformed_col_idx < mask_dims[1] as isize)
                    {
                        *value = image_arr
                            [[transformed_row_idx as usize, transformed_col_idx as usize]]
                            as u64;
                    } else {
                        println!("Spot ({row_idx},{col_idx}) maps to ({transformed_row_idx},{transformed_col_idx}) which is outside the instance mask whose shape is {} x {}.", mask_dims[0], mask_dims[1]);
                    }
                }
                let spot_mask_path: NpyFile = rover.make_path("spot_mask_from_user.npy");
                let writer = spot_mask_path.buf_writer()?;
                polygon_mask.write_npy(writer)?;
                Ok(PreprocessInstanceMaskStageOutputs {
                    spot_mask_from_user: Some(spot_mask_path),
                })
            }
            (false, false, true) => {
                println!("Putting together segmentation from barcode to cell mapping!");
                let bc_and_cell_path = args.square_barcode_to_cell_map.unwrap();
                let feature_slice = FeatureSliceH5::open(&args.hd_feature_slice)?;
                let metadata = feature_slice.metadata();
                println!("num rows {}; num cols {}", metadata.nrows, metadata.ncols);

                let mut polygon_mask = Array2::<u64>::zeros((metadata.nrows, metadata.ncols));
                for barcode_and_cell in bc_and_cell_path
                    .read()
                    .context("Invalid barcode to cell mapping")?
                {
                    let square_barcode = SquareBinIndex::from_str(
                        barcode_and_cell.barcode.as_str(),
                    )
                    .context("barcode to cell map CSV do not all contain square barcodes")?;
                    match  (square_barcode, barcode_and_cell.id){
                        (SquareBinIndex {
                            col,
                            row,
                            size_um: 2,
                        }, Some(cell))  => {
                            let cell_id = cell.to_id()?;
                            assert!(
                                cell_id > 0,
                                "{} is assigned cell-id {}, which is not allowed.",
                                barcode_and_cell.barcode,
                                cell_id
                            );
                            polygon_mask[[row, col]] = cell_id as u64;
                        }
                        (SquareBinIndex {
                            col: _,
                            row: _,
                            size_um: 2,
                        }, None)  => (),
                        _ =>bail!("Found an non 2micron square barcode in square_barcode_to_cell_map."),
                    };
                }
                let spot_mask_path: NpyFile = rover.make_path("spot_mask_from_user.npy");
                let writer = spot_mask_path.buf_writer()?;
                polygon_mask.write_npy(writer)?;
                Ok(PreprocessInstanceMaskStageOutputs {
                    spot_mask_from_user: Some(spot_mask_path),
                })
            }
            _ => bail!("Can provide at most one of an instance mask tiff, instance mask npy, or a CSV with barcode to cell ID map."),
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test() -> Result<()> {
        let cell1: Cell = serde_json::from_str(r"777")?;
        assert_eq!(777, cell1.to_id()?);
        let cell2: Cell = serde_json::from_str(r#""cellid_000000079-1""#)?;
        assert_eq!(79, cell2.to_id()?);
        Ok(())
    }
}
