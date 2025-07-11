//! Martian stage WRITE_SPATIAL_BARCODE_INDEX
//! Assign a distinct integer to each barcode sequence.
#![allow(missing_docs)]

use anyhow::Result;
use barcode::binned::SquareBinIndex;
use barcode::whitelist::ReqStrand;
use barcode::{Barcode, BcSeq, WhitelistSpec};
use cr_types::{BarcodeIndexFormat, BarcodeIndexOutput};
use itertools::Itertools;
use martian::prelude::*;
use martian_derive::{make_mro, MartianStruct};
use serde::{Deserialize, Serialize};
use slide_design::VisiumHdSlide;
use std::path::{Path, PathBuf};

// Could also take in a chemistry def
#[derive(Clone, Deserialize, MartianStruct)]
pub struct StageInputs {
    barcode_whitelist: Option<String>,
    visium_hd_slide_name: Option<String>,
}

impl StageInputs {
    fn num_barcodes(&self) -> Result<usize> {
        match (&self.barcode_whitelist, &self.visium_hd_slide_name) {
            (Some(whitelist), None) => {
                let whitelist = WhitelistSpec::TxtFile {
                    name: whitelist.clone(),
                    translation: false,
                    strand: ReqStrand::Forward,
                };
                Ok(whitelist.as_source()?.iter()?.count())
            }
            (None, Some(slide_name)) => {
                let visium_hd_slide = VisiumHdSlide::from_name_and_layout(slide_name, None)?;
                Ok(visium_hd_slide.num_spots(None))
            }
            _ => unreachable!(
                "Exactly one of barcode_whitelist or visium_hd_slide_name must be provided."
            ),
        }
    }
    fn barcode_index(&self, path: &Path) -> Result<BarcodeIndexOutput> {
        match (&self.barcode_whitelist, &self.visium_hd_slide_name) {
            (Some(whitelist), None) => {
                let whitelist = WhitelistSpec::TxtFile {
                    name: whitelist.clone(),
                    translation: false,
                    strand: ReqStrand::Forward,
                };
                whitelist.as_source()?.iter()?.process_results(|bc_iter| {
                    BarcodeIndexFormat::write(
                        path,
                        bc_iter
                            .map(|(seq, _, _)| {
                                Barcode::with_seq(1, BcSeq::from_bytes(seq.as_bytes()), true)
                            })
                            .sorted()
                            .dedup(),
                    )
                })?
            }
            (None, Some(slide_name)) => {
                let visium_hd_slide = VisiumHdSlide::from_name_and_layout(slide_name, None)?;
                BarcodeIndexFormat::write(
                    path,
                    visium_hd_slide
                        .spots()
                        .map(|spot| {
                            Barcode::with_square_bin_index(
                                1,
                                SquareBinIndex {
                                    row: spot.row(),
                                    col: spot.col(),
                                    size_um: visium_hd_slide.spot_pitch_u32(),
                                },
                            )
                        })
                        .sorted()
                        .dedup(),
                )
            }
            _ => unreachable!(
                "Exactly one of barcode_whitelist or visium_hd_slide_name must be provided."
            ),
        }
    }
}

#[derive(Clone, Serialize, Deserialize, MartianStruct)]
pub struct StageOutputs {
    barcode_index_output: BarcodeIndexOutput,
}

pub struct WriteSpatialBarcodeIndex;

#[make_mro(volatile = strict)]
impl MartianStage for WriteSpatialBarcodeIndex {
    type StageInputs = StageInputs;
    type StageOutputs = StageOutputs;
    type ChunkInputs = MartianVoid;
    type ChunkOutputs = MartianVoid;

    fn split(
        &self,
        args: Self::StageInputs,
        _rover: MartianRover,
    ) -> Result<StageDef<Self::ChunkInputs>> {
        let num_barcodes = args.num_barcodes()?;
        let bytes_per_barcode = 200.0;
        let mem_gib = (1.0 + bytes_per_barcode * (num_barcodes as f64) / 1024.0 / 1024.0 / 1024.0)
            .ceil() as isize;
        println!("num_barcodes={num_barcodes},mem_gib={mem_gib}");
        Ok(StageDef::with_join_resource(Resource::with_mem_gb(mem_gib)))
    }

    fn main(
        &self,
        _args: Self::StageInputs,
        _chunk_args: MartianVoid,
        _rover: MartianRover,
    ) -> Result<Self::ChunkOutputs> {
        unreachable!()
    }

    fn join(
        &self,
        args: Self::StageInputs,
        _chunk_defs: Vec<MartianVoid>,
        _chunk_outs: Vec<MartianVoid>,
        rover: MartianRover,
    ) -> Result<Self::StageOutputs> {
        Ok(StageOutputs {
            barcode_index_output: args
                .barcode_index(&rover.make_path::<PathBuf>("barcode_index"))?,
        })
    }
}
