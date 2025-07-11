//! Martian stage WRITE_BINNED_H5_MATRIX.
#![allow(missing_docs)]

use anyhow::Result;
use cr_h5::count_matrix::write_matrix_h5;
use cr_lib::FeatureReferenceFormat;
use cr_types::chemistry::{ChemistryDefs, ChemistryDefsExt};
use cr_types::{BarcodeIndexFormat, CountShardFile, GemWell, H5File};
use martian::{MartianMain, MartianRover};
use martian_derive::{make_mro, MartianStruct};
use martian_filetypes::FileTypeRead;
use serde::{Deserialize, Serialize};

pub struct WriteBinnedH5Matrix;
#[derive(Deserialize, Clone, MartianStruct)]
pub struct WriteBinnedH5MatrixStageInputs {
    pub gem_well: GemWell,
    pub counts: Vec<CountShardFile>,
    pub feature_reference: FeatureReferenceFormat,
    pub chemistry_defs: ChemistryDefs,
    pub sample_id: String,
    pub barcode_index: BarcodeIndexFormat,
}
#[derive(Serialize, Deserialize, Clone, MartianStruct)]
pub struct WriteBinnedH5StageOutputs {
    pub matrix: H5File,
}

#[make_mro(mem_gb = 12, volatile = strict)]
impl MartianMain for WriteBinnedH5Matrix {
    type StageInputs = WriteBinnedH5MatrixStageInputs;
    type StageOutputs = WriteBinnedH5StageOutputs;

    fn main(&self, args: Self::StageInputs, rover: MartianRover) -> Result<Self::StageOutputs> {
        let raw_feature_bc_matrix = rover.make_path("raw_feature_bc_matrix");
        write_matrix_h5(
            &raw_feature_bc_matrix,
            &args.counts,
            &args.feature_reference.read()?,
            &args.sample_id,
            &args.chemistry_defs.description(),
            args.gem_well,
            &args.barcode_index.read()?,
            &rover.pipelines_version(),
        )?;
        Ok(Self::StageOutputs {
            matrix: raw_feature_bc_matrix,
        })
    }
}
