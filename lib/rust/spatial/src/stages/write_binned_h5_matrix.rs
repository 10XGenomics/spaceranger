//! Martian stage WRITE_BINNED_H5_MATRIX.

use anyhow::Result;
use cr_h5::count_matrix::write_matrix_h5;
use cr_lib::stages::write_h5_matrix::{StageInputs, StageOutputs};
use cr_types::chemistry::ChemistryDefsExt;
use martian::{MartianMain, MartianRover};
use martian_derive::make_mro;
use martian_filetypes::FileTypeRead;

pub struct WriteBinnedH5Matrix;

#[make_mro(mem_gb = 5, volatile = strict)]
impl MartianMain for WriteBinnedH5Matrix {
    type StageInputs = StageInputs;
    type StageOutputs = StageOutputs;

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
        Ok(StageOutputs {
            matrix: raw_feature_bc_matrix,
        })
    }
}
