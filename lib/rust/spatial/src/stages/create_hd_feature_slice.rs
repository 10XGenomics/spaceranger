//! CreateHdFeatureSlice stage code

use anyhow::Result;
use cr_h5::count_matrix::CountMatrixFile;
use cr_types::H5File;
use hd_feature_slice::io::FeatureSliceH5Writer;
use hd_feature_slice::metadata::TransformMatrices;
use martian::{MartianRover, MartianStage, MartianVoid, Resource, StageDef};
use martian_derive::{make_mro, martian_filetype, MartianStruct};
use martian_filetypes::json_file::JsonFile;
use martian_filetypes::FileTypeRead;
use serde::{Deserialize, Serialize};
use slide_design::{VisiumHdLayout, VisiumHdSlide};

martian_filetype! {JpgFile, "jpg"}

#[derive(Debug, Clone, Serialize, Deserialize, MartianStruct)]
pub struct UmiRegOutsSubset {
    transform_matrices: JsonFile<TransformMatrices>,
    tissue_on_spots: JpgFile,
}

#[derive(Debug, Clone, Serialize, Deserialize, MartianStruct)]
pub struct CreateHdFeatureSliceStageInputs {
    sample_id: String,
    sample_desc: Option<String>,
    raw_matrix_h5: CountMatrixFile,
    hd_layout_data_json: Option<JsonFile<VisiumHdLayout>>,
    visium_hd_slide_name: String,
    barcode_summary_h5: H5File,
    umi_registration_outs: Option<UmiRegOutsSubset>,
}

#[derive(Debug, Clone, Serialize, Deserialize, MartianStruct)]
pub struct CreateHdFeatureSliceStageOutputs {
    hd_feature_slice: H5File,
}

pub struct CreateHdFeatureSlice;

#[make_mro(volatile = strict)]
impl MartianStage for CreateHdFeatureSlice {
    type StageInputs = CreateHdFeatureSliceStageInputs;
    type StageOutputs = CreateHdFeatureSliceStageOutputs; // Use `MartianVoid` if empty
    type ChunkInputs = MartianVoid;
    type ChunkOutputs = MartianVoid;

    fn split(
        &self,
        args: Self::StageInputs,
        _rover: MartianRover,
    ) -> Result<StageDef<Self::ChunkInputs>> {
        Ok(StageDef::with_join_resource(Resource::with_mem_gb(
            (4.0 + args.raw_matrix_h5.estimate_mem_gib()? * 2.0).ceil() as isize,
        )))
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
        CreateHdFeatureSliceStageInputs {
            sample_id,
            sample_desc,
            raw_matrix_h5,
            hd_layout_data_json,
            visium_hd_slide_name,
            barcode_summary_h5,
            umi_registration_outs,
        }: Self::StageInputs,
        _chunk_defs: Vec<Self::ChunkInputs>,
        _chunk_outs: Vec<Self::ChunkOutputs>,
        rover: MartianRover,
    ) -> Result<Self::StageOutputs> {
        let hd_feature_slice: H5File = rover.make_path("hd_feature_slice");

        let raw_matrix = raw_matrix_h5.read()?;
        let layout = hd_layout_data_json.map(|json| json.read()).transpose()?;
        let slide = VisiumHdSlide::from_name_and_layout(&visium_hd_slide_name, layout)?;
        let grid_index_of_barcode =
            FeatureSliceH5Writer::compute_grid_index_of_barcode(&raw_matrix);

        let transform_matrices = umi_registration_outs
            .as_ref()
            .map(|umi_reg| umi_reg.transform_matrices.read())
            .transpose()?;

        let writer = FeatureSliceH5Writer::new(&hd_feature_slice)?;
        writer.write_attributes(&rover.pipelines_version())?;
        writer.write_metadata(sample_id, sample_desc, &slide, transform_matrices)?;
        writer.write_feature_ref(raw_matrix.feature_reference())?;
        writer.write_feature_slices(
            &raw_matrix,
            &slide,
            &grid_index_of_barcode,
            raw_matrix.feature_reference().target_set.as_ref(),
        )?;
        writer.write_reads_group(&barcode_summary_h5, &grid_index_of_barcode)?;

        if let Some(umi_reg) = umi_registration_outs {
            writer.write_microscope_image_on_spots(&umi_reg.tissue_on_spots)?;
        }

        Ok(CreateHdFeatureSliceStageOutputs { hd_feature_slice })
    }
}
