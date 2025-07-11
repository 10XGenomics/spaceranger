#![allow(missing_docs)]
use anyhow::Result;
use serde::{Deserialize, Serialize};
use slide_design::{Transform, VisiumHdSlide};

/// Different flavors of data stored in the H5
#[derive(Default, Serialize, Deserialize, Debug, Clone, Copy)]
#[serde(rename_all = "snake_case")]
pub enum MatrixDataKind {
    #[default]
    RawCounts,
    Categorical,
    BinaryMask,
    GrayImageU8,
    Float,
}

#[derive(Default, Serialize, Deserialize, Clone, Copy)]
#[serde(rename_all = "snake_case")]
pub enum MatrixDataDomain {
    #[default]
    AllSpots,
    FilteredSpots,
}

fn default_binning_scale() -> u32 {
    1
}

#[derive(Serialize, Deserialize, Clone, Copy)]
pub struct MatrixMetadata {
    pub kind: MatrixDataKind,
    pub domain: MatrixDataDomain,
    #[serde(default = "default_binning_scale")]
    pub binning_scale: u32,
}

impl Default for MatrixMetadata {
    fn default() -> Self {
        MatrixMetadata {
            kind: MatrixDataKind::RawCounts,
            domain: MatrixDataDomain::AllSpots,
            binning_scale: 1,
        }
    }
}

impl MatrixMetadata {
    pub fn raw_counts() -> Self {
        MatrixMetadata::default()
    }
    pub fn image() -> Self {
        MatrixMetadata {
            kind: MatrixDataKind::GrayImageU8,
            domain: MatrixDataDomain::AllSpots,
            binning_scale: 1,
        }
    }
}

/// Note: Transforms map spot indices to positions of spot centers in
/// *corner-based* sub-pixel image coordinates, and vice versa. This means the
/// sub-spot coordinate system is *center-based*.
#[derive(Serialize, Deserialize, Clone)]
pub struct TransformMatrices {
    pub spot_colrow_to_microscope_colrow: Option<Transform>,
    pub microscope_colrow_to_spot_colrow: Option<Transform>,
    pub spot_colrow_to_cytassist_colrow: Option<Transform>,
    pub cytassist_colrow_to_spot_colrow: Option<Transform>,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct Metadata {
    #[serde(default)]
    pub sample_id: String,
    pub sample_desc: Option<String>,
    pub nrows: usize,
    pub ncols: usize,
    pub slide_name: String,
    pub spot_pitch: f64,
    pub hd_layout_json: Option<String>,
    pub transform_matrices: Option<TransformMatrices>,
}

impl Metadata {
    pub fn new(
        sample_id: String,
        sample_desc: Option<String>,
        slide: &VisiumHdSlide,
        transform_matrices: Option<TransformMatrices>,
    ) -> Result<Self> {
        Ok(Metadata {
            sample_id,
            sample_desc,
            nrows: slide.num_rows(None),
            ncols: slide.num_cols(None),
            slide_name: slide.name().to_string(),
            spot_pitch: slide.spot_pitch() as f64,
            hd_layout_json: slide.layout_str()?,
            transform_matrices,
        })
    }
}
