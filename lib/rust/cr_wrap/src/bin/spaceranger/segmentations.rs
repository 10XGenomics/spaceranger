use anyhow::{bail, Result};
use cr_wrap::utils::CliPath;
use serde::Serialize;
use std::str::FromStr;

const TIF_EXTENSION: &str = "tif";
const TIFF_EXTENSION: &str = "tiff";
const BTF_EXTENSION: &str = "btf";
const NPY_EXTENSION: &str = "npy";
const CSV_EXTENSION: &str = "csv";
const GEOJSON_EXTENSION: &str = "geojson";

#[derive(Debug, Clone)]
pub(crate) enum UserProvidedSegmentation {
    Tif(CliPath),
    Npy(CliPath),
    Csv(CliPath),
    Geojson(CliPath),
}

impl FromStr for UserProvidedSegmentation {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        let path: CliPath = s.parse()?;
        match path.extension().and_then(|ext| ext.to_str()) {
            Some(TIF_EXTENSION) | Some(TIFF_EXTENSION) | Some(BTF_EXTENSION) => {
                Ok(UserProvidedSegmentation::Tif(path))
            }
            Some(NPY_EXTENSION) => Ok(UserProvidedSegmentation::Npy(path)),
            Some(CSV_EXTENSION) => Ok(UserProvidedSegmentation::Csv(path)),
            Some(GEOJSON_EXTENSION) => Ok(UserProvidedSegmentation::Geojson(path)),
            _ => {
                bail!("Segmentation file must be either an instance mask TIFF \
                (extensions '.{TIF_EXTENSION}', '.{TIFF_EXTENSION}', '.{BTF_EXTENSION}') \
                or an instance mask NPY (extension '.{NPY_EXTENSION}') \
                or a GeoJSON with a FeatureCollection of polygons (extension '.{GEOJSON_EXTENSION}') \
                or CSV with a barcode to cell ID map (extension '.{CSV_EXTENSION}.')")
            }
        }
    }
}

impl UserProvidedSegmentation {
    pub(crate) fn instance_mask_tiff(&self) -> Option<CliPath> {
        if let UserProvidedSegmentation::Tif(path) = self {
            Some(path.clone())
        } else {
            None
        }
    }

    pub(crate) fn instance_mask_npy(&self) -> Option<CliPath> {
        if let UserProvidedSegmentation::Npy(path) = self {
            Some(path.clone())
        } else {
            None
        }
    }

    pub(crate) fn square_barcode_to_cell_map(&self) -> Option<CliPath> {
        if let UserProvidedSegmentation::Csv(path) = self {
            Some(path.clone())
        } else {
            None
        }
    }

    pub(crate) fn user_provided_segmentation_geojson(&self) -> Option<CliPath> {
        if let UserProvidedSegmentation::Geojson(path) = self {
            Some(path.clone())
        } else {
            None
        }
    }
}

#[derive(Serialize, Clone, Default)]

pub(crate) struct SegmentationInputs {
    user_provided_segmentations: Option<CliPath>,
    square_barcode_to_cell_map: Option<CliPath>,
    instance_mask_tiff: Option<CliPath>,
    instance_mask_npy: Option<CliPath>,
    max_nucleus_diameter_px: Option<u32>,
    barcode_assignment_distance_micron: Option<u32>,
}

impl SegmentationInputs {
    pub(crate) fn from_cli_inputs(
        custom_segmentation_file: Option<UserProvidedSegmentation>,
        max_nucleus_diameter_px: Option<u32>,
        barcode_assignment_distance_micron: Option<u32>,
    ) -> Self {
        Self {
            user_provided_segmentations: custom_segmentation_file
                .as_ref()
                .and_then(UserProvidedSegmentation::user_provided_segmentation_geojson),
            square_barcode_to_cell_map: custom_segmentation_file
                .as_ref()
                .and_then(UserProvidedSegmentation::square_barcode_to_cell_map),
            instance_mask_tiff: custom_segmentation_file
                .as_ref()
                .and_then(UserProvidedSegmentation::instance_mask_tiff),
            instance_mask_npy: custom_segmentation_file.and_then(|x| x.instance_mask_npy()),
            max_nucleus_diameter_px,
            barcode_assignment_distance_micron,
        }
    }
}
