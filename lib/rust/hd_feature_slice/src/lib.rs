//! Interface with the HD feature slice H5 file.
//! Should be in sync with lib/python/cellranger/spatial/hd_feature_slice.py
#![allow(missing_docs)]

use anyhow::{bail, Context, Result};
use cr_h5::feature_reference_io::from_h5;
use cr_types::reference::feature_reference::{FeatureDef, FeatureReference};
use dzi::PixelRange;
use hdf5::types::VarLenUnicode;
use hdf5::{Attribute, H5Type};
use io::group_names::{
    CLUSTERING, EIGHT_MICRON_SQUARE_PREFIX, GENE_EXPRESSION_PREFIX, GRAPHCLUST_SUFFIX,
    SECONDARY_ANALYSIS,
};
use itertools::Itertools;
use ndarray::Array2;
use num_traits::Zero;
use serde::de::DeserializeOwned;
use slide_design::{Transform, VisiumHdLayout, VisiumHdSlide};
use std::collections::HashMap;
use std::ops::AddAssign;
use std::path::Path;

pub mod coo_matrix;
pub mod io;
pub mod metadata;
pub mod scaling;
pub mod view;

use crate::coo_matrix::CooMatrix;
use crate::metadata::{MatrixDataKind, MatrixMetadata, Metadata};

/// Possible values that are stored in the feature slice h5
#[derive(Clone, Debug)]
pub enum MatrixValue {
    RawCounts(Array2<u32>),
    BinaryMask(Array2<u8>),
    Categorical(Array2<u32>),
    GrayImageU8(Array2<u8>),
    Float(Array2<f64>),
}

impl MatrixValue {
    pub fn kind(&self) -> MatrixDataKind {
        match self {
            MatrixValue::RawCounts(_) => MatrixDataKind::RawCounts,
            MatrixValue::BinaryMask(_) => MatrixDataKind::BinaryMask,
            MatrixValue::Categorical(_) => MatrixDataKind::Categorical,
            MatrixValue::GrayImageU8(_) => MatrixDataKind::GrayImageU8,
            MatrixValue::Float(_) => MatrixDataKind::Float,
        }
    }
}

pub(crate) const METADATA_JSON_ATTR_NAME: &str = "metadata_json";

/// Load an attribute which stores a json string and deserialize it into the
/// appropriate type
fn _load_json_attr<T: DeserializeOwned>(attr: Attribute) -> Result<T> {
    Ok(serde_json::from_str(
        attr.read_scalar::<VarLenUnicode>()?.as_str(),
    )?)
}

/// Interface for reading the feature slice H5
pub struct FeatureSliceH5 {
    file: hdf5::File,
    metadata: Metadata,
    feature_ref: FeatureReference,
}

impl FeatureSliceH5 {
    pub fn slide(&self) -> Result<VisiumHdSlide> {
        VisiumHdSlide::from_name_and_layout(
            &self.metadata.slide_name,
            self.metadata
                .hd_layout_json
                .as_ref()
                .map(|s| VisiumHdLayout::from_json_str(s))
                .transpose()
                .map_err(|x| x.0)?,
        )
    }
    pub fn metadata(&self) -> &Metadata {
        &self.metadata
    }

    pub fn link_exists(&self, link_path: &str) -> bool {
        self.file.link_exists(link_path)
    }

    pub fn has_eight_micron_gex_graph_clust(&self) -> bool {
        self.link_exists(
            format!(
                "{SECONDARY_ANALYSIS}/{CLUSTERING}/{EIGHT_MICRON_SQUARE_PREFIX}_{GENE_EXPRESSION_PREFIX}_{GRAPHCLUST_SUFFIX}"
            )
            .as_str(),
        )
    }

    /// Transform matrix from the microscope image to the spot
    /// Only present for spatial runs
    pub fn transform_microscope_to_spot(&self) -> Option<&Transform> {
        self.metadata
            .transform_matrices
            .as_ref()
            .and_then(|t| t.microscope_colrow_to_spot_colrow.as_ref())
    }

    /// Transform matrix from the spots to microscope image
    /// Only present for spatial runs
    pub fn transform_spot_to_microscope(&self) -> Option<&Transform> {
        self.metadata
            .transform_matrices
            .as_ref()
            .and_then(|t| t.spot_colrow_to_microscope_colrow.as_ref())
    }

    /// Transform matrix from the spots to cytassist image
    /// Only present for spatial runs
    pub fn transform_spot_to_cytassist(&self) -> Option<&Transform> {
        self.metadata
            .transform_matrices
            .as_ref()
            .and_then(|t| t.spot_colrow_to_cytassist_colrow.as_ref())
    }

    /// Compute the bin scale given a bin size. Return an error
    /// if the bin size in not a multiple of spot pitch
    pub fn bin_scale_from_um(&self, bin_size_um: f64) -> Result<usize> {
        let scale = bin_size_um / self.metadata.spot_pitch;
        if scale.fract() != 0.0 {
            bail!(
                "Bin size should be a multiple of spot pitch {}",
                self.metadata.spot_pitch
            )
        }
        Ok(scale as usize)
    }

    fn _load_metadata(file: &hdf5::File) -> Result<Metadata> {
        _load_json_attr(file.attr(METADATA_JSON_ATTR_NAME)?)
    }

    /// Open a feature slice h5 from the given path
    pub fn open(path: &Path) -> Result<Self> {
        let file = hdf5::File::open(path)?;
        Ok(FeatureSliceH5 {
            metadata: Self::_load_metadata(&file)?,
            feature_ref: from_h5(&file.group("features")?)?,
            file,
        })
    }

    /// Buffer to store data at the specific bin scale
    pub fn allocate_buffer<T: Clone + Zero>(&self, bin_scale: usize) -> Array2<T> {
        Array2::zeros((
            self.metadata.nrows.div_ceil(bin_scale),
            self.metadata.ncols.div_ceil(bin_scale),
        ))
    }

    fn _get_indices_from_name_or_id(&self, feature_name_or_id: &[String]) -> Result<Vec<usize>> {
        let mut ensembl_id_map = HashMap::new();
        for FeatureDef {
            index, name, id, ..
        } in &self.feature_ref.feature_defs
        {
            let value = (*index, id);
            for key in [name, id] {
                ensembl_id_map
                    .entry(key.to_lowercase())
                    .or_insert_with(Vec::new)
                    .push(value);
            }
        }
        feature_name_or_id
            .iter()
            .map(|name_or_id| {
                match ensembl_id_map
                    .get(&name_or_id.to_lowercase())
                    .map(Vec::as_slice)
                {
                    Some([]) => unreachable!(),
                    Some([(index, _)]) => Ok(*index),
                    Some(multiple) => bail!(
                        "'{name_or_id}' matches multiple features. Specify one of '{}' instead",
                        multiple.iter().map(|(_, id)| id).join(", ")
                    ),
                    None => bail!("Feature '{name_or_id}' is not known"),
                }
            })
            .collect()
    }

    /// Sum up feature counts from a list of name of ensembl ID. Returns an error if the
    /// name/id is not found or not unique
    pub fn load_sum_feature_counts_from_names(
        &self,
        feature_name_or_id: &[String],
        bin_scale: usize,
    ) -> Result<Array2<u32>> {
        let indices = self._get_indices_from_name_or_id(feature_name_or_id)?;
        self.load_sum_feature_counts(&indices, bin_scale)
    }

    pub fn load_sum_feature_counts(
        &self,
        feature_indices: &[usize],
        bin_scale: usize,
    ) -> Result<Array2<u32>> {
        let feat_slice_group = self.file.group("feature_slices")?;
        let mut sum_feature_counts = self.allocate_buffer(bin_scale);
        for &index in feature_indices.iter().unique() {
            if let Ok(feat_group) = feat_slice_group.group(&index.to_string()) {
                for ([row, col], count) in
                    CooMatrix::<u32>::load_from_h5_group(&feat_group)?.into_iter()
                {
                    sum_feature_counts[[row / bin_scale, col / bin_scale]] += count;
                }
            }
        }
        Ok(sum_feature_counts)
    }

    fn _collect_into<T: Clone + Zero + H5Type + AddAssign + Default>(
        &self,
        group: hdf5::Group,
        bin_scale: usize,
    ) -> Result<Array2<T>> {
        let mut arr = self.allocate_buffer(bin_scale);
        for (index, count) in CooMatrix::<T>::load_from_h5_group(&group)?.into_iter() {
            arr[index] += count;
        }
        Ok(arr)
    }

    // Returns an error if the requested group is not storing counts
    pub fn load_counts_from_group_name(
        &self,
        group_name: &str,
        bin_scale: usize,
    ) -> Result<Array2<u32>> {
        assert!(bin_scale != 0);
        let counts = match self.load_slice_from_group_name(group_name)? {
            MatrixValue::RawCounts(c) => c,
            data => bail!("Expected counts, found {:?}", data.kind()),
        };
        if bin_scale == 1 {
            return Ok(counts);
        }
        let mut buffer = self.allocate_buffer(bin_scale);
        for ((row, col), count) in counts.indexed_iter() {
            buffer[[row / bin_scale, col / bin_scale]] += count;
        }
        Ok(buffer)
    }

    pub fn load_slice_from_group_name(&self, group_name: &str) -> Result<MatrixValue> {
        let group = self
            .file
            .group(group_name)
            .context(format!("While opening {group_name}"))?;
        let metadata = group
            .attr(METADATA_JSON_ATTR_NAME)
            .map_err(anyhow::Error::from)
            .and_then(_load_json_attr::<MatrixMetadata>)
            .unwrap_or_default();
        Ok(match metadata.kind {
            MatrixDataKind::RawCounts => {
                MatrixValue::RawCounts(self._collect_into(group, metadata.binning_scale as usize)?)
            }
            MatrixDataKind::Categorical => MatrixValue::Categorical(
                self._collect_into(group, metadata.binning_scale as usize)?,
            ),
            MatrixDataKind::BinaryMask => {
                MatrixValue::BinaryMask(self._collect_into(group, metadata.binning_scale as usize)?)
            }
            MatrixDataKind::GrayImageU8 => MatrixValue::GrayImageU8(
                self._collect_into(group, metadata.binning_scale as usize)?,
            ),
            MatrixDataKind::Float => {
                MatrixValue::Float(self._collect_into(group, metadata.binning_scale as usize)?)
            }
        })
    }

    pub fn load_total_umi(&self, bin_scale: usize) -> Result<Array2<u32>> {
        let mut umis = self.allocate_buffer(bin_scale);

        if let Ok(group) = self.file.group("umis/total") {
            for ([row, col], count) in CooMatrix::<u32>::load_from_h5_group(&group)?.into_iter() {
                umis[[row / bin_scale, col / bin_scale]] += count;
            }
        }

        Ok(umis)
    }

    pub fn load_total_reads(&self, bin_scale: usize) -> Result<Array2<u32>> {
        const GROUP_NAME: &str = "reads/sequenced";
        if !self.file.link_exists(GROUP_NAME) {
            return Ok(self.allocate_buffer(bin_scale));
        }
        self.load_counts_from_group_name(GROUP_NAME, bin_scale)
    }

    /// Given a pixel range in the microscope image, return an iterator of indices
    /// in the microscope image alond with the spot indices if it exists
    pub fn spot_indices_of_microscope_pixel_range(
        &self,
        pixel_range: &PixelRange,
    ) -> impl Iterator<Item = MicroscopeSpotIndexMap> {
        let shape = (pixel_range.nrows(), pixel_range.ncols());

        let spots_nrows = self.metadata.nrows as isize;
        let spots_ncols = self.metadata.ncols as isize;

        let x0 = pixel_range.x.start;
        let y0 = pixel_range.y.start;

        let transform = self.transform_microscope_to_spot().unwrap().clone();

        ndarray::indices(shape)
            .into_iter()
            .filter_map(move |(y, x)| {
                // Pixel coordinates are corner-based. Add 0.5 to get the middle of the pixel
                let pxl_center_x = (x + x0) as f64 + 0.5;
                let pxl_center_y = (y + y0) as f64 + 0.5;
                let (xt, yt) = transform.apply((pxl_center_x, pxl_center_y));
                let xt = xt.round() as isize;
                let yt = yt.round() as isize;
                if (xt >= 0 && xt < spots_ncols) && (yt >= 0 && yt < spots_nrows) {
                    Some(MicroscopeSpotIndexMap {
                        spot_row_col: (yt as usize, xt as usize),
                        microscope_row_col: (y, x),
                    })
                } else {
                    None
                }
            })
    }

    pub fn counts_in_microscope_pixel_range(
        &self,
        spot_counts: &Array2<u32>,
        pixel_range: &PixelRange,
    ) -> Array2<u32> {
        let mut array = Array2::zeros((pixel_range.nrows(), pixel_range.ncols()));

        for MicroscopeSpotIndexMap {
            spot_row_col,
            microscope_row_col,
        } in self.spot_indices_of_microscope_pixel_range(pixel_range)
        {
            array[microscope_row_col] = spot_counts[spot_row_col];
        }
        array
    }
}

pub struct MicroscopeSpotIndexMap {
    pub spot_row_col: (usize, usize),
    pub microscope_row_col: (usize, usize),
}
