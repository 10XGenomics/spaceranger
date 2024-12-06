use crate::coo_matrix::CooMatrix;
use crate::metadata::{MatrixMetadata, Metadata, TransformMatrices};
use anyhow::Result;
use barcode::binned::SquareBinIndex;
use barcode::Barcode;
use cr_h5::count_matrix::{CountMatrix, RawCount};
use cr_types::reference::feature_reference::{FeatureReference, TargetSet};
use hdf5::types::VarLenUnicode;
use hdf5::{File, Group};
use itertools::zip_eq;
use ndarray::Array2;
use slide_design::{GridIndex2D, VisiumHdSlide};
use std::path::Path;
use std::str::FromStr;

pub mod attribute_names {
    pub const METADATA_JSON: &str = "metadata_json";
    pub const FILETYPE: &str = "filetype";
    pub const VERSION: &str = "version";
    pub const SOFTWARE_VERSION: &str = "software_version";
}
pub mod attributes {
    pub const FEATURE_SLICE_FILETYPE: &str = "feature_slice";
    pub const FEATURE_SLICE_VERSION: &str = "0.1.0";
}

pub mod group_names {
    pub const FEATURE_REFERENCE: &str = "features";
    pub const FEATURE_SLICES: &str = "feature_slices";

    pub const UMIS: &str = "umis";
    pub const TOTAL_UMIS: &str = "total";

    pub const READS: &str = "reads";

    pub const IMAGES: &str = "images";
    pub const MICROSCOPE_IMAGE: &str = "microscope";
}

pub struct FeatureSliceH5Writer {
    file: File,
}

impl FeatureSliceH5Writer {
    pub fn new(file: &Path) -> Result<Self> {
        let file = File::append(file)?;
        Ok(Self { file })
    }

    pub fn write_metadata(
        &self,
        sample_id: String,
        sample_desc: Option<String>,
        slide: &VisiumHdSlide,
        transform_matrices: Option<TransformMatrices>,
    ) -> Result<()> {
        let metadata = Metadata::new(sample_id, sample_desc, slide, transform_matrices)?;
        let metadata_json = serde_json::to_string(&metadata)?;
        cr_h5::scalar_attribute(
            &self.file,
            attribute_names::METADATA_JSON,
            VarLenUnicode::from_str(&metadata_json)?,
        )?;
        Ok(())
    }

    pub fn write_metadata_json(&self, metadata_json: &str) -> Result<()> {
        self.file
            .attr(attribute_names::METADATA_JSON)?
            .as_writer()
            .write(&ndarray::Array0::from_shape_vec(
                (),
                vec![VarLenUnicode::from_str(metadata_json)?],
            )?)?;
        Ok(())
    }

    pub fn write_reads_group(
        &self,
        barcode_summary_h5: &Path,
        grid_index_of_barcode: &[GridIndex2D],
    ) -> Result<()> {
        let barcode_summary = File::open(barcode_summary_h5)?;
        let reads_group = self.file.create_group(group_names::READS)?;
        for (bc_summary_name, feat_slice_group_name) in [
            ("sequenced_reads", "sequenced"),
            (
                "barcode_corrected_sequenced_reads",
                "barcode_corrected_sequenced",
            ),
            ("unmapped_reads", "unmapped_reads"),
            (
                "_multi_transcriptome_split_mapped_barcoded_reads",
                "split_mapped",
            ),
            (
                "_multi_transcriptome_half_mapped_barcoded_reads",
                "half_mapped",
            ),
        ] {
            if !barcode_summary.link_exists(bc_summary_name) {
                continue;
            }
            let data = barcode_summary.dataset(bc_summary_name)?.read_1d::<u32>()?;
            CooMatrix::from_iter(zip_eq(grid_index_of_barcode, data)).write_to_h5(
                &mut reads_group.create_group(feat_slice_group_name)?,
                MatrixMetadata::raw_counts(),
            )?;
        }
        Ok(())
    }

    pub fn write_attributes(&self, software_version: &str) -> Result<()> {
        cr_h5::scalar_attribute(
            &self.file,
            attribute_names::FILETYPE,
            VarLenUnicode::from_str(attributes::FEATURE_SLICE_FILETYPE)?,
        )?;
        cr_h5::scalar_attribute(
            &self.file,
            attribute_names::VERSION,
            VarLenUnicode::from_str(attributes::FEATURE_SLICE_VERSION)?,
        )?;
        cr_h5::scalar_attribute(
            &self.file,
            attribute_names::SOFTWARE_VERSION,
            VarLenUnicode::from_str(software_version)?,
        )?;
        Ok(())
    }

    pub fn write_feature_ref(&self, feature_ref: &FeatureReference) -> Result<()> {
        let mut group = self.file.create_group(group_names::FEATURE_REFERENCE)?;
        cr_h5::feature_reference_io::to_h5(feature_ref, &mut group)?;
        Ok(())
    }
    pub fn compute_grid_index_of_barcode(matrix: &CountMatrix) -> Vec<GridIndex2D> {
        matrix
            .barcodes()
            .iter()
            .map(|bc| {
                let SquareBinIndex { row, col, .. } =
                    Barcode::from_str(bc).unwrap().spatial_index();
                GridIndex2D {
                    row: row as u32,
                    col: col as u32,
                }
            })
            .collect()
    }
    pub fn write_feature_slices(
        &self,
        matrix: &CountMatrix,
        slide: &VisiumHdSlide,
        grid_index_of_barcode: &[GridIndex2D],
        target_set: Option<&TargetSet>,
    ) -> Result<()> {
        let group = self.file.create_group(group_names::FEATURE_SLICES)?;
        let mut feature_slices = vec![CooMatrix::<u32>::default(); matrix.num_features()];
        let mut total_umis = Array2::<u32>::zeros((slide.num_rows(None), slide.num_cols(None)));

        println!("Iterating over the matrix");
        for RawCount {
            count,
            barcode_idx,
            feature_idx,
        } in matrix.raw_counts()
        {
            let grid_index = &grid_index_of_barcode[barcode_idx];
            feature_slices[feature_idx].insert(grid_index, count as u32);
            if target_set.map_or(true, |trgt_set| trgt_set.is_on_target(feature_idx as u32)) {
                total_umis[[grid_index.row as usize, grid_index.col as usize]] += count as u32;
            }
        }
        println!("Writing feature slice");
        for (feature_idx, slice) in feature_slices
            .into_iter()
            .enumerate()
            .filter(|(_, slice)| !slice.is_empty())
        {
            if feature_idx % 100 == 0 {
                println!("Writing feature slice {feature_idx}");
            }
            slice.write_to_h5(
                &mut group.create_group(&feature_idx.to_string())?,
                MatrixMetadata::raw_counts(),
            )?;
        }
        println!("Writing total umis");
        let umis_group = self.file.create_group(group_names::UMIS)?;
        CooMatrix::from(total_umis).write_to_h5(
            &mut umis_group.create_group(group_names::TOTAL_UMIS)?,
            MatrixMetadata::raw_counts(),
        )?;
        Ok(())
    }
    pub fn write_microscope_image_on_spots(&self, image_path: &Path) -> Result<()> {
        self.write_image_on_spots(image_path, group_names::MICROSCOPE_IMAGE)
    }
    fn get_or_create_group(&self, group_name: &str) -> Result<Group> {
        if self.file.link_exists(group_name) {
            Ok(self.file.group(group_name)?)
        } else {
            Ok(self.file.create_group(group_name)?)
        }
    }
    fn create_image_subgroup(&self, subgroup_name: &str) -> Result<Group> {
        Ok(self
            .get_or_create_group(group_names::IMAGES)?
            .create_group(subgroup_name)?)
    }
    fn write_image_on_spots(&self, image_path: &Path, group_name: &str) -> Result<()> {
        let img = image::open(image_path)?.into_luma8();
        let nrows = img.height() as usize;
        let ncols = img.width() as usize;
        let arr = Array2::from_shape_vec((nrows, ncols), img.into_raw())?;
        CooMatrix::from(arr).write_to_h5(
            &mut self.create_image_subgroup(group_name)?,
            MatrixMetadata::image(),
        )?;
        Ok(())
    }
}

#[cfg(test)]
#[allow(dead_code)]
mod tests {
    use super::*;
    use anyhow::ensure;
    use hdf5::types::FixedAscii;
    use hdf5::{Container, File, Group};
    use itertools::Itertools;
    use pretty_assertions::assert_eq;

    fn compare_feature_slice_files(file1: &Path, file2: &Path) -> Result<()> {
        let file1 = File::open(file1)?;
        let file2 = File::open(file2)?;
        compare_groups(&file1, &file2)?;
        Ok(())
    }

    fn read_bytes(c: &Container) -> Result<Vec<u8>> {
        Ok(match c.dtype()?.to_descriptor().unwrap() {
            hdf5::types::TypeDescriptor::Float(_) => todo!(),
            hdf5::types::TypeDescriptor::VarLenUnicode => {
                c.read_scalar::<VarLenUnicode>()?.as_bytes().to_vec()
            }
            hdf5::types::TypeDescriptor::Integer(_) | hdf5::types::TypeDescriptor::Unsigned(_) => {
                c.read_raw::<u8>()?
            }
            hdf5::types::TypeDescriptor::Boolean => todo!(),
            hdf5::types::TypeDescriptor::Enum(_) => todo!(),
            hdf5::types::TypeDescriptor::Compound(_) => todo!(),
            hdf5::types::TypeDescriptor::FixedArray(_, _) => todo!(),
            hdf5::types::TypeDescriptor::FixedAscii(_) => c
                .read_raw::<FixedAscii<2048>>()?
                .iter()
                .flat_map(FixedAscii::as_bytes)
                .copied()
                .collect(),
            hdf5::types::TypeDescriptor::FixedUnicode(_) => todo!(),
            hdf5::types::TypeDescriptor::VarLenArray(_) => c
                .read_raw::<VarLenUnicode>()?
                .iter()
                .flat_map(VarLenUnicode::as_bytes)
                .copied()
                .collect(),
            hdf5::types::TypeDescriptor::VarLenAscii => todo!(),
        })
    }

    fn compare_container(c1: &Container, c2: &Container) -> Result<()> {
        ensure!(c1.name() == c2.name(), "Container names do not match");
        ensure!(
            c2.shape() == c2.shape(),
            "Container {} shapes do not match",
            c1.name()
        );
        println!(
            "Container {} types: {:?} and {:?}",
            c1.name(),
            c1.dtype()?.to_descriptor().unwrap(),
            c2.dtype()?.to_descriptor().unwrap()
        );

        if c1.name() == "/features/target_sets/Visium Mouse Transcriptome Probe Set v2.0" {
            return Ok(());
        }

        let v1 = read_bytes(c1)?;
        let v2 = read_bytes(c2)?;

        ensure!(v1 == v2, "Container {} values do not match", c1.name());

        Ok(())
    }

    fn sorted_coo(group: &Group) -> Result<Vec<(usize, usize, u32)>> {
        Ok(CooMatrix::<u32>::load_from_h5_group(group)?
            .into_iter()
            .map(|([r, c], d)| (r, c, d))
            .sorted()
            .collect())
    }

    fn compare_groups(group1: &Group, group2: &Group) -> Result<()> {
        // Compare the names of the groups

        println!("Comparing {} and {}", group1.name(), group2.name());
        ensure!(group1.name() == group2.name(), "Group names do not match");
        // ensure!(group1.len() == group2.len(), "Group lengths do not match");

        // Compare the attributes in the groups
        for attr_name in group1
            .attr_names()?
            .into_iter()
            .chain(group2.attr_names()?)
            .unique()
        {
            println!("Comparing attribute {attr_name}");
            let attr1 = group1.attr(&attr_name)?;
            let attr2 = group2.attr(&attr_name)?;

            if attr_name == attribute_names::METADATA_JSON {
                let json1: serde_json::Value = serde_json::from_slice(&read_bytes(&attr1)?)?;
                let json2: serde_json::Value = serde_json::from_slice(&read_bytes(&attr2)?)?;
                assert_eq!(json1, json2, "Attribute {attr_name} values do not match");
            } else {
                compare_container(&attr1, &attr2)?;
            }
        }

        println!("Looking at datasets");

        if group1.name() == "/umis/total" {
            let data1 = sorted_coo(group1)?;
            let data2 = sorted_coo(group2)?;
            assert_eq!(data1, data2, "Total UMIs do not match");
            return Ok(());
        }

        let member_names: Vec<_> = group1
            .member_names()?
            .into_iter()
            .chain(group2.member_names()?)
            .unique()
            .collect();

        // Compare the datasets in the groups
        for dataset_name in &member_names {
            println!("Comparing dataset {dataset_name}");
            let dataset1 = group1.dataset(dataset_name);
            let dataset2 = group2.dataset(dataset_name);
            if dataset1.is_err() && dataset2.is_err() {
                continue;
            }
            ensure!(
                dataset1.is_ok() && dataset2.is_ok(),
                "Dataset {} does not exist in both groups",
                dataset_name
            );
            let dataset1 = dataset1?;
            let dataset2 = dataset2?;

            compare_container(&dataset1, &dataset2)?;
        }

        println!("Looking at subgroups");

        // Compare the subgroups in the groups
        for subgroup_name in member_names.iter().take(2000) {
            let subgroup1 = group1.group(subgroup_name);
            let subgroup2 = group2.group(subgroup_name);
            if subgroup1.is_err() && subgroup2.is_err() {
                continue;
            }
            ensure!(
                subgroup1.is_ok() && subgroup2.is_ok(),
                "Subgroup {} does not exist in both groups",
                subgroup_name
            );
            compare_groups(&subgroup1?, &subgroup2?)?;
        }

        Ok(())
    }
}
