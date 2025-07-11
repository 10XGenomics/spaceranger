#![allow(missing_docs)]
use crate::metadata::MatrixMetadata;
use crate::METADATA_JSON_ATTR_NAME;
use anyhow::Result;
use hdf5::types::VarLenUnicode;
use hdf5::{Extent, H5Type};
use itertools::izip;
use ndarray::{Array1, Array2};
use num_traits::Zero;
use slide_design::GridIndex2D;
use std::str::FromStr;

#[derive(Default, Clone)]
pub(crate) struct CooMatrix<T: H5Type + Zero + Default> {
    row: Vec<u32>,
    col: Vec<u32>,
    data: Vec<T>,
}

impl<T: H5Type + Zero + Default + Copy> From<Array2<T>> for CooMatrix<T> {
    fn from(array: Array2<T>) -> Self {
        let mut coo_matrix = Self::default();
        for ((r, c), &d) in array.indexed_iter() {
            coo_matrix.insert(
                &GridIndex2D {
                    row: r as u32,
                    col: c as u32,
                },
                d,
            );
        }
        coo_matrix
    }
}

impl<T: H5Type + Zero + Default> CooMatrix<T> {
    pub(crate) const ROW_DATASET_NAME: &'static str = "row";
    pub(crate) const COL_DATASET_NAME: &'static str = "col";
    pub(crate) const DATA_DATASET_NAME: &'static str = "data";

    pub fn is_empty(&self) -> bool {
        self.row.is_empty()
    }

    pub(crate) fn insert(&mut self, GridIndex2D { row, col }: &GridIndex2D, data: T) {
        if !data.is_zero() {
            self.row.push(*row);
            self.col.push(*col);
            self.data.push(data);
        }
    }

    pub(crate) fn from_iter<'a>(iter: impl Iterator<Item = (&'a GridIndex2D, T)>) -> Self {
        let mut coo_matrix = Self::default();
        for (index, data) in iter {
            coo_matrix.insert(index, data);
        }
        coo_matrix
    }

    fn write_to<D: H5Type>(group: &mut hdf5::Group, name: &str, data: Vec<D>) -> Result<()> {
        let data = Array1::from_vec(data);

        group
            .new_dataset::<D>()
            .shuffle()
            .deflate(1)
            .shape(Extent::new(data.len(), None))
            .chunk(10000)
            .create(name)?
            .as_writer()
            .write(data.view())?;
        Ok(())
    }

    pub(crate) fn write_to_h5(
        self,
        group: &mut hdf5::Group,
        metadata: MatrixMetadata,
    ) -> Result<()> {
        let metadata = VarLenUnicode::from_str(&serde_json::to_string(&metadata)?)?;
        cr_h5::scalar_attribute(group, METADATA_JSON_ATTR_NAME, metadata)?;
        Self::write_to(group, Self::ROW_DATASET_NAME, self.row)?;
        Self::write_to(group, Self::COL_DATASET_NAME, self.col)?;
        Self::write_to(group, Self::DATA_DATASET_NAME, self.data)?;
        Ok(())
    }

    pub(crate) fn into_iter(self) -> impl Iterator<Item = ([usize; 2], T)> {
        izip!(self.row, self.col, self.data).map(move |(r, c, d)| ([r as usize, c as usize], d))
    }

    pub(crate) fn load_from_h5_group(group: &hdf5::Group) -> Result<Self> {
        let (row, offset) = group
            .dataset(Self::ROW_DATASET_NAME)?
            .read_1d::<u32>()?
            .into_raw_vec_and_offset();
        assert_eq!(offset, Some(0));
        let (col, offset) = group
            .dataset(Self::COL_DATASET_NAME)?
            .read_1d::<u32>()?
            .into_raw_vec_and_offset();
        assert_eq!(offset, Some(0));
        let (data, offset) = group
            .dataset(Self::DATA_DATASET_NAME)?
            .read_1d::<T>()?
            .into_raw_vec_and_offset();
        assert_eq!(offset, Some(0));
        Ok(CooMatrix { row, col, data })
    }
}
