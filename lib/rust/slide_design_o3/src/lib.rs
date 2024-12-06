#![allow(non_snake_case)]

use numpy::PyArray2;
use pyanyhow::Result;
use pyo3::prelude::*;
use slide_design::{
    BarcodePartEmbeddings, CircularFiducial, GridIndex2D, OligoPart, Point2D, Synthesis, Transform,
    VisiumHdLayout, VisiumHdSlide,
};

#[pyclass]
struct PySpot {
    #[pyo3(get)]
    center: PyPoint2D,
    #[pyo3(get)]
    grid_index: PyGridIndex2D,
    #[pyo3(get)]
    barcode: Option<String>,
}

#[derive(Clone, Copy)]
#[pyclass]
struct PyGridIndex2D {
    #[pyo3(get)]
    row: u32,
    #[pyo3(get)]
    col: u32,
}

impl From<GridIndex2D> for PyGridIndex2D {
    fn from(GridIndex2D { row, col }: GridIndex2D) -> Self {
        PyGridIndex2D { row, col }
    }
}

#[derive(Clone, Copy)]
#[pyclass]
struct PyPoint2D {
    #[pyo3(get)]
    x: f32,
    #[pyo3(get)]
    y: f32,
}

impl From<Point2D> for PyPoint2D {
    fn from(Point2D { x, y }: Point2D) -> Self {
        PyPoint2D { x, y }
    }
}

#[derive(Clone)]
#[pyclass]
struct PyCircularFiducial {
    #[pyo3(get)]
    center: PyPoint2D,
    #[pyo3(get)]
    code: u32,
}

impl From<CircularFiducial> for PyCircularFiducial {
    fn from(
        CircularFiducial {
            center,
            code,
            rings: _,
        }: CircularFiducial,
    ) -> Self {
        PyCircularFiducial {
            center: center.unwrap().into(),
            code,
        }
    }
}

#[pyclass]
struct VisiumHdSlideWrapper(VisiumHdSlide);

#[pymethods]
impl VisiumHdSlideWrapper {
    #[new]
    fn from_name_and_layout(slide_name: &str, layout: Option<VisiumHdLayout>) -> Result<Self> {
        Ok(VisiumHdSlideWrapper(VisiumHdSlide::from_name_and_layout(
            slide_name, layout,
        )?))
    }

    fn name(&self) -> &str {
        self.0.name()
    }

    fn layout_str(&self) -> Result<Option<String>> {
        Ok(self.0.layout_str()?)
    }

    fn layout(&self) -> Option<VisiumHdLayout> {
        self.0.layout().clone()
    }

    fn spot_pitch(&self) -> f32 {
        self.0.spot_pitch()
    }

    fn spot_size(&self) -> f32 {
        self.0.spot_size()
    }

    #[pyo3(signature = (with_barcode = false))]
    fn spots(&self, with_barcode: bool) -> Vec<PySpot> {
        self.0
            .spots()
            .map(|spot| {
                let grid_index = spot.grid_index();
                let barcode = if with_barcode {
                    Some(self.0.barcode(spot.barcode_index.as_ref().unwrap()))
                } else {
                    None
                };
                PySpot {
                    center: spot.center.unwrap().into(),
                    grid_index: grid_index.into(),
                    barcode,
                }
            })
            .collect()
    }

    fn spot_xy_with_transform<'py>(
        &self,
        py: Python<'py>,
        transform: &PyArray2<f64>,
    ) -> &'py PyArray2<f64> {
        let transform = Transform::from_owned_array(transform.to_owned_array());
        PyArray2::from_vec2(
            py,
            &self
                .0
                .spots()
                .map(|spot| {
                    let center = spot.center.unwrap();
                    let (x, y) = transform.apply((center.x as f64, center.y as f64));
                    vec![x, y]
                })
                .collect::<Vec<_>>(),
        )
        .unwrap()
    }

    fn has_two_part_barcode(&self) -> bool {
        self.0.has_two_part_barcode()
    }

    fn num_rows(&self, binning_scale: Option<u32>) -> usize {
        self.0.num_rows(binning_scale)
    }

    fn num_cols(&self, binning_scale: Option<u32>) -> usize {
        self.0.num_cols(binning_scale)
    }

    fn num_spots(&self, binning_scale: Option<u32>) -> usize {
        self.0.num_spots(binning_scale)
    }

    fn grid_size(&self, binning_scale: Option<u32>) -> PyGridIndex2D {
        self.0.grid_size(binning_scale).into()
    }

    pub fn circular_fiducials(&self) -> Vec<PyCircularFiducial> {
        self.0
            .circular_fiducials()
            .iter()
            .cloned()
            .map(Into::into)
            .collect()
    }

    pub fn circular_fiducial_ring_widths(&self) -> Option<Vec<f32>> {
        self.0.circular_fiducial_ring_widths().map(<[f32]>::to_vec)
    }

    pub fn circular_fiducial_outer_radius(&self) -> Option<f32> {
        self.0.circular_fiducial_outer_radius()
    }

    pub fn bc1_oligos(&self) -> Vec<String> {
        self.0.oligos(OligoPart::Bc1).to_vec()
    }

    pub fn bc2_oligos(&self) -> Vec<String> {
        self.0.oligos(OligoPart::Bc2).to_vec()
    }

    pub fn bc1_embeddings<'py>(&self, py: Python<'py>) -> &'py PyArray2<bool> {
        PyArray2::from_vec2(py, &self.embeddings(|s| s.bc1.as_ref().unwrap())).unwrap()
    }

    pub fn bc2_embeddings<'py>(&self, py: Python<'py>) -> &'py PyArray2<bool> {
        PyArray2::from_vec2(py, &self.embeddings(|s| s.bc2.as_ref().unwrap())).unwrap()
    }

    pub fn transform_spot_colrow_to_xy<'py>(&self, py: Python<'py>) -> &'py PyArray2<f64> {
        PyArray2::from_owned_array(py, self.0.transform_spot_colrow_to_xy().inner())
    }
}

impl VisiumHdSlideWrapper {
    fn embeddings<F: Fn(&Synthesis) -> &BarcodePartEmbeddings>(&self, f: F) -> Vec<Vec<bool>> {
        let synthesis = self
            .0
            .synthesis()
            .expect("Synthesis information not found in slide design file!");
        f(synthesis)
            .embeddings
            .iter()
            .map(|e| e.embedding.clone())
            .collect()
    }
}

/// Use the slide design code from python
#[pymodule]
fn slide_design_o3(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<VisiumHdSlideWrapper>()?;
    m.add_class::<VisiumHdLayout>()?;
    Ok(())
}
