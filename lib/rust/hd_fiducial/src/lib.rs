use fiducial::fiducial_detector::{turing_detect_fiducial, FidDetectionParameter};
use fiducial::fiducial_registration::{similarity_transform_components, Coord};
use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

#[pyfunction]
#[pyo3(name = "turing_detect_fiducial")]
#[allow(clippy::too_many_arguments)]
fn turing_detect_fiducial_py(
    img: PyReadonlyArray2<'_, f32>,
    search_start: Option<usize>,
    search_end: Option<usize>,
    edge_threshold: Option<f32>,
    parallel_threshold: Option<f32>,
    hough_accu_threshold: Option<f32>,
    circle_fit_support: Option<usize>,
    outlier_threshold: Option<f32>,
    label: Option<[f32; 3]>,
) -> Vec<(usize, [f32; 2])> {
    //let image = ImageUtil::read_r_channel(path);
    let image = img.as_array().to_owned();
    let param = FidDetectionParameter::new_py(
        search_start.unwrap_or(2),
        search_end.unwrap_or(10),
        edge_threshold.unwrap_or(0.1),
        parallel_threshold.unwrap_or(-0.9),
        hough_accu_threshold.unwrap_or(20.0),
        circle_fit_support.unwrap_or(10),
        outlier_threshold.unwrap_or(0.4),
        label,
    );
    turing_detect_fiducial(&image, &param, &None)
}

#[pyfunction]
fn fit_2d_similarity_transform(
    target_x: PyReadonlyArray1<'_, f64>,
    target_y: PyReadonlyArray1<'_, f64>,
    moving_x: PyReadonlyArray1<'_, f64>,
    moving_y: PyReadonlyArray1<'_, f64>,
) -> (f64, f64, f64, f64) {
    let tx = target_x.as_array();
    let ty = target_y.as_array();
    let target_xy: Vec<_> = tx
        .iter()
        .zip(ty.iter())
        .map(|(x, y)| Coord(*x, *y))
        .collect();

    let mx = moving_x.as_array();
    let my = moving_y.as_array();
    let moving_xy: Vec<_> = mx
        .iter()
        .zip(my.iter())
        .map(|(x, y)| Coord(*x, *y))
        .collect();

    similarity_transform_components(&moving_xy, &target_xy)
}

#[pymodule]
fn hd_fiducial(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(turing_detect_fiducial_py, m)?)?;
    m.add_function(wrap_pyfunction!(fit_2d_similarity_transform, m)?)?;
    Ok(())
}
