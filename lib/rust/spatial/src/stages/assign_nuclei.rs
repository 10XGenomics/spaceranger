//! AssignNuclei stage code
//! Martian stage ASSIGN_NUCLEI
#![allow(missing_docs)]

use crate::{H5File, NpyFile};
use anyhow::{Context, Result};
use hd_feature_slice::FeatureSliceH5;
use itertools::{iproduct, Itertools};
use martian::{MartianFileType, MartianMain, MartianRover};
use martian_derive::{make_mro, MartianStruct};
use ndarray::{stack, Array2, Axis};
use ndarray_npy::{ReadNpyExt, WriteNpyExt};
use serde::{Deserialize, Serialize};
use std::fs::{copy, hard_link, File};

// Maximum distance of a spot center from a polygon to be considered
// for nuclear assignment
const MAX_NUCLEAR_ASSIGNMENT_DISTANCE_MICRON: usize = 8;

#[derive(Debug, Clone, Serialize, Deserialize, MartianStruct)]
pub struct AssignNucleiStageInputs {
    // Image with segmentation number a spot is in. Segmentations are numbered
    // from 1 onwards and 0 indicates that the spot is in no segment.
    pub spot_mask: Option<NpyFile>,
    pub hd_feature_slice: Option<H5File>,
    // Some(0) indicates no barcode assignment. Some(x) expands by x microns
    // None expands by max distance
    pub barcode_assignment_distance_micron: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize, MartianStruct)]
pub struct AssignNucleiStageOutputs {
    expanded_spot_mask: Option<NpyFile>,
    // Matrix with the minimum squared euclidean distance of
    // a spot to a nucleus in spot-coordinate space.
    // f64::INFINITY for spots which have no nucleus within range
    minimum_distance_mask: Option<NpyFile>,
    // Coordinate of the closest nucleus to a spot.
    // This is the spot itself for spots under nuclei
    // and (-1, -1) for spots which are not assigned to nuclei
    closest_object_mask: Option<NpyFile>,
}

// This is our stage struct
pub struct AssignNuclei;

#[make_mro(mem_gb = 4, volatile = strict)]
impl MartianMain for AssignNuclei {
    type StageInputs = AssignNucleiStageInputs;
    type StageOutputs = AssignNucleiStageOutputs;
    fn main(&self, args: Self::StageInputs, rover: MartianRover) -> Result<Self::StageOutputs> {
        match (
            args.spot_mask,
            args.hd_feature_slice,
            args.barcode_assignment_distance_micron,
        ) {
            (Some(spot_mask_path), _, Some(0)) => {
                let expanded_spot_mask_path: NpyFile = rover.make_path("expanded_spot_mask.npy");
                hard_link(&spot_mask_path, &expanded_spot_mask_path).or_else(|_| {
                    println!("Ran into trouble while hard linking {spot_mask_path:?} to {expanded_spot_mask_path:?}");
                    copy(&spot_mask_path, &expanded_spot_mask_path).map(|_| ())
                }).with_context(||
                format!("Ran into trouble while hard linking and copying {spot_mask_path:?} to {expanded_spot_mask_path:?}"))?;

                Ok(AssignNucleiStageOutputs {
                    expanded_spot_mask: Some(expanded_spot_mask_path),
                    minimum_distance_mask: None,
                    closest_object_mask: None,
                })
            }
            (
                Some(spot_mask_path),
                Some(hd_feature_slice_path),
                max_nuclear_assignment_distance_micron,
            ) => {
                let feature_slice = FeatureSliceH5::open(&hd_feature_slice_path)?;
                let metadata = feature_slice.metadata();
                println!("num rows {}; num cols {}", metadata.nrows, metadata.ncols);

                let search_distance_spot_space = (max_nuclear_assignment_distance_micron
                    .unwrap_or(MAX_NUCLEAR_ASSIGNMENT_DISTANCE_MICRON)
                    as f64
                    / metadata.spot_pitch)
                    .ceil() as i64;
                let search_distance_squared_spot_space = search_distance_spot_space.pow(2);
                println!(
                    "max search euclidean distance in spot space {search_distance_spot_space}, ceiled sq euclidean distance {search_distance_squared_spot_space}"
                );

                let segmentation_mask = Array2::<u64>::read_npy(File::open(spot_mask_path)?)?;

                // Search space vec sorted by squared euclidean distance
                let search_space_vec: Vec<_> = iproduct!(
                    -search_distance_spot_space..=search_distance_spot_space,
                    -search_distance_spot_space..=search_distance_spot_space
                )
                .filter(|(x, y)| x.pow(2) + y.pow(2) <= search_distance_squared_spot_space)
                .sorted_by_key(|(x, y)| x.pow(2) + y.pow(2))
                .collect();
                println!(
                    "Searching a maximum of {} neighbours",
                    search_space_vec.len()
                );
                println!("Search vectors: {:?}", &search_space_vec);

                let mut extended_segmentation_mask = segmentation_mask.clone();
                // Minimum squared euclidean distance of a spot from a spot under nucleus
                let mut minimum_sq_euclidean_distance_array =
                    f64::INFINITY * Array2::<f64>::ones(segmentation_mask.dim());

                let temp_array = -1 * Array2::<i64>::ones(segmentation_mask.dim());
                // Closest spot under nucleus to a spot. (-1, -1) used as sentinel value
                // for spots not assigned to nuclei.
                let mut closest_object_location =
                    stack![Axis(2), temp_array, temp_array].to_owned();

                for ((x, y), &val) in segmentation_mask.indexed_iter() {
                    if val > 0 {
                        closest_object_location[[x, y, 0]] = x as i64;
                        closest_object_location[[x, y, 1]] = y as i64;
                        minimum_sq_euclidean_distance_array[[x, y]] = 0.0;
                    }
                }

                let nrows = minimum_sq_euclidean_distance_array.nrows() as i64;
                let ncols = minimum_sq_euclidean_distance_array.ncols() as i64;

                for ((x, y), minimum_object_distance) in
                    minimum_sq_euclidean_distance_array.indexed_iter_mut()
                {
                    if x % 100 == 0 && y % 100 == 0 {
                        println!("doing bounds expansion for row={x} col={y}");
                    }
                    let nearest_neighbour = search_space_vec
                        .iter()
                        // as dx, dy are sorted in ascending order by their distances
                        // the first non-None is the one we assign to
                        .find_map(|(dx, dy)| {
                            let (sx, sy) = (x as i64 + dx, y as i64 + dy);
                            if 0 <= sx
                                && sx < nrows
                                && 0 <= sy
                                && sy < ncols
                                && segmentation_mask[[sx as usize, sy as usize]] > 0
                            {
                                Some((
                                    sx as usize,
                                    sy as usize,
                                    segmentation_mask[[sx as usize, sy as usize]],
                                    dx.pow(2) + dy.pow(2),
                                ))
                            } else {
                                None
                            }
                        });

                    if let Some((
                        sx,
                        sy,
                        nearest_object_id,
                        nearest_object_min_sq_euclidiean_dist,
                    )) = nearest_neighbour
                    {
                        *minimum_object_distance = nearest_object_min_sq_euclidiean_dist as f64;
                        extended_segmentation_mask[[x, y]] = nearest_object_id;
                        closest_object_location[[x, y, 0]] = sx as i64;
                        closest_object_location[[x, y, 1]] = sy as i64;
                    }
                }

                let expanded_spot_mask_path: NpyFile = rover.make_path("expanded_spot_mask.npy");
                let writer = expanded_spot_mask_path.buf_writer()?;
                extended_segmentation_mask.write_npy(writer)?;

                let minimum_distance_mask_path: NpyFile =
                    rover.make_path("minimum_distance_mask.npy");
                let writer = minimum_distance_mask_path.buf_writer()?;
                minimum_sq_euclidean_distance_array.write_npy(writer)?;

                let closest_object_mask_path: NpyFile = rover.make_path("closest_object_mask.npy");
                let writer = closest_object_mask_path.buf_writer()?;
                closest_object_location.write_npy(writer)?;

                Ok(AssignNucleiStageOutputs {
                    expanded_spot_mask: Some(expanded_spot_mask_path),
                    minimum_distance_mask: Some(minimum_distance_mask_path),
                    closest_object_mask: Some(closest_object_mask_path),
                })
            }
            (_, _, _) => Ok(AssignNucleiStageOutputs {
                expanded_spot_mask: None,
                minimum_distance_mask: None,
                closest_object_mask: None,
            }),
        }
    }
}
