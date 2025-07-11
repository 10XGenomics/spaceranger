//! PreprocessNucleusSegmentationGeojson stage code
//! Martian stage PREPROCESS_NUCLEUS_SEGMENTATION_GEOJSON
#![allow(missing_docs)]

use crate::{GeoJsonFile, H5File, NpyFile};
use anyhow::{bail, Context, Result};
use bio::data_structures::interval_tree::IntervalTree;
use bio::utils::Interval;
use geo::{point, Contains};
use hd_feature_slice::FeatureSliceH5;
use itertools::{iproduct, Itertools};
use martian::{MartianFileType, MartianRover, MartianStage, Resource, StageDef};
use martian_derive::{make_mro, MartianStruct};
use martian_filetypes::json_file::JsonFile;
use martian_filetypes::FileTypeWrite;
use ndarray::{concatenate, Array2, ArrayBase, Axis};
use ndarray_npy::{ReadNpyExt, WriteNpyExt};
use roi::qupath::{write_geojson_collection_sequential_names, NamedPolygon};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs::{copy, hard_link, metadata as fs_metadata, File};

#[derive(Debug, Deserialize)]
pub struct BarcodeAndCell {
    pub barcode: String,
    pub id: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, MartianStruct)]
pub struct PreprocessNucleusSegmentationGeojsonStageInputs {
    pub segmentations: Option<GeoJsonFile>,
    pub spot_mask_from_user: Option<NpyFile>,
    pub hd_feature_slice: H5File,
}

#[derive(Debug, Clone, Serialize, Deserialize, MartianStruct)]
pub struct PreprocessNucleusSegmentationGeojsonStageOutputs {
    processed_segmentations: Option<JsonFile<Vec<NamedPolygon>>>,
    named_segmentations: Option<GeoJsonFile>,
    // Image with segmentation number a spot is in. Segmentations are numbered
    // from 1 onwards and 0 indicates that the spot is in no segment.
    spot_mask: Option<NpyFile>,
}

#[derive(Debug, Serialize, Deserialize, MartianStruct)]
pub struct PreprocessNucleusSegmentationGeojsonChunkInputs {
    start_row_coordinate: usize,
    end_row_coordinate: usize,
}

#[derive(Debug, Serialize, Deserialize, MartianStruct)]
pub struct PreprocessNucleusSegmentationGeojsonChunkOutputs {
    mask_chunk_image: NpyFile,
}

const ROW_AXIS_CHUNK_SIZE: usize = 100;

// This is our stage struct
pub struct PreprocessNucleusSegmentationGeojson;

#[make_mro(mem_gb = 4, volatile = strict, stage_name = PREPROCESS_NUCLEUS_SEGMENTATION_GEOJSON)]
impl MartianStage for PreprocessNucleusSegmentationGeojson {
    type StageInputs = PreprocessNucleusSegmentationGeojsonStageInputs;
    type StageOutputs = PreprocessNucleusSegmentationGeojsonStageOutputs;
    type ChunkInputs = PreprocessNucleusSegmentationGeojsonChunkInputs;
    type ChunkOutputs = PreprocessNucleusSegmentationGeojsonChunkOutputs;

    fn split(
        &self,
        args: Self::StageInputs,
        _rover: MartianRover,
    ) -> Result<StageDef<Self::ChunkInputs>> {
        let mem_gib = 4;
        if let Some(segmentations) = args.segmentations {
            let feature_slice = FeatureSliceH5::open(&args.hd_feature_slice)?;
            let metadata = feature_slice.metadata();
            println!("num rows {}; num cols {}", metadata.nrows, metadata.ncols);

            let file_size_in_bytes = fs_metadata(segmentations)?.len();
            let file_size_in_gb =
                (file_size_in_bytes as f64 / (1024. * 1024. * 1024.)).ceil() as isize;

            let chunk_start_points: Vec<_> = (0..metadata.nrows)
                .step_by(ROW_AXIS_CHUNK_SIZE)
                .chain([metadata.nrows])
                .collect();

            Ok(chunk_start_points
                .windows(2)
                .map(|a| {
                    (
                        Self::ChunkInputs {
                            start_row_coordinate: a[0],
                            end_row_coordinate: a[1],
                        },
                        Resource::with_mem_gb(mem_gib + file_size_in_gb),
                    )
                })
                .collect::<StageDef<_>>()
                .join_resource(Resource::with_mem_gb(mem_gib + file_size_in_gb)))
        } else {
            Ok(StageDef::new())
        }
    }

    fn main(
        &self,
        args: Self::StageInputs,
        chunk_args: Self::ChunkInputs,
        rover: MartianRover,
    ) -> Result<Self::ChunkOutputs> {
        let segmentations_in = args.segmentations.unwrap();

        let polygons = NamedPolygon::load_geojson_collection(&segmentations_in)?;

        let feature_slice = FeatureSliceH5::open(&args.hd_feature_slice)?;
        let metadata = feature_slice.metadata();
        println!("num rows {}; num cols {}", metadata.nrows, metadata.ncols);

        let transform_hires_to_spot = feature_slice
            .transform_microscope_to_spot()
            .expect("No transform matrix in feature slice");
        let transformed_polygons: Vec<_> = polygons
            .into_iter()
            .map(|x| x.map_coords(|c| transform_hires_to_spot.apply(c.x_y()).into()))
            .collect();

        // Polygons are numbered from 1; as 0 is aused as a sentinel for barcodes assigned to no polygon
        let transformed_polygons_to_index: HashMap<_, _> = transformed_polygons
            .iter()
            .enumerate()
            .map(|(ind, poly)| (poly, ind + 1))
            .collect();

        let mut xaxis_interval_tree = IntervalTree::<isize, &NamedPolygon>::new();
        let mut yaxis_interval_tree = IntervalTree::<isize, &NamedPolygon>::new();

        for polygon in &transformed_polygons {
            let bd_rect = polygon.bounding_box();
            let xmin = bd_rect.min().x.floor() as isize;
            let ymin = bd_rect.min().y.floor() as isize;
            let xmax = bd_rect.max().x.ceil() as isize;
            let ymax = bd_rect.max().y.ceil() as isize;

            xaxis_interval_tree.insert(Interval::new(xmin..(xmax + 1))?, polygon);
            yaxis_interval_tree.insert(Interval::new(ymin..(ymax + 1))?, polygon);
        }

        println!("Building row interval trees");
        let polygons_by_row_coordinate: Vec<_> = (0..metadata.nrows)
            .map(|y| {
                let common_row_polygons: Result<HashSet<&NamedPolygon>> = Ok(yaxis_interval_tree
                    .find(Interval::new((y as isize - 1)..(y as isize + 1))?)
                    .map(|w| *w.data())
                    .collect());
                common_row_polygons
            })
            .collect::<Result<Vec<_>>>()?;

        println!("Building column interval trees");
        let polygons_by_col_coordinate: Vec<_> = (0..metadata.ncols)
            .map(|x| {
                let common_col_polygons: Result<HashSet<&NamedPolygon>> = Ok(xaxis_interval_tree
                    .find(Interval::new((x as isize - 1)..(x as isize + 1))?)
                    .map(|w| *w.data())
                    .collect());
                common_col_polygons
            })
            .collect::<Result<Vec<_>>>()?;

        let mut polygon_mask = Array2::<u64>::zeros((
            chunk_args.end_row_coordinate - chunk_args.start_row_coordinate,
            metadata.ncols,
        ));
        for (row_coord, col_coord) in iproduct!(
            chunk_args.start_row_coordinate..chunk_args.end_row_coordinate,
            0..metadata.ncols
        ) {
            if row_coord % 100 == 0 && col_coord % 100 == 0 {
                println!("in row={row_coord} col={col_coord}");
            }
            let candidate_polygons = polygons_by_row_coordinate[row_coord]
                .intersection(&polygons_by_col_coordinate[col_coord]);

            let polygons_containing_point: Vec<_> = candidate_polygons
                .filter(|poly| {
                    poly.geometry
                        .contains(&point!(x: col_coord as f64, y: row_coord as f64))
                })
                .sorted_by(|a, b| Ord::cmp(&b.name, &a.name))
                .collect();

            if !polygons_containing_point.is_empty() {
                let polygon_id = transformed_polygons_to_index[*polygons_containing_point[0]];
                polygon_mask[[row_coord - chunk_args.start_row_coordinate, col_coord]] =
                    polygon_id as u64;
            }
        }

        let spot_mask_path: NpyFile = rover.make_path("spot_mask.npy");
        let writer = spot_mask_path.buf_writer()?;
        polygon_mask.write_npy(writer)?;

        Ok(PreprocessNucleusSegmentationGeojsonChunkOutputs {
            mask_chunk_image: spot_mask_path,
        })
    }

    fn join(
        &self,
        args: Self::StageInputs,
        _chunk_defs: Vec<Self::ChunkInputs>,
        chunk_outs: Vec<Self::ChunkOutputs>,
        rover: MartianRover,
    ) -> Result<Self::StageOutputs> {
        match (args.segmentations, args.spot_mask_from_user) {
            (Some(_), Some(_)) => bail!(
                "Only one of cell segmentation GeoJSON or a spot instance mask map can be input. Both found."
            ),
            (Some(segmentations_in), _) => {
                println!("Putting together segmentation from geojson");
                let polygons = NamedPolygon::load_geojson_collection(&segmentations_in)?;
                let feature_slice = FeatureSliceH5::open(&args.hd_feature_slice)?;
                let transform_hires_to_spot = feature_slice
                    .transform_microscope_to_spot()
                    .expect("No transform matrix in feature slice");

                let named_segmentations: GeoJsonFile =
                    rover.make_path("named_segmentations.geojson");
                write_geojson_collection_sequential_names(&polygons, &named_segmentations)?;

                let transformed_polygons: Vec<_> = polygons
                    .into_iter()
                    .map(|x| x.map_coords(|c| transform_hires_to_spot.apply(c.x_y()).into()))
                    .collect();
                let processed_segmentations: JsonFile<_> =
                    rover.make_path("processed_segmentations");

                let list_of_mtx: Vec<_> = chunk_outs
                    .into_iter()
                    .map(|x| Ok(Array2::<u64>::read_npy(File::open(x.mask_chunk_image)?)?))
                    .collect::<Result<Vec<_>>>()?;
                let list_of_mtx_view: Vec<_> = list_of_mtx.iter().map(ArrayBase::view).collect();
                let mtx = concatenate(Axis(0), &list_of_mtx_view)?;
                let spot_mask_path: NpyFile = rover.make_path("spot_mask.npy");
                let writer = spot_mask_path.buf_writer()?;
                mtx.write_npy(writer)?;

                processed_segmentations.write(&transformed_polygons)?;
                Ok(PreprocessNucleusSegmentationGeojsonStageOutputs {
                    processed_segmentations: Some(processed_segmentations),
                    spot_mask: Some(spot_mask_path),
                    named_segmentations: Some(named_segmentations),
                })
            }
            (None, Some(spot_mask_from_user)) => {
                let spot_mask_path: NpyFile = rover.make_path("spot_mask.npy");
                hard_link(&spot_mask_from_user, &spot_mask_path).or_else(|_| {
                    println!("Ran into trouble while hard linking {spot_mask_from_user:?} to {spot_mask_path:?}");
                    copy(&spot_mask_from_user, &spot_mask_path).map(|_| ())
                }).with_context(||
                format!("Ran into trouble while hard linking and copying {spot_mask_from_user:?} to {spot_mask_path:?}"))?;
                Ok(PreprocessNucleusSegmentationGeojsonStageOutputs {
                    processed_segmentations: None,
                    spot_mask: Some(spot_mask_path),
                    named_segmentations: None,
                })
            }
            _ => Ok(PreprocessNucleusSegmentationGeojsonStageOutputs {
                processed_segmentations: None,
                spot_mask: None,
                named_segmentations: None,
            }),
        }
    }
}
