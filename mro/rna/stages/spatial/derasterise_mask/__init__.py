#
# Copyright (c) 2024 10X Genomics, Inc. All rights reserved.
#
"""Derasterise a mask using find contours."""

import itertools
from functools import partial

import cv2 as cv
import geojson
import martian
import numpy as np
import shapely

from cellranger.spatial.hd_feature_slice import HdFeatureSliceIo
from cellranger.spatial.segmentation_constants import GEOJSON_CELL_ID_KEY

__MRO__ = """
stage DERASTERISE_MASK(
    in  npy     segmentation_spot_mask,
    in  h5      hd_feature_slice,
    out geojson segmentation_tissue_image_geojson,
    src py      "stages/spatial_pd/derasterise_mask",
) split (
    in  int     chunk_ind_start,
    in  int     chunk_ind_end,
    out geojson chunk_segmentation_tissue_image_geojson,
) using (
    mem_gb = 2,
)
"""

GEOJSON_PROCESSING_CHUNK_SIZE = 10_000


def contour_to_polygon(contour):
    """Convert a contour to a polygon."""
    if contour.shape[0] == 1:
        return shapely.Point(contour)
    elif contour.shape[0] == 2:
        return shapely.LineString(contour)
    else:
        return shapely.Polygon(contour)


def apply_perspective_transform(x, y, mat: np.ndarray) -> tuple[float, float]:
    """Apply perspective transform to an (x,y)-coordinate.

    Args:
        x (int | float): X-co-ordinate
        y (int | float): Y-co-ordinate
        mat (np.ndarray): Transform matrix

    Returns:
        tuple[float, float]: Transformed co-ordinates
    """
    raw = mat.dot(np.array([x, y, 1]))
    return (
        raw[0] / raw[2],
        raw[1] / raw[2],
    )


def round_polygon_coordinates(polygon, decimals):
    """Round coordinates of a shapely polygon to a given decimal precision."""
    return shapely.Polygon(np.round(np.array(polygon.exterior.coords), decimals))


def find_manhattan_contour(mask: np.ndarray) -> np.ndarray:
    """Find manhattan contour of a connected object."""
    mask = mask.astype(np.uint8)
    # Use only roi containing the largest connected component
    cntrs, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cntr = cntrs[0] if len(cntrs) == 1 else sorted(cntrs, key=cv.contourArea)[-1]
    c_min, r_min, w, h = cv.boundingRect(cntr)
    roi_mask = mask[r_min : r_min + h, c_min : c_min + w]
    # Upsample the mask by a factor of 2 using nearest neighbor
    # In the auxiliary mask, one pixel in the original mask becomes 4 pixels, representing its four corners
    # This is necessary for preventing cv.findContours from ignoring one-pixel wide gaps in the object structure
    aux_cntrs, _ = cv.findContours(
        roi_mask.repeat(2, axis=0).repeat(2, axis=1),
        cv.RETR_EXTERNAL,
        cv.CHAIN_APPROX_SIMPLE,
    )
    aux_cntr = aux_cntrs[0].reshape(-1, 2)
    # Bring vertices back to the coordinate space before upsampling
    # Adjust four corners accordingly so that there won't be any diagonal edges
    manh_cntr = aux_cntr // 2 + 0.5 * ((aux_cntr % 2 == 1) * 2 - 1)
    # Deduplicate adjacent vertices
    non_dups = np.any(np.diff(manh_cntr, axis=0) != 0, axis=1)
    manh_cntr = np.vstack([manh_cntr[0], manh_cntr[1:][non_dups]])
    # Back to full mask coordinates
    manh_cntr += np.array([c_min, r_min])
    return manh_cntr


def split(args):
    if not args.segmentation_spot_mask or not args.hd_feature_slice:
        return {
            "chunks": [],
            "join": {"__mem_gb": 1},
        }

    segmentation_spot_mask = np.load(args.segmentation_spot_mask)
    unique_cell_ids = [
        int(x) for x in np.unique(segmentation_spot_mask) if x != 0
    ]  # x=0 is sentinel for no cell.
    num_cells = len(unique_cell_ids)
    processing_boundaries = list(np.arange(0, num_cells, GEOJSON_PROCESSING_CHUNK_SIZE)) + [
        num_cells
    ]

    chunks = [
        {
            "chunk_ind_start": int(start_ind),
            "chunk_ind_end": int(end_ind),
            "__mem_gb": 8,
            "__vmem_gb": 128,
        }
        for (start_ind, end_ind) in itertools.pairwise(processing_boundaries)
    ]
    return {
        "chunks": chunks,
        "join": {"__mem_gb": 8, "__vmem_gb": 128},
    }


def main(args, outs):
    cv.setNumThreads(1)
    segmentation_spot_mask = np.load(args.segmentation_spot_mask)
    unique_cell_ids = [
        int(x) for x in np.unique(segmentation_spot_mask) if x != 0
    ]  # x=0 is sentinel for no cell.

    with HdFeatureSliceIo(args.hd_feature_slice) as hd_feature_slice:
        transform_mtx = (
            hd_feature_slice.metadata.transform_matrices.get_spot_colrow_to_tissue_image_colrow_transform()
        )
    co_ordinate_transform_fn = partial(apply_perspective_transform, mat=transform_mtx)

    def spot_colrow_to_tissue_image_colrow_transform(segmentation):
        return np.array([co_ordinate_transform_fn(x[0], x[1]) for x in segmentation])

    feature_list = []
    for cell_id in unique_cell_ids[args.chunk_ind_start : args.chunk_ind_end]:
        if cell_id % 1000 == 1:
            print(cell_id)
        cntr = find_manhattan_contour(segmentation_spot_mask == cell_id)
        shapely_poly = contour_to_polygon(cntr)

        # Transforming from spot coordinates to tissue image coordinates
        tissue_image_shapely_poly = shapely.transform(
            shapely_poly, spot_colrow_to_tissue_image_colrow_transform
        )
        tissue_image_shapely_poly = round_polygon_coordinates(tissue_image_shapely_poly, 1)

        feature = geojson.Feature(
            geometry=tissue_image_shapely_poly,
            properties={GEOJSON_CELL_ID_KEY: cell_id},
        )
        feature_list.append(feature)

    feature_collection = geojson.FeatureCollection(feature_list)
    with open(outs.chunk_segmentation_tissue_image_geojson, "w") as f:
        geojson.dump(feature_collection, f)


def join(args, outs, _chunk_defs, chunk_outs):
    if not args.segmentation_spot_mask or not args.hd_feature_slice:
        martian.clear(outs)
        return

    all_features_list = []
    for chunk_out in chunk_outs:
        with open(chunk_out.chunk_segmentation_tissue_image_geojson) as f:
            chunk_feature_collection = geojson.load(f)
        all_features_list.extend(chunk_feature_collection.features)

    feature_collection = geojson.FeatureCollection(all_features_list)
    with open(outs.segmentation_tissue_image_geojson, "w") as f:
        geojson.dump(feature_collection, f)
