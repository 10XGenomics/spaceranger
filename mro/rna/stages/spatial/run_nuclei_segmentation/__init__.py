#
# Copyright (c) 2024 10X Genomics, Inc. All rights reserved.
#
"""Run nucleus segmentation on tissue image."""

import json
import os
from zipfile import ZIP_DEFLATED

import geojson
import martian
import numpy as np
import onnxruntime as ort
import tifffile
from geojson import Feature, FeatureCollection, Polygon

import cellranger.spatial.tiffer as tiffer
from cellranger.spatial.bounding_box import BoundingBox
from cellranger.spatial.segmentation_constants import GEOJSON_CELL_ID_KEY
from cellranger.spatial.stardist.model2d import StarDist2D
from cellranger.spatial.stardist.prepare import MinMaxNormalizer
from cellranger.spatial.stardist.stardist_utils import (
    STARDIST_BLOCK_MIN_OVERLAP,
    STARDIST_BLOCK_SIZE,
    STARDIST_DEFAULT_MODEL_NAME,
    STARDIST_DEFAULT_MODEL_PATH,
    STARDIST_N_TILES_PER_BLOCK,
    STARDIST_TILE_SIZE,
)

__MRO__ = """
stage RUN_NUCLEI_SEGMENTATION(
    in  file    normalized_tissue_image,
    in  int     max_nucleus_diameter_px,
    in  json    crop_bbox_used,
    in  json    input_image_bbox,
    out tiff    nucleus_instance_mask,
    out geojson nucleus_segmentations,
    out json    segment_nuclei_metrics,
    out int     num_nuclei_detected,
    out int     max_nucleus_diameter_px_used,
    src py      "stages/spatial/run_nuclei_segmentation",
) split (
) using (
    mem_gb   = 2,
    vmem_gb  = 64,
    volatile = strict,
)
"""


def split(args):
    max_dim = tiffer.get_max_image_dimension(args.normalized_tissue_image)
    normalized_image_mem_est = tiffer.call_tiffer_mem_estimate_gb(
        args.normalized_tissue_image, max_dim
    )
    img_max_dim = max_dim
    if args.input_image_bbox:
        with open(args.input_image_bbox) as f:
            bbox_image = BoundingBox.new(**json.load(f))
        martian.log_info(f"Bounding box of image {bbox_image}.")
        img_max_dim = max(bbox_image.maxc, bbox_image.maxr)
    original_image_mem_est = tiffer.call_tiffer_mem_estimate_gb(
        args.normalized_tissue_image, img_max_dim
    )
    print(f"{max_dim=} {img_max_dim=} {normalized_image_mem_est=} {original_image_mem_est=}")
    mem_gb_estimate = normalized_image_mem_est + original_image_mem_est
    mem_gb_estimate = max(mem_gb_estimate, 8)
    return {
        "chunks": [],
        "join": {
            "__threads": 8,
            "__mem_gb": mem_gb_estimate,
            "__vmem_gb": max(2 * mem_gb_estimate, 64),
        },
    }


def join(args, outs, _chunk_defs, _chunk_outs) -> None:  # pylint: disable=too-many-locals
    image_info = tiffer.call_tiffer_info(args.normalized_tissue_image)
    if image_info.get(tiffer.TIFFER_INFO_FORMAT_KEY) == tiffer.TIFFER_JPEG_VALUE:
        martian.log_info("Reading JPEG via cv.")
        pages = image_info[tiffer.TIFFER_INFO_PAGE_KEY]
        image_dims = (
            pages[0][tiffer.TIFFER_INFO_HEIGHT_KEY] * pages[0][tiffer.TIFFER_INFO_WIDTH_KEY]
        )
        os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(image_dims + 1)
        from cellranger.spatial.image_util import (  # pylint: disable=import-outside-toplevel
            cv_read_rgb_image,
        )

        image = cv_read_rgb_image(args.normalized_tissue_image)
    else:
        martian.log_info("Reading normalised tiffer image.")
        image = tifffile.imread(args.normalized_tissue_image)

    martian.log_info(f"Input image has size {image.shape=}.")
    spatial_dims = np.array(image.shape[:2])

    num_thread = martian.get_threads_allocation()
    onnx_session_opts = ort.SessionOptions()
    onnx_session_opts.intra_op_num_threads = num_thread
    onnx_session_opts.execution_mode = ort.ExecutionMode.ORT_PARALLEL
    onnx_session_opts.inter_op_num_threads = num_thread
    martian.log_info(f"Using {num_thread=} for onnx operations.")
    stardist_model = StarDist2D(
        None,
        name=STARDIST_DEFAULT_MODEL_NAME,
        basedir=STARDIST_DEFAULT_MODEL_PATH,
        onnx_session_opts=onnx_session_opts,
    )

    max_nucleus_diameter_px = (
        args.max_nucleus_diameter_px if args.max_nucleus_diameter_px else STARDIST_BLOCK_MIN_OVERLAP
    )
    outs.max_nucleus_diameter_px_used = max_nucleus_diameter_px
    outs.num_nuclei_detected = 0

    if spatial_dims.min() < STARDIST_BLOCK_SIZE:
        # If image is smaller in any dimension than a block, use normal predict
        # Dynamically calculating n_tiles based on input image dimension
        n_tiles = (spatial_dims // STARDIST_TILE_SIZE) + 1
        n_tiles = (n_tiles[0], n_tiles[1], 1)
        martian.log_info(f"Running predict_instances with {n_tiles=}.")
        labels, polys = stardist_model.predict_instances(
            image,
            axes="YXC",
            normalizer=MinMaxNormalizer(),
            n_tiles=n_tiles,
        )
        num_nuclei_too_large = 0
    else:
        martian.log_info("Running predict_instances_big.")
        labels, polys, num_nuclei_too_large = stardist_model.predict_instances_big(
            image,
            axes="YXC",
            block_size=STARDIST_BLOCK_SIZE,
            min_overlap=max_nucleus_diameter_px,
            normalizer=MinMaxNormalizer(),
            n_tiles=STARDIST_N_TILES_PER_BLOCK,
        )

    row_offset, col_offset, bbox_used = 0, 0, None
    if args.crop_bbox_used:
        with open(args.crop_bbox_used) as f:
            bbox_used = BoundingBox.new(**json.load(f))
        martian.log_info(f"Bounding box of fiducials {bbox_used}.")
        row_offset = bbox_used.minr
        col_offset = bbox_used.minc

    if labels is not None:
        if bbox_used and args.input_image_bbox:
            with open(args.input_image_bbox) as f:
                bbox_image = BoundingBox.new(**json.load(f))
            martian.log_info(f"Bounding box of image {bbox_image}.")
            pad_width = (
                (row_offset, bbox_image.maxr - bbox_used.maxr),
                (col_offset, bbox_image.maxc - bbox_used.maxc),
            )
            martian.log_info(f"label shape going in  {labels.shape}.")
            martian.log_info(f"Unboxing using pads {pad_width}.")
            labels = np.pad(
                array=labels,
                pad_width=pad_width,
                mode="constant",
                constant_values=0,
            )
            martian.log_info(f"label shape after unboxing  {labels.shape}.")
        tifffile.imwrite(
            outs.nucleus_instance_mask, labels.astype(np.uint32), compression=ZIP_DEFLATED
        )
    else:
        tifffile.imwrite(
            outs.nucleus_instance_mask,
            np.zeros(image.shape, dtype=np.uint32),
            compression=ZIP_DEFLATED,
        )

    segment_nuclei_metrics = {
        "num_nuclei_too_large": num_nuclei_too_large,
    }
    with open(outs.segment_nuclei_metrics, "w") as f:
        json.dump(segment_nuclei_metrics, f, indent=4)

    # TODO: make this more efficient
    if polys is not None:
        feature_collection = []
        for i, coords in enumerate(polys["coord"]):
            coords = list(
                map(
                    tuple,
                    np.stack([coords[1, :] + col_offset, coords[0, :] + row_offset], axis=1).astype(
                        float
                    ),
                )
            )
            # QuPath requires closed polygon
            coords = [coords + [coords[0]]]
            feature = Feature(geometry=Polygon(coords), properties={GEOJSON_CELL_ID_KEY: i})
            feature_collection.append(feature)

        martian.log_info(f"StarDist segmented {len(feature_collection)} nuclei.")
        outs.num_nuclei_detected = len(feature_collection)
        feature_collection = FeatureCollection(feature_collection)

        with open(outs.nucleus_segmentations, "w") as f:
            geojson.dump(feature_collection, f)

    else:
        martian.log_info("No nuclei segmented. Not writing out a geojson.")
        outs.nucleus_segmentations = None
        outs.nucleus_instance_mask = None
        outs.num_nuclei_detected = 0
