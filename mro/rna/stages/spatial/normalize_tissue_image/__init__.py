# Copyright (c) 2025 10x Genomics, Inc. All rights reserved.
"""Run tiffer normalise on tissue image."""

import json
import os
from dataclasses import asdict

import martian

import cellranger.cr_io as cr_io
import cellranger.spatial.tiffer as tiffer
from cellranger.spatial.bounding_box import BoundingBox

__MRO__ = """
stage NORMALIZE_TISSUE_IMAGE(
    in  file[] tissue_image_paths,
    in  json   fiducial_bounding_box_on_tissue_image,
    out file   normalized_tissue_image,
    out json   crop_bbox_used,
    out json   input_image_bbox,
    src py     "stages/spatial/normalize_tissue_image",
) split (
) using (
    mem_gb   = 2,
    vmem_gb  = 64,
    volatile = strict,
)
"""
MINIMUM_SMALLEST_DIM_TO_CROP = 1024


def split(args):
    if len(args.tissue_image_paths) != 1:
        raise ValueError("expected a tissue image with one 3 channel image")
    max_dim = tiffer.get_max_image_dimension(args.tissue_image_paths[0])
    mem_gb_estimate = tiffer.call_tiffer_mem_estimate_gb(args.tissue_image_paths[0], max_dim)
    mem_gb_estimate = max(mem_gb_estimate, 8)
    return {
        "chunks": [],
        "join": {
            "__mem_gb": mem_gb_estimate,
            "__vmem_gb": max(2 * mem_gb_estimate, 64),
        },
    }


def join(args, outs, _chunk_defs, _chunk_outs):
    image_info = tiffer.call_tiffer_info(args.tissue_image_paths[0])
    _, img_ext = os.path.splitext(args.tissue_image_paths[0])
    tmp_image_path = martian.make_path(f"temp_image_path{img_ext}").decode()
    if image_info.get(tiffer.TIFFER_INFO_FORMAT_KEY) == tiffer.TIFFER_JPEG_VALUE:
        martian.log_info("JPEG image found. Outputing it.")
        cr_io.hardlink_with_fallback(args.tissue_image_paths[0], tmp_image_path)
        outs.crop_bbox_used = None
        outs.input_image_bbox = None
    else:
        bbox_image = None
        if not args.fiducial_bounding_box_on_tissue_image:
            bbox = None
        else:
            with open(args.fiducial_bounding_box_on_tissue_image) as f:
                bbox_in = BoundingBox.new(**json.load(f))
                martian.log_info(f"Bounding box of fiducials {bbox_in}.")

            if not (pages := image_info[tiffer.TIFFER_INFO_PAGE_KEY]) or len(pages) > 1:
                raise ValueError("Expect an image with exactly one page for segmenting nuclei")
            bbox_image = BoundingBox.from_image_dimensions(
                width=pages[0][tiffer.TIFFER_INFO_WIDTH_KEY],
                height=pages[0][tiffer.TIFFER_INFO_HEIGHT_KEY],
            )
            martian.log_info(f"Bounding box of image {bbox_image}.")

            if not bbox_in.non_overlapping(bbox_image):
                bbox = bbox_in.intersect(bbox_image)
                if bbox.smallest_dimension() <= MINIMUM_SMALLEST_DIM_TO_CROP:
                    martian.log_info(f"Not using {bbox} not being used as it is too small.")
                    bbox = None
            else:
                martian.log_info(
                    f"Bounding box of fiducials {bbox_in} does not intersect with image with bbox {bbox_image}."
                )
                bbox = None
        martian.log_info(f"Using bounding box {bbox} to crop.")
        martian.log_info("Running tiffer normalize.")
        tiffer.call_tiffer_normalize(
            image_path=args.tissue_image_paths[0], full_output_path=tmp_image_path, crop_box=bbox
        )

        if bbox:
            with open(outs.crop_bbox_used, "w") as f:
                json.dump(asdict(bbox), f)
        else:
            outs.crop_bbox_used = None
        if bbox_image:
            with open(outs.input_image_bbox, "w") as f:
                json.dump(asdict(bbox_image), f)
        else:
            outs.input_image_bbox = None
    outs.normalized_tissue_image = tmp_image_path
