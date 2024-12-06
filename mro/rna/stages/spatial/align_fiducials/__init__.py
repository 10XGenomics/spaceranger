#!/usr/bin/env python3
#
# Copyright (c) 2021 10X Genomics, Inc. All rights reserved.
#

"""Detect and align fiducials for all slide types."""

from __future__ import annotations

import json
import os

import cv2
import martian

import cellranger.spatial.image_util as image_util
from cellranger.spatial.fiducial import (
    construct_roi,
    fiducial_detection,
    fiducial_registration,
    write_qc_fig,
    write_reg_err_fig,
)
from cellranger.spatial.fiducial_alignment import isolate_perspective_transform
from cellranger.spatial.loupe_util import LoupeParser
from cellranger.spatial.pipeline_mode import PipelineMode
from cellranger.spatial.slide_design_o3 import (  # pylint: disable=no-name-in-module, import-error
    VisiumHdLayout,
    VisiumHdSlideWrapper,
)
from tenkit import safe_json

__MRO__ = """
stage ALIGN_FIDUCIALS(
    in  PipelineMode pipeline_mode,
    in  png          fiducials_detection_image,
    in  json         gpr_spots_data_json,
    in  json         hd_layout_data_json,
    in  json         loupe_spots_data_json,
    in  string       visium_hd_slide_name,
    in  string       reorientation_mode,
    in  json         crop_info_json,
    in  bool         is_visium_hd,
    out json         registered_spots_data_json,
    out json         fiducial_alignment_metrics,
    out json         transform_matrix,
    out map<file>    qc_detected_fiducials_images,
    out jpg          qc_aligned_fiducials_image,
    out jpg          qc_fiducial_error_image,
    src py           "stages/spatial/align_fiducials",
) split (
) using (
    volatile = strict,
) retain (
    qc_detected_fiducials_images,
)
"""


def split(args):
    mem_gb = max(
        10.0 if args.is_visium_hd else 4.0,
        LoupeParser.estimate_mem_gb_from_json_file(args.loupe_spots_data_json),
        LoupeParser.estimate_mem_gb_from_json_file(args.gpr_spots_data_json),
        LoupeParser.estimate_mem_gb_from_json_file(args.hd_layout_data_json),
    )
    return {
        "chunks": [],
        "join": {
            "__mem_gb": mem_gb,
            "__vmem_gb": max(16.0, mem_gb + 3.0),
        },
    }


def join(args, outs, _chunk_defs, _chunk_outs):
    cv2.setNumThreads(martian.get_threads_allocation())
    fid_detect_img = image_util.cv_read_image_standard(args.fiducials_detection_image)

    metrics = {}
    if args.loupe_spots_data_json and os.path.exists(args.loupe_spots_data_json):
        spots_data = LoupeParser(args.loupe_spots_data_json)
        if args.is_visium_hd:
            # Checking if the loupe file was created using --unknown-slide, but it was a HD run
            # and we have HD slide ID specified elsewhere.
            assert spots_data.is_hd_unknown_slide() ^ (
                (args.hd_layout_data_json is not None) and os.path.exists(args.hd_layout_data_json)
            ), (
                "Inconsistency between loupe file and pipeline. "
                "One was generated using --unknown-slide and other was not. "
                f"Loupe file unknown slide: {spots_data.is_hd_unknown_slide()}. "
                f"Layout file passed in: "
                f"{((args.hd_layout_data_json is not None) and os.path.exists(args.hd_layout_data_json))}"
            )

        if spots_data.has_spot_transform():
            with open(outs.transform_matrix, "w") as f:
                safe_json.dump_numpy(spots_data.get_spot_transform(), f, pretty=True)
        else:
            outs.transform_matrix = None

        metrics["alignment_method"] = "Manual Alignment"
    elif args.gpr_spots_data_json or args.visium_hd_slide_name:
        pipeline_mode = PipelineMode(**args.pipeline_mode)
        try:
            pipeline_mode.validate()
        except ValueError:
            martian.throw(f"Invalid pipeline mode of {pipeline_mode}")
        if args.gpr_spots_data_json and os.path.exists(args.gpr_spots_data_json):
            spots_data = LoupeParser(args.gpr_spots_data_json)
            slide = None
        else:
            slide = VisiumHdSlideWrapper(
                slide_name=args.visium_hd_slide_name,
                layout=(
                    VisiumHdLayout.from_json(args.hd_layout_data_json)
                    if args.hd_layout_data_json
                    else None
                ),
            )
            spots_data = LoupeParser.from_visium_hd_slide(slide)

        with open(args.crop_info_json) as f:
            crop_info_dict = json.load(f)
        roi_mask = construct_roi(crop_info_dict, fid_detect_img.shape)
        detect_fid_dict, detection_metrics, qc_img_paths_dict = fiducial_detection(
            pipeline_mode,
            fid_detect_img,
            roi_mask,
            slide,
        )
        outs.qc_detected_fiducials_images = qc_img_paths_dict

        spots_data, reg_metrics, out_frac, transform_mat, overlap = fiducial_registration(
            pipeline_mode,
            spots_data,
            detect_fid_dict,
            args.reorientation_mode,
            fid_detect_img.shape,
        )
        if args.is_visium_hd:
            write_reg_err_fig(
                fid_detect_img,
                spots_data.get_fiducials_data(),
                detect_fid_dict,
                transform_mat,
                outs.qc_fiducial_error_image,
            )
            outs.fid_perp_tmat = isolate_perspective_transform(
                spots_data.get_fiducials_data(),
                detect_fid_dict,
                overlap,
            )

        if out_frac > 0.05:
            raise RuntimeError(
                "Fiducial alignment seems to have failed - too many points rotated out of the image space. "
                + "Please use the manual alignment tool in Loupe Browser"
            )
        metrics.update(detection_metrics)
        metrics.update(reg_metrics)
        with open(outs.transform_matrix, "w") as f:
            safe_json.dump_numpy(transform_mat, f, pretty=True)

    else:
        raise RuntimeError("No fiducials design file")

    if args.is_visium_hd:
        assert spots_data.has_hd_slide(), "Processing HD slide, but loupe file did not contain it."
    spots_data.save_to_json(outs.registered_spots_data_json)
    write_qc_fig(fid_detect_img, spots_data, outs.qc_aligned_fiducials_image)
    with open(outs.fiducial_alignment_metrics, "w") as f:
        safe_json.dump_numpy(metrics, f, pretty=True)
