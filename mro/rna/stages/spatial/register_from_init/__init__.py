# Copyright (c) 2021 10x Genomics, Inc. All rights reserved.
"""Tissue registration stage."""

import json
import os

import cv2
import martian
import numpy as np
from skimage import io

import cellranger.cr_io as cr_io
import tenkit.safe_json as tk_safe_json
from cellranger.spatial.loupe_util import LoupeParser
from cellranger.spatial.tissue_regist import (
    FEATURE_MATCHING,
    ITK_ERROR_PREFIX,
    register_from_init_transform,
)
from cellranger.spatial.transform import (
    convert_transform_corner_to_center,
    normalize_perspective_transform,
)

__MRO__ = """
stage REGISTER_FROM_INIT(
    in  png       tissue_detection_image,
    in  tiff      registration_target_image,
    in  json      loupe_tissue_registration_json,
    in  json      crop_info_json,
    in  json      initial_transform_info_json,
    in  float[][] fid_perp_tmat,
    in  bool      is_visium_hd,
    in  float     pixel_size_target_to_cyta_ratio,
    in  bool      is_pd,
    out json      final_transform_json,
    out json      sitk_registration_metrics,
    out jpg       max_mutual_info_init_debug,
    out string    itk_error_string,
    src py        "stages/spatial/register_from_init",
) split (
    in  float[][] init_transform_mat,
    in  string    init_method_used,
    out float     metric,
    out string    chunk_stop_description,
    out float[]   transform_mat,
    out jpg       chunk_max_mutual_info_init_debug,
) using (
    mem_gb   = 8,
    volatile = strict,
)
"""


def split(args):
    with open(args.initial_transform_info_json) as f:
        info = json.load(f)
        init_method_used = info["init_method_used"]
        init_transform_list = info["init_transform_list"]
    chunks = []
    join_mem_gb = 8
    mem_gb_chunk = 12
    for init_transform in init_transform_list:
        chunks.append(
            {
                "init_transform_mat": init_transform,
                "init_method_used": init_method_used,
                "__mem_gb": mem_gb_chunk,
                "__vmem_gb": 64,
            }
        )
    return {
        "chunks": chunks,
        "join": {
            "__mem_gb": join_mem_gb,
        },
    }


def main(args, outs):
    cv2.setNumThreads(martian.get_threads_allocation())
    regist_target_img = io.imread(args.registration_target_image).astype(np.float32)
    cyta_img = io.imread(args.tissue_detection_image).astype(np.float32)
    if len(cyta_img.shape) != 2:
        raise ValueError("CytAssist image for registration should have dimension of 2")
    if len(regist_target_img.shape) != 2:
        raise ValueError("Registration target image should have dimension of 2")
    cyta_img = (cyta_img - np.min(cyta_img)) / (np.max(cyta_img) - np.min(cyta_img))
    if args.is_visium_hd and args.fid_perp_tmat:
        center_based_perp_tmat = convert_transform_corner_to_center(np.array(args.fid_perp_tmat))
        # shape[::-1] because cv2 uses (width, height) convention
        cyta_img = cv2.warpPerspective(cyta_img, center_based_perp_tmat, cyta_img.shape[::-1])
    regist_target_img = (regist_target_img - np.min(regist_target_img)) / (
        np.max(regist_target_img) - np.min(regist_target_img)
    )

    learning_rate = 1.0 if args.init_method_used == FEATURE_MATCHING else 10.0
    transform_mat, metric, stop_description = register_from_init_transform(
        regist_target_img,
        cyta_img,
        np.array(args.init_transform_mat),
        learning_rate,
        outs.chunk_max_mutual_info_init_debug,
    )

    if args.is_visium_hd and args.fid_perp_tmat:
        transform_mat = transform_mat @ np.array(args.fid_perp_tmat)

    outs.metric = metric
    outs.transform_mat = tk_safe_json.json_sanitize(transform_mat.flatten(order="C"))
    outs.chunk_stop_description = stop_description


def join(args, outs, chunk_defs, chunk_outs):
    if args.loupe_tissue_registration_json and os.path.exists(args.loupe_tissue_registration_json):
        cyta_info = LoupeParser(args.loupe_tissue_registration_json)
        # The tissue transform matrix in loupe maps from microscope image to cytassist image
        # invert the matrix to map from cytassist image to microscope image
        transform_mat = normalize_perspective_transform(
            np.linalg.inv(cyta_info.get_cyta_transform())
        )
        metrics = {
            "tissue_registration_type": "Manual registration",
            "tissue_image_scale_um_per_pixel": args.pixel_size_target_to_cyta_ratio,
        }
        outs.target_tissue_detection_debug = None
        outs.cytassist_tissue_detection_debug = None
        outs.max_mutual_info_init_debug = None
        outs.itk_error_string = None
    else:
        best_chunk = min(
            chunk_outs,
            key=lambda chunk_out: chunk_out.metric if chunk_out.metric is not None else np.inf,
        )
        transform_mat = np.array(best_chunk.transform_mat).reshape((3, 3), order="C")
        metrics = {
            "tissue_registration_type": "Automatic registration",
            "tissue_registration_mmi_metric": best_chunk.metric,
            "tissue_registration_optimizer_stop_description": best_chunk.chunk_stop_description,
            "tissue_image_scale_um_per_pixel": args.pixel_size_target_to_cyta_ratio,
        }
        cr_io.hardlink_with_fallback(
            best_chunk.chunk_max_mutual_info_init_debug, outs.max_mutual_info_init_debug
        )

        outs.itk_error_string = (
            best_chunk.chunk_stop_description
            if best_chunk.chunk_stop_description
            and str(best_chunk.chunk_stop_description).startswith(ITK_ERROR_PREFIX)
            else None
        )

    output = {"tissue_transform": transform_mat.tolist()}
    with open(outs.final_transform_json, "w") as f:
        json.dump(output, f)
    with open(outs.sitk_registration_metrics, "w") as f:
        json.dump(metrics, f)
