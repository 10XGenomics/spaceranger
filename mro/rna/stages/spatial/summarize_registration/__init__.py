#
# Copyright (c) 2024 10X Genomics, Inc. All rights reserved.
#
"""Summary stage for tissue registration."""
import json
import os

import cv2
import numpy as np
import skimage
from skimage import io

import cellranger.cr_io as cr_io
from cellranger.spatial.tissue_regist_qc import (
    create_tissue_regist_qc_img,
    float_image_to_ubyte,
    save_tissue_regist_qc_img,
)
from cellranger.spatial.transform import (
    convert_transform_corner_to_center,
)

__MRO__ = """
stage SUMMARIZE_REGISTRATION(
    in  png       tissue_detection_image,
    in  tiff      registration_target_image,
    in  float[][] fid_perp_tmat,
    in  bool      is_visium_hd,
    in  bool      is_pd,
    in  json      fm_tissue_registration_metrics,
    in  json      sitk_tissue_registration_metrics,
    in  json      final_transform_json,
    out json      tissue_registration_metrics,
    out tiff      resampled_cyta_img,
    out jpg       qc_registered_tissue_image,
    out jpg       qc_resampled_cyta_img,
    out jpg       qc_regist_target_img,
    src py        "stages/spatial/summarize_registration",
) using (
    mem_gb   = 8,
    volatile = strict,
)
"""


def main(args, outs):
    if not os.path.exists(args.registration_target_image):
        raise ValueError("Registration target image does not exist.")

    regist_target_img = io.imread(args.registration_target_image).astype(np.uint8)
    cyta_img = io.imread(args.tissue_detection_image).astype(np.uint8)

    if args.is_visium_hd and args.fid_perp_tmat:
        center_based_perp_tmat = convert_transform_corner_to_center(np.array(args.fid_perp_tmat))
        # shape[::-1] because cv2 uses (width, height) convention
        cyta_img = cv2.warpPerspective(cyta_img, center_based_perp_tmat, cyta_img.shape[::-1])

    with open(args.final_transform_json) as f:
        transform_mat = json.load(f)["tissue_transform"]

    qc_type = "color" if args.is_pd else "checkerboard"
    qc_img, resampled_cyta_img = create_tissue_regist_qc_img(
        regist_target_img,
        cyta_img,
        np.linalg.inv(transform_mat),
        qc_img_type=qc_type,
    )
    save_tissue_regist_qc_img(qc_img, outs.qc_registered_tissue_image)

    if args.is_pd:
        io.imsave(outs.resampled_cyta_img, resampled_cyta_img)

    downscaled_resampled_cyta_img = skimage.transform.rescale(
        resampled_cyta_img, 0.33, anti_aliasing=False, preserve_range=True
    )
    downscaled_target_img = skimage.transform.rescale(
        regist_target_img, 0.33, anti_aliasing=False, preserve_range=True
    )

    io.imsave(
        outs.qc_resampled_cyta_img,
        float_image_to_ubyte(downscaled_resampled_cyta_img),
        quality=95,
    )
    io.imsave(
        outs.qc_regist_target_img,
        float_image_to_ubyte(downscaled_target_img),
        quality=95,
    )

    if args.fm_tissue_registration_metrics and os.path.exists(args.fm_tissue_registration_metrics):
        with open(args.fm_tissue_registration_metrics) as f:
            fm_metrics = json.load(f)
        with open(args.sitk_tissue_registration_metrics) as f:
            sitk_metrics = json.load(f)
        sitk_metrics.update(fm_metrics)
        with open(outs.tissue_registration_metrics, "w") as f:
            json.dump(sitk_metrics, f)
    else:
        cr_io.hardlink_with_fallback(
            args.sitk_tissue_registration_metrics, outs.tissue_registration_metrics
        )
