# Copyright (c) 2021 10x Genomics, Inc. All rights reserved.
"""Generate plots for Image Alignment QC in HD."""

import json
import os
import shutil

import numpy as np
import skimage
from skimage import io

from cellranger.spatial.hd_feature_slice import HdFeatureSliceIo, TransformMatrices
from cellranger.spatial.image_util import cv_read_image_standard, normalized_image_from_counts
from cellranger.spatial.transform import (
    convert_transform_corner_to_center,
    normalize_perspective_transform,
    scale_matrix,
    translation_matrix,
)

__MRO__ = """
stage PROCESS_HD_ALIGNMENT(
    in  h5   hd_feature_slice_h5,
    in  json fiducial_transform,
    in  json tissue_registration_transform,
    in  json scalefactors,
    in  tiff cytassist_image,
    in  tiff microscope_image,
    out png  cytassist_image_on_spots,
    out png  umi_cytassist_checkerboard,
    out png  log_umi_image,
    out png  microscope_image_on_spots,
    out png  umi_microscope_checkerboard,
    out h5   hd_feature_slice_h5_out,
    src py   "stages/spatial/process_hd_alignment",
) using (
    mem_gb   = 2,
    vmem_gb  = 16,
    volatile = strict,
)
"""


def _write_qc_images(log_umi_img, img, transform_matrix, resampled_out_path, checkerboard_out_path):
    center_based_transform = convert_transform_corner_to_center(transform_matrix)
    sk_transform = skimage.transform.ProjectiveTransform(matrix=center_based_transform)
    resampled_img = skimage.transform.warp(
        img, sk_transform, output_shape=log_umi_img.shape, preserve_range=True
    ).astype(np.uint8)
    checkerboard_img = skimage.util.compare_images(
        log_umi_img, resampled_img, method="checkerboard", n_tiles=(10, 10)
    )

    io.imsave(resampled_out_path, resampled_img)
    io.imsave(checkerboard_out_path, normalized_image_from_counts(checkerboard_img))


def main(args, outs):
    shutil.copyfile(args.hd_feature_slice_h5, outs.hd_feature_slice_h5_out, follow_symlinks=False)

    if args.cytassist_image is None or not os.path.exists(args.cytassist_image):
        outs.cytassist_image_on_spots = None
        outs.umi_cytassist_checkerboard = None
        outs.log_umi_image = None
        outs.microscope_image_on_spots = None
        outs.umi_microscope_checkerboard = None
        outs.transform_matrices = None
        return

    with open(args.fiducial_transform) as f:
        fid_transform = np.array(json.load(f))

    with open(args.scalefactors) as f:
        scalefactors = json.load(f)

    feature_slice = HdFeatureSliceIo(args.hd_feature_slice_h5)
    slide = feature_slice.slide()
    log_umi_img = normalized_image_from_counts(feature_slice.total_umis(), log1p=True, invert=True)
    cytassist_img = cv_read_image_standard(args.cytassist_image)

    transform_spot_colrow_to_cytassist_colrow = normalize_perspective_transform(
        fid_transform @ slide.transform_spot_colrow_to_xy()
    )

    io.imsave(outs.log_umi_image, log_umi_img)
    _write_qc_images(
        log_umi_img,
        cytassist_img,
        transform_spot_colrow_to_cytassist_colrow @ translation_matrix(-0.5, -0.5),
        outs.cytassist_image_on_spots,
        outs.umi_cytassist_checkerboard,
    )

    transform_matrices = TransformMatrices()
    transform_matrices.set_spot_to_cytassist_transform(transform_spot_colrow_to_cytassist_colrow)

    if args.tissue_registration_transform and os.path.exists(args.tissue_registration_transform):
        with open(args.tissue_registration_transform) as f:
            tissue_registration_transform = np.array(json.load(f))

        transform_spot_colrow_to_microscope_colrow = normalize_perspective_transform(
            tissue_registration_transform @ transform_spot_colrow_to_cytassist_colrow
        )

        transform_matrices.set_spot_to_microscope_transform(
            transform_spot_colrow_to_microscope_colrow
        )
        microscope_image = cv_read_image_standard(args.microscope_image).astype(np.uint8)
        _write_qc_images(
            log_umi_img,
            microscope_image,
            normalize_perspective_transform(
                scale_matrix(scalefactors["regist_target_img_scalef"])
                @ transform_spot_colrow_to_microscope_colrow
                @ translation_matrix(-0.5, -0.5),
            ),
            outs.microscope_image_on_spots,
            outs.umi_microscope_checkerboard,
        )
    else:
        outs.microscope_image_on_spots = None
        outs.umi_microscope_checkerboard = None

    HdFeatureSliceIo(h5_path=outs.hd_feature_slice_h5_out, open_mode="a").set_transform_matrices(
        transform_matrices
    )
