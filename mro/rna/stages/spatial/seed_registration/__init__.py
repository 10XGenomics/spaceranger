#
# Copyright (c) 2024 10X Genomics, Inc. All rights reserved.
#
"""Feature matching stage for tissue registration."""

import json
import os

import cv2
import martian
import numpy as np
from PIL import Image
from skimage import io

import cellranger.cr_io as cr_io
from cellranger.spatial.image_util import (
    CYT_IMG_PIXEL_SIZE,
    generate_fiducial_mask,
    get_tiff_pixel_size,
)
from cellranger.spatial.loupe_util import LoupeParser
from cellranger.spatial.tissue_regist import (
    CENTROID_ALIGNMENT,
    FEATURE_MATCHING,
    INIT_METHOD_USED_KEY,
    INIT_TRANSFORM_LIST_KEY,
    TARGET_IMG_MAX_SIZE_UM,
    TARGET_IMG_MIN_SIZE_UM,
    get_centroid_alignment_transforms,
)
from cellranger.spatial.tissue_regist_qc import (
    create_tissue_regist_qc_img,
    save_tissue_regist_qc_img,
)
from cellranger.spatial.transform import (
    convert_transform_corner_to_center,
    crop_img,
)
from cellranger.spatial.valis.registration import (
    FM_CANONICAL_PIXEL_SIZE,
    FM_NUM_FINAL_MATCHES_KEY,
    MAXIMUM_TISSUE_IMAGE_SCALING,
    feature_matching_based_rigid_registration,
)

__MRO__ = """
stage SEED_REGISTRATION(
    in  png       tissue_detection_image,
    in  tiff      registration_target_image,
    in  json      loupe_tissue_registration_json,
    in  json      crop_info_json,
    in  json      registered_spots_data_json,
    in  float[][] fid_perp_tmat,
    in  bool      is_visium_hd,
    in  float     tissue_image_pixel_size,
    in  bool      is_pd,
    in  bool      skip_feature_matching_init,
    out json      initial_transform_info_json,
    out float     pixel_size_target_to_cyta_ratio,
    out json      feature_matching_metrics,
    out jpg       cytassist_tissue_detection_debug,
    out jpg       target_tissue_detection_debug,
    out jpg       matched_features_debug,
    out jpg       feature_matching_registered_tissue_image,
    src py        "stages/spatial/seed_registration",
) split (
    in  float     cyta_scale_factor,
    in  float     target_scale_factor,
    out float     metric,
    out float[][] init_transform_mat,
    out json      chunk_feature_matching_metrics,
    out jpg       chunk_matched_features_debug,
    out jpg       chunk_feature_matching_registered_tissue_image,
) using (
    mem_gb   = 8,
    volatile = strict,
)
"""


def split(args):
    # Decide the scaling factors for the customer image
    if not os.path.exists(args.registration_target_image):
        raise ValueError("Registration target image does not exist.")

    # Compute scale suggestion to both FM and SITK-based registration
    reg_target_pixel_size = (
        args.tissue_image_pixel_size
        if args.tissue_image_pixel_size
        else get_tiff_pixel_size(args.registration_target_image)
    )
    martian.log_info(f"Tissue image pixel size read in from tiff {reg_target_pixel_size}.")

    chunks = []
    join_mem_gb = 8
    mem_gb_chunk = 12
    cyta_scale_factor = CYT_IMG_PIXEL_SIZE / FM_CANONICAL_PIXEL_SIZE
    if args.loupe_tissue_registration_json and os.path.exists(args.loupe_tissue_registration_json):
        pass
    elif not args.skip_feature_matching_init:
        reg_target_pixel_size_too_large = False
        if (
            reg_target_pixel_size is not None
            and reg_target_pixel_size > MAXIMUM_TISSUE_IMAGE_SCALING * FM_CANONICAL_PIXEL_SIZE
        ):
            reg_target_pixel_size_too_large = True
            martian.log_info(
                f"Tissue image target pixel size too large. Ignoring tissue size and using line search"
                f"{reg_target_pixel_size=} max size handled: {MAXIMUM_TISSUE_IMAGE_SCALING * FM_CANONICAL_PIXEL_SIZE}"
            )

        if reg_target_pixel_size is not None and not reg_target_pixel_size_too_large:
            # If known, use the provided target pixel size for scaling and if the target scale factor
            # is less than 10. Bounding the scale factor as else will run out of mem
            target_scale_factor = reg_target_pixel_size / FM_CANONICAL_PIXEL_SIZE
            chunks.append(
                {
                    "cyta_scale_factor": cyta_scale_factor,
                    "target_scale_factor": target_scale_factor,
                    "__mem_gb": mem_gb_chunk,
                    "__vmem_gb": 64,
                }
            )
        else:
            # Guess the scale using a linear search within bounds
            regist_target_img = Image.open(args.registration_target_image)  # Lazy loading
            target_short_dim = min(regist_target_img.width, regist_target_img.height)
            target_img_sizes_um = np.arange(TARGET_IMG_MIN_SIZE_UM, TARGET_IMG_MAX_SIZE_UM, 4000.0)
            target_scale_factor_list = (
                target_img_sizes_um / target_short_dim / FM_CANONICAL_PIXEL_SIZE
            )
            for target_scale_factor in target_scale_factor_list:
                chunks.append(
                    {
                        "cyta_scale_factor": cyta_scale_factor,
                        "target_scale_factor": target_scale_factor,
                        "__mem_gb": mem_gb_chunk,
                        "__vmem_gb": 64,
                    }
                )

    return {
        "chunks": chunks,
        "join": {
            "__mem_gb": join_mem_gb,
            "__vmem_gb": join_mem_gb + 14,
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

    if not args.skip_feature_matching_init and os.path.exists(args.registered_spots_data_json):
        # Generate a mask that covers the fiducial locations
        spots_data = LoupeParser(args.registered_spots_data_json)
        fiducial_mask = 255 - generate_fiducial_mask(
            cyta_img.shape[:2],
            spots_data.get_fiducials_imgxy(),
            spots_data.get_fiducials_diameter(),
        )
    else:
        fiducial_mask = None

    if args.is_visium_hd and args.fid_perp_tmat:
        # cv2.warpPerspective uses center-based sub-pixel coordinates
        center_based_perp_tmat = convert_transform_corner_to_center(np.array(args.fid_perp_tmat))
        # shape[::-1] because cv2 uses (width, height) convention
        cyta_img = cv2.warpPerspective(cyta_img, center_based_perp_tmat, cyta_img.shape[::-1])
        if fiducial_mask is not None:
            fiducial_mask = cv2.warpPerspective(
                fiducial_mask, center_based_perp_tmat, fiducial_mask.shape[::-1]
            )

    with open(args.crop_info_json) as f:
        crop_info = json.load(f)
    cropped_cyta_img, cyta_crop_mat = crop_img(cyta_img, crop_info)
    if fiducial_mask is not None:
        fiducial_mask, _ = crop_img(fiducial_mask, crop_info)

    fm_success = False
    martian.log_info("Begin feature matching based registration.")
    cropped_cyta_to_target_mat, fm_metrics = feature_matching_based_rigid_registration(
        regist_target_img=regist_target_img,
        cyta_img=cropped_cyta_img,
        fiducial_mask=fiducial_mask,
        target_scale_factor=args.target_scale_factor,
        cyta_scale_factor=args.cyta_scale_factor,
        # TODO: incorporate this image into pd_test and cs pipelines
        matched_features_debug=outs.chunk_matched_features_debug,
    )
    fm_success = cropped_cyta_to_target_mat is not None
    if fm_success:
        martian.log_info("Feature matching succeeded.")
        cyta_to_target_mat = cropped_cyta_to_target_mat @ cyta_crop_mat
        qc_img, _ = create_tissue_regist_qc_img(
            regist_target_img,
            cyta_img,
            np.linalg.inv(cyta_to_target_mat),
            qc_img_type="color" if args.is_pd else "checkerboard",
        )
        save_tissue_regist_qc_img(qc_img, outs.chunk_feature_matching_registered_tissue_image)
        # From cropped cyta to target
        outs.init_transform_mat = cyta_to_target_mat.tolist()
        outs.metric = fm_metrics[FM_NUM_FINAL_MATCHES_KEY]
    else:
        martian.log_warn("Feature matching failed.")
        outs.metric, outs.init_transform_mat = None, None
    with open(outs.chunk_feature_matching_metrics, "w") as f:
        json.dump(fm_metrics, f)


def join(args, outs, chunk_defs, chunk_outs):
    reg_target_pixel_size = (
        args.tissue_image_pixel_size
        if args.tissue_image_pixel_size
        else get_tiff_pixel_size(args.registration_target_image)
    )
    pixel_size_target_to_cyta_ratio = (
        reg_target_pixel_size / CYT_IMG_PIXEL_SIZE if reg_target_pixel_size else None
    )
    outs.pixel_size_target_to_cyta_ratio = pixel_size_target_to_cyta_ratio

    if args.loupe_tissue_registration_json and os.path.exists(args.loupe_tissue_registration_json):
        with open(outs.initial_transform_info_json, "w") as f:
            json.dump({INIT_TRANSFORM_LIST_KEY: [], INIT_METHOD_USED_KEY: ""}, f)
        return

    fm_success = False
    if len(chunk_outs) > 0:
        # Best is one with the most matched features
        best_chunk = max(
            chunk_outs,
            key=lambda chunk_out: chunk_out.metric if chunk_out.metric is not None else 0,
        )
        fm_success = best_chunk.init_transform_mat is not None
        cr_io.hardlink_with_fallback(
            best_chunk.chunk_feature_matching_metrics, outs.feature_matching_metrics
        )
        cr_io.hardlink_with_fallback(
            best_chunk.chunk_matched_features_debug, outs.matched_features_debug
        )
        if os.path.exists(best_chunk.chunk_feature_matching_registered_tissue_image):
            cr_io.hardlink_with_fallback(
                best_chunk.chunk_feature_matching_registered_tissue_image,
                outs.feature_matching_registered_tissue_image,
            )

    if fm_success:
        martian.log_info(f"Best chunk from feature matching has metric {best_chunk.metric:.4f}.")
        init_transform_list = [best_chunk.init_transform_mat]
    else:
        martian.log_info("Proceeding with centroid alignment initializations.")
        regist_target_img = io.imread(args.registration_target_image).astype(np.float32)
        cyta_img = io.imread(args.tissue_detection_image).astype(np.float32)

        if args.is_visium_hd and args.fid_perp_tmat:
            # cv2.warpPerspective uses center-based sub-pixel coordinates
            center_based_perp_tmat = convert_transform_corner_to_center(
                np.array(args.fid_perp_tmat)
            )
            # shape[::-1] because cv2 uses (width, height) convention
            cyta_img = cv2.warpPerspective(cyta_img, center_based_perp_tmat, cyta_img.shape[::-1])

        with open(args.crop_info_json) as f:
            crop_info = json.load(f)
        cropped_cyta_img, cyta_crop_mat = crop_img(cyta_img, crop_info)

        init_transform_list = [
            (mat @ cyta_crop_mat).tolist()
            for mat in get_centroid_alignment_transforms(
                regist_target_img,
                cropped_cyta_img,
                pixel_size_target_to_cyta_ratio,
                outs.target_tissue_detection_debug,
                outs.cytassist_tissue_detection_debug,
            )
        ]

    with open(outs.initial_transform_info_json, "w") as f:
        json.dump(
            {
                INIT_TRANSFORM_LIST_KEY: init_transform_list,
                INIT_METHOD_USED_KEY: FEATURE_MATCHING if fm_success else CENTROID_ALIGNMENT,
            },
            f,
        )
