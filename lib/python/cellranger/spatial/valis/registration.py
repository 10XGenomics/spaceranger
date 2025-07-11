#
# Copyright (c) 2024 10X Genomics, Inc. All rights reserved.
#
"""Top level interface to feature matching based rigid registration."""
from dataclasses import dataclass, field
from typing import NamedTuple

import cv2
import martian
import numpy as np
from skimage import io, transform

from cellranger.spatial.tissue_regist import (
    adjust_image_contrast,
)
from cellranger.spatial.transform import (
    get_scaled_similarity_transform,
    reflection_matrix,
    warp_img,
    warp_xy,
)
from cellranger.spatial.valis.feature_matcher import (
    MIN_MATCHES,
    FeatureDetectorDescriptor,
    match_and_filter,
)
from cellranger.spatial.valis.visualization import draw_matches

FM_CONTAINS_REFLECTION_KEY = "fm_contains_reflection"
FM_MEDIAN_MATCH_DISTANCE_KEY = "fm_normalized_median_match_distance"
FM_NUM_FINAL_MATCHES_KEY = "fm_num_final_matches"
FM_ESTIMATED_SCALE_KEY = "fm_estimated_cyta_to_target_scale"
FM_CANONICAL_PIXEL_SIZE = 10.0
MAXIMUM_TISSUE_IMAGE_SCALING = (
    1.0  # Max scaling of FM_CANONICAL_PIXEL_SIZE that we allow the tissue image scale to be
)


@dataclass
class ZImage:
    """Class that stores info about an image, including the rigid registration parameters.

    Args:
    image (np.ndarray): grayscale image undergone preprocessing that will be used for feature detection.
    feature_detect_mask (np.ndarray | None, optional): mask of pixel value 0 or 255 specifying where feature detection should take place. Defaults to None.
    reflection_mat (np.ndarray): the reflection matrix. Defaults to np.identity(3).
    transform_mat (np.ndarray): the reflected-source-to-target transform matrix. Defaults to np.identity(3).
    """

    image: np.ndarray
    feature_detect_mask: np.ndarray | None = None
    reflection_mat: np.ndarray = field(default_factory=lambda: np.identity(3))
    transform_mat: np.ndarray = field(default_factory=lambda: np.identity(3))


class ValisRegistrationParameters(NamedTuple):
    """Parameters for feature matching registration."""

    check_for_reflections: bool
    max_num_features: int
    feature_detector: cv2.Feature2D | None
    feature_descriptor: cv2.Feature2D
    cross_check_matches: bool
    ratio_test_threshold: float
    ransac_reproj_threshold: int
    ransac_max_iters: int
    transform_class: transform.EuclideanTransform
    max_median_distance: float


class MatchResults(NamedTuple):
    """Results for a matching attempt."""

    metrics: dict
    num_matches: int
    transform_mat: np.ndarray
    reflected_over_x: bool
    debug_matches_image: np.ndarray


def rigid_register(
    src_img_obj: ZImage,
    dst_img_obj: ZImage,
    params: ValisRegistrationParameters,
) -> MatchResults:
    """Rigidly align the source image to the target.

    Args:
        src_img_obj (ZImage): ZImage object of the source image.
        dst_img_obj (ZImage): ZImage object of the target image.
        params (ValisRegistrationParameters): Valis registration parameters.

    Returns:
        MatchResults: matching results of the best choice of reflection.
    """

    def _try_register(refl_x: bool) -> MatchResults:
        refl_y = False
        reflection_mat = reflection_matrix(refl_x, refl_y, src_img_obj.image.shape)
        reflected_img = warp_img(
            src_img_obj.image,
            reflection_mat,
        )
        if src_img_obj.feature_detect_mask is not None:
            reflected_mask = warp_img(src_img_obj.feature_detect_mask, reflection_mat)
        else:
            reflected_mask = None

        reflected_src_xy, reflected_desc = feature_detector.detect_and_compute(
            reflected_img, reflected_mask
        )

        filtered_match_info, metrics = match_and_filter(
            reflected_desc, reflected_src_xy, dst_desc, dst_xy, params
        )
        metrics[FM_CONTAINS_REFLECTION_KEY] = refl_x or refl_y
        msg = f"Reflection over (x, y)-axis = ({refl_x}, {refl_y})"
        if filtered_match_info.n_matches < MIN_MATCHES:
            martian.log_warn(f"{msg}: not enough good matches were found.")
            metrics[FM_MEDIAN_MATCH_DISTANCE_KEY] = int(np.cumprod(dst_img_obj.image.shape)[-1])
            transform_mat = np.identity(3)
        else:
            transformer = params.transform_class()
            transformer.estimate(
                filtered_match_info.matched_kp1_xy, filtered_match_info.matched_kp2_xy
            )
            transform_mat = transformer.params
            # Compute normalized median registration error
            # Calculated in cyta space because target can have large area not imaged in cyta
            warped_dst_xy = warp_xy(
                filtered_match_info.matched_kp2_xy, np.linalg.inv(transformer.params)
            )
            median_dist = np.median(
                np.linalg.norm(warped_dst_xy - filtered_match_info.matched_kp1_xy, axis=1)
            )
            src_diag = np.sqrt(np.sum(np.power(reflected_img.shape, 2)))
            norm_median_dist = median_dist / src_diag
            metrics[FM_MEDIAN_MATCH_DISTANCE_KEY] = norm_median_dist
            martian.log_info(f"{msg}: {norm_median_dist=}, {filtered_match_info.n_matches=}.")

        return MatchResults(
            metrics=metrics,
            num_matches=filtered_match_info.n_matches,
            transform_mat=transform_mat,
            reflected_over_x=refl_x,
            debug_matches_image=draw_matches(
                reflected_img,
                filtered_match_info.matched_kp1_xy,
                dst_img_obj.image,
                filtered_match_info.matched_kp2_xy,
                unfiltered_kp1_xy=reflected_src_xy,
                unfiltered_kp2_xy=dst_xy,
            ),
        )

    if params.check_for_reflections:
        # Technically only need to check one mirroring configuration due to the rotation-invariance of the features used
        rx_to_check = [False, True]
    else:
        rx_to_check = [False]

    feature_detector = FeatureDetectorDescriptor(
        params.feature_detector, params.feature_descriptor, params.max_num_features
    )

    # Only need to compute this once since dst image is not tested for reflection
    dst_xy, dst_desc = feature_detector.detect_and_compute(
        dst_img_obj.image, dst_img_obj.feature_detect_mask
    )

    # Find the best reflection and return its matching results
    return min(
        map(_try_register, rx_to_check),
        key=lambda res: res.metrics[FM_MEDIAN_MATCH_DISTANCE_KEY],
    )


def feature_matching_based_rigid_registration(  # pylint: disable=too-many-locals
    regist_target_img: np.ndarray,
    cyta_img: np.ndarray,
    *,
    fiducial_mask: np.ndarray | None = None,
    target_scale_factor: float,
    cyta_scale_factor: float,
    matched_features_debug: str | None = None,
) -> tuple[np.ndarray | None, dict]:
    """Rigid registration using feature matching.

    Args:
        regist_target_img (np.ndarray): grayscale target tissue image.
        cyta_img (np.ndarray): grayscale CytaAssist image.
        fiducial_mask (np.ndarray | None, optional): binary mask used in masking out fiducial locations in feature detection. Defaults to None.
        target_scale_factor (float): used for scaling the target image. Defaults to None.
        cyta_scale_factor (float): used for scaling the cyta image. Defaults to None.
        matched_features_debug (str | None, optional): path to write the feature matching image to. Defaults to None.

    Returns:
        tuple[np.ndarray | None, dict]: cyta-to-target transformation matrix and metrics.
    """
    params = ValisRegistrationParameters(
        check_for_reflections=True,
        max_num_features=1000,  # Features used for matching
        feature_detector=None,
        feature_descriptor=cv2.SIFT_create(5000),  # Features detected regardless mask
        cross_check_matches=True,
        ratio_test_threshold=0.8,
        ransac_reproj_threshold=7,
        ransac_max_iters=4000,
        transform_class=transform.SimilarityTransform,
        max_median_distance=0.002,  # Normalized distance against image diagonal
    )
    target_input_shape = regist_target_img.shape
    cyta_input_shape = cyta_img.shape
    assert target_scale_factor is not None
    processed_target_img = transform.rescale(
        regist_target_img, target_scale_factor, preserve_range=True
    ).astype(np.uint8)
    assert cyta_scale_factor is not None
    processed_cyta_img = transform.rescale(cyta_img, cyta_scale_factor, preserve_range=True).astype(
        np.uint8
    )
    if fiducial_mask is not None:
        fiducial_mask = transform.rescale(
            fiducial_mask, cyta_scale_factor, preserve_range=True
        ).astype(np.uint8)

    processed_target_img = adjust_image_contrast(processed_target_img, clip_limit=0.01, invert=True)
    processed_cyta_img = adjust_image_contrast(processed_cyta_img, clip_limit=0.01, invert=True)

    target_img_obj = ZImage(processed_target_img, None)
    cyta_img_obj = ZImage(processed_cyta_img, fiducial_mask)

    match_results = rigid_register(cyta_img_obj, target_img_obj, params)
    match_results.metrics[FM_NUM_FINAL_MATCHES_KEY] = match_results.num_matches

    if matched_features_debug is not None:
        io.imsave(matched_features_debug, match_results.debug_matches_image)

    if match_results.metrics[FM_NUM_FINAL_MATCHES_KEY] < MIN_MATCHES:
        martian.log_warn(
            "Feature Matching based registration has failed: could not find enough good matches in any reflection."
        )
        return None, match_results.metrics

    if match_results.metrics[FM_MEDIAN_MATCH_DISTANCE_KEY] > params.max_median_distance:
        martian.log_warn(
            f"Feature Matching based registration has failed: the max acceptable normalized median match distance is {params.max_median_distance}, but was calculated as being {match_results.metrics[FM_MEDIAN_MATCH_DISTANCE_KEY]}."
        )
        return None, match_results.metrics

    # Scale transformation to input-scale
    cyta_to_target_mat = get_scaled_similarity_transform(
        mat=match_results.transform_mat,
        src_shape_rc=cyta_input_shape,
        dst_shape_rc=target_input_shape,
        transformation_src_shape_rc=cyta_img_obj.image.shape,
        transformation_dst_shape_rc=target_img_obj.image.shape,
    )
    # Add back reflection if necessary
    cyta_to_target_mat = cyta_to_target_mat @ reflection_matrix(
        match_results.reflected_over_x, False, cyta_input_shape
    )
    final_scale = transform.AffineTransform(cyta_to_target_mat).scale
    match_results.metrics[FM_ESTIMATED_SCALE_KEY] = np.mean(final_scale)

    return cyta_to_target_mat, match_results.metrics
