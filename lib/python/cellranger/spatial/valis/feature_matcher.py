#
# Copyright (c) 2024 10X Genomics, Inc. All rights reserved.
#
"""Functions and classes to detect, match and filter image features."""
from dataclasses import dataclass
from typing import NamedTuple

import cv2
import numpy as np
from skimage import transform
from sklearn import metrics as skmetrics

from cellranger.spatial.transform import warp_xy

MIN_MATCHES = 5  # Instead of 4 because 4 undesirably guarantees match distance to be 0
FM_NUM_FEATURES_KEY = "fm_num_features_detected"
FM_AFTER_CROSS_CHECK_KEY = "fm_num_matches_after_cross_check"
FM_AFTER_RATIO_TEST_KEY = "fm_num_matches_after_ratio_test"
FM_AFTER_RANSAC_KEY = "fm_num_matches_after_ransac"
FM_AFTER_TUKEY_KEY = "fm_num_matches_after_tukey"


@dataclass
class FeatureDetectorDescriptor:
    """Class for feature detection and description.

    Note that in some cases, such as KAZE, kp_detector can also detect features. However, in other cases, there may need to be a separate feature detector (like BRISK or ORB) and feature descriptor (like VGG).
    """

    kp_detector: cv2.Feature2D | None
    kp_descriptor: cv2.Feature2D
    max_num_features: int

    def detect_and_compute(
        self, image: np.ndarray, mask: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Detect the features in the image.

        Args:
            image (np.ndarray): image in which the features will be detected. Should be a 2D uint8 image.
            mask (np.ndarray | None, optional): binary image with same shape as image, where foreground > 0, and background = 0. If provided, feature detection will only be performed on the foreground. Defaults to None.

        Returns:
            tuple[np.ndarray, np.ndarray]: (N, 2) array positions of keypoints in xy corrdinates and (N, M) array containing descriptor vectors each of length M.
        """
        if self.kp_detector is not None:
            detected_kp = self.kp_detector.detect(image, mask=mask)
            keypoints, descriptors = self.kp_descriptor.compute(image, detected_kp)
        else:
            keypoints, descriptors = self.kp_descriptor.detectAndCompute(image, mask=mask)

        # Retain keypoints and corresponding descriptors with highest responses
        if descriptors.shape[0] > self.max_num_features:
            response = np.array([x.response for x in keypoints])
            keep_idx = np.argsort(response)[::-1][0 : self.max_num_features]
            keypoints = [keypoints[i] for i in keep_idx]
            descriptors = descriptors[keep_idx, :]

        kp_pos_xy = np.array([k.pt for k in keypoints])
        return kp_pos_xy, descriptors


def filter_matches_ransac(
    kp1_xy: np.ndarray,
    kp2_xy: np.ndarray,
    ransac_val: int,
    max_iters: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Use OpenCV RANSAC to eliminate outliers."""
    if kp1_xy.shape[0] > MIN_MATCHES:
        _, mask = cv2.findHomography(kp1_xy, kp2_xy, cv2.RANSAC, ransac_val, maxIters=max_iters)
        good_idx = np.where(mask.reshape(-1) == 1)[0]
        filtered_src_points = kp1_xy[good_idx, :]
        filtered_dst_points = kp2_xy[good_idx, :]
        return filtered_src_points, filtered_dst_points, good_idx
    else:
        return np.array([]), np.array([]), np.array([])


def filter_matches_tukey(
    src_xy: np.ndarray, dst_xy: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Use Tukey's method to eliminate the probable outliers."""
    tform = transform.SimilarityTransform()
    tform.estimate(src=src_xy, dst=dst_xy)
    warped_xy = warp_xy(src_xy, tform.params)
    distances = np.linalg.norm(warped_xy - dst_xy, axis=1)

    quant_1 = np.quantile(distances, 0.25)
    quant_3 = np.quantile(distances, 0.75)
    iqr = quant_3 - quant_1
    outer_fence = 3 * iqr
    outer_fence_le = quant_1 - outer_fence
    outer_fence_ue = quant_3 + outer_fence

    good_idx = np.nonzero(np.logical_and(distances > outer_fence_le, distances < outer_fence_ue))[0]
    src_xy_inlier = src_xy[good_idx, :]
    dst_xy_inlier = dst_xy[good_idx, :]
    return src_xy_inlier, dst_xy_inlier, good_idx


def match_descriptors(
    descriptors1: np.ndarray,
    descriptors2: np.ndarray,
    cross_check: bool = False,
    max_ratio: float = 1.0,
) -> tuple[np.ndarray, dict]:
    """Match the descriptors across an image pair.

    Note: for binary descriptors, Hamming distance is used. For all else, Euclidian distance is used.

    Args:
        descriptors1 (np.ndarray): descriptors found in the source image.
        descriptors2 (np.ndarray): descriptors found in the target image.
        cross_check (bool, optional): if True, the matched keypoint pair is returned only if keypoint2 is the best match for keypoint1 in second image and keypoint1 is the best match for keypoint2 in first image. Defaults to False.
        max_ratio (float, optional): enbales the ratio test if smaller than 1.0. Defaults to 1.0.

    Returns:
        tuple[np.ndarray, dict]: (N, 2) array of the indices of the matches for source and target, and metrics.
    """
    if descriptors1.shape[1] != descriptors2.shape[1]:
        raise ValueError("Descriptor length must equal.")

    if np.issubdtype(descriptors1.dtype, np.bool_):
        metric = "hamming"
    else:
        metric = "euclidean"
    distances = skmetrics.pairwise_distances(descriptors1, descriptors2, metric=metric)
    indices1 = np.arange(descriptors1.shape[0])
    indices2 = np.argmin(distances, axis=1)
    metrics = {FM_NUM_FEATURES_KEY: distances.shape}

    if cross_check:
        matches1 = np.argmin(distances, axis=0)
        mask = indices1 == matches1[indices2]
        indices1 = indices1[mask]
        indices2 = indices2[mask]
        metrics[FM_AFTER_CROSS_CHECK_KEY] = indices1.size

    # Ratio test
    if max_ratio < 1.0:
        best_distances = distances[indices1, indices2]
        distances[indices1, indices2] = np.inf
        second_best_indices2 = np.argmin(distances[indices1], axis=1)
        second_best_distances = distances[indices1, second_best_indices2]
        second_best_distances[second_best_distances == 0] = np.finfo(np.double).eps
        ratio = best_distances / second_best_distances
        mask = ratio < max_ratio
        indices1 = indices1[mask]
        indices2 = indices2[mask]
        metrics[FM_AFTER_RATIO_TEST_KEY] = indices1.size

    return np.column_stack((indices1, indices2)), metrics


class MatchInfo(NamedTuple):
    """Matching results."""

    matched_kp1_xy: np.ndarray
    matched_desc1: np.ndarray
    matches12: np.ndarray
    matched_kp2_xy: np.ndarray
    matched_desc2: np.ndarray
    matches21: np.ndarray
    n_matches: int


def match_and_filter(  # pylint: disable=too-many-locals
    desc1: np.ndarray,
    kp1_xy: np.ndarray,
    desc2: np.ndarray,
    kp2_xy: np.ndarray,
    params: NamedTuple,
) -> tuple[MatchInfo, dict]:
    """Produce and filter matches given keypoints and their descriptors.

    Args:
        desc1 (np.ndarray): descriptors in source image.
        kp1_xy (np.ndarray): keypoints detected in source image.
        desc2 (np.ndarray): descriptors in target image.
        kp2_xy (np.ndarray): keypoints detected in target image.
        params (NamedTuple): Valis registration parameters.

    Returns:
        tuple[MatchInfo, dict]: matching results before and after outlier rejection, and metrics.
    """
    matches, metrics = match_descriptors(
        desc1, desc2, cross_check=params.cross_check_matches, max_ratio=params.ratio_test_threshold
    )

    desc1_match_idx = matches[:, 0]
    matched_kp1_xy = kp1_xy[desc1_match_idx, :]
    matched_desc1 = desc1[desc1_match_idx, :]

    desc2_match_idx = matches[:, 1]
    matched_kp2_xy = kp2_xy[desc2_match_idx, :]
    matched_desc2 = desc2[desc2_match_idx, :]

    filtered_matched_kp1_xy, filtered_matched_kp2_xy, good_idx = filter_matches_ransac(
        matched_kp1_xy, matched_kp2_xy, params.ransac_reproj_threshold, params.ransac_max_iters
    )
    metrics[FM_AFTER_RANSAC_KEY] = good_idx.size
    # Do additional filtering to remove other outliers that may have been missed by RANSAC
    if len(good_idx) > MIN_MATCHES:
        filtered_matched_kp1_xy, filtered_matched_kp2_xy, good_idx = filter_matches_tukey(
            filtered_matched_kp1_xy, filtered_matched_kp2_xy
        )
        metrics[FM_AFTER_TUKEY_KEY] = good_idx.size

    if len(good_idx) >= MIN_MATCHES:
        filterd_matched_desc1 = matched_desc1[good_idx, :]
        filterd_matched_desc2 = matched_desc2[good_idx, :]
        good_matches12 = desc1_match_idx[good_idx]
        good_matches21 = desc2_match_idx[good_idx]
    else:
        filterd_matched_desc1 = np.array([], dtype=float)
        filterd_matched_desc2 = np.array([], dtype=float)
        good_matches12 = np.array([], dtype=int)
        good_matches21 = np.array([], dtype=int)

    # Record filtered matches
    filtered_match_info = MatchInfo(
        matched_kp1_xy=filtered_matched_kp1_xy,
        matched_desc1=filterd_matched_desc1,
        matches12=good_matches12,
        matched_kp2_xy=filtered_matched_kp2_xy,
        matched_desc2=filterd_matched_desc2,
        matches21=good_matches21,
        n_matches=len(good_matches12),
    )

    return filtered_match_info, metrics
