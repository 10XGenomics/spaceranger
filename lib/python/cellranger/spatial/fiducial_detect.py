#
# Copyright (c) 2021 10X Genomics, Inc. All rights reserved.
#
"""Fiducial detection algorithm and util functions."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, TypedDict

import cv2
import numpy as np
from scipy.signal import argrelmax, argrelmin
from scipy.stats import gaussian_kde, mode

import cellranger.metrics_names as metrics_names
from cellranger.spatial.hd_fiducial import turing_detect_fiducial

if TYPE_CHECKING:
    from cellranger.spatial.slide_design_o3 import VisiumHdSlideWrapper


def get_max_threshold(image: np.ndarray):
    """Gets the max threshold used for blob detector params.

    Args:
        image (np.ndarray): the input image being thresholded.
    """
    img_th, _ = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)
    # the image background is greater than 200 unless something else fails.
    # ensure that if we have lots of tissue, the threshold is at least 128.
    img_th = np.min([img_th, 128])
    # use the threshold to exclude dark regions from estimation of
    # the background intensity value (the mode of the bright parts of the image)
    background_intensity = mode(image[image > img_th].ravel())[0]
    # return a value slightly less than the background as max threshold
    if background_intensity < 200:
        return 255
    return int(0.97 * background_intensity)


def get_blob_size_threshold(blob_sizes, mindist):
    """Given blob size distribution, and minimum distance between blobs, obtain a threshold on blob size."""
    _, bins = np.histogram(blob_sizes, bins=50, density=True)
    density = gaussian_kde(blob_sizes)
    signal = density(bins)

    # get peaks and valleys in smoothed histogram
    peaks = argrelmax(signal)[0]
    valleys = argrelmin(signal)[0]

    # find up to 3 tallest peaks reverse sorted by size. Discard 1st peak if tiny and set farthest relevant peak
    farthest_peaks = sorted(
        sorted(zip(signal[peaks], peaks), reverse=True)[0 : min(3, len(peaks))],
        key=lambda v: v[1],
        reverse=True,
    )
    farthest_peak = (
        farthest_peaks[1][1]
        if farthest_peaks[0][0] < 0.12 * np.max(signal) and len(farthest_peaks) > 2
        else farthest_peaks[0][1]
    )

    # find valley prior to farthest peak and return as threshold on blob size
    threshold = (
        bins[valleys[valleys < farthest_peak][-1]]
        if np.any(valleys < farthest_peak)
        else bins[farthest_peak] - mindist
    )
    return threshold, bins[farthest_peak]


class KeyPointDict(TypedDict):
    x: float
    y: float
    size: float


def blob_detect_fiducials(
    image: np.ndarray, minsize: float = 10, roi_mask: np.ndarray = None
) -> tuple[list[KeyPointDict], dict[str, np.ndarray], dict[str, int]]:
    """Fiducial detection by blob detection.

    Based on opencv method: simpleblobDetector. Detects features of minimum 10px/blob
    by first standardizing the image to fixed pixel dimensions, then performing blob detection,
    and then filtering blobs detected based on their size in a data-driven manner.

    Args:
        image (np.ndarray): image with fiducials
        minsize (float): minimum blob size. Defaults to 10.
        roi_mask (np.ndarray): a 2D array of 0 and 1 with the same size as the image. 1 marks
            region of interest.

    Returns:
        Tuple[List[Dict], Dict, Dict]: the detected points, the dict for qc figures,
            and dict of metrics
    """
    if roi_mask is not None and roi_mask.shape != image.shape:
        raise ValueError(
            f"The shape of roi_mask {roi_mask.shape} is different with image {image.shape}"
        )

    # blob detector params
    params = cv2.SimpleBlobDetector_Params()
    params.filterByCircularity = True
    params.minCircularity = 0.8  # keep regular circular shapes
    params.filterByInertia = True
    params.minInertiaRatio = 0.5  # circle has inertia 1, ellipse 0-1, line 0
    params.filterByColor = False  # treat as greyscale
    params.thresholdStep = 1.0
    params.minThreshold = 10
    params.maxThreshold = 255  # no effect, but image max is 255
    # params.maxThreshold = get_max_threshold(image)

    # aggressively detect blobs
    detector: cv2.SimpleBlobDetector = cv2.SimpleBlobDetector_create(params)

    # below - scaling minsize for blurring slightly because this was optimized for 3K images and now we're at 2K
    # This was necessary to avoid obliterating contrast in some images.
    keypoints: Sequence[cv2.KeyPoint] = detector.detect(
        cv2.blur(image, (int(minsize * 0.75 + 0.5), int(minsize * 0.75 + 0.5))).astype("uint8")
    )

    # signal processing to find cutoff size and filter
    seed_keypoints = []
    added_keypoints = []
    if len(keypoints) > 0:
        threshold, mean_size = get_blob_size_threshold(
            np.array([x.size for x in keypoints]), minsize
        )
        seed_keypoints = [k for k in keypoints if k.size >= threshold]
        other_keypoints = [k for k in keypoints if k.size < threshold]
        final_keypoints = seed_keypoints[:]

        if len(seed_keypoints) < 400:
            ## do an iteration to grab more keypoints within distance approx 2*mean fiducial size. Can be slow
            for seed_kp in seed_keypoints:
                for other_kp in other_keypoints:
                    dist = np.sqrt(
                        (seed_kp.pt[0] - other_kp.pt[0]) ** 2
                        + (seed_kp.pt[1] - other_kp.pt[1]) ** 2
                    )

                    if 1.7 * mean_size <= dist <= 2.3 * mean_size:
                        final_keypoints += [other_kp]
        added_keypoints = list(set(final_keypoints).difference(set(seed_keypoints)))

        # sort-uniq the final keypoints, sorting based on location
        # to ensure stability for testing
        keypoints = []
        for final_kp in sorted(final_keypoints, key=lambda k: (k.pt[0], k.pt[1])):
            if len(keypoints) > 0 and final_kp == keypoints[-1]:
                continue
            if roi_mask is None:
                keypoints.append(final_kp)
            elif roi_mask[int(final_kp.pt[1]), int(final_kp.pt[0])] == 1:
                keypoints.append(final_kp)

    # generate qc image for both round.
    qc_img_dict = {
        "1": generate_keypoint_img(image, keypoints),
        "2": generate_keypoint_img(image, seed_keypoints),
        "3": generate_keypoint_img(image, added_keypoints),
    }
    detection_dict = [KeyPointDict(x=kp.pt[0], y=kp.pt[1], size=kp.size) for kp in keypoints]

    metrics = {
        metrics_names.DETECTED_FIDUCIALS: len(keypoints),
        f"{metrics_names.DETECTED_FIDUCIALS}_1": len(seed_keypoints),
        f"{metrics_names.DETECTED_FIDUCIALS}_2": len(added_keypoints),
    }

    return detection_dict, qc_img_dict, metrics


def generate_keypoint_img(img: np.ndarray, point_list: list):
    """Generate qc image with key points."""
    qc_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    for keypoint in point_list:
        x = int(keypoint.pt[0])
        y = int(keypoint.pt[1])
        cv2.circle(qc_img, (x, y), int(0.5 * keypoint.size), (0, 255, 0), 3)
    return qc_img


def detect_visium_hd_fiducials(
    fid_detect_img: np.ndarray, slide: VisiumHdSlideWrapper
) -> tuple[list[KeyPointDict], dict, dict[str, int]]:
    """Fiducial detection for Visium HD slides."""
    image = fid_detect_img / np.max(fid_detect_img)

    fid_radius = slide.circular_fiducial_outer_radius()
    label = [w / fid_radius for w in slide.circular_fiducial_ring_widths()]

    decoded_centers = turing_detect_fiducial(image.astype(np.float32), label=label)
    if len(decoded_centers) == 0:
        raise ValueError("No fiducials detected")

    decodings = [i[0] for i in decoded_centers]
    fid_center_list = np.asarray([i[1] for i in decoded_centers])

    num_design_fiducials = len(slide.circular_fiducials())
    num_detected_fiducials = len(fid_center_list)
    metrics = {
        metrics_names.DETECTED_FIDUCIALS: num_detected_fiducials,
        metrics_names.FIDUCIAL_DETECTION_RATE: num_detected_fiducials / num_design_fiducials,
    }

    detected = [
        KeyPointDict(name=coding, x=x, y=y) for (coding, (x, y)) in zip(decodings, fid_center_list)
    ]

    return detected, {}, metrics
