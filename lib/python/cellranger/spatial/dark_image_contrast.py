#
# Copyright (c) 2024 10X Genomics, Inc. All rights reserved.
#

"""Functions for dealing with contrast enhancement for immunoflourescence images."""

from typing import NamedTuple

import cv2
import numpy as np

UPPER_LIMIT_FRACTION_PIXELS = 1 / 10
LOWER_LIMIT_FRACTION_PIXELS = 1 / 5000
MINIMUM_PIXEL_VALUE_RANGE_TO_ENHANCE = 10
MAX_PIXEL_VALUE = np.iinfo(np.uint8).max


class ClipLimits(NamedTuple):
    lower: int | None = None
    upper: int | None = None


def get_clip_limits(img: np.ndarray) -> ClipLimits:
    """Compute clip limits used for contrast enhancement.

    The algorithm here is ported over from ImageJ's FIJI.
    The lower threshold is the lowest pixel value that is seen in more than
    1/5000 pixels but less than 10% of pixels.
    The upper threshold for clipping is the largest pixel value that is seen in
    in more than 1/5000 pixels but less than 10% of pixels.
    The numbers 1/5000 and 1/10 are magic numbers we picked up from ImageJ.

    The input image is expected to be a three channel image of dtype uint8.
    """
    brightness_values = np.max(img, axis=2).flatten()
    num_pixels = len(brightness_values)
    brightness_histogram, _ = np.histogram(
        brightness_values, bins=range(MAX_PIXEL_VALUE + 2)
    )  # Need to have bins 0:256 to have 256 bins - as rightmost value included

    lower_hist_threshold = num_pixels * LOWER_LIMIT_FRACTION_PIXELS
    upper_hist_threshold = int(num_pixels * UPPER_LIMIT_FRACTION_PIXELS)

    prospective_cutoff_values = np.flatnonzero(
        (brightness_histogram > lower_hist_threshold)
        & (brightness_histogram <= upper_hist_threshold)
    )
    if not prospective_cutoff_values.shape[0]:
        return ClipLimits()
    return ClipLimits(
        lower=np.min(prospective_cutoff_values), upper=np.max(prospective_cutoff_values)
    )


def enhance_immunoflourescence_contrast(img: np.ndarray) -> np.ndarray:
    """Enhance contrast of immunoflourence image.

    This clips pixels  at thresholds picked according to
    the algorithm used in ImageJ and ported over to loupe. In addition we perform
    the contrast enhancement only if the lower limit is  more than 10 units smaller
    than the upper limit.

    The input image is expected to be a three channel image of dtype uint8.
    Returns an image of dtype uint8 and same shape as input image.
    """
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(
            f"Expected a 3-channel image of shape (X_shape, Y_shape, 3). Received image of shape: {img.shape}"
        )
    if img.dtype != np.uint8:
        raise ValueError(
            f"Expect only an 8-bit image while enhancing IF contrast. Got an image of type {img.dtype}"
        )

    lower_limit, upper_limit = get_clip_limits(img=img)
    if (
        lower_limit is None
        or not upper_limit
        or upper_limit - lower_limit <= MINIMUM_PIXEL_VALUE_RANGE_TO_ENHANCE
    ):
        return img

    lut = np.interp(
        range(MAX_PIXEL_VALUE + 1),
        [0, lower_limit, upper_limit, MAX_PIXEL_VALUE],
        [0, 0, MAX_PIXEL_VALUE, MAX_PIXEL_VALUE],
    ).astype(np.uint8)

    return cv2.LUT(img, lut)
