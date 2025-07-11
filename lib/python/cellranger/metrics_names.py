# Copyright (c) 2019 10X Genomics, Inc. All rights reserved.
"""This is centralized location to host all metrics."""

SUSPECT_ALIGNMENT = "suspect_alignment"
REORIENTATION_NEEDED = "reorientation_needed"
DETECTED_FIDUCIALS = "num_fiducials_detected"
DECODED_FIDUCIALS = "num_fiducials_decoded"
FIDUCIAL_DETECTION_RATE = "frac_fiducials_detected"
FIDUCIAL_DECODING_RATE = "frac_fiducials_decoded_given_detected"
OUTLIERS_DETECTED = "num_incorrect_decodings"
HD_LAYOUT_OFFSET_ABOVE_THRESHOLD = "hd_layout_offset.is_predicted_offset_outside_range"
HD_LAYOUT_OFFSET_FELLBACK_TO_PRE_REFINED_FIT = (
    "hd_layout_offset.is_pre_refinement_offset_good_but_fit_bad"
)
