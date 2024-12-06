#!/usr/bin/env python
#
# Copyright (c) 2024 10X Genomics, Inc. All rights reserved.
#

"""Returns flags for websummary alerts."""

from cellranger.spatial.loupe_util import LoupeParser
from cellranger.spatial.tiffer import call_tiffer_info

__MRO__ = """
stage WEBSUMMARY_ALERTS(
    in  path   loupe_alignment_file,
    in  file[] cytassist_image_paths,
    in  bool   is_visium_hd,
    out bool   slide_id_mismatch,
    src py     "stages/spatial/websummary_alerts",
)
"""


def main(args, outs):
    outs.slide_id_mismatch = slide_id_mismatch_alert(
        args.loupe_alignment_file, args.cytassist_image_paths, args.is_visium_hd
    )


def slide_id_mismatch_alert(loupe_alignment_file, cytassist_image_paths, is_visium_hd):
    """Checks whether the Slide ID entered in Loupe matches the metadata in the cytassist image.

    Args:
        loupe_alignment_file: Loupe Alignment File
        cytassist_image_paths (list): Path to cytassist image
        is_visium_hd (bool): True if Visium HD run

    Returns:
        bool: True if there is a mismatch
    """
    if not (loupe_alignment_file and is_visium_hd and cytassist_image_paths):
        return False
    loupe_data = LoupeParser(loupe_alignment_file)
    metadata = call_tiffer_info(cytassist_image_paths[0])
    meta_slide_id = metadata.get("slideSerial")
    meta_capture_area = metadata.get("captureArea")
    if loupe_data.has_serial_number() and meta_slide_id:
        return (
            meta_slide_id != loupe_data.get_serial_number()
            or meta_capture_area != loupe_data.get_area_id()
        )
    return False
