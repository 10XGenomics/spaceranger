#
# Copyright (c) 2024 10X Genomics, Inc. All rights reserved.
#

"""Functions related to fixing data affected by a bug in CytAssist firmware 2.1.z.w."""

import martian

from cellranger.spatial.cytassist_constants import CYTA_HD_IMAGE_WIDTH
from cellranger.spatial.tiffer import (
    TIFFER_INFO_CYTA_SW_KEY,
    TIFFER_INFO_PAGE_KEY,
    TIFFER_INFO_WIDTH_KEY,
    try_call_tiffer_info,
)


def test_for_cytassist_image_fix(image_path):
    """Use get cytassist firmware version and check whether red shift fix needs to be performed."""
    image_info = None
    try:
        image_info = try_call_tiffer_info(image_path)
    except RuntimeError as exc:
        martian.log_info(f"error reading cytassist image for metadata, but continuing on:\n{exc}")
    if image_info is not None:
        swver = image_info.get(TIFFER_INFO_CYTA_SW_KEY, "")
        cyta_pages = image_info.get(TIFFER_INFO_PAGE_KEY)
        if cyta_pages and len(cyta_pages) == 1:
            return (
                swver.startswith("2.1.")
                and not swver.endswith("corrected")
                and cyta_pages[0].get(TIFFER_INFO_WIDTH_KEY) == CYTA_HD_IMAGE_WIDTH
            )

    return False
