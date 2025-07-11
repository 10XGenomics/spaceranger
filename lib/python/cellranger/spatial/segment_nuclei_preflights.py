#
# Copyright (c) 2025 10X Genomics, Inc. All rights reserved.
#
"""Functions to do preflights for SPATIAL_SEGMENT_NUCLEI_CS."""


from dataclasses import dataclass
from typing import Self

import cellranger.spatial.tiffer as tiffer


@dataclass
class ShouldSegmentSufficientStats:
    """Variables we extract from tissue image to decide if it is segmentable."""

    tissue_segmentable: bool
    is_jpeg: bool
    is_multi_page_tiff: bool
    is_grayscale: bool
    is_eight_bit_pixel: bool

    @classmethod
    def new(cls, tissue_image_path: str | bytes) -> Self:
        """Extract information we need to decide if we can do nucleus segmentation."""
        image_info = tiffer.call_tiffer_info(tissue_image_path)
        is_jpeg = image_info[tiffer.TIFFER_INFO_FORMAT_KEY] == tiffer.TIFFER_JPEG_VALUE
        is_multi_page_tiff = len(pages := image_info.get(tiffer.TIFFER_INFO_PAGE_KEY, [{}])) > 1
        is_grayscale = (
            pages[0].get(tiffer.TIFFER_INFO_COLORMODE_KEY) == tiffer.TIFFER_GRAYSCALE_VALUE
        )
        is_eight_bit_pixel = pages[0].get(tiffer.TIFFER_INFO_PIXEL_DEPTH_KEY) == 8

        tissue_segmentable = is_jpeg or not (  # Has to either be a JPEG
            is_multi_page_tiff  # Or a TIFF with one page
            or is_grayscale  # which is not grayscale
            or not is_eight_bit_pixel  # which has 8 bit pixels
        )
        return cls(
            tissue_segmentable=tissue_segmentable,
            is_jpeg=is_jpeg,
            is_multi_page_tiff=is_multi_page_tiff,
            is_grayscale=is_grayscale,
            is_eight_bit_pixel=is_eight_bit_pixel,
        )
