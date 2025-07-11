#
# Copyright (c) 2019 10X Genomics, Inc. All rights reserved.
#
"""Preflights for the Segment Nuclei pipeline."""
import os

import cellranger.spatial.preflight as cr_sp_preflight
from cellranger.spatial.segment_nuclei_preflights import ShouldSegmentSufficientStats

__MRO__ = """
stage SPATIAL_SEGMENT_NUCLEI_PREFLIGHTS(
    in  file tissue_image,
    in  int  max_nucleus_diameter_px,
    src py   "stages/spatial/spatial_segment_nuclei_preflights",
) using (
    vmem_gb = 32,
)
"""


def main(args, _outs):
    if not args.tissue_image:
        raise cr_sp_preflight.PreflightException("No tissue image provided to spaceranger segment.")
    elif not os.path.exists(args.tissue_image):
        raise cr_sp_preflight.PreflightException(
            f"Tissue image provided does not exist at {args.tissue_image}"
        )

    if args.max_nucleus_diameter_px and (
        args.max_nucleus_diameter_px < 1 or args.max_nucleus_diameter_px > 1024
    ):
        raise cr_sp_preflight.PreflightException(
            f"max_nucleus_diameter_px must be between 1 and 1024. Provided value: {args.max_nucleus_diameter_px}"
        )

    should_segment = ShouldSegmentSufficientStats.new(args.tissue_image)
    if not should_segment.tissue_segmentable:
        if not should_segment.is_jpeg and should_segment.is_multi_page_tiff:
            raise cr_sp_preflight.PreflightException(
                "Cannot perform segmentation on provided image because it is a multi-page TIFF. Provided TIFFs must be single page."
            )
        elif not should_segment.is_jpeg and should_segment.is_grayscale:
            raise cr_sp_preflight.PreflightException(
                "Cannot perform segmentation on provided image because it is a grayscale TIFF. Provided image must be 8-bit RGB."
            )
        elif not should_segment.is_jpeg and should_segment.is_eight_bit_pixel:
            raise cr_sp_preflight.PreflightException(
                "Cannot perform segmentation on provided image because it's pixel depth is not 8-bits. Provided image must be 8-bit RGB."
            )
        else:
            raise cr_sp_preflight.PreflightException(
                f"The provided image is not segmentable: {args.tissue_image}"
            )
