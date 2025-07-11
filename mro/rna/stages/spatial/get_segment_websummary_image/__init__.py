#
# Copyright (c) 2025 10X Genomics, Inc. All rights reserved.
#

"""Get websummary image for spaceranger segment."""

import json

import cellranger.spatial.tiffer as tiffer
from cellranger.spatial.data_utils import HIRES_VISIUM_HD_MAX_DIM_DEFAULT

__MRO__ = """
stage GET_SEGMENT_WEBSUMMARY_IMAGE(
    in  file tissue_image,
    out png  tissue_hires_image,
    out json barebones_scalefactors,
    src py   "stages/spatial/get_segment_websummary_image",
) split (
) using (
    vmem_gb = 32,
)
"""


def split(args):
    if not args.tissue_image:
        raise ValueError("Need a tissue image for segment websummary")
    mem_gb_estimate = tiffer.call_tiffer_mem_estimate_gb(
        args.tissue_image, HIRES_VISIUM_HD_MAX_DIM_DEFAULT
    )

    image_info = tiffer.call_tiffer_info(args.tissue_image)
    is_jpeg = image_info[tiffer.TIFFER_INFO_FORMAT_KEY] == tiffer.TIFFER_JPEG_VALUE
    if is_jpeg:
        mem_gb_estimate += 4

    mem_gb_estimate = max(mem_gb_estimate, 8)
    return {
        "chunks": [],
        "join": {
            "__mem_gb": mem_gb_estimate,
            "__vmem_gb": max(2 * mem_gb_estimate, 64),
        },
    }


def join(args, outs, _chunk_defs, _chunk_outs):
    hires_img_dict = tiffer.call_tiffer_resample(
        args.tissue_image, HIRES_VISIUM_HD_MAX_DIM_DEFAULT, outs.tissue_hires_image
    )
    hr_scalef = hires_img_dict[tiffer.TIFFER_RESAMPLE_SCALE_KEY]

    with open(outs.barebones_scalefactors, "w") as f:
        json.dump({"tissue_hires_scalef": hr_scalef}, f, indent=4)
