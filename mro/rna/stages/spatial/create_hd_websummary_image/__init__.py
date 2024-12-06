# Copyright (c) 2021 10x Genomics, Inc. All rights reserved.
"""Generate the tissue image for the HD websummary."""

import json

import cv2

from cellranger.spatial.data_utils import DARK_IMAGES_CHANNELS, HD_WS_MAX_DIM
from cellranger.spatial.image_util import normalized_image_from_counts, shrink_to_max

__MRO__ = """
stage CREATE_HD_WEBSUMMARY_IMAGE(
    in  png   tissue_hires_image,
    in  json  scalefactors,
    in  int   dark_images,
    out jpg   websummary_tissue_image,
    out float websummary_tissue_image_scale,
    src py    "stages/spatial/create_hd_websummary_image",
) using (
    volatile = strict,
)
"""


def main(args, outs):
    tissue_hires_image = cv2.imread(args.tissue_hires_image, cv2.IMREAD_COLOR)
    if args.dark_images == DARK_IMAGES_CHANNELS:
        tissue_hires_image = normalized_image_from_counts(
            tissue_hires_image, log1p=True, invert=True
        )

    hires_to_ws_scale = shrink_to_max(tissue_hires_image.shape, HD_WS_MAX_DIM)
    tissue_ws_image = cv2.resize(
        tissue_hires_image, (0, 0), fx=hires_to_ws_scale, fy=hires_to_ws_scale
    )

    cv2.imwrite(outs.websummary_tissue_image, tissue_ws_image, [cv2.IMWRITE_JPEG_QUALITY, 90])

    with open(args.scalefactors) as f:
        scalefactors = json.load(f)
    outs.websummary_tissue_image_scale = hires_to_ws_scale * scalefactors["tissue_hires_scalef"]
