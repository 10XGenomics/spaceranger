# Copyright (c) 2021 10x Genomics, Inc. All rights reserved.
"""Produce a UMI image at the given binning scale from the raw matrix and HD slide."""


from __future__ import annotations

import cv2
import numpy as np
from skimage import io

from cellranger.spatial.cytassist_constants import CYTA_IMAGE_DIM
from cellranger.spatial.hd_feature_slice import HdFeatureSliceIo

__MRO__ = """
stage CREATE_TOTAL_UMI_IMAGE(
    in  h5   hd_feature_slice_h5,
    in  int  binning_scale,
    out png  umi_image,
    out png  log_umi_image,
    out tiff log_umi_image_3k,
    out tiff umi_image_3k,
    src py   "stages/spatial/create_total_umi_image",
) using (
    mem_gb   = 3,
    vmem_gb  = 16,
    volatile = strict,
)
"""


def invert_and_normalize_image(image):
    maxcount = np.amax(image)
    return ((maxcount - image) * (255 / maxcount)).astype("uint8")


def resize_and_save(img, out_file):
    io.imsave(
        out_file,
        cv2.resize(img, (CYTA_IMAGE_DIM, CYTA_IMAGE_DIM), interpolation=cv2.INTER_NEAREST),
    )


def main(args, outs):
    binning_scale: int = args.binning_scale if args.binning_scale is not None else 1
    umi_img = HdFeatureSliceIo(args.hd_feature_slice_h5).total_umis(binning_scale).astype("float")

    inverted_umi_img = invert_and_normalize_image(umi_img)
    io.imsave(outs.umi_image, inverted_umi_img)
    resize_and_save(inverted_umi_img, outs.umi_image_3k)

    umi_img = np.log1p(umi_img)
    umi_img = invert_and_normalize_image(umi_img)

    io.imsave(outs.log_umi_image, umi_img)
    resize_and_save(umi_img, outs.log_umi_image_3k)
