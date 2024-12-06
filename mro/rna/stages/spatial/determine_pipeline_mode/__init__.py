#!/usr/bin/env python3
#
# Copyright (c) 2021 10X Genomics, Inc. All rights reserved.
#

"""Stage to determine the subpipeline mode."""

import martian

from cellranger.spatial.pipeline_mode import PipelineMode, Product, SlideType

__MRO__ = """
stage DETERMINE_PIPELINE_MODE(
    in  file[]       tissue_image_paths,
    in  file[]       cytassist_image_paths,
    in  string       visium_hd_slide_name,
    in  string       barcode_whitelist,
    out PipelineMode pipeline_mode,
    out bool         is_visium_hd,
    out bool         is_visium_sd,
    src py           "stages/spatial/determine_pipeline_mode",
) using (
    volatile = strict,
)
"""

XL_WHITELISTS = (  # THIS NEEDS TO GO AND COME FROM THE GAL FILE INSTEAD
    "pseudo-xl-v1",
    "thor-XL-v1",
    "thor-XL-v2",
    "visium-v5",
)


def main(args, outs):
    """Determine both product and slide."""
    # Determine product.
    if args.cytassist_image_paths:
        product = Product.CYT
    elif args.tissue_image_paths:
        if args.visium_hd_slide_name:
            product = Product.VISIUM_HD_NOCYT_PD
        else:
            product = Product.VISIUM
    else:
        martian.throw("Pipeline called with no image")

    # Determine slide.
    if args.visium_hd_slide_name:
        slide = SlideType.VISIUM_HD
    elif args.barcode_whitelist in XL_WHITELISTS:
        slide = SlideType.XL
    else:
        slide = SlideType.VISIUM

    pipeline_mode = PipelineMode(product=product, slide=slide)
    try:
        pipeline_mode.validate()
    except ValueError:
        martian.throw(f"Invalid pipeline mode of {pipeline_mode}")
    outs.pipeline_mode = pipeline_mode._asdict()
    outs.is_visium_hd = product == Product.VISIUM_HD_NOCYT_PD or (
        product == Product.CYT and slide == SlideType.VISIUM_HD
    )
    outs.is_visium_sd = not outs.is_visium_hd
