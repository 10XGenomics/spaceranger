#!/usr/bin/env python3
#
# Copyright (c) 2021 10X Genomics, Inc. All rights reserved.
#

"""Calculate IF intensity within area of each spot for each channel if an IF image."""

from __future__ import annotations

import csv

import numpy as np

import cellranger.spatial.fluorescence_detection as fd
from cellranger.spatial.data_utils import (
    DARK_IMAGES_CHANNELS,
    HIRES_MAX_DIM_DEFAULT,
    HIRES_MAX_DIM_DICT,
)
from cellranger.spatial.loupe_util import LoupeParser
from cellranger.spatial.pipeline_mode import PipelineMode

__MRO__ = """
stage CALCULATE_FLUORESCENCE_INTENSITY(
    in  file[]       tissue_image_paths,
    in  int          dark_images,
    in  json         scalefactors_json,
    in  csv          tissue_positions,
    in  string       barcode_whitelist,
    in  string[]     image_page_names,
    in  path         loupe_alignment_file,
    in  PipelineMode pipeline_mode,
    out csv          barcode_fluorescence_intensity,
    src py           "stages/spatial_pd/calculate_fluorescence_intensity_pd",
) split (
) using (
    mem_gb   = 1,
    threads  = 1,
    volatile = strict,
)
"""


def split(args):
    """Check if dark image channels are used and proceed if so."""
    if args.dark_images == DARK_IMAGES_CHANNELS and args.tissue_positions:
        chunk_def = {"__mem_gb": 24, "__vmem_gb": 32, "__threads": 16}
    else:
        chunk_def = {}
    return {"chunks": [], "join": chunk_def}


def join(args, outs, _chunk_def, _chunk_outs):
    if args.dark_images != DARK_IMAGES_CHANNELS or args.tissue_positions is None:
        outs.barcode_fluorescence_intensity = None
        return
    image_page_names = None
    pages_to_remove = []

    # if there is a loupe alignment file that has page names use that as the record of evidence
    if args.loupe_alignment_file:
        loupe_data = LoupeParser(args.loupe_alignment_file)
        image_page_names = loupe_data.get_image_page_names()
        pages_to_remove = loupe_data.get_remove_image_pages()

    # If the user provided image page names from the command line or pipeline_paramaters_json, use it
    if args.image_page_names:
        image_page_names = args.image_page_names

    # calculate IF intensity in each spot for each channel.
    ## Determine the pipeline mode use this to determine how the images should be downsampled
    pipeline_mode = PipelineMode(**args.pipeline_mode)
    max_dim = HIRES_MAX_DIM_DICT.get(pipeline_mode, HIRES_MAX_DIM_DEFAULT)

    spot_summary = fd.summarize_spot_pixels(
        tissue_image_paths=args.tissue_image_paths,
        tissue_positions_csv=args.tissue_positions,
        scalefactors_json=args.scalefactors_json,
        downsample_size=max_dim,
        image_page_names=image_page_names,
        pages_to_remove=pages_to_remove,
    )
    # Calculate the mean and sd fluorescence intensity for each spot on each page
    mean_sd_dict = fd.calculate_if_mean_sd(spot_summary[0])
    for name, values in mean_sd_dict.items():
        mean_sd_dict[name] = ["NA" if np.isnan(y) else y for y in values]
    # Combine barcodes and tissue positions with IF mean and sd for output to csv
    spot_summary_out = spot_summary[1].copy()
    spot_summary_out.update(mean_sd_dict)
    # Write out the CSV
    with open(outs.barcode_fluorescence_intensity, "w") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(spot_summary_out.keys())
        writer.writerows(zip(*spot_summary_out.values()))
