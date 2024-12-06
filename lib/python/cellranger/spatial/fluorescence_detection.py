# Copyright (c) 2022 10X Genomics, Inc. All rights reserved.
#
"""Functions used in image fluorescence detection."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator, Sequence
from typing import Any

import cv2
import numpy as np
from skimage.morphology import disk

import cellranger.spatial.data_utils as du
import cellranger.spatial.tiffer as tiffer


def calculate_if_mean_sd(
    channel_dict: dict[Any, Sequence[float] | Sequence[int]]
) -> dict[str, Iterator[float]]:
    """For each IF image page, calculate the spot level mean and sd pixel.

    intensity.

    Args:
        channel_dict: A dictionary of lists with the pixel values of each
        spot. The first entry of summarize_spot_pixels()

    Returns:
        A dictionary of means and sd for each spot per channel
    """
    output = {}
    for channel_name, spot_pixel_values in channel_dict.items():
        output[channel_name + "_mean"] = map(np.mean, spot_pixel_values)
        output[channel_name + "_stdev"] = map(np.std, spot_pixel_values)
    return output


# pylint: disable=too-many-locals
def summarize_spot_pixels(
    tissue_image_paths: Sequence[str],
    tissue_positions_csv: str,
    scalefactors_json: str | bytes,
    image_page_names: list[str],
    pages_to_remove: list[str],
    downsample_size: float | None = None,
) -> tuple[dict[str, list[int]], dict[str, list[str]]]:
    """Returns means and standard deviations for each spot and channel for the.

    given IF image. Will downsample each page and returns a dictionary of
    means and sd for each spot per channel, a dictionary of barcodes, a
    dictionary of tissue positions (in/out)

    Args:
        tissue_image_paths: path to the original image.
        tissue_positions_csv: path to tissue positions csv.
        scalefactors_json: path to scalefactors json.
        downsample_size: In pixels, the size to which each page should be.
        resized to on the longest edge according to the tissue_hires_image.
        image_page_names: a list of user provided page names.
        Example = ["DAPI","FITC"]
        pages_to_remove: a list of image pages to be removed from the analysis.
        This is provided in the loupe manual alignment file.

    Returns:
        channel_dict: dictionary of lists with the pixel values of each spot
        which can be used with calculate_if_mean_sd().
        barcode_tissue: dictionary of barcodes and tissue positions (in/out).
    """
    barcode_tissue_positions = du.read_tissue_positions_csv(tissue_positions_csv)
    barcodes = [str(bc, "utf-8") for bc in barcode_tissue_positions.index]
    barcode_tissue_positions = barcode_tissue_positions.values.tolist()
    barcode_tissue_positions = [[bc] + data for bc, data in zip(barcodes, barcode_tissue_positions)]

    # For use with writing out the CSV
    barcode_tissue = {
        "barcode": [item[0] for item in barcode_tissue_positions],
        "in_tissue": [int(item[1]) for item in barcode_tissue_positions],
    }
    # Get the scale factors needed
    with open(scalefactors_json) as f:
        scalefactors = json.load(f)

    # create a mask for each spot
    oligo_roi = disk(
        int(
            scalefactors["spot_diameter_fullres"] * scalefactors["tissue_hires_scalef"],
        ),
    )

    pages = tiffer.call_tiffer_get_num_pages(tissue_image_paths[0])

    # Calculate the IF intensities for each page.
    # If a list of pages to remove comes from Loupe (removeImagePages) remove
    # those page index values assuming the names and the length of the image
    # pages are the same.
    channel_dict: dict[str, list[int]] = {}
    xy_idxs = (5, 4)
    for page_num in range(pages):
        if pages_to_remove and page_num in pages_to_remove:
            continue
        if_image = cv2.imread(
            tiffer.call_tiffer_resample(
                tissue_image_paths[0],
                downsample_size,
                f"{os.path.splitext(os.path.basename(tissue_image_paths[0]))[0]}_{page_num}.tiff",
                page=page_num,
            )["outpath"],
            cv2.IMREAD_UNCHANGED,
        )
        # Pull the y and x coordinates from the tissue positions list and get
        # the pixes associated with each disk
        oligo_data = []
        for barcode in barcode_tissue_positions:
            row_pixels = (
                int(
                    (float(barcode[xy_idxs[1]]) * scalefactors["tissue_hires_scalef"])
                    - (oligo_roi.shape[0] / 2)
                )
                + np.where(oligo_roi)[0]
            )
            col_pixels = (
                int(
                    (float(barcode[xy_idxs[0]]) * scalefactors["tissue_hires_scalef"])
                    - (oligo_roi.shape[1] / 2)
                )
                + np.where(oligo_roi)[1]
            )
            keep_idxs = np.where(
                (row_pixels >= 0)
                * (col_pixels >= 0)
                * (row_pixels < if_image.shape[0])
                * (col_pixels < if_image.shape[1])
                > 0
            )[0]
            row_pixels = row_pixels[keep_idxs]
            col_pixels = col_pixels[keep_idxs]
            if np.sum(keep_idxs) == 0:
                oligo_data.append(np.nan)
            else:
                oligo_data.append(
                    if_image[
                        row_pixels,
                        col_pixels,
                    ]
                )
        # if the user provided page names and those page names are the right
        # number use the names
        # if not use the default "channelX" nomenclature.
        if image_page_names and pages == len(image_page_names):
            channel_dict[image_page_names[page_num]] = oligo_data
        else:
            # keep the +1 so that the user doesn't see any changes to the
            # current convention.
            channel_dict[f"channel{page_num + 1}"] = oligo_data

    return channel_dict, barcode_tissue
