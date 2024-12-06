#!/usr/bin/env python3
#
# Copyright (c) 2023 10X Genomics, Inc. All rights reserved.
#

"""Functions to generate the final output of oligos and barcodes."""

from __future__ import annotations

import csv
import json
from collections import Counter

from cellranger.spatial.data_utils import TISSUE_POSITIONS_HEADER
from cellranger.spatial.image import WebImage

# Tissue positions csv header
TISSUE_POSITIONS_HEADER = ",".join(TISSUE_POSITIONS_HEADER) + "\n"

GEM_GROUP = 1


def read_barcode_coordinates(barcode_coordinates_file):
    """Read the barcode_whitelist file.

    Return a dict with spatial coordinates (row and column as integers)
    as keys and barcode sequnce as values.

    Args:
        barcode_coordinates_file (path): Path to the barcode coordinates text file.

    Returns:
        barcodes (dict): Dictionary with (row, column, dup#) as keys and barcode sequences as values.
            dup# is added to support custom slides where we want to associate multiple
            barcodes to the same spot.
    """
    # read barcode coordinates
    barcodes = {}
    coords = Counter()
    with open(barcode_coordinates_file) as barcode_coordinates:
        for line in barcode_coordinates:
            if "#" in line:
                continue
            fields = line.strip().split(maxsplit=2)
            col, row = int(fields[1]) - 1, int(fields[2]) - 1
            barcodes[(row, col, coords[(row, col)])] = fields[0]  # note row, col, dup#
            coords[(row, col)] += 1
    return barcodes


def save_tissue_position_list(
    oligo_list: list,
    tissue_oligos_flag: list[bool],
    barcodes: dict[tuple[int, int, int], str],
    filepath: str,
    gemgroup: int = GEM_GROUP,
) -> list[str]:
    """Save the barcode and oligo information for future sub-pipelines.

    Args:
        oligo_list (np.ndarray): (n, 4) shape. contain (row, col, row-position, col-position)
        tissue_oligos_flag (List[bool]): mark whether each oligo is under tissue or not
        barcodes (Dict): corresponding barcodes for each oligo spot
        filepath (str): path to save the txt file to
        gemgroup (int, optional): Defaults to 1.

    Returns:
        List[str]: the tissue position list to be saved
    """
    tissue_barcodes = []
    with open(filepath, "w") as f:
        f.write(TISSUE_POSITIONS_HEADER)
        for (row, col, rowpos, colpos), in_tissue in zip(oligo_list, tissue_oligos_flag):
            row = int(row)
            col = int(col)
            dup = 0
            done = False
            while not done:
                if (row, col, dup) in barcodes:
                    barcode = f"{barcodes[(row, col, dup)]}-{gemgroup!s}"
                    if int(in_tissue) > 0:
                        tissue_barcodes.append(barcode)
                    f.write(f"{barcode},{int(in_tissue)},{row},{col},{rowpos},{colpos}\n")
                    dup += 1
                else:
                    done = True
    return tissue_barcodes


def spots_outside_image(
    tissue_positions: str, tissue_lowres_image: str, scale_factors: str
) -> dict:
    """Calculate the fraction of spots with pixel coordinates outside the image boundaries.

    Args:
        tissue_positions (str): The path to the CSV file containing spot coordinates.
        tissue_lowres_image (str): The path to the low res tissue image
        scale_factors (str): The path to the scale_factors.json

    Returns:
        dict: The fraction of spot coordinates with at least one value outside the image boundaries.

    """
    # Get lowres image dimensions
    img = WebImage(tissue_lowres_image)
    low_res_height = img.height
    low_res_width = img.width

    # Read in scale_factors
    with open(scale_factors) as f:
        scale_factor_dict = json.load(f)
    # Calculate height and width of original res image
    original_res_height = low_res_height / scale_factor_dict["tissue_lowres_scalef"]
    original_res_width = low_res_width / scale_factor_dict["tissue_lowres_scalef"]

    # Open the CSV file and read it line by line
    total_count = 0
    outside_count = 0
    with open(tissue_positions) as file:
        reader = csv.DictReader(file)
        for row in reader:
            pxl_row_in_fullres = float(row["pxl_row_in_fullres"])
            pxl_col_in_fullres = float(row["pxl_col_in_fullres"])
            in_tissue = int(row["in_tissue"])

            if in_tissue == 1:
                total_count += 1
                if any(
                    (
                        pxl_row_in_fullres < 0,
                        pxl_row_in_fullres >= original_res_height,
                        pxl_col_in_fullres < 0,
                        pxl_col_in_fullres >= original_res_width,
                    )
                ):
                    outside_count += 1

    fraction_outside_image = round(outside_count / total_count, ndigits=3)
    return {"fraction_bc_outside_image": fraction_outside_image}


def patterened_barcode_filenames(pattern_type: int) -> tuple[str, str]:
    """Centralize construction of pattern barcode filenames.

    returns: affected, unaffected nearby
    """
    return f"pat{pattern_type}_affected_barcodes", f"pat{pattern_type}_nearby_barcodes"
