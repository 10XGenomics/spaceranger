#
# Copyright (c) 2025 10X Genomics, Inc. All rights reserved.
#
"""Data loading utilities which depend on pandas."""

import os

import pandas as pd

from cellranger.spatial.data_utils import (
    IMAGEX_LOWRES,
    IMAGEY_LOWRES,
    TISSUE_POSITIONS_HEADER,
    TISSUE_POSITIONS_HEADER_TYPES,
    get_scalefactors,
)


def get_lowres_coordinates(tissue_positions_csv: str, scalefactors_json: str) -> pd.DataFrame:
    """Return a pandas data frame that is just like the tissue_positions_csv but has the lowres scaled image coordinates.

    Args:
        tissue_positions_csv (str): Path to the tissue_positions.csv
        scalefactors_json (str): Path to the scalefactors_json.json

    Returns:
        pd.DataFrame:
    """
    coords = read_tissue_positions_csv(tissue_positions_csv)

    # read in scalefactors json and adjust coords for downsampled image
    scalef = get_scalefactors(scalefactors_json)["tissue_lowres_scalef"]
    coords[IMAGEY_LOWRES] = coords["pxl_row_in_fullres"] * scalef
    coords[IMAGEX_LOWRES] = coords["pxl_col_in_fullres"] * scalef
    return coords


def estimate_mem_gb_pandas_csv(csv_filename: str | None) -> float:
    """Memory required to load the CSV such as tissue positions, filtered barcodes using pandas.

    If the input file foes not exists, returns 0

    Args:
        csv_filename (str | None): Filename

    Returns:
        float: memory in GB
    """
    if csv_filename is None or (not os.path.exists(csv_filename)):
        return 0.0
    # Empirically estimated memory by loading the csv file and checking
    # the RSS used
    mem_gb_per_gb_on_disk = 4.1
    file_size_gb = os.path.getsize(csv_filename) / (1024**3)
    return mem_gb_per_gb_on_disk * file_size_gb


def read_tissue_positions_csv(tissue_positions_fn) -> pd.DataFrame:
    # output dir to search for a file name
    # raw data
    # file name
    """Read the tissue positions csv as a pandas dataframe.

    Args:
        tissue_positions_fn (str): Filename

    Returns:
        pd.DataFrame: Csv as a dataframe
    """
    # For backwards compatibility
    ## First check if the file has a header. If there are digits there is no header
    with open(tissue_positions_fn) as f:
        first_line = f.readline()

    no_header = any(map(str.isdigit, first_line))

    # Set the kwargs according to the header state
    kwargs = {"names": TISSUE_POSITIONS_HEADER} if no_header else {"header": 0}

    coords = pd.read_csv(
        tissue_positions_fn,
        **kwargs,
        dtype=TISSUE_POSITIONS_HEADER_TYPES,
        sep=",",
    )
    coords["barcode"] = coords["barcode"].str.encode(encoding="ascii")
    coords = coords.set_index("barcode")
    return coords
