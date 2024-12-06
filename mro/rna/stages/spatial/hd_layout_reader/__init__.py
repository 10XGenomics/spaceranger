#!/usr/bin/env python3
#
# Copyright (c) 2023 10X Genomics, Inc. All rights reserved.
#

"""Read hd_layout file and output expected fiducial and spots position."""

from __future__ import annotations

import os
import shutil

from cellranger.spatial import data_utils, slide
from cellranger.spatial.data_utils import SLIDE_ID_EXCEPTIONS
from cellranger.spatial.slide_design_o3 import (  # pylint: disable=no-name-in-module, import-error
    VisiumHdLayout,
)

__MRO__ = """
stage HD_LAYOUT_READER(
    in  string slide_serial_capture_area,
    in  vlf    hd_layout_file,
    in  string visium_hd_slide_name,
    in  json   loupe_hd_slide_layout_json,
    in  bool   is_pd,
    out json   hd_layout_data_json,
    src py     "stages/spatial/hd_layout_reader",
) using (
    volatile = strict,
)
"""


def main(args, outs):
    outs.hd_layout_data_json = hd_layout_reader(
        args.slide_serial_capture_area,
        args.hd_layout_file,
        args.visium_hd_slide_name,
        args.loupe_hd_slide_layout_json,
        args.is_pd,
        outs.hd_layout_data_json,
    )


def hd_layout_reader(
    slide_sample_area_id: str | None,
    hd_layout_file_in: str | os.PathLike,
    visium_hd_slide_name: str | None,
    loupe_hd_slide_layout_json: str | None,
    is_pd: bool,
    hd_layout_data_out: str,
) -> str | None:
    """Entry point for stage to run the Go-based hd_layout file reader and provide JSON output.

    Args:
        slide_sample_area_id (str): input slide id of the form <batch_number>-<serial_number>-<area_id>
        loupe_hd_slide_layout_json (str | None): Slide layout loaded from Loupe manual alignment file
        hd_layout_file_in (str): None (for automatic retrieval or manual pathway) or hd_layout file pathname
        visium_hd_slide_name (str): used to retrive g
        is_pd (bool): is this a PD run?
        hd_layout_data_out (str): pathname for JSONified hd_layout data output to the pipeline
    """
    # Run hd_layout_reader with appropriate parameters for various scenarios
    if loupe_hd_slide_layout_json and os.path.exists(loupe_hd_slide_layout_json):
        shutil.copyfile(loupe_hd_slide_layout_json, hd_layout_data_out)
        return hd_layout_data_out

    if visium_hd_slide_name is None:
        return None

    if not slide_sample_area_id:
        return None

    # files path for this stage
    out_path = os.path.dirname(hd_layout_data_out)

    if data_utils.is_hd_slide(slide_sample_area_id):
        slide_sample_id, area_id = data_utils.parse_slide_sample_area_id(slide_sample_area_id)
        if hd_layout_file_in:
            load_file = hd_layout_file_in
        elif is_pd and slide_sample_id in SLIDE_ID_EXCEPTIONS:
            return None
        else:
            slide.call_hd_layout_reader(slide_sample_id, out_path)
            load_file = os.path.join(out_path, slide_sample_id + ".vlf")

        hd_layout_data = VisiumHdLayout.from_vlf_and_area(load_file, area_id)
        hd_layout_data.save_as_json(hd_layout_data_out)
        return hd_layout_data_out
    else:
        raise ValueError(f"{slide_sample_area_id} is not a recognized hd slide ID")
