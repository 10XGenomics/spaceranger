#!/usr/bin/env python3
#
# Copyright (c) 2022 10X Genomics, Inc. All rights reserved.
#

"""Read gpr file and output expected fiducial and spots position."""

from __future__ import annotations

import json
import os

import martian

from cellranger.spatial import data_utils, slide

__MRO__ = """
stage GPR_READER(
    in  string slide_serial_capture_area,
    in  gpr    gpr_file,
    in  string barcode_whitelist,
    in  json   loupe_spots_data_json,
    out json   gpr_spots_data_json,
    src py     "stages/spatial/gpr_reader",
) split using (
)
"""


def main(args, outs):
    gpr_reader(
        args.slide_serial_capture_area,
        args.gpr_file,
        args.barcode_whitelist,
        args.loupe_spots_data_json,
        outs.gpr_spots_data_json,
    )


def gpr_reader(
    slide_sample_area_id: str,
    gpr_file_in: str,
    barcode_whitelist: str,
    loupe_spots_data_json: str | None,
    gpr_data_out: str,
):
    """Entry point for stage to run the Go-based gpr file reader and provide JSON output.

    Args:
        slide_sample_area_id (str): input slide id of the form <batch_number>-<serial_number>-<area_id>
        loupe_spots_data_json (str | None): None for automatic pathway
        gpr_file_in (str): None (for automatic retrieval or manual pathway) or gpr file pathname
        barcode_whitelist (str): used to retrive gal file
        gpr_data_out (str): pathname for JSONified GPR data output to the pipeline
    """
    # Run gprreader with appropriate parameters for various scenarios
    if loupe_spots_data_json and os.path.exists(loupe_spots_data_json):
        return

    # files path for this stage
    out_path = os.path.dirname(gpr_data_out)

    # return if no galfile for specified barcode whitelist
    galfile = data_utils.get_galfile_path(barcode_whitelist)
    if not os.path.exists(galfile):
        martian.log_info(f"gpr_reader - no GAL file for whitelist {barcode_whitelist}")
        return

    if slide_sample_area_id and data_utils.is_production_slide(slide_sample_area_id):
        slide_sample_id, area_id = data_utils.parse_slide_sample_area_id(slide_sample_area_id)
        if gpr_file_in:
            slide.call_gprreader("read", gpr_file_in, area_id, out_path)
            _, basename_ext = os.path.split(gpr_file_in)
            basename, _ = os.path.splitext(basename_ext)
            gpr_data = data_utils.read_from_json(
                os.path.join(out_path, basename + "_" + area_id + ".json")
            )
        else:
            slide.call_gprreader("fetch", slide_sample_id, area_id, out_path)
            gpr_data = data_utils.read_from_json(slide_sample_id + "_" + area_id + ".json")
    else:  # default case
        galfile = data_utils.get_galfile_path(barcode_whitelist)
        slide.call_gprreader("default", galfile, None, out_path)
        gpr_data = data_utils.read_from_json("default.json")

    # Regardless of where the GPR is coming from, check it has the required data
    if not gpr_data["spots"]["oligo"] or not gpr_data["spots"]["fiducial"]:
        martian.exit("The slidefile is missing either oligo or fiducial spot information.")

    gpr_data = gpr_data["spots"]
    with open(gpr_data_out, "w") as json_out:
        json.dump(gpr_data, json_out)
