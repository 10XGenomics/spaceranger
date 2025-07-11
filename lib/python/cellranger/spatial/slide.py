#!/usr/bin/env python
#
# Copyright (c) 2020 10X Genomics, Inc. All rights reserved.
#

"""Read slide metadata."""

from __future__ import annotations

import os
import subprocess
import tempfile

import martian

import tenkit.log_subprocess as tk_subproc
from cellranger.constants import BARCODE_WHITELIST_PATH
from cellranger.spatial.data_utils import get_galfile_path, read_from_json
from cellranger.spatial.visium_hd_schema_pb2 import (  # pylint: disable=no-name-in-module, import-error
    VisiumHdSlideDesign,
)


def call_hd_layout_reader(input_file, output_path):
    """Run hd layout reader with given params."""
    # Find by relative path in lib/bin
    hd_layout_reader = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
        "bin",
        "slidelayoutreader",
    )
    call = [hd_layout_reader, "fetch", input_file, output_path]

    try:
        tk_subproc.check_output(call, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as err:
        if err.returncode == 3:
            hd_layout_url = err.output.split(b"Attempted URL:", 1)[1].split()[0]
            no_internet_msg = (
                "We failed to download slide layout data for your slide from the Internet, "
                "possibly due to network connection issues. If you are on an internal network "
                "that has restricted access to the Internet, you can download the slide layout "
                f"data for your slide from {hd_layout_url} and provide that file to the --slidefile option to "
                "spaceranger count. If you cannot retrieve this file, you can use the "
                "--unknown-slide option."
            )
            martian.exit(
                f"Could not retrieve slide layout data: \n{err.output.decode()}\n{no_internet_msg}"
            )
        else:
            martian.exit(f"Could not retrieve slide layout data: \n{err.output.decode()}")


def call_gprreader(mode, input_file, area, output_path):
    """Run gprreader with given parameters."""
    # Find gprreader by relative path in lib/bin
    gprreader = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
        "bin",
        "gprreader",
    )
    if mode == "info_gpr":
        call = [gprreader, "info", input_file, output_path, "--gpr"]
    else:
        call = [gprreader, mode, input_file, output_path]

        if area is not None:  # allow modes without an area, such as galparse
            call.extend(["--area", area])

    try:
        tk_subproc.check_output(call, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as err:
        if mode == "fetch":
            if err.returncode == 3:
                gpr_url = err.output.split(b"Attempted URL:", 1)[1].split()[0]
                no_internet_msg = (
                    "We failed to download slide layout data for your slide from the Internet, "
                    "possibly due to network connection issues. If you are on an internal network "
                    "that has restricted access to the Internet, you can download the slide layout "
                    f"data for your slide from {gpr_url} and provide that file to the --slidefile option to "
                    "spaceranger count. If you cannot retrieve this file, you can use the "
                    "--unknown-slide option."
                )
                martian.exit(
                    f"Could not retrieve slide layout data: \n{err.output.decode()}\n{no_internet_msg}"
                )
            else:
                martian.exit(f"Could not retrieve slide layout data: \n{err.output.decode()}")
        else:
            martian.exit(f"Could not read or decode slide layout file: \n{err.output.decode()}")


def call_hd_layout_reader_info(input_file) -> dict:
    """Given a .vlf file, return the file metadata."""
    return read_from_json(input_file)


def call_gprreader_info_gpr(input_file) -> dict:
    """Given a .gpr file, return the file metadata by calling gprreader."""
    # Please dont create tempfile in a context manager, as that leads to python
    # trying delete the file, which can cause issues when operating off an NFS
    # and prevents martian from accounting for the disk usage.
    out_path = tempfile.mkdtemp()
    call_gprreader("info_gpr", input_file, None, out_path)

    _, basename_ext = os.path.split(input_file)
    basename, _ = os.path.splitext(basename_ext)
    gpr_json = os.path.join(out_path, basename + ".json")
    gpr_data = read_from_json(gpr_json)

    return gpr_data


def load_gal_for_whitelist(wl_name: str):
    """Given a whitelist name, read and return a GAL file."""
    gal_path = get_galfile_path(wl_name)
    # Please dont create tempfile in a context manager, as that leads to python
    # trying delete the file, which can cause issues when operating off an NFS
    out_path = tempfile.mkdtemp()
    call_gprreader("galparse", gal_path, None, out_path)
    gal_json = read_from_json(os.path.join(out_path, wl_name + ".json"))

    if "areas" not in gal_json.keys():
        martian.exit(f"read malformed slide layout (GAL) file from {wl_name}")

    return gal_json


def get_capture_areas_for_whitelist(wl_name: str):
    """Given the name of a whitelist (e.g. visium-v1), read the GAL file and return a list of.

    valid capture areas
    """
    gal_json = load_gal_for_whitelist(wl_name)

    return gal_json["areas"].keys()


def get_whitelist_for_slide_id(slide_id: str):
    """Give a slide ID, e.g. V10J14, return a barcode whitelist defaulting to visium-v1 to accomodate research slides."""
    if slide_id and len(slide_id) > 1 and slide_id[0] == "V":
        return f"visium-v{slide_id[1]}"
    else:
        return "visium-v1"


def get_capture_areas_from_hd_layout_info_json(hd_layout_info_json) -> list:
    """Given a decoded HD Layout file as JSON, return the capture area names."""
    return list(hd_layout_info_json["capture_areas"].keys())


def get_capture_areas_from_gpr_info_json(gpr_info_json) -> list:
    """Given a decoded GPR file as JSON, return the capture area names."""
    return list(gpr_info_json["blockMaps"]["oligos"].keys())


SLIDE_EXTENSION = ".slide"


def read_hd_slide_design(slide_name: str):
    """Load HD slide design protobuf file."""
    slide_path = os.path.join(BARCODE_WHITELIST_PATH, slide_name + SLIDE_EXTENSION)
    with open(slide_path, "rb") as f:
        design = VisiumHdSlideDesign()
        design.ParseFromString(f.read())
    return design
