# Copyright (c) 2025 10X Genomics, Inc. All rights reserved.
"""Helper stage to pick the final registered spot data json."""

import os

from cellranger.cr_io import hardlink_with_fallback

__MRO__ = """
stage PICK_REGISTERED_SPOT_DATA(
    in  json align_fiducials_registered_spots_data_json,
    in  json e2e_registered_spots_data_json,
    in  json raw_hd_layout_data_json,
    in  json e2e_hd_layout_data_json,
    out json registered_spots_data_json,
    out json hd_layout_data_json,
    src py   "stages/spatial/pick_registered_spot_data",
)
"""


def main(args, outs):
    if args.e2e_registered_spots_data_json is not None and os.path.exists(
        args.e2e_registered_spots_data_json
    ):
        hardlink_with_fallback(args.e2e_registered_spots_data_json, outs.registered_spots_data_json)
    else:
        assert args.align_fiducials_registered_spots_data_json is not None
        assert os.path.exists(args.align_fiducials_registered_spots_data_json)
        hardlink_with_fallback(
            args.align_fiducials_registered_spots_data_json, outs.registered_spots_data_json
        )

    found = False
    for layout_json in [
        args.e2e_hd_layout_data_json,
        args.raw_hd_layout_data_json,
    ]:
        if layout_json is not None and os.path.exists(layout_json):
            found = True
            hardlink_with_fallback(layout_json, outs.hd_layout_data_json)
            break

    if not found:
        outs.hd_layout_data_json = None
