#
# Copyright (c) 2025 10X Genomics, Inc. All rights reserved.
#
"""Set up writing spatial directory for the segmented outs."""

import json

__MRO__ = """
stage SETUP_WRITING_SPATIAL_DIRECTORY(
    in  json scalefactors_in,
    in  bool disable_downstream_analysis_in,
    out bool disable_downstream_analysis,
    out json scalefactors,
    src py   "stages/spatial/setup_writing_spatial_directory",
)
"""

SCALE_FACTOR_KEYS_TO_REMOVE = ["spot_diameter_fullres", "bin_size_um"]


def main(args, outs):
    outs.disable_downstream_analysis = args.disable_downstream_analysis_in is None or bool(
        args.disable_downstream_analysis_in
    )  # disable if disable_downstream_analysis_in is true, or null because upstream stages were disabled

    if not args.scalefactors_in:
        outs.scalefactors = None
        return

    with open(args.scalefactors_in) as f:
        scale_factors_dict = json.load(f)
    for key in SCALE_FACTOR_KEYS_TO_REMOVE:
        scale_factors_dict.pop(key, None)
    with open(outs.scalefactors, "w") as f:
        json.dump(scale_factors_dict, f, indent=4)
