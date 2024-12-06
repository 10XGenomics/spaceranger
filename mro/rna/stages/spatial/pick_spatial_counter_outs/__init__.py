#!/usr/bin/env python
#
# Copyright (c) 2023 10X Genomics, Inc. All rights reserved.
#
"""Pick the CS output files appropriately for SD and HD."""


__MRO__ = """
stage PICK_SPATIAL_COUNTER_OUTS(
    in  h5          filtered_feature_bc_matrix_h5_in,
    in  path        filtered_feature_bc_matrix_mex_in,
    in  h5          raw_feature_bc_matrix_h5_in,
    in  path        raw_feature_bc_matrix_mex_in,
    in  h5          raw_probe_bc_matrix_h5_in,
    in  bool        is_visium_hd,
    in  map<cloupe> cloupe_files,
    in  int         custom_bin_size,
    in  html        sd_web_summary,
    in  html        hd_web_summary,
    out h5          filtered_feature_bc_matrix_h5_out,
    out path        filtered_feature_bc_matrix_mex_out,
    out h5          raw_feature_bc_matrix_h5_out,
    out path        raw_feature_bc_matrix_mex_out,
    out h5          raw_probe_bc_matrix_h5_out,
    out cloupe      cloupe_008um,
    out cloupe      cloupe_custom,
    out html        web_summary,
    src py          "stages/spatial/pick_spatial_counter_outs",
) using (
    volatile = strict,
)
"""

import os

from cellranger.cr_io import hardlink_with_fallback

BIN_NAME_TEMPLATE = "square_{:03}um"
BIN_SIZE_8UM = 8


def main(args, outs):
    for file_prefix in [
        "filtered_feature_bc_matrix_h5",
        "filtered_feature_bc_matrix_mex",
        "raw_feature_bc_matrix_h5",
        "raw_feature_bc_matrix_mex",
        "raw_probe_bc_matrix_h5",
    ]:
        in_file = getattr(args, file_prefix + "_in")
        outs_key = file_prefix + "_out"
        if (not args.is_visium_hd) and (in_file is not None) and os.path.exists(in_file):
            hardlink_with_fallback(in_file, getattr(outs, outs_key))
        else:
            setattr(outs, outs_key, None)

    cloupe_selections: list[tuple[str, str]] = [
        (BIN_NAME_TEMPLATE.format(BIN_SIZE_8UM), "cloupe_008um"),
    ]
    if args.custom_bin_size is not None:
        cloupe_selections.append((BIN_NAME_TEMPLATE.format(args.custom_bin_size), "cloupe_custom"))

    for bin_name, outs_key in cloupe_selections:
        # cloupe file at8 µm
        if (
            args.is_visium_hd
            and args.cloupe_files is not None
            and args.cloupe_files.get(bin_name, None) is not None
            and os.path.exists(args.cloupe_files[bin_name])
        ):
            # No need to hardlink because the cloupe files are pushed to the output directory
            # at each bin level
            setattr(outs, outs_key, args.cloupe_files[bin_name])
        else:
            setattr(outs, outs_key, None)

    web_summary = args.hd_web_summary if args.is_visium_hd else args.sd_web_summary
    if web_summary is not None and os.path.exists(web_summary):
        hardlink_with_fallback(web_summary, outs.web_summary)
