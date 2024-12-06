#!/usr/bin/env python
#
# Copyright (c) 2024 10X Genomics, Inc. All rights reserved.
#
"""Pick base level probe BC matrix if in the base bin level."""

import martian

import cellranger.cr_io as cr_io

BASE_BIN_SCALE = 1

__MRO__ = """
stage PICK_PROBE_BC_MATRIX(
    in  h5  raw_probe_bc_matrix_base_bin,
    in  int bin_scale,
    out h5  raw_probe_bc_matrix,
    src py  "stages/spatial/pick_probe_bc_matrix",
)
"""


def main(args, outs):
    if not args.raw_probe_bc_matrix_base_bin or args.bin_scale != BASE_BIN_SCALE:
        outs.raw_probe_bc_matrix = None
        return

    cr_io.hardlink_with_fallback(
        args.raw_probe_bc_matrix_base_bin, martian.make_path(outs.raw_probe_bc_matrix).decode()
    )
