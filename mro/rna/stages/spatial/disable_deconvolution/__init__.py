# Copyright (c) 2024 10x Genomics, Inc. All rights reserved.
"""Decides if spot deconvolution should be disabled."""

__MRO__ = """
stage DISABLE_DECONVOLUTION(
    in  bool disable_gex,
    in  bool is_visium_hd,
    out bool disable_deconvolution,
    src py   "stages/spatial/disable_deconvolution",
)
"""


def main(args, outs):
    outs.disable_deconvolution = bool(args.disable_gex) or bool(args.is_visium_hd)
