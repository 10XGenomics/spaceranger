# Copyright (c) 2024 10x Genomics, Inc. All rights reserved.
"""Decides if spot deconvolution should be disabled."""

__MRO__ = """
stage DISABLE_SD_ONLY_STAGES(
    in  bool disable_gex,
    in  bool is_visium_hd,
    out bool disable_deconvolution,
    out bool disable_sd_extra_metrics,
    src py   "stages/spatial/disable_sd_only_stages",
)
"""


def main(args, outs):
    outs.disable_deconvolution = bool(args.disable_gex) or bool(args.is_visium_hd)
    outs.disable_sd_extra_metrics = bool(args.disable_gex) or bool(args.is_visium_hd)
