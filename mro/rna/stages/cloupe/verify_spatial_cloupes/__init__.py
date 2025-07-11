#!/usr/bin/env python
#
# Copyright (c) 2020 10X Genomics, Inc. All rights reserved.
#


import subprocess

import martian

import cellranger.constants as cr_constants
import tenkit.log_subprocess as tk_subproc

__MRO__ = """
stage VERIFY_SPATIAL_CLOUPES(
    in  map[] sample_defs,
    src py    "stages/cloupe/verify_spatial_cloupes",
)
"""


def main(args, outs):
    """VERIFY_SPATIAL_CLOUPES checks the cloupes in a spatial aggr.

    sample definition to check if each is a valid spatial cloupe file.

    Args:
        args: MRO args
        outs: MRO outs
    """
    loupe_files = {
        sample_def[cr_constants.AGG_CLOUPE_FIELD]
        for sample_def in args.sample_defs
        if cr_constants.AGG_CLOUPE_FIELD in sample_def
    }
    joined_files = ",".join(loupe_files)
    verify_call = [
        "crconverter",
        "verify",
        "--spatial-cloupe-paths",
        joined_files,
    ]
    try:
        tk_subproc.check_output(verify_call, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        martian.exit(e.output)
