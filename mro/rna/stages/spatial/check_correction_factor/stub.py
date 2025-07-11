#
# Copyright (c) 2023 10X Genomics, Inc. All rights reserved.
#
"""Placeholder."""

__MRO__ = """
stage CHECK_CORRECTION_FACTOR(
    in  V1PatternFixArgs v1_pattern_fix,
    in  h5               filtered_fbm,
    src py               "stages/spatial/check_correction_factor",
) using (
    volatile = strict,
)
"""


# pylint: disable=unused-argument
def main(args, outs):
    pass
