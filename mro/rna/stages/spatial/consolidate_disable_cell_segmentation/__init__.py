#
# Copyright (c) 2025 10X Genomics, Inc. All rights reserved.
#
"""OR two boolean variables."""

__MRO__ = """
stage CONSOLIDATE_DISABLE_CELL_SEGMENTATION(
    in  bool disable_downstream_segmentation_processing_in,
    in  bool no_secondary_analysis_in,
    out bool no_secondary_analysis,
    src py   "stages/spatial/consolidate_disable_cell_segmentation",
)
"""


def main(args, outs):
    outs.no_secondary_analysis = (
        args.disable_downstream_segmentation_processing_in is None
        or args.disable_downstream_segmentation_processing_in
        or args.no_secondary_analysis_in
    )
