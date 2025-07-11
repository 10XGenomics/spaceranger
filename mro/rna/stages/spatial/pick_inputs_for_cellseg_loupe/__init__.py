#
# Copyright (c) 2025 10X Genomics, Inc. All rights reserved.
#
"""Pick tissue positions of the base bin."""

import martian

import cellranger.cr_io as cr_io
from cellranger.spatial.hd_feature_slice import HdFeatureSliceIo, bin_name_from_bin_size_um

__MRO__ = """
stage PICK_INPUTS_FOR_CELLSEG_LOUPE(
    in  map<csv>  tissue_positions,
    in  map<json> scalefactors,
    in  h5        hd_feature_slice,
    out csv       base_tissue_positions,
    out json      base_scalefactors,
    src py        "stages/spatial/pick_inputs_for_cellseg_loupe",
) using (
    volatile = strict,
)
"""


def main(args, outs):
    if not args.hd_feature_slice or not args.tissue_positions or not args.scalefactors:
        martian.clear(outs)
        return

    with HdFeatureSliceIo(args.hd_feature_slice) as feature_slice:
        base_bin_size = feature_slice.metadata.spot_pitch

    base_bin_name = bin_name_from_bin_size_um(base_bin_size)
    base_tissue_positions = args.tissue_positions.get(base_bin_name)
    base_scalefactors = args.scalefactors.get(base_bin_name)
    if base_tissue_positions:
        cr_io.hardlink_with_fallback(base_tissue_positions, outs.base_tissue_positions)
    else:
        outs.base_tissue_positions = None

    if base_scalefactors:
        cr_io.hardlink_with_fallback(base_scalefactors, outs.base_scalefactors)
    else:
        outs.base_scalefactors = None
