# Copyright (c) 2025 10x Genomics, Inc. All rights reserved.
"""Generate inputs needed for the End-to-End Image Alignment Accuracy card."""

import numpy as np

from cellranger.spatial.hd_feature_slice import HdFeatureSliceIo

__MRO__ = """
stage BUILD_HD_END_TO_END_ALIGNMENT_INPUTS(
    in  h5  hd_feature_slice_h5,
    out npy primary_bin_mask,
    out npy primary_bin_total_umis,
    out npy spot_colrow_to_tissue_image_colrow_transform,
    src py  "stages/spatial/build_hd_end_to_end_alignment_inputs",
)
"""

BIN_SCALE_8UM = 4


def main(args, outs):
    with HdFeatureSliceIo(args.hd_feature_slice_h5) as feat_slice:
        np.save(outs.primary_bin_mask, feat_slice.read_filtered_mask(BIN_SCALE_8UM))
        np.save(outs.primary_bin_total_umis, feat_slice.total_umis(BIN_SCALE_8UM))
        np.save(
            outs.spot_colrow_to_tissue_image_colrow_transform,
            feat_slice.metadata.transform_matrices.get_spot_colrow_to_tissue_image_colrow_transform(),
        )
