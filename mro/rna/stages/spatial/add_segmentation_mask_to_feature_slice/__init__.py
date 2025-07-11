# Copyright (c) 2025 10x Genomics, Inc. All rights reserved.

"""Update feature slice H5 with outs.

- add segmentation masks to the feature slice matrix.
"""

import shutil

from cellranger.spatial.hd_feature_slice import HdFeatureSliceIo

__MRO__ = """
stage ADD_SEGMENTATION_MASKS_TO_FEATURE_SLICE(
    in  h5   hd_feature_slice_h5_in,
    in  npy  segmentation_mask,
    in  npy  cell_segmentation_mask,
    in  npy  minimum_distance_mask,
    in  npy  closest_object_mask,
    in  bool disable_downstream_analysis,
    in  bool is_pd,
    out h5   hd_feature_slice_h5_out      "HD Feature Slice"  "hd_feature_slice.h5",
    src py   "stages/spatial/add_segmentation_mask_to_feature_slice",
) using (
    mem_gb   = 12,
    volatile = strict,
)
"""


def main(args, outs):
    if args.hd_feature_slice_h5_in is None:
        outs.hd_feature_slice_h5_out = None
        return
    shutil.copyfile(
        args.hd_feature_slice_h5_in, outs.hd_feature_slice_h5_out, follow_symlinks=False
    )

    # We check every one of the inputs and write them out into the feature slice if they exist
    if args.disable_downstream_analysis or not any(
        (
            args.segmentation_mask,
            args.cell_segmentation_mask,
            args.minimum_distance_mask,
            args.closest_object_mask,
        )
    ):
        return

    with HdFeatureSliceIo(h5_path=outs.hd_feature_slice_h5_out, open_mode="a") as feature_slice_io:
        feature_slice_io.write_segmentation_masks(
            segmentation_mask_path=args.segmentation_mask,
            cell_segmentation_mask_path=args.cell_segmentation_mask,
            minimum_distance_mask_path=args.minimum_distance_mask if args.is_pd else None,
            closest_object_mask_path=args.closest_object_mask if args.is_pd else None,
        )
