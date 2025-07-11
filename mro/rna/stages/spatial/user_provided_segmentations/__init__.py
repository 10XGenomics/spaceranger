#
# Copyright (c) 2025 10X Genomics, Inc. All rights reserved.
#

"""Check if user provide segmentations."""

__MRO__ = """
stage USER_PROVIDED_SEGMENTATIONS(
    in  csv     square_barcode_to_cell_map,
    in  tiff    instance_mask_tiff,
    in  npy     instance_mask_npy,
    in  geojson user_provided_segmentations,
    out bool    segmentation_from_user,
    src py      "stages/spatial/user_provided_segmentations",
)
"""


def main(args, outs):
    outs.segmentation_from_user = (
        bool(args.square_barcode_to_cell_map)
        or bool(args.instance_mask_tiff)
        or bool(args.instance_mask_npy)
        or bool(args.user_provided_segmentations)
    )
