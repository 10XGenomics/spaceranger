# Copyright (c) 2025 10x Genomics, Inc. All rights reserved.

"""Set up segmentation processing inputs."""


import cellranger.cr_io as cr_io

__MRO__ = """
stage SETUP_SEGMENTATION_PROCESSING(
    in  geojson user_provided_segmentations,
    in  geojson pipeline_generated_segmentations,
    in  csv     square_barcode_to_cell_map,
    in  int     barcode_assignment_distance_micron_in,
    out int     barcode_assignment_distance_micron,
    out geojson nucleus_segmentations,
    src py      "stages/spatial/setup_segmentation_processing",
)
"""


def main(args, outs):
    if args.user_provided_segmentations:
        cr_io.hardlink_with_fallback(args.user_provided_segmentations, outs.nucleus_segmentations)
    elif args.pipeline_generated_segmentations:
        cr_io.hardlink_with_fallback(
            args.pipeline_generated_segmentations, outs.nucleus_segmentations
        )
    else:
        outs.nucleus_segmentations = None

    # Expansion distance is whatever the user gives us.
    # If the give us no distance, we use the default unless we
    # are give a barcode -> cell ID map; in which case we set it to 0 (no expansio)
    outs.barcode_assignment_distance_micron = args.barcode_assignment_distance_micron_in
    if args.square_barcode_to_cell_map and args.barcode_assignment_distance_micron_in is None:
        outs.barcode_assignment_distance_micron = 0
