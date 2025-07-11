#
# Copyright (c) 2025 10X Genomics, Inc. All rights reserved.
#
"""Make sure all detected nuclei have only one connected component."""

import martian
import numpy as np
from skimage.measure import label  # pylint: disable = no-name-in-module

__MRO__ = """
stage CLEAN_SPOT_MASK(
    in  npy raw_segmentation_spot_mask,
    out npy segmentation_spot_mask,
    src py  "stages/spatial/clean_spot_mask",
) using (
    mem_gb   = 4,
    volatile = strict,
)
"""


def main(args, outs):
    if not args.raw_segmentation_spot_mask:
        martian.clear(outs)
        return

    raw_segmentation_spot_mask = np.load(args.raw_segmentation_spot_mask).astype(np.int64)
    label_img = label(raw_segmentation_spot_mask, background=0, connectivity=2)

    # skimage.measure.label takes an instance mask and renames it into instances of connected
    # components. The issue ends up being that we do not know which label corresponds
    # to which cell_id in the original spot mask or even which cell IDs have multiple connected
    # components.
    # We thus compute the maximum and minimum label_id in each cell-ID. And use the
    # fact that a cell-ID has disconnected components if and only if the maximum label-ID
    # corresponding to a cell-ID is different from the minimum label-ID corresponding to
    # that cell-ID.
    # np.ufunc.at allows us to compute the minimum and maximum very fast.
    max_cell_id_plus_one = int(raw_segmentation_spot_mask.max() + 1)
    max_label_in_cell_ids = np.zeros(max_cell_id_plus_one)
    np.maximum.at(max_label_in_cell_ids, raw_segmentation_spot_mask, label_img)

    min_label_in_cell_ids = np.ones(max_cell_id_plus_one) * max_cell_id_plus_one
    map_max_cell_id_plus_one_to_zero = np.vectorize(lambda x: x if x != max_cell_id_plus_one else 0)
    np.minimum.at(min_label_in_cell_ids, raw_segmentation_spot_mask, label_img)
    min_label_in_cell_ids = map_max_cell_id_plus_one_to_zero(min_label_in_cell_ids)

    cell_ids_with_disconnected_components = np.nonzero(
        max_label_in_cell_ids - min_label_in_cell_ids
    )[0]

    for cell_id in (
        x for x in cell_ids_with_disconnected_components if x
    ):  # ignoring 0 as it is background
        print(f"disconnected label {cell_id}")
        unique_elements, counts = np.unique(
            label_img[raw_segmentation_spot_mask == cell_id], return_counts=True
        )  # find labels corresponding to a cell id and its counts
        max_count_index = np.argmax(counts)  # Find the index of the maximum count
        label_to_keep = unique_elements[max_count_index]
        raw_segmentation_spot_mask += cell_id * (
            (label_img == label_to_keep).astype(dtype=np.int64)
            - (raw_segmentation_spot_mask == cell_id).astype(
                dtype=np.int64
            )  # subtracting out things not in the largest component
        )

    np.save(outs.segmentation_spot_mask, raw_segmentation_spot_mask.astype(np.uint64))
