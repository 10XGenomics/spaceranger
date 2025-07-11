# Copyright (c) 2025 10x Genomics, Inc. All rights reserved.
"""Generate inputs to crconvert with segmentations."""

import h5py as h5
import martian
from scipy.sparse import csr_matrix

from cellranger.spatial.hd_feature_slice import (
    CELL_SEGMENTATION_MASK_NAME,
    SEGMENTATION_GROUP_NAME,
    HdFeatureSliceIo,
)

__MRO__ = """
stage GENERATE_LOUPE_SEGMENTATION_PRIMITIVES(
    in  h5 hd_feature_slice,
    out h5 spatial_cell_segment_mask,
    src py "stages/spatial/generate_loupe_segmentation_primitives",
) using (
    mem_gb   = 4,
    volatile = strict,
)
"""


def main(args, outs):

    if not args.hd_feature_slice:
        martian.clear(outs)
        return

    with HdFeatureSliceIo(args.hd_feature_slice) as feature_slice:
        cell_segmentation_mask_dataset = f"{SEGMENTATION_GROUP_NAME}/{CELL_SEGMENTATION_MASK_NAME}"
        if (
            cell_segmentation_mask_dataset not in feature_slice.h5_file
            or not feature_slice.metadata.transform_matrices
        ):
            martian.clear(outs)
            return

        cell_segmentation_mask = feature_slice.load_counts_from_group_name(
            cell_segmentation_mask_dataset, 1
        )
        transform = (
            feature_slice.metadata.transform_matrices.get_spot_colrow_to_tissue_image_colrow_transform()
        )

    sparse_cell_segmentation_mask = csr_matrix(cell_segmentation_mask)

    with h5.File(outs.spatial_cell_segment_mask, "w") as f:
        group = f.create_group("matrix")
        group.create_dataset("data", data=sparse_cell_segmentation_mask.data, dtype="uint32")
        group.create_dataset("indices", data=sparse_cell_segmentation_mask.indices, dtype="uint32")
        group.create_dataset("indptr", data=sparse_cell_segmentation_mask.indptr, dtype="uint32")
        group.create_dataset("shape", data=sparse_cell_segmentation_mask.shape, dtype="uint32")

        f.create_dataset("transform", data=transform)
