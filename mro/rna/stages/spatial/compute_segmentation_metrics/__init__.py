#!/usr/bin/env python
#
# Copyright (c) 2025 10x Genomics, Inc. All rights reserved.
#
"""Compute segmentation metrics for Visium HD data."""

import json

import martian
import numpy as np

import cellranger.fast_utils as fast_utils  # pylint: disable=no-name-in-module,unused-import
import cellranger.matrix as cr_matrix
import cellranger.molecule_counter as cr_mc
import cellranger.spatial.hd_feature_slice as hd_fs
from cellranger.rna.library import GENE_EXPRESSION_LIBRARY_TYPE
from cellranger.spatial.hd_feature_slice import (
    CELL_SEGMENTATION_MASK_NAME,
    NUCLEUS_SEGMENTATION_MASK_NAME,
    SEGMENTATION_GROUP_NAME,
)
from cellranger.spatial.segmentation_constants import (
    FILTERED_CELLS,
    FRACTION_COUNTS_PER_CELL,
    FRACTION_NUCLEI_EXPANDED,
    FRACTION_READS_IN_CELLS,
    MEAN_COUNTS_PER_CELL,
    MEAN_READS_PER_CELL,
    MEDIAN_CELL_AREA,
    MEDIAN_COUNTS_PER_CELL,
    MEDIAN_GENES_PER_CELL,
    MEDIAN_NUCLEUS_AREA,
)

__MRO__ = """
stage COMPUTE_SEGMENTATION_METRICS(
    in  h5   filtered_feature_cell_matrix,
    in  h5   hd_feature_slice,
    in  h5   molecule_info,
    in  int  max_nucleus_diameter_px,
    out json segmentation_metrics,
    src py   "stages/spatial/compute_segmentation_metrics",
) split (
) using (
    volatile = strict,
)
"""


def split(args):
    if not all((args.filtered_feature_cell_matrix, args.hd_feature_slice, args.molecule_info)):
        return {"chunks": [], "join": {}}

    filtered_mem = cr_matrix.CountMatrix.get_mem_gb_from_matrix_h5(
        args.filtered_feature_cell_matrix
    )
    join_mem_gb = int(np.ceil(filtered_mem))

    return {
        "chunks": [],
        "join": {
            "__mem_gb": join_mem_gb,
        },
    }


def join(args, outs, _chunk_defs, _chunk_outs):  # pylint: disable=too-many-locals
    if not all((args.filtered_feature_cell_matrix, args.hd_feature_slice, args.molecule_info)):
        martian.clear(outs)
        return
    # Metrics
    filtered_cells = cr_matrix.CountMatrix.count_cells_from_h5(args.filtered_feature_cell_matrix)

    filtered_feature_cell_matrix = cr_matrix.CountMatrix.load_h5_file(
        args.filtered_feature_cell_matrix
    )
    filtered_bc_ids_set = set(
        fast_utils.CellId(barcode=bc.decode()).id for bc in filtered_feature_cell_matrix.bcs
    )

    filtered_num_features_per_bc = filtered_feature_cell_matrix.get_numfeatures_per_bc()
    median_genes_per_cell = (
        np.median(filtered_num_features_per_bc) if filtered_num_features_per_bc.size else 0.0
    )

    filtered_num_counts_per_bc = filtered_feature_cell_matrix.get_subselected_counts(
        library_type="Gene Expression"
    )
    median_counts_per_cell = (
        np.median(filtered_num_counts_per_bc) if filtered_num_counts_per_bc.size else 0.0
    )
    mean_counts_per_cell = (
        np.mean(filtered_num_counts_per_bc) if filtered_num_counts_per_bc.size else 0.0
    )

    with hd_fs.HdFeatureSliceIo(args.hd_feature_slice) as feature_slice:
        cell_mask = hd_fs.CooMatrix.from_hdf5(
            feature_slice.h5_file[SEGMENTATION_GROUP_NAME][CELL_SEGMENTATION_MASK_NAME]
        )
        squares_per_cell_2um = feature_slice.squares_per_cell_nuc(
            cell_or_nucleus=hd_fs.SegmentationKind.CELL
        )
        squares_per_filtered_cell = {x: squares_per_cell_2um[x] for x in filtered_bc_ids_set}
        barcode_area = int(feature_slice.metadata.spot_pitch**2)
        area_per_cell = np.array(
            [value * barcode_area for value in squares_per_filtered_cell.values()]
        )
        median_cell_area = np.median(area_per_cell) if area_per_cell.size else 0.0

        # Median nucleus area
        nucleus_segmentation_mask = hd_fs.CooMatrix.from_hdf5(
            feature_slice.h5_file[SEGMENTATION_GROUP_NAME][NUCLEUS_SEGMENTATION_MASK_NAME]
        )
        squares_per_nucleus_2um = feature_slice.squares_per_cell_nuc(
            cell_or_nucleus=hd_fs.SegmentationKind.NUCLEUS
        )
        squares_per_filtered_nucleus = {x: squares_per_nucleus_2um[x] for x in filtered_bc_ids_set}
        area_per_nucleus = np.array(
            [value * barcode_area for value in squares_per_filtered_nucleus.values()]
        )
        median_nucleus_area = np.median(area_per_nucleus) if area_per_nucleus.size else 0.0

        # Total umis
        total_umi = np.sum(feature_slice.total_umis())

        # Reads per cell
        reads_per_cell = feature_slice.bin_reads_per_cell_nuc(
            cell_or_nucleus=hd_fs.SegmentationKind.CELL
        )

        # Mean reads in cells (only using reads in cells)
        mean_reads_per_cell = round(np.mean(reads_per_cell), 1) if reads_per_cell.size else 0.0

        # Fraction of nuclei expanded
        feature_slice_rows = feature_slice.nrows()
        feature_slice_cols = feature_slice.ncols()
        dense_nucleus_segmentation_mask = nucleus_segmentation_mask.to_ndarray(
            ncols=feature_slice_cols, nrows=feature_slice_rows
        )
        dense_cell_mask = cell_mask.to_ndarray(ncols=feature_slice_cols, nrows=feature_slice_rows)

        num_nuclei = np.count_nonzero(np.unique(dense_nucleus_segmentation_mask))
        num_expanded_nuclei = np.count_nonzero(
            np.unique(dense_cell_mask - dense_nucleus_segmentation_mask)
        )
        fraction_nuclei_expanded = round(num_expanded_nuclei / num_nuclei, 4) if num_nuclei else 0.0

    # Total reads per library
    with cr_mc.MoleculeCounter.open(args.molecule_info, "r") as mc:
        raw_reads_per_lib = np.array(mc.get_raw_read_pairs_per_library())
        gex_index = mc.get_library_indices_by_type()[GENE_EXPRESSION_LIBRARY_TYPE]

    # Fraction of reads in cells
    fraction_reads_in_cells = float(
        round((sum(reads_per_cell) / raw_reads_per_lib[gex_index])[0], 1)
    )

    # Fraction of UMIs in expanded cells
    fraction_counts_per_cell = (
        round(
            sum(filtered_feature_cell_matrix.get_counts_per_bc()) / total_umi,
            4,
        )
        if total_umi
        else 0.0
    )

    cell_segmentation_metrics = {
        FILTERED_CELLS: filtered_cells,  # number of expanded cells with >=1 UMI
        FRACTION_COUNTS_PER_CELL: fraction_counts_per_cell,
        FRACTION_READS_IN_CELLS: fraction_reads_in_cells,
        MEAN_COUNTS_PER_CELL: mean_counts_per_cell,
        MEAN_READS_PER_CELL: mean_reads_per_cell,
        MEDIAN_GENES_PER_CELL: median_genes_per_cell,
        MEDIAN_COUNTS_PER_CELL: median_counts_per_cell,
        MEDIAN_CELL_AREA: median_cell_area,
        MEDIAN_NUCLEUS_AREA: median_nucleus_area,
        FRACTION_NUCLEI_EXPANDED: fraction_nuclei_expanded,
        "max_nucleus_diameter_px": args.max_nucleus_diameter_px,
    }

    with open(outs.segmentation_metrics, "w") as f:
        json.dump(cell_segmentation_metrics, f, indent=4)
