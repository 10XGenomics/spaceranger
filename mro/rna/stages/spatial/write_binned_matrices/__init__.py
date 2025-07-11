#!/usr/bin/env python
#
# Copyright (c) 2021 10X Genomics, Inc. All rights reserved.
#

"""Write filtered and mtx versions starting from the raw h5 matrix."""

import json

import martian

import cellranger.h5_constants as h5_constants
import cellranger.matrix as cr_matrix
import cellranger.rna.matrix as rna_matrix

__MRO__ = """
stage WRITE_BINNED_MATRICES(
    in  string            sample_id,
    in  h5                raw_matrix_h5,
    in  map<ChemistryDef> chemistry_defs,
    in  json              filtered_bin_barcodes,
    out h5                filtered_matrices_h5,
    out path              filtered_matrices_mex,
    out path              raw_matrices_mex,
    src py                "stages/spatial/write_binned_matrices",
) split (
)
"""


def split(args):
    mem_gb = cr_matrix.CountMatrix.get_mem_gb_from_matrix_h5(
        args.raw_matrix_h5, scale=h5_constants.VIS_HD_MATRIX_MEM_GB_MULTIPLIER
    )
    mem_gb = (
        2 * int(mem_gb + 1) + 6
    )  # twice the memory of raw matrix as we store the raw and filtered matrix

    return {
        "chunks": [],
        "join": {
            "__mem_gb": mem_gb,
        },
    }


def join(args, outs, _chunk_defs, _chunk_outs):
    with open(args.filtered_bin_barcodes) as f:
        filtered_bcs = json.load(f)

    raw_matrix = cr_matrix.CountMatrix.load_h5_file(args.raw_matrix_h5)
    filtered_matrix = raw_matrix.select_barcodes_by_seq([bc.encode() for bc in filtered_bcs])

    # subset the filtered matrix to only targeted genes
    is_targeted = raw_matrix.feature_ref.has_target_features()
    if is_targeted:
        target_features = raw_matrix.feature_ref.get_target_feature_indices()
        filtered_matrix = filtered_matrix.remove_genes_not_on_list(target_features)

    del filtered_bcs

    rna_matrix.save_mex(raw_matrix, outs.raw_matrices_mex, martian.get_pipelines_version())
    del raw_matrix

    assert len(args.chemistry_defs) == 1
    chemistry_def = args.chemistry_defs.popitem()[1]

    matrix_attrs = cr_matrix.make_matrix_attrs_count(
        args.sample_id, [1], chemistry_def["description"]
    )
    filtered_matrix.save_h5_file(
        outs.filtered_matrices_h5,
        extra_attrs=matrix_attrs,
        sw_version=martian.get_pipelines_version(),
    )

    rna_matrix.save_mex(
        filtered_matrix, outs.filtered_matrices_mex, martian.get_pipelines_version()
    )
