#!/usr/bin/env python3
#
# Copyright (c) 2023 10X Genomics, Inc. All rights reserved
#
"""Make Antibody QC plots."""

from __future__ import annotations

import cellranger.feature.utils as feature_utils
import cellranger.matrix as cr_matrix
import cellranger.websummary.isotypes as isotypes
from cellranger.rna.library import ANTIBODY_LIBRARY_TYPE, GENE_EXPRESSION_LIBRARY_TYPE

__MRO__ = """
stage ANTIBODY_QC_PLOTS(
    in  h5   filtered_matrix,
    in  h5   raw_matrix,
    out json raw_normalized_heatmap,
    out json isotype_scatter,
    out json gex_fbc_correlation_heatmap,
    out json ab_qc_summary,
    src py   "stages/spatial/antibody_qc_plots",
) split (
) using (
    volatile = strict,
)
"""


def split(args):
    if not args.raw_matrix or not args.filtered_matrix:
        chunk_def = {}
    else:
        mem_gb_filtered = cr_matrix.CountMatrix.get_mem_gb_from_matrix_h5(args.filtered_matrix)
        mem_gb_raw = cr_matrix.CountMatrix.get_mem_gb_from_matrix_h5(args.raw_matrix)
        total_mem = int(mem_gb_filtered + mem_gb_raw + 1)
        chunk_def = {"__mem_gb": total_mem, "__vmem_gb": total_mem + 3}

    return {"chunks": [], "join": chunk_def}


def join(args, outs, _chunk_def, _chunk_outs):
    """This stage.

    Inputs:
        filtered_matrix: path to the filtered count matrix h5 file
        raw_matrix: path to the raw count matrix h5 file
        is_antibody: is this an antibody sample
        is_spatial: is this a spatial sample

    Outputs:
        raw_normalized_heatmap: Plot that represents correlations between antibody features on a raw scale and isotype normalized scale
        isotype_scatter: Plot that represents per barcode total counts between isotype and non-isotype features
        gex_fbc_correlation_heatmap: Plot that represents the expression correlation between RNA features and their isotype normalized protein products
    """
    if not args.raw_matrix or not args.filtered_matrix:
        outs.raw_normalized_heatmap = None
        outs.isotype_scatter = None
        outs.gex_fbc_correlation_heatmap = None
        outs.ab_qc_summary = None
        return
    # read in the data
    filtered_barcodes = cr_matrix.CountMatrix.load_bcs_from_h5(args.filtered_matrix)
    filtered_matrix = cr_matrix.CountMatrix.load_h5_file(args.filtered_matrix)
    raw_matrix = cr_matrix.CountMatrix.load_h5_file(args.raw_matrix)

    ## Filter the raw matrix to only tissue associated barcodes
    raw_matrix = raw_matrix.select_barcodes_by_seq(list(filtered_barcodes))

    # if we have Ab library type, proceed to make make the QC plots
    library_types = {fd.feature_type for fd in raw_matrix.feature_ref.feature_defs}
    if ANTIBODY_LIBRARY_TYPE in library_types:
        # Get AB features
        antibody_features = {
            x.name
            for x in raw_matrix.feature_ref.feature_defs
            if x.feature_type == ANTIBODY_LIBRARY_TYPE
        }

        # If we have GEX library type as well, make GEX-AB QC plots
        gex_ab_feature_overlap = []
        if GENE_EXPRESSION_LIBRARY_TYPE in library_types:
            gex_features = {
                x.name
                for x in raw_matrix.feature_ref.feature_defs
                if x.feature_type == GENE_EXPRESSION_LIBRARY_TYPE
            }
            gex_ab_feature_overlap = list(gex_features & antibody_features)

        if gex_ab_feature_overlap:
            gex_fbc_correlation_heatmap = isotypes.make_gex_fbc_correlation_heatmap(filtered_matrix)
            feature_utils.write_json_from_dict(
                gex_fbc_correlation_heatmap, outs.gex_fbc_correlation_heatmap
            )
        else:
            outs.gex_fbc_correlation_heatmap = None

        # Make sure there are isotype controls to make the plots that should have them
        control_raw_abs_ids = {
            f.id
            for f in raw_matrix.feature_ref.feature_defs
            if f.feature_type == ANTIBODY_LIBRARY_TYPE
            and f.tags.get("isotype_control", "FALSE") == "TRUE"
        }
        if control_raw_abs_ids:
            # make the plots if we have valid gex and ab features
            raw_normalized_heatmap = isotypes.make_ab_ab_correlation_heatmap(
                raw_matrix=raw_matrix, filtered_matrix=filtered_matrix
            )
            isotype_scatter, r_squared = isotypes.make_fbc_isotype_correlation_scatter(
                raw_matrix, filtered_barcodes
            )
            feature_utils.write_json_from_dict(raw_normalized_heatmap, outs.raw_normalized_heatmap)
            feature_utils.write_json_from_dict(isotype_scatter, outs.isotype_scatter)
            # write the scatter corr metric
            ab_qc_summary = {"ANTIBODY_isotype_nonisotype_r_squared": r_squared}
            feature_utils.write_json_from_dict(ab_qc_summary, outs.ab_qc_summary)
        else:
            outs.raw_normalized_heatmap = None
            outs.isotype_scatter = None
            outs.ab_qc_summary = None
            outs.gex_fbc_correlation_heatmap = None
    else:
        outs.raw_normalized_heatmap = None
        outs.isotype_scatter = None
        outs.gex_fbc_correlation_heatmap = None
        outs.ab_qc_summary = None
        return
