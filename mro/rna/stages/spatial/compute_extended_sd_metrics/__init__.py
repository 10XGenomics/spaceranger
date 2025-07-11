#
# Copyright (c) 2022 10X Genomics, Inc. All rights reserved.
#

"""Compute extended metrics from sd runs."""

import json
import math

import h5py as h5
import martian

import cellranger.matrix as cr_matrix
from cellranger.h5_constants import (
    BC_SUMMARY_BARCODE_KEY,
    BC_SUMMARY_GEX_TOTAL_READS_KEY,
)
from cellranger.rna.library import GENE_EXPRESSION_LIBRARY_TYPE
from cellranger.spatial.image_util import SD_SPOT_TISSUE_AREA_UM2

__MRO__ = """
stage COMPUTE_EXTENDED_SD_METRICS(
    in  h5   feature_bc_matrix,
    in  h5   barcode_summary,
    out json extended_metrics,
    src py   "stages/spatial/compute_extended_sd_metrics",
) split (
)
"""


def split(args):
    if not args.barcode_summary or not args.feature_bc_matrix:
        join_def = {}
    else:
        mem_gb = int(cr_matrix.CountMatrix.get_mem_gb_from_matrix_h5(args.feature_bc_matrix)) + 1
        join_def = {"__mem_gb": mem_gb, "__vmem_gb": mem_gb * 2}
    return {"chunks": [], "join": join_def}


def join(args, outs, _chunk_def, _chunk_outs):
    if not args.feature_bc_matrix or not args.barcode_summary:
        martian.clear(outs)
        return

    cmatrix = cr_matrix.CountMatrix.load_h5_file(args.feature_bc_matrix).select_features_by_type(
        GENE_EXPRESSION_LIBRARY_TYPE
    )
    total_gex_umis = cmatrix.m.sum()
    total_area_under_tissue_sq_um = len(cmatrix.bcs) * SD_SPOT_TISSUE_AREA_UM2
    total_area_under_tissue_sq_mm = total_area_under_tissue_sq_um / 1_000_000.0

    filtered_bcs_set = set(x.decode() for x in cmatrix.bcs)

    with h5.File(args.barcode_summary) as barcode_summary_fl:
        barcodes = (x.decode("utf-8") for x in barcode_summary_fl[BC_SUMMARY_BARCODE_KEY])
        total_reads = (int(x) for x in barcode_summary_fl[BC_SUMMARY_GEX_TOTAL_READS_KEY])

        total_gex_reads = sum(x for (y, x) in zip(barcodes, total_reads) if y in filtered_bcs_set)

    umis_per_mm_sq_tissue = total_gex_umis / total_area_under_tissue_sq_mm
    umis_per_um_sq_tissue = total_gex_umis / total_area_under_tissue_sq_um
    read_per_mm_sq_tissue = total_gex_reads / total_area_under_tissue_sq_mm
    read_per_um_sq_tissue = total_gex_reads / total_area_under_tissue_sq_um
    metrics_dict = {
        "umis_per_mm_sq_tissue": (
            umis_per_mm_sq_tissue if not math.isnan(umis_per_mm_sq_tissue) else 0.0
        ),
        "umis_per_um_sq_tissue": (
            umis_per_um_sq_tissue if not math.isnan(umis_per_um_sq_tissue) else 0.0
        ),
        "read_per_mm_sq_tissue": (
            read_per_mm_sq_tissue if not math.isnan(read_per_mm_sq_tissue) else 0.0
        ),
        "read_per_um_sq_tissue": (
            read_per_um_sq_tissue if not math.isnan(read_per_um_sq_tissue) else 0.0
        ),
    }

    with open(outs.extended_metrics, "w") as f:
        json.dump(metrics_dict, f, indent=4)
