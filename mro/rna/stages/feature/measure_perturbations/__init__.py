#!/usr/bin/env python
#
# Copyright (c) 2018 10X Genomics, Inc. All rights reserved
#

import csv

import numpy as np

from cellranger.analysis.diffexp import save_differential_expression_csv
from cellranger.feature.crispr.measure_perturbations import (
    get_perturbation_efficiency,
    read_and_validate_feature_ref,
    save_perturbation_efficiency_summary,
    save_top_perturbed_genes,
)
from cellranger.feature.utils import all_files_present
from cellranger.matrix import CountMatrix
from cellranger.rna.library import CRISPR_LIBRARY_TYPE, GENE_EXPRESSION_LIBRARY_TYPE

SUMMARY_FILE_NAME = "transcriptome_analysis"

__MRO__ = """
stage MEASURE_PERTURBATIONS(
    in  csv  protospacer_calls_per_cell,
    in  h5   filtered_feature_counts_matrix,
    in  csv  feature_reference,
    in  bool by_feature,
    out csv  perturbation_efficiencies,
    out path perturbation_effects_path,
    src py   "stages/feature/measure_perturbations",
) split (
) using (
    volatile = strict,
)
"""


def split(args):
    with open(args.protospacer_calls_per_cell) as f:
        num_feature_calls = len(set(x["feature_call"] for x in csv.DictReader(f)))

    feature_ref = CountMatrix.load_feature_ref_from_h5_file(args.filtered_feature_counts_matrix)
    num_gex_features = feature_ref.get_count_of_feature_type(GENE_EXPRESSION_LIBRARY_TYPE)
    num_crispr_features = feature_ref.get_count_of_feature_type(CRISPR_LIBRARY_TYPE)

    num_features, num_barcodes, nnz = CountMatrix.load_dims_from_h5(
        args.filtered_feature_counts_matrix
    )
    matrix_mem_gib = CountMatrix.get_mem_gb_from_matrix_dim(num_barcodes, nnz, scale=1)

    # The DEG are stored in a dense matrix of shape 3 * num_gex_features * num_feature_calls.
    # When args.by_feature is False, the shape is num_target_calls rather than num_feature_calls,
    # which is strictly less than num_feature_calls, so we're overestimating VMEM for this case.
    vmem_deg_gib = round(24 * num_gex_features * num_feature_calls / 1024**3, 1)
    vmem_gib = 9 + 1.4 * matrix_mem_gib + vmem_deg_gib

    # The RSS usage is 2 to 4 times less than VMEM, presumably due to its access pattern of
    # the dense DEG matrix.
    mem_gib = 9 + 2.6 * matrix_mem_gib if args.by_feature else 4 + 1.9 * matrix_mem_gib

    print(
        f"{num_gex_features=},{num_crispr_features=},{num_features=},{num_barcodes=},{nnz=},"
        f"{matrix_mem_gib=},{mem_gib=},{num_feature_calls=},{vmem_deg_gib=},{vmem_gib=}"
    )
    vmem_gib = max(vmem_gib, mem_gib)
    # FIXME: Temporarily set __mem_gb to vmem_gib to avoid OOM errors
    return {"chunks": [], "join": {"__mem_gb": vmem_gib, "__vmem_gb": vmem_gib, "__threads": 5}}


def join(args, outs, _chunk_defs, _chunk_outs):
    np.random.seed(0)

    list_file_paths = [
        args.protospacer_calls_per_cell,
        args.filtered_feature_counts_matrix,
        args.feature_reference,
    ]
    if not (all_files_present(list_file_paths)):
        outs.perturbation_efficiencies = None
        return

    feature_count_matrix = CountMatrix.load_h5_file(args.filtered_feature_counts_matrix)
    gex_count_matrix = feature_count_matrix.select_features_by_type(GENE_EXPRESSION_LIBRARY_TYPE)
    target_info = read_and_validate_feature_ref(args.feature_reference)
    if target_info is None:
        outs.perturbation_efficiencies = None
        return

    perturbation_result = get_perturbation_efficiency(
        target_info,
        args.protospacer_calls_per_cell,
        feature_count_matrix,
        by_feature=args.by_feature,
    )

    if perturbation_result is None:
        outs.perturbation_efficiencies = None
        return
    (
        results_per_perturbation,
        results_all_perturbations,
        fold_change_per_perturbation,
    ) = perturbation_result

    save_perturbation_efficiency_summary(
        outs.perturbation_efficiencies,
        fold_change_per_perturbation,
        args.by_feature,
    )
    save_top_perturbed_genes(outs.perturbation_effects_path, results_per_perturbation)
    save_differential_expression_csv(
        None,
        results_all_perturbations,
        gex_count_matrix,
        outs.perturbation_effects_path,
        cluster_names=list(results_per_perturbation.keys()),
        file_name=SUMMARY_FILE_NAME,
    )
    # this call assumes that results_all_perturbations is an OrderedDict, hence can get ordered names from keys()
