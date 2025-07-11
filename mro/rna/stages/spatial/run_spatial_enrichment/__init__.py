#
# Copyright (c) 2020 10X Genomics, Inc. All rights reserved.
#
"""Calculates Moran's I and p value for the normalized filtered gene count matrix.

The results can be used to identify spatially enriched genes and gene expression spatial patterns.
"""
from __future__ import annotations

import pickle as cPickle

import h5py
import martian
import numpy as np
import pandas as pd
import scipy
from six import ensure_str
from statsmodels.stats.multitest import multipletests

import cellranger.analysis.morans_i as cr_morans_i
import cellranger.analysis.stats as analysis_stats
import cellranger.matrix as cr_matrix
import cellranger.spatial.spatial_pandas_utils as spatial_pandas_utils
import tenkit.safe_json as tk_safe_json
from cellranger.analysis.morans_i import KNN_NEIGHBORS, MORANS_I_PERMUTATIONS
from cellranger.spatial.data_utils import IMAGEX_LOWRES, IMAGEY_LOWRES
from tenkit.stats import robust_divide

__MRO__ = """
stage RUN_SPATIAL_ENRICHMENT(
    in  h5     filtered_gene_bc_matrices_h5,
    in  csv    tissue_positions,
    in  json   image_scale_factors,
    out pickle chunked_spatial_enrichment_files,
    out csv    spatial_enrichment_csv,
    out json   spatial_enrichment_json,
    src py     "stages/spatial_pd/run_spatial_enrichment",
) split (
    in  pickle weight_matrix_file,
    in  h5     submatrix_path,
    in  pickle feature_def_path,
    in  int    row_start,
    in  int    total_rows,
    in  bool   has_secondary_name,
) using (
    mem_gb   = 2,
    volatile = strict,
)
"""
# ~32k genes => ~16 chunks
FEATURES_PER_CHUNK = 2000
MIN_BCS_FOR_ANALYSIS = KNN_NEIGHBORS + 1
MAX_BCS_FOR_ANALYSIS = 50000
MEM_GB_PER_BC = 0.0008

# Limits for for summary stats in json
## Limit features to those observed in a minimum number of barcodes
BCS_PER_FEATURE = 10
## Limit features to those with minimum number of total UMI associated with any feature
FEATURE_UMI = 20
## Arbitrary cutoffs used in metrics for summary of spatial auto-correlation
MODERATE_I = 0.2
HIGH_I = 0.4

SPATIAL_SUBMATRIX_NAME = "spatial_submatrix"

SECONDARY_NAME_TAG = "secondary_name"
SECONDARY_NAME_COL = "Feature Secondary Name"


def split(args):
    chunks = []
    bcs = cr_matrix.CountMatrix.load_bcs_from_h5(args.filtered_gene_bc_matrices_h5)
    if (
        len(bcs) < MIN_BCS_FOR_ANALYSIS
        or len(bcs) > MAX_BCS_FOR_ANALYSIS
        or args.tissue_positions is None
    ):
        return {"chunks": []}
    coords = spatial_pandas_utils.get_lowres_coordinates(
        args.tissue_positions, args.image_scale_factors
    )
    spots = coords.loc[bcs]
    weight_matrix = cr_morans_i.get_neighbors_weight_matrix(
        spots[[IMAGEY_LOWRES, IMAGEX_LOWRES]].values
    )
    weight_matrix_file = martian.make_path("weight_matrix.pickle")
    with open(weight_matrix_file, "wb") as f:
        cPickle.dump(weight_matrix, f, cPickle.HIGHEST_PROTOCOL)
    filtered_matrix = cr_matrix.CountMatrix.load_h5_file(args.filtered_gene_bc_matrices_h5)
    # if the matrix has no counts return
    if scipy.sparse.csr_matrix.sum(filtered_matrix.m) == 0:
        return {"chunks": []}
    features = filtered_matrix.feature_ref.feature_defs
    has_secondary_name = any(SECONDARY_NAME_TAG in feature.tags for feature in features)

    # Normalize the matrix
    filtered_matrix_norm = cr_morans_i.normalize_morani_matrix(filtered_matrix)

    for row_start in range(0, filtered_matrix_norm.shape[0], FEATURES_PER_CHUNK):
        row_end = min(row_start + FEATURES_PER_CHUNK, filtered_matrix_norm.shape[0])
        # Write the submatrix to an h5 file
        submatrix_path = martian.make_path(f"{row_start}_spatial_submatrix.h5")
        # dense matrix used for Moran's I calc.
        chunk_matrix = filtered_matrix_norm[row_start:row_end, :].toarray()
        with h5py.File(submatrix_path, "w") as f:
            f.create_dataset(SPATIAL_SUBMATRIX_NAME, data=chunk_matrix)
        # Write the features to a pickle file
        feature_def_path = martian.make_path(f"{row_start}_spatial_feature_def.pickle")
        with open(feature_def_path, "wb") as f:
            cPickle.dump(features[row_start:row_end], f, cPickle.HIGHEST_PROTOCOL)
        chunks.append(
            {
                "weight_matrix_file": weight_matrix_file,
                "submatrix_path": submatrix_path,
                "feature_def_path": feature_def_path,
                "row_start": row_start,
                "total_rows": row_end - row_start,
                "has_secondary_name": has_secondary_name,
            }
        )

    mem_gb = int(np.ceil(len(bcs) * MEM_GB_PER_BC))
    return {
        "chunks": chunks,
        "join": {"__mem_gb": mem_gb, "__threads": 1},
    }


def main(args, outs):
    with h5py.File(args.submatrix_path, "r") as f:
        submatrix = f[SPATIAL_SUBMATRIX_NAME][:]
    with open(args.feature_def_path, "rb") as f:
        feature_def = cPickle.load(f)
    with open(args.weight_matrix_file, "rb") as f:
        weight_matrix = cPickle.load(f)

    res = {
        "Index": [],
        "Feature ID": [],
        "Feature Name": [],
        "Feature Type": [],
        "I": [],
        "P value": [],
    }
    if args.has_secondary_name:
        res[SECONDARY_NAME_COL] = []

    if MORANS_I_PERMUTATIONS > 0:
        res["P value Perm"] = []

    for i in range(args.total_rows):
        moransi_res = cr_morans_i.calculate_morans_i(submatrix[i, :], weight_matrix)
        res["Index"].append(feature_def[i].index)
        res["Feature ID"].append(ensure_str(feature_def[i].id))
        res["Feature Name"].append(feature_def[i].name)
        if args.has_secondary_name:
            res[SECONDARY_NAME_COL].append(feature_def[i].tags.get(SECONDARY_NAME_TAG, ""))
        res["Feature Type"].append(feature_def[i].feature_type)
        res["I"].append(moransi_res[0])
        # Use the p-value generated from variance under randomization.
        res["P value"].append(moransi_res[1])
        if MORANS_I_PERMUTATIONS > 0:
            res["P value Perm"].append(moransi_res[2])
    outs.chunked_spatial_enrichment_files = martian.make_path(
        "chunked_spatial_enrichment_files.pickle"
    )
    with open(outs.chunked_spatial_enrichment_files, "wb") as f:
        cPickle.dump(res, f, cPickle.HIGHEST_PROTOCOL)


# pylint: disable=too-many-locals
def join(args, outs, chunk_defs, chunk_outs):
    if len(chunk_outs) == 0:
        martian.clear(outs)
        return
    chunk_res_names = [chunk.chunked_spatial_enrichment_files for chunk in chunk_outs]
    frames = []
    for chunk_res_name in chunk_res_names:
        with open(chunk_res_name, "rb") as f:
            chunk_enrichment = cPickle.load(f)
            frames.append(pd.DataFrame(chunk_enrichment, columns=sorted(chunk_enrichment.keys())))
    result = pd.concat(frames)
    cleaned_res = result[~np.isnan(result.I)]
    cleaned_res.insert(
        cleaned_res.columns.get_loc("P value") + 1,
        "Adjusted p value",
        multipletests(cleaned_res["P value"], method="fdr_bh")[1],
        True,
    )
    if MORANS_I_PERMUTATIONS > 0:
        cleaned_res.insert(
            cleaned_res.columns.get_loc("P value Perm") + 1,
            "Adjusted p value Perm",
            multipletests(cleaned_res["P value Perm"], method="fdr_bh")[1],
            True,
        )

    # use the unnormalized matrix to count number of UMI and Barcodes for each feature
    filtered_matrix = cr_matrix.CountMatrix.load_h5_file(args.filtered_gene_bc_matrices_h5)
    filtered_matrix_norm = analysis_stats.normalize_by_umi(filtered_matrix)
    feature_counts_per_feature = filtered_matrix.get_counts_per_feature()
    barcodes_detected_per_feature = filtered_matrix.get_numbcs_per_feature()

    # Median Normalized Average Counts for consistency with how diff expression is presented
    mna_counts = np.squeeze(np.asarray(scipy.sparse.csr_matrix.mean(filtered_matrix_norm, 1)))

    metadata = pd.DataFrame(
        {
            "Index": range(filtered_matrix.features_dim),
            "Feature Counts in Spots Under Tissue": feature_counts_per_feature,
            "Median Normalized Average Counts": mna_counts,
            "Barcodes Detected per Feature": barcodes_detected_per_feature,
        }
    )
    cleaned_res = cleaned_res.merge(metadata, on="Index", how="left")
    cleaned_res = cleaned_res.sort_values("I", ascending=False)
    cleaned_res = cleaned_res.drop(["Index"], axis=1)

    # We want the SECONDARY_NAME_COL as the last column if it exists
    if SECONDARY_NAME_COL in cleaned_res.columns:
        column = cleaned_res.pop(SECONDARY_NAME_COL)
        # Check if all the elements in SECONDARY_NAME_COL are empty. if not add the column to the end of the df
        if not all(elem == "" for elem in list(column)):
            cleaned_res.insert(len(cleaned_res.columns), SECONDARY_NAME_COL, column)
    cleaned_res.to_csv(outs.spatial_enrichment_csv, index=False)
    outs.chunked_spatial_enrichment_h5 = None
    # print out a few summary metrics on the top of the distribution
    n_features = cleaned_res.shape[0]
    cleaned_res = cleaned_res.loc[
        (cleaned_res["Barcodes Detected per Feature"] >= BCS_PER_FEATURE)
        & (cleaned_res["Feature Counts in Spots Under Tissue"] >= FEATURE_UMI)
    ]
    mi_p5 = (
        cleaned_res.iloc[int(0.05 * n_features)]["I"]
        if int(0.05 * n_features) < cleaned_res.shape[0]
        else np.nan
    )
    mi_p10 = (
        cleaned_res.iloc[int(0.10 * n_features)]["I"]
        if int(0.10 * n_features) < cleaned_res.shape[0]
        else np.nan
    )
    mi_top20_median = (
        np.nanmedian(cleaned_res.iloc[0:20]["I"]) if cleaned_res.shape[0] >= 20 else np.nan
    )
    mi_top50_median = (
        np.nanmedian(cleaned_res.iloc[0:50]["I"]) if cleaned_res.shape[0] >= 50 else np.nan
    )
    # arbitrary threshold of 0.2 -> moderate MI, 0.4 -> high MI
    num_genes_mod_morans_i = cleaned_res.loc[cleaned_res["I"] >= MODERATE_I].shape[0]
    num_genes_high_morans_i = cleaned_res.loc[cleaned_res["I"] >= HIGH_I].shape[0]
    summary_stats = {
        "p5_morans_i": mi_p5,
        "p10_morans_i": mi_p10,
        "top20_median_morans_i": mi_top20_median,
        "top50_median_morans_i": mi_top50_median,
        "frac_genes_mod_morans_i": robust_divide(num_genes_mod_morans_i, n_features),
        "frac_genes_high_morans_i": robust_divide(num_genes_high_morans_i, n_features),
    }
    with open(outs.spatial_enrichment_json, "w") as f:
        tk_safe_json.dump_numpy(summary_stats, f, indent=4, sort_keys=True)
