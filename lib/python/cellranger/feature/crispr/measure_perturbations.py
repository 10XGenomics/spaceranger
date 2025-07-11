# Copyright (c) 2019 10x Genomics, Inc. All rights reserved.
# Keeping the annotation import for now until PEP 563 is mandatory in a future Python version
# Check: https://peps.python.org/pep-0563/
# Check: https://github.com/astral-sh/ruff/issues/7214
from __future__ import annotations

import csv
import os
import sys
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from cellranger.analysis.analysis_types import DifferentialExpression
from cellranger.analysis.diffexp import get_local_sseq_params, sseq_differential_expression
from cellranger.constants import FILTER_LIST
from cellranger.rna.library import GENE_EXPRESSION_LIBRARY_TYPE
from tenkit.stats import robust_divide

if TYPE_CHECKING:
    from typing import Any

    fc_t = tuple[float, float, int, int]
    ci_t = tuple[np.float64, np.float64]
    np_1d_array_int64 = np.ndarray[tuple[int], np.dtype[np.int64]]
    np_1d_array_intp = np.ndarray[tuple[int], np.dtype[np.intp]]
    from cellranger.matrix import CountMatrix

pd.set_option("compute.use_numexpr", False)


@dataclass
class TargetInfo:
    target_id: str
    gene_id: str
    gene_name: str


@dataclass
class ProtospacerCall:
    feature_call: str
    gene_id: str


@dataclass
class PerturbationResult:
    perturbation_name: str
    target_or_gene_name: str
    log2_fc: float
    p_val: float
    lower_bound: np.float64
    upper_bound: np.float64
    num_cells_with_perb: int
    mean_umi_count_perb: float
    num_cells_with_control: int
    mean_umi_count_control: float


PERT_EFFI_SUMM_COLS_BY_FEAT = (
    "Perturbation",
    "Target Guide",
    "Log2 Fold Change",
    "p Value",
    "Log2 Fold Change Lower Bound",
    "Log2 Fold Change Upper Bound",
    "Cells with Perturbation",
    "Mean UMI Count Among Cells with Perturbation",
    "Cells with Non-Targeting Guides",
    "Mean UMI Count Among Cells with Non-Targeting Guides",
)
PERT_EFFI_SUMM_COLS_BY_TRGT = (
    PERT_EFFI_SUMM_COLS_BY_FEAT[:1] + ("Target Gene",) + PERT_EFFI_SUMM_COLS_BY_FEAT[2:]
)
NUM_BOOTSTRAPS = 500  # number of bootstrap draws to do for calculating
# empirical confidence intervals for perturbation efficiencies
CI_LOWER_BOUND = 5.0  # CI lower bound (ie percentile value) for perturbation efficiencies
CI_UPPER_BOUND = 95.0  # CI upper bound (ie percentile value) for perturbation efficiencies
# for which we can't or won't compute perturbation efficiencies
CONTROL_LIST = ("Non-Targeting",)  # Target IDs used for specifying control perturbations
MIN_NUMBER_CELLS_PER_PERTURBATION = 10  # Minimum number of cells a perturbation has to have before we compute differential expression for it
MIN_COUNTS_PERTURBATION = 5
MIN_COUNTS_CONTROL = 5

UMI_NUM_TRIES = 10  # Number of initial points to try for GMM-fitting
UMI_MIX_INIT_SD = 0.25  # Intial standard deviation for GMM components


TOP_GENES_SUMMARY_MAP = OrderedDict(
    [
        ("Gene Name", "Gene Name"),
        ("Gene ID", "Gene ID"),
        ("log2_fold_change", "Log2 Fold Change"),
        ("adjusted_p_value", "Adjusted p-value"),
    ]
)
NUM_TOP_GENES = None


def read_and_validate_feature_ref(
    feature_reference: str,
) -> None | dict[str, TargetInfo]:
    """Returns a dict of target_id to TargetInfo."""
    with open(feature_reference) as csv_file:
        csv_reader = csv.reader(csv_file)
        # Read the header
        col_name_to_idx = {n: i for i, n in enumerate(next(csv_reader))}

        if "target_gene_id" not in col_name_to_idx.keys():
            sys.stderr.write(
                "feature_reference CSV does not have target_gene_id column which "
                + "is a requirement for measuring perturbation efficiencies"
            )
            return None

        assert (
            "id" in col_name_to_idx.keys()
        ), f'This feature_reference CSV does not have "id" column: {feature_reference}'

        # If target_gene_name is not present, use target_gene_id as target_gene_name
        if "target_gene_name" not in col_name_to_idx.keys():
            col_name_to_idx["target_gene_name"] = col_name_to_idx["target_gene_id"]

        target_info = {
            row[col_name_to_idx["id"]]: TargetInfo(
                target_id=row[col_name_to_idx["id"]],
                gene_id=row[col_name_to_idx["target_gene_id"]],
                gene_name=row[col_name_to_idx["target_gene_name"]],
            )
            for row in csv_reader
        }
    if not any(v.gene_id == "Non-Targeting" for v in target_info.values()):
        sys.stderr.write(
            "Non-Targeting guides required as controls for differential expression calculations"
        )
        return None
    return target_info


def save_perturbation_efficiency_summary(
    outpath: str,
    fold_change_per_perturbation: dict[str, dict[str, PerturbationResult]],
    by_feature: bool,
) -> None:
    with open(outpath, "w", newline="") as outfile:
        writer = csv.writer(outfile, dialect="unix", quoting=csv.QUOTE_MINIMAL)
        writer.writerow(PERT_EFFI_SUMM_COLS_BY_FEAT if by_feature else PERT_EFFI_SUMM_COLS_BY_TRGT)
        rows = sorted(
            (
                (
                    fc.perturbation_name,
                    fc.target_or_gene_name,
                    fc.log2_fc,
                    fc.p_val,
                    fc.lower_bound,
                    fc.upper_bound,
                    fc.num_cells_with_perb,
                    fc.mean_umi_count_perb,
                    fc.num_cells_with_control,
                    fc.mean_umi_count_control,
                )
                for fold_changes in fold_change_per_perturbation.values()
                for fc in fold_changes.values()
            ),
            key=lambda x: x[2],
        )
        writer.writerows(rows)


def save_top_perturbed_genes(
    base_dir: str,
    results_per_perturbation: OrderedDict[str, pd.DataFrame],
):
    if not results_per_perturbation:
        return
    os.makedirs(base_dir, exist_ok=True)
    fn = os.path.join(base_dir, "top_perturbed_genes.csv")

    list_df_results = []
    summary_df_columns = []
    for name, de_result in results_per_perturbation.items():
        list_df_results.append(de_result)
        summary_df_columns += [f"Perturbation: {name}, {s}" for s in TOP_GENES_SUMMARY_MAP.values()]
    summary_df = pd.concat(list_df_results, ignore_index=True, axis=1)
    summary_df.columns = summary_df_columns
    summary_df.to_csv(fn, index=False)


def sanitize_perturbation_results(
    res_table: pd.DataFrame,
) -> pd.DataFrame:

    # at least 1 count amongst all the control cells
    res_table = res_table[res_table["sum_b"] > 0]
    # at least the minimum number of counts in either category (perturbation or control)
    res_table = res_table[
        (res_table["sum_a"] >= MIN_COUNTS_PERTURBATION) | (res_table["sum_b"] >= MIN_COUNTS_CONTROL)
    ]
    res_table["abs_log2_fold_change"] = np.abs(res_table["log2_fold_change"])
    # sort by abs log2 fold change, adjusted_p_value, Gene Name, in that order
    res_table.sort_values(
        by=["abs_log2_fold_change", "adjusted_p_value", "Gene Name"],
        ascending=[False, True, True],
        inplace=True,
    )
    res_table = res_table[list(TOP_GENES_SUMMARY_MAP)][:NUM_TOP_GENES]
    res_table.reset_index(drop=True, inplace=True)

    return res_table


def _get_bc_targets_dict(
    target_info: dict[str, TargetInfo],
    protospacers_per_cell_path: str,
) -> dict[str, ProtospacerCall]:
    """Get the dict of barcode pairs.

    The values are dicts with information about the targets of the features assgined to the barcode.

    Returns:
        a dict of bc:{} pairs. All barcodes which have been assigned at least 1 protospacer are keys in this dict.

    Note:
        barcodes without any protospacers will not be present in this dict.
    """
    bc_targets_dict: dict[str, ProtospacerCall] = {}

    with open(protospacers_per_cell_path) as csv_file:
        csv_reader = csv.reader(csv_file)
        # Read the header
        col_name_to_idx = {n: i for i, n in enumerate(next(csv_reader))}

        for row in csv_reader:
            feature_call = row[col_name_to_idx["feature_call"]]
            num_features = int(row[col_name_to_idx["num_features"]])
            cell_barcode = row[col_name_to_idx["cell_barcode"]]

            if feature_call in target_info:
                # single feature
                gene_id = target_info[feature_call].gene_id
            else:
                gene_id: str = feature_call

            if num_features > 1:
                # multiple features
                this_features = feature_call.split("|")
                gene_ids = [target_info[x].gene_id for x in this_features]

                if set(gene_ids) == set(CONTROL_LIST):
                    gene_id = "Non-Targeting"
                    # each feature is a non-targeting guide, and so the cell is a control cell
                else:
                    gene_ids = sorted(set(gene_ids).difference(FILTER_LIST))
                    if gene_ids:
                        gene_id = "|".join(gene_ids)
                    else:
                        gene_id = "Ignore"

            bc_targets_dict[cell_barcode] = ProtospacerCall(
                feature_call=feature_call,
                gene_id=gene_id,
            )
    return bc_targets_dict


def _should_filter(
    perturbation_name: str,
    target_id_name_map: dict[str, str],
):
    target_tuple = _get_target_id_from_name(perturbation_name, target_id_name_map)
    return all(x in FILTER_LIST for x in target_tuple[1])


def _get_target_id_from_name(
    this_perturbation_name: str, target_id_name_map: dict[str, str]
) -> tuple[list[str], list[str]]:
    if "|" not in this_perturbation_name:
        return ([this_perturbation_name], [target_id_name_map[this_perturbation_name]])

    p_names = this_perturbation_name.split("|")

    return (p_names, [target_id_name_map[p_name] for p_name in p_names])


def get_perturbation_efficiency(
    target_info: dict[str, TargetInfo],
    protospacers_per_cell_path: str,
    feature_count_matrix: CountMatrix,
    by_feature: bool = False,
) -> (
    tuple[
        OrderedDict[str, pd.DataFrame],
        DifferentialExpression,
        dict[str, dict[str, PerturbationResult]],
    ]
    | None
):
    """Calculates log2 fold change and empirical confidence intervals for log2 fold change for target genes.

    Args:
        targets (list[TargetInfo]): list of TargetInfo dataclass objects
        protospacers_per_cell (str): path to the protospacer calls per cell CSV file
        feature_count_matrix (CountMatrix): Feature Barcode Matrix
        by_feature (bool): if True, cells are grouped by the combination of protospacers
                           present in them, rather than the gene targets of those protospacers.
                           Right now, the pipeline measures perturbations both by feature
                           (by_feature=True) or by target (by_feature=False) by calling
                           the MEASURE_PERTURBATIONS_PD stage twice using the "as" keyword in
                           Martian 3.0

    Returns:
        results_per_perturbation (OrderedDict[str, DataFrame]): dict[perturbation_name, DE results]
        results_all_perturbations (DifferentialExpression): All DE results
        fold_change_per_perturbation (dict[str, dict[str, FoldChange]]): (perturbation_name, dict[target/feature name, FoldChange])
    """
    matrix = feature_count_matrix.select_features_by_type(GENE_EXPRESSION_LIBRARY_TYPE)

    (target_calls, perturbation_keys) = _get_ps_clusters(
        target_info,
        protospacers_per_cell_path,
        [bc.decode() for bc in matrix.bcs],
        by_feature,
    )

    target_to_gene_id = {
        target.target_id if by_feature else target.gene_name: target.gene_id
        for target in target_info.values()
    }
    results_per_perturbation: OrderedDict[str, pd.DataFrame] = OrderedDict()
    # MEASURE_PERTURBATIONS assumes this dict to be ordered
    filter_cluster_indices = [x for x in perturbation_keys if perturbation_keys[x] in FILTER_LIST]
    n_clusters = len(perturbation_keys)
    n_effective_clusters = n_clusters - len(filter_cluster_indices)

    # Create a numpy array with 3*k columns, where k is the number of perturbations
    # k = n_clusters - 2 since we don't need to report results for ignore and non-targeting
    # each group of 3 columns is mean, log2, pvalue for cluster i
    all_de_results = np.zeros((matrix.features_dim, 3 * n_effective_clusters))

    nt_indices = [x for x in perturbation_keys if perturbation_keys[x] == "Non-Targeting"]

    if len(nt_indices) == 0:
        return None
    nt_index = nt_indices[0]
    control_num_cells = sum(target_calls == nt_index)

    in_control_cluster = target_calls == nt_index
    feature_defs = matrix.feature_ref.feature_defs
    gene_ids = [feature_def.id.decode() for feature_def in feature_defs]
    gene_names = [feature_def.name for feature_def in feature_defs]

    fold_change_per_perturbation: dict[str, dict[str, PerturbationResult]] = {}
    cluster_counter = 1
    column_counter = 0
    for cluster, perturbation_name in perturbation_keys.items():
        if (cluster in filter_cluster_indices) or _should_filter(
            perturbation_name, target_to_gene_id
        ):
            continue

        in_cluster = target_calls == cluster
        group_a = np.flatnonzero(in_cluster)

        if len(group_a) < MIN_NUMBER_CELLS_PER_PERTURBATION:
            continue

        group_b = np.flatnonzero(in_control_cluster)
        sys.stdout.flush()

        both_conditions = np.concatenate([group_a, group_b])
        local_matrix = matrix.select_barcodes(both_conditions)

        (
            local_sseq_params,
            new_group_a,
            new_group_b,
            matrix_groups,
        ) = get_local_sseq_params(matrix.m, group_a, group_b)

        de_result = sseq_differential_expression(
            matrix_groups, new_group_a, new_group_b, local_sseq_params
        )
        assert de_result is not None
        de_result["Gene ID"] = gene_ids
        de_result["Gene Name"] = gene_names

        all_de_results[:, 0 + 3 * (cluster_counter - 1)] = de_result["sum_a"] / len(group_a)
        all_de_results[:, 1 + 3 * (cluster_counter - 1)] = de_result["log2_fold_change"]
        all_de_results[:, 2 + 3 * (cluster_counter - 1)] = de_result["adjusted_p_value"]
        column_counter += 3

        num_cells = sum(target_calls == cluster)

        fold_change_per_perturbation[perturbation_name] = _get_log2_fold_change(
            perturbation_name,
            num_cells,
            control_num_cells,
            de_result,
            target_to_gene_id,
            local_matrix,
            new_group_a,
            new_group_b,
            local_sseq_params,
        )

        de_result = sanitize_perturbation_results(de_result)
        results_per_perturbation[perturbation_name] = de_result

        cluster_counter += 1

    all_de_results = all_de_results[:, 0:column_counter]
    results_all_perturbations = DifferentialExpression(all_de_results)

    return (
        results_per_perturbation,
        results_all_perturbations,
        fold_change_per_perturbation,
    )


def _get_log2_fold_change(
    perturbation_name: str,
    num_cells: int,
    control_num_cells: int,
    results: pd.DataFrame,
    target_to_gene_id: dict[str, str],
    matrix: CountMatrix,
    group_a: np_1d_array_intp,
    group_b: np_1d_array_intp,
    local_params: dict[str, Any],
) -> dict[str, PerturbationResult]:
    (this_names, this_ids) = _get_target_id_from_name(
        perturbation_name,
        target_to_gene_id,
    )

    perturbation_results: dict[str, PerturbationResult] = {}

    for name, target in zip(this_names, this_ids):
        if target in FILTER_LIST:
            continue
        deg_result = results.loc[results["Gene ID"] == target]
        if deg_result.empty:
            continue

        lower_bound, upper_bound = _get_fold_change_cis(
            matrix,
            target,
            group_a,
            group_b,
            local_params,
        )
        log2_fold_change = deg_result["log2_fold_change"].values[0]
        p_value = deg_result["p_value"].values[0]
        sum_a = deg_result["sum_a"].values[0]
        sum_b = deg_result["sum_b"].values[0]
        perturbation_results[name] = PerturbationResult(
            perturbation_name=perturbation_name,
            target_or_gene_name=name,
            log2_fc=log2_fold_change,
            p_val=p_value,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            num_cells_with_perb=num_cells,
            mean_umi_count_perb=robust_divide(sum_a, num_cells),
            num_cells_with_control=control_num_cells,
            mean_umi_count_control=robust_divide(sum_b, control_num_cells),
        )

    return perturbation_results


def _get_fold_change_cis(
    matrix: CountMatrix,
    target: str,
    cond_a: np_1d_array_intp,
    cond_b: np_1d_array_intp,
    computed_params: dict[str, Any],
) -> tuple[np.float64, np.float64]:
    # filter the matrix to select only the target gene
    this_matrix = matrix.select_features_by_ids([target.encode()])

    x = this_matrix.m

    log2_fold_change_vals = list(range(NUM_BOOTSTRAPS))
    for i in range(NUM_BOOTSTRAPS):
        this_cond_a = np.random.choice(cond_a, size=len(cond_a), replace=True)
        this_cond_b = np.random.choice(cond_b, size=len(cond_b), replace=True)
        x_a = x[:, this_cond_a]
        x_b = x[:, this_cond_b]

        # Size factors
        size_factor_a = np.sum(computed_params["size_factors"][cond_a])
        size_factor_b = np.sum(computed_params["size_factors"][cond_b])

        gene_sums_a = np.squeeze(np.asarray(x_a.sum(axis=1)))
        gene_sums_b = np.squeeze(np.asarray(x_b.sum(axis=1)))

        log2_fold_change_vals[i] = np.log2((1 + gene_sums_a) / (1 + size_factor_a)) - np.log2(
            (1 + gene_sums_b) / (1 + size_factor_b)
        )

    return (
        np.percentile(log2_fold_change_vals, CI_LOWER_BOUND),
        np.percentile(log2_fold_change_vals, CI_UPPER_BOUND),
    )


def _get_ps_clusters(
    target_info: dict[str, TargetInfo],
    protospacers_per_cell_path: str,
    barcodes: list[str],
    by_feature: bool = True,
) -> tuple[np_1d_array_int64, dict[int, str]]:
    """Returns a tuple (target_calls, perturbation_keys).

    Args:
        target_calls (np.array(int)): identifies the perturbation assigned to
            each cell in the gene-barcode matrix
        perturbation_keys (dict): (cluster_number:perturbation_name) pairs

    Returns:
        target_calls (np_1d_array_int64): integer perturbation target labels starting from 1
        perturbation_keys (dict): dict[cluster_number, perturbation_name]
    """
    bc_targets_dict = _get_bc_targets_dict(target_info, protospacers_per_cell_path)
    for bc in barcodes:
        if bc not in bc_targets_dict:
            bc_targets_dict[bc] = ProtospacerCall(
                feature_call="None",
                gene_id="None",
            )

    if by_feature:
        return _get_ps_clusters_by_feature(bc_targets_dict, barcodes)
    else:
        return _get_ps_clusters_by_target(target_info, bc_targets_dict, barcodes)


def _get_ps_clusters_by_target(
    target_info: dict[str, TargetInfo],
    bc_targets_dict: dict[str, ProtospacerCall],
    barcodes: list[str],
) -> tuple[np_1d_array_int64, dict[int, str]]:

    gene_id_to_gene_name = {v.gene_id: v.gene_name for v in target_info.values()}

    def get_target_name(gene_id: str) -> str:
        sep = "|"
        if sep not in gene_id:
            return gene_id_to_gene_name.get(gene_id, gene_id)
        return "|".join([gene_id_to_gene_name.get(gid, gid) for gid in gene_id.split(sep)])

    gene_ids = [bc_targets_dict[bc].gene_id for bc in barcodes]
    unique_gene_ids = sorted(set(gene_ids))

    gene_id_to_idx = {gene_id: idx for (idx, gene_id) in enumerate(unique_gene_ids, start=1)}
    target_calls = np.asarray([gene_id_to_idx[x] for x in gene_ids], dtype=np.int64)

    perturbation_keys = {idx: get_target_name(gid) for (gid, idx) in gene_id_to_idx.items()}

    return (target_calls, perturbation_keys)


def _get_ps_clusters_by_feature(
    bc_targets_dict: dict[str, ProtospacerCall],
    barcodes: list[str],
) -> tuple[np_1d_array_int64, dict[int, str]]:

    def _get_feature_from_pc(protospacer_call: ProtospacerCall) -> str:
        if protospacer_call.gene_id not in FILTER_LIST:
            return protospacer_call.feature_call

        if protospacer_call.gene_id == "None":
            return "Ignore"

        return protospacer_call.gene_id

    features = [_get_feature_from_pc(bc_targets_dict[bc]) for bc in barcodes]
    unique_features = sorted(set(features))

    feature_to_idx = {feat: idx for (idx, feat) in enumerate(unique_features, start=1)}
    target_calls = np.asarray([feature_to_idx[feat] for feat in features], dtype=np.int64)

    perturbation_keys = {b: a for a, b in feature_to_idx.items()}

    return (target_calls, perturbation_keys)
