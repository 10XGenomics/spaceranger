#!/usr/bin/env python
#
# Copyright (c) 2025 10x Genomics, Inc. All rights reserved.
#
"""Compute segmentation plots for Visium HD data."""
from collections import Counter

import martian
import numpy as np

import cellranger.altair_plot_utils as alt_plot_utils
import cellranger.altair_utils as alt_utils
import cellranger.feature.utils as feature_utils
import cellranger.matrix as cr_matrix
import cellranger.spatial.hd_cs_websummary_plt_utils as hd_cs_websummary_plt_utils
import cellranger.spatial.hd_feature_slice as hd_fs
from cellranger.cell_typing.cas_metrics import get_df_from_analysis, sample_df
from cellranger.spatial.hd_feature_slice import (
    CELL_SEGMENTATION_MASK_NAME,
    SEGMENTATION_GROUP_NAME,
)

__MRO__ = """
stage SEGMENTATION_ANALYSIS_PLOTS(
    in  h5      filtered_feature_cell_matrix,
    in  h5      hd_feature_slice,
    in  path    analysis,
    out json    cell_area_chart,
    out json    features_per_bc_chart,
    out json    counts_per_bc_chart,
    out json    segmentation_umap_chart,
    src py      "stages/spatial/segmentation_analysis_plots",
) split (
) using (
    volatile = strict,
)
"""


def generate_counts_bc_list(data, idx_name: str, key_name: str):
    """Generates a list of dictionaries from the given data, where each dictionary contains keys specified by `idx_name` and `key_name`.

    Args:
        data (iterable): The input data to be converted into a dictionary list.
        idx_name (str): The key name to use for the index
        key_name (str): The key name to use for the values

    Returns:
        list[dict]: A list of dictionaries where each dictionary represents an item
        from the input data, with keys `idx_name` and `key_name`.
    """
    data_dict = dict(enumerate(data))
    return alt_utils.convert_to_dict_list(data_dict.items(), idx_name, key_name)


def split(args):
    if any(arg is None for arg in args):
        return {"chunks": [], "join": {}}

    filtered_mem = cr_matrix.CountMatrix.get_mem_gb_from_matrix_h5(
        args.filtered_feature_cell_matrix
    )
    join_mem_gb = int(np.ceil(filtered_mem)) + 4

    return {
        "chunks": [],
        "join": {
            "__mem_gb": join_mem_gb,
        },
    }


def join(args, outs, _chunk_defs, _chunk_outs):  # pylint: disable=too-many-locals
    if any(arg is None for arg in args):
        martian.clear(outs)
        return

    filtered_feature_cell_matrix = cr_matrix.CountMatrix.load_h5_file(
        args.filtered_feature_cell_matrix
    )
    with hd_fs.HdFeatureSliceIo(args.hd_feature_slice) as feature_slice:
        cell_segmentation_mask = hd_fs.CooMatrix.from_hdf5(
            feature_slice.h5_file[SEGMENTATION_GROUP_NAME][CELL_SEGMENTATION_MASK_NAME]
        )
        barcode_size = int(feature_slice.metadata.spot_pitch**2)

    squares_per_cell_2um = Counter(np.array(cell_segmentation_mask.data))
    areas_of_cell_sq_um = {k: v * barcode_size for k, v in squares_per_cell_2um.items()}
    cell_area_list = alt_utils.convert_to_dict_list(areas_of_cell_sq_um.items(), "idx", "cell_area")

    features_per_bc_list = generate_counts_bc_list(
        filtered_feature_cell_matrix.get_numfeatures_per_bc(), "idx", "num_features"
    )
    counts_per_bc_list = generate_counts_bc_list(
        filtered_feature_cell_matrix.get_counts_per_bc(), "idx", "num_counts"
    )

    # Strip the "idx" key from the dictionaries
    # This reduces the size of the JSON file and is not needed for the plots
    cell_area_list = [{"cell_area": d["cell_area"]} for d in cell_area_list]
    features_per_bc_list = [{"num_features": d["num_features"]} for d in features_per_bc_list]
    counts_per_bc_list = [{"num_counts": d["num_counts"]} for d in counts_per_bc_list]

    # Create the charts
    cell_area_chart = alt_plot_utils.create_histogram(
        data_list=cell_area_list,
        field="cell_area",
        maxbins=200,
        round_multiple=8,
        x_title="Cell area (μm²)",
        tooltip_title="Cell area (μm²)",
        plot_width=375,
    )

    features_per_bc_chart = alt_plot_utils.create_histogram(
        data_list=features_per_bc_list,
        field="num_features",
        maxbins=200,
        round_multiple=10,
        x_title="Genes per Cell",
        tooltip_title="Genes per Cell",
        plot_width=375,
    )

    counts_per_bc_chart = alt_plot_utils.create_histogram(
        data_list=counts_per_bc_list,
        field="num_counts",
        maxbins=200,
        round_multiple=10,
        x_title="UMIs per Cell",
        tooltip_title="UMIs per Cell",
        plot_width=375,
    )

    # UMAP plot
    segmentation_umap_df = get_df_from_analysis(args.analysis, get_umap_coords=True)
    segmentation_umap_df_sampled = sample_df(
        segmentation_umap_df, group_by_key="clusters", max_samples=20000
    )
    # Apply 10x cluster colors
    num_clusters = segmentation_umap_df_sampled["clusters"].unique()
    cluster_colors = []
    for cluster_num in range(1, len(num_clusters) + 1):
        rgb_clor = hd_cs_websummary_plt_utils.cluster_color_rgb(cluster_num, len(num_clusters))
        cluster_colors.append(rgb_clor)  # Keep RGB format for raster application

    segmentation_umap_chart = alt_plot_utils.create_umap_plot(
        segmentation_umap_df_sampled,
        plotting_variable="clusters",
        color_range=[hd_cs_websummary_plt_utils.rgb_to_hex(c) for c in cluster_colors],
        title="Cluster",
    )

    plots = list(
        map(
            alt_utils.chart_to_json,
            [
                cell_area_chart,
                features_per_bc_chart,
                counts_per_bc_chart,
                segmentation_umap_chart,
            ],
        )
    )
    feature_utils.write_json_from_dict(plots[0], outs.cell_area_chart)
    feature_utils.write_json_from_dict(plots[1], outs.features_per_bc_chart)
    feature_utils.write_json_from_dict(plots[2], outs.counts_per_bc_chart)
    feature_utils.write_json_from_dict(plots[3], outs.segmentation_umap_chart)
