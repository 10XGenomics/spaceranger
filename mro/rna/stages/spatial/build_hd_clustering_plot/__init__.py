# Copyright (c) 2021 10x Genomics, Inc. All rights reserved.
"""Generate the cluster plot in the HD websummary."""

import json
import os
from dataclasses import dataclass

import martian
import numpy as np
import skimage
from PIL import Image

from cellranger.analysis.singlegenome import UMAP_NAME, SingleGenomeAnalysis
from cellranger.spatial.hd_cs_websummary_plt_utils import (
    cluster_color_rgb,
    plot_umap_image,
    rgb_to_hex,
)
from cellranger.spatial.hd_feature_slice import (
    GENE_EXPRESSION_CLUSTERING,
    GRAPHCLUSTERING_NAME,
    HdFeatureSliceIo,
    bin_size_um_from_bin_name,
)
from cellranger.spatial.image import base64_encode_image
from cellranger.spatial.transform import (
    css_transform_list,
    normalize_perspective_transform,
    scale_matrix,
    translation_matrix,
)
from cellranger.websummary.analysis_tab_core import (
    SPATIAL_DIFFEXP_TABLE_HELP,
    diffexp_table_from_analysis,
)
from cellranger.websummary.react_components import ClusteringData, ReactComponentEncoder
from cellranger.websummary.zoom import InitialZoomPan

__MRO__ = """
stage BUILD_HD_CLUSTERING_PLOT(
    in  h5        hd_feature_slice,
    in  jpg       websummary_tissue_image,
    in  float     websummary_tissue_image_scale,
    in  map<path> analysis_h5,
    out json      cluster_plot,
    src py        "stages/spatial/build_hd_clustering_plot",
) using (
    mem_gb   = 12,
    volatile = strict,
)
"""

TISSUE_IMAGE_DISPLAY_WIDTH = 400


@dataclass
class SingleClusterData:
    """Data for a single cluster in a clustering plot."""

    cluster_name: str
    hex_color: str
    spatial_plot: str
    umap_plot: str


@dataclass
class SpatialPlotProps:
    """Props for a spatial plot."""

    title: str
    tissue_image: str
    tissue_css_transform: list[float]
    spot_css_transform: list[float]
    width: int
    height: int
    initial_zoom_pan: InitialZoomPan


@dataclass
class UmapPlotProps:
    """Props for a UMAP plot."""

    title: str


@dataclass
class HdClusteringPlot:
    """Clustering plots for a single bin size and clustering algorithm."""

    spatial_plot_props: SpatialPlotProps
    umap_plot_props: UmapPlotProps
    clusters: list[SingleClusterData]


@dataclass
class DifferentialExpressionTable:
    """Differential expression table for a single bin size and clustering algorithm."""

    table: dict


@dataclass
class DifferentialExpression:
    """Differential expression table for a single bin size and clustering algorithm."""

    title: dict
    table: DifferentialExpressionTable


@dataclass
class SingleClusteringData:
    """The cluster plots and differential expression table for a single clustering algorithm."""

    hd_clustering_plot: HdClusteringPlot
    differential_expression: DifferentialExpression


@dataclass
class SingleBinLevelData:
    """Clustering results for a single bin size."""

    bin_name: str
    clustering_data: SingleClusteringData


@dataclass
class AllBinLevelsData:
    """Clustering results for all bin sizes."""

    bin_plots: list[SingleBinLevelData]

    @staticmethod
    def create(
        feature_slice, analysis_h5_folders, tissue_image, tissue_image_scale
    ):  # pylint: disable=too-many-locals
        """Create the clustering plot data for all bin sizes."""
        _, tissue_image_width = skimage.io.imread(tissue_image).shape[:2]
        websummary_scale = TISSUE_IMAGE_DISPLAY_WIDTH / tissue_image_width

        bin_plots = []
        cluster_algo_key, cluster_algo_name = (GRAPHCLUSTERING_NAME, "Graph-based")
        # Ignore k-means to reduce websummary size
        # + [
        #     (f"kmeans_{i}_clusters", f"K-means (K={i})") for i in range(2, 11)
        # ]
        for bin_name, analysis_h5_folder in analysis_h5_folders.items():
            if analysis_h5_folder is None or not os.path.exists(analysis_h5_folder):
                continue
            bin_size_um = bin_size_um_from_bin_name(bin_name)
            binning_scale = int(bin_size_um / feature_slice.metadata.spot_pitch)
            analysis = SingleGenomeAnalysis.load_h5(
                os.path.join(analysis_h5_folder, "analysis.h5"),
                "pca",
                [UMAP_NAME],
            )

            spot_colrow_to_tissue_image_colrow = (
                feature_slice.metadata.transform_matrices.get_spot_colrow_to_tissue_image_colrow_transform()
            )
            spot_transform = normalize_perspective_transform(
                scale_matrix(websummary_scale * tissue_image_scale)
                @ spot_colrow_to_tissue_image_colrow
                # spot_colrow is center-based and we need a corner-based transform for CSS
                @ translation_matrix(-0.5, -0.5)
                @ scale_matrix(binning_scale)
            )

            umap_x, umap_y = feature_slice.get_umap(binning_scale)
            initial_zoom_pan = InitialZoomPan.compute(
                umap_x, spot_transform, TISSUE_IMAGE_DISPLAY_WIDTH
            )

            clustering_key = f"{GENE_EXPRESSION_CLUSTERING}_{cluster_algo_key}"
            clustering = feature_slice.get_clustering(
                binning_scale=binning_scale, clustering_method=cluster_algo_key
            )

            num_clusters = np.max(clustering)

            clustering_data = SingleClusteringData(
                differential_expression=DifferentialExpression(
                    title=SPATIAL_DIFFEXP_TABLE_HELP,
                    table=DifferentialExpressionTable(
                        table=json.loads(
                            json.dumps(
                                ClusteringData(
                                    key=clustering_key,
                                    clustering=analysis.clusterings[clustering_key],
                                    data=diffexp_table_from_analysis(
                                        analysis.differential_expression[clustering_key],
                                        analysis.clusterings[clustering_key],
                                        analysis,
                                        is_hd=True,
                                    ),
                                ).data,
                                cls=ReactComponentEncoder,
                            )
                        )
                    ),
                ),
                hd_clustering_plot=HdClusteringPlot(
                    clusters=[
                        AllBinLevelsData.build_cluster_data(
                            umap_x, umap_y, clustering, cluster_num, num_clusters
                        )
                        for cluster_num in range(1, num_clusters + 1)
                    ],
                    umap_plot_props=UmapPlotProps(
                        title=f"UMAP Projection of {bin_size_um}µm bins colored by {cluster_algo_name} clustering",
                    ),
                    spatial_plot_props=SpatialPlotProps(
                        title=f"Tissue plot with {bin_size_um}µm bins colored by {cluster_algo_name} clustering",
                        tissue_image=base64_encode_image(tissue_image),
                        tissue_css_transform=css_transform_list(scale_matrix(websummary_scale)),
                        spot_css_transform=css_transform_list(spot_transform),
                        width=TISSUE_IMAGE_DISPLAY_WIDTH,
                        height=TISSUE_IMAGE_DISPLAY_WIDTH,
                        initial_zoom_pan=initial_zoom_pan,
                    ),
                ),
            )
            bin_plots.append(
                SingleBinLevelData(
                    bin_name=f"{bin_size_um} µm bin",
                    clustering_data=clustering_data,
                )
            )
        return AllBinLevelsData(
            bin_plots=bin_plots,
        )

    @staticmethod
    def umap_plot_xy_limits(umap_x, umap_y):
        """Compute the XY limits for the UMAP plot."""
        min_xy = min(np.min(umap_x), np.min(umap_y))
        max_xy = max(np.max(umap_x), np.max(umap_y))
        delta_xy = max_xy - min_xy
        min_xy = min_xy - 0.01 * delta_xy
        max_xy = max_xy + 0.01 * delta_xy
        return min_xy, max_xy

    @staticmethod
    def build_cluster_data(umap_x, umap_y, clustering, cluster_num, num_clusters):
        """Build the data for a single cluster."""
        min_xy, max_xy = AllBinLevelsData.umap_plot_xy_limits(umap_x, umap_y)
        nrows, ncols = clustering.shape
        spatial_plot_fname = martian.make_path("spatial_plot.png").decode()
        umap_plot_fname = martian.make_path("umap_plot.png").decode()
        color_rgb = cluster_color_rgb(cluster_num, num_clusters)
        color_hex = rgb_to_hex(color_rgb)
        cluster_mask = clustering == cluster_num
        # SPATIAL IMAGE

        img = np.zeros((nrows, ncols, 4), dtype=np.uint8)
        img[cluster_mask, -1] = 255
        img[cluster_mask, :-1] = color_rgb

        Image.fromarray(img).save(spatial_plot_fname)

        # UMAP IMAGE
        plot_umap_image(
            umap_x[cluster_mask], umap_y[cluster_mask], umap_plot_fname, color_hex, min_xy, max_xy
        )

        cluster_data = SingleClusterData(
            cluster_name=f"Cluster {cluster_num}",
            hex_color=color_hex,
            spatial_plot=base64_encode_image(spatial_plot_fname),
            umap_plot=base64_encode_image(umap_plot_fname),
        )

        return cluster_data


def main(args, outs):
    if args.analysis_h5 is None or all(x is None for x in args.analysis_h5.values()):
        outs.cluster_plot = None
        return

    clustering_selector = AllBinLevelsData.create(
        HdFeatureSliceIo(args.hd_feature_slice),
        analysis_h5_folders=args.analysis_h5,
        tissue_image=args.websummary_tissue_image,
        tissue_image_scale=args.websummary_tissue_image_scale,
    )
    with open(outs.cluster_plot, "w") as f:
        json.dump(clustering_selector, f, indent=2, default=lambda x: x.__dict__)
