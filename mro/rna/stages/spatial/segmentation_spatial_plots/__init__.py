#!/usr/bin/env python
#
# Copyright (c) 2025 10x Genomics, Inc. All rights reserved.
#
"""Compute segmentation plots for Visium HD data."""
import csv
import json
import os
from collections import defaultdict

import cv2
import martian
import numpy as np
import skimage
from PIL import Image

import cellranger.spatial.hd_cs_websummary_plt_utils as hd_cs_websummary_plt_utils
import cellranger.spatial.hd_feature_slice as hd_fs
import cellranger.spatial.multi_layered_images as mli
import cellranger.websummary.polygon_utils as polygon_utils
from cellranger.fast_utils import CellId
from cellranger.spatial.image import base64_encode_image
from cellranger.spatial.transform import (
    transform_pts_2d,
)
from cellranger.spatial.webp_image_utils import convert_img_to_webp

__MRO__ = """
stage SEGMENTATION_SPATIAL_PLOTS(
    in  h5      hd_feature_slice,
    in  path    analysis_csv,
    in  geojson cell_segmentations,
    in  png     tissue_hires_image,
    in  json    scale_factors_json,
    out json    spatial_segmentation_chart,
    src py      "stages/spatial/segmentation_spatial_plots",
) using (
    mem_gb   = 16,
    volatile = strict,
)
"""

SEGMENTATION_IMAGE_PADDING = 30

CLUSTER_KEY = "Cluster"
BARCODE_KEY = "Barcode"

ANALYSIS_CSV_CLUSTERING_NAME = "clustering"
ANALYSIS_CSV_GRAPHCLUST_CLUSTERING_DIR_NAME = "gene_expression_graphclust"
ANALYSIS_CSV_CLUSTERING_CSV_NAME = "clusters.csv"


def main(args, outs):  # pylint: disable=too-many-locals, too-many-statements
    if not all((args.cell_segmentations, args.tissue_hires_image, args.scale_factors_json)):
        martian.clear(outs)
        return

    with open(args.cell_segmentations) as f:
        cell_segmentations = json.load(f)
    with open(args.scale_factors_json) as f:
        scale_factors = json.load(f)
    tissue_hires_image_scale = scale_factors["tissue_hires_scalef"]
    for feature in cell_segmentations["features"]:
        feature["geometry"] = polygon_utils.scale_geometry(
            feature["geometry"], tissue_hires_image_scale
        )

    tissue_image_height, tissue_image_width = skimage.io.imread(args.tissue_hires_image).shape[:2]
    # Extract and clean the cell_id
    if args.analysis_csv and os.path.exists(
        cluster_csv_path := os.path.join(
            args.analysis_csv,
            ANALYSIS_CSV_CLUSTERING_NAME,
            ANALYSIS_CSV_GRAPHCLUST_CLUSTERING_DIR_NAME,
            ANALYSIS_CSV_CLUSTERING_CSV_NAME,
        )
    ):
        cell_cluster_dict = {}
        with open(cluster_csv_path) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                bc_num = CellId(row[BARCODE_KEY]).id
                cell_cluster_dict[bc_num] = int(row[CLUSTER_KEY])
        # Apply 10x cluster colors
        num_clusters = sorted(set(cell_cluster_dict.values()))  # Ensure clusters are sorted
        cluster_colors = {
            i + 1: hd_cs_websummary_plt_utils.cluster_color_rgb(i + 1, len(num_clusters))
            for i in range(len(num_clusters))
        }
        # Map each cell_id to its RGB cluster color
        cluster_color_dict = {
            cell_id: cluster_colors[cluster_idx]
            for cell_id, cluster_idx in cell_cluster_dict.items()
        }
        default_color = None
    else:
        cluster_color_dict, default_color = {}, hd_cs_websummary_plt_utils.cluster_color_rgb(1, 1)

    # Group all polygon coordinates by edge_color
    coords_by_color = defaultdict(list)
    for feature in cell_segmentations["features"]:
        if feature["geometry"]["type"] != "Polygon":
            continue
        cell_id = feature["properties"].get("cell_id")
        edge_color = cluster_color_dict.get(cell_id, default_color)
        if edge_color is None:
            continue
        coords = np.array(feature["geometry"]["coordinates"][0], dtype=np.int32)
        coords_by_color[tuple(edge_color)].append(coords)

    # initialize the rasters
    geometry_raster = np.zeros((tissue_image_height, tissue_image_width, 4), dtype=np.uint8)

    box_coords = None
    if args.hd_feature_slice:
        with hd_fs.HdFeatureSliceIo(args.hd_feature_slice) as feature_slice:
            spot_colrow_to_tissue_image_colrow = (
                feature_slice.metadata.transform_matrices.get_spot_colrow_to_tissue_image_colrow_transform()
            )
            nrows = feature_slice.nrows()
            ncols = feature_slice.ncols()
        # Get the capture area
        corners = np.array([[0, 0], [ncols, 0], [0, nrows], [ncols, nrows]])
        transformed_corners = (
            transform_pts_2d(corners, spot_colrow_to_tissue_image_colrow) * tissue_hires_image_scale
        )

        # Get the capture area dimensions
        box_coords = np.array(
            [
                transformed_corners[0],
                transformed_corners[1],
                transformed_corners[3],
                transformed_corners[2],
                transformed_corners[0],  # Close the box
            ],
            dtype=np.int32,
        )

    # Get the segmentation layer
    for _, (edge_color, poly_list) in enumerate(coords_by_color.items(), start=1):
        # Draw outlines on the geometry_raster if needed
        for coords in poly_list:
            cv2.polylines(
                geometry_raster,
                [coords],
                isClosed=True,
                color=tuple(int(c) for c in edge_color) + (255,),
                thickness=1,
            )

    geometry_img = Image.fromarray(geometry_raster, mode="RGBA")
    cell_segmentations_encoded = polygon_utils.make_path_and_encode(
        geometry_img,
        "cell_segmentation",
        lossless=True,
        crop_box=box_coords,
        padding=SEGMENTATION_IMAGE_PADDING if box_coords is not None else 0,
    )
    segmentation_image = [
        mli.LabeledImage(
            label="Segmentations",
            image=cell_segmentations_encoded,
            color=None,
            css_transform=None,
        )
    ]

    cluster_layers = None
    if args.analysis_csv:
        # For each cluster color: draw, save to .webp, then discard raster to save mem
        saved_clusters = []  # to record (cluster_name, edge_color) tuples
        cluster_color_mapping = {tuple(v): k for k, v in cluster_colors.items()}
        for edge_color, poly_list in coords_by_color.items():
            fill_raster = np.zeros_like(geometry_raster)

            # Draw outlines on the geometry_raster if needed
            for coords in poly_list:
                # Fill cluster raster
                cv2.fillPoly(
                    fill_raster,
                    [coords],
                    color=tuple(int(c) for c in edge_color) + (255,),
                )

            # Convert to PIL Image and write out
            img = Image.fromarray(fill_raster, mode="RGBA")
            cluster_idx = cluster_color_mapping.get(edge_color)
            cluster_name = f"Cluster{cluster_idx}"
            file_prefix = f"{cluster_name}_cell_segmentation_fill"
            polygon_utils.make_path_and_encode(
                img,
                file_prefix,
                lossless=True,
                crop_box=box_coords,
                padding=SEGMENTATION_IMAGE_PADDING if box_coords is not None else 0,
            )
            # Encode the filled layers and make a labeled image
            # Record for later layer assembly
            saved_clusters.append((cluster_name, edge_color))
        # Ensure saved_clusters is sorted by cluster number
        saved_clusters.sort(key=lambda x: int(x[0].replace("Cluster", "")))

        # Build the multilayer cluster_layers
        cluster_layers = mli.Layer(
            name="Clusters",
            images=[
                mli.LabeledImage(
                    label=cluster_name,
                    image=base64_encode_image(f"{cluster_name}_cell_segmentation_fill.webp"),
                    color=hd_cs_websummary_plt_utils.rgb_to_hex(edge_color),
                    css_transform=None,
                )
                for cluster_name, edge_color in saved_clusters
            ],
            initial_opacity=0.5,
        )

    webp_tissue_image = convert_img_to_webp(
        args.tissue_hires_image,
        martian.make_path("webp_tissue_image.webp").decode(),
        quality=20,
        crop_box=box_coords,
        padding=SEGMENTATION_IMAGE_PADDING if box_coords is not None else 0,
    )
    with Image.open(webp_tissue_image) as img:
        img_height, img_width = img.height, img.width
    # Store all the images
    tissue_image = [
        mli.LabeledImage(
            label="Tissue image",
            image=base64_encode_image(webp_tissue_image),
            color=None,
            css_transform=None,
        )
    ]

    layers = [
        mli.Layer(name="Tissue image", images=tissue_image),
        mli.Layer(name="Segmentations", images=segmentation_image),
    ]
    if cluster_layers:
        layers.append(cluster_layers)

    final_layers = mli.MultiLayerImages(
        focus=mli.InitialFocus(
            x=0,
            y=0,
            width=img_width,
            height=img_height,
        ),
        layers=layers,
        fullScreen=False,
        checkAll=True,
        showFullScreenButton=False,
        legendTitle=None,
        legendWidthPx=125,
    )

    with open(outs.spatial_segmentation_chart, "w") as f:
        json.dump(final_layers, f, indent=2, default=lambda x: x.__dict__)
