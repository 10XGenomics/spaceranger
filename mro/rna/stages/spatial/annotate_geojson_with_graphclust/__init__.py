#
# Copyright (c) 2024 10X Genomics, Inc. All rights reserved.
#
"""Dumps graphclust clusters into geoJSON."""

import csv
import os

import geojson
import martian
import matplotlib
import numpy as np

from cellranger.colors import COLORS_20, COLORS_40
from cellranger.fast_utils import CellId  # pylint: disable=no-name-in-module,unused-import
from cellranger.spatial.segmentation_constants import (
    GEOJSON_CELL_ID_KEY,
    GEOJSON_CLASSIFICATION_COLOR_KEY,
    GEOJSON_CLASSIFICATION_KEY,
    GEOJSON_CLASSIFICATION_NAME_KEY,
)

__MRO__ = """
stage ANNOTATE_GEOJSON_WITH_GRAPHCLUST(
    in  path    analysis_csv,
    in  geojson nucleus_segmentations,
    out geojson annotated_segmentations,
    src py      "stages/spatial/annotate_geojson_with_graphclust",
) split (
) using (
    mem_gb = 8,
)
"""

CLUSTER_KEY = "Cluster"
BARCODE_KEY = "Barcode"

ANALYSIS_CSV_CLUSTERING_NAME = "clustering"
ANALYSIS_CSV_GRAPHCLUST_CLUSTERING_DIR_NAME = "gene_expression_graphclust"
ANALYSIS_CSV_CLUSTERING_CSV_NAME = "clusters.csv"


def split(args):
    if (
        not args.nucleus_segmentations
        or not args.analysis_csv
        or not os.path.exists(args.analysis_csv)
        or not os.path.exists(
            os.path.join(
                args.analysis_csv,
                ANALYSIS_CSV_CLUSTERING_NAME,
                ANALYSIS_CSV_GRAPHCLUST_CLUSTERING_DIR_NAME,
                ANALYSIS_CSV_CLUSTERING_CSV_NAME,
            )
        )
    ):
        join_def = {}
    else:
        mem_gb = os.path.getsize(args.nucleus_segmentations) / (1024**3) * 6 + 8
        join_def = {"__mem_gb": mem_gb, "__vmem_gb": mem_gb * 2}
    return {"chunks": [], "join": join_def}


def join(args, outs, _chunk_def, _chunk_outs):
    if (
        not args.nucleus_segmentations
        or not args.analysis_csv
        or not os.path.exists(args.analysis_csv)
        or not os.path.exists(
            os.path.join(
                args.analysis_csv,
                ANALYSIS_CSV_CLUSTERING_NAME,
                ANALYSIS_CSV_GRAPHCLUST_CLUSTERING_DIR_NAME,
                ANALYSIS_CSV_CLUSTERING_CSV_NAME,
            )
        )
    ):
        martian.clear(outs)
        return

    cluster_csv_path = os.path.join(
        args.analysis_csv,
        ANALYSIS_CSV_CLUSTERING_NAME,
        ANALYSIS_CSV_GRAPHCLUST_CLUSTERING_DIR_NAME,
        ANALYSIS_CSV_CLUSTERING_CSV_NAME,
    )

    bc_to_cluster_dct = {}
    with open(cluster_csv_path) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            bc_num = int(CellId(row[BARCODE_KEY]).id)
            bc_to_cluster_dct[bc_num] = row[CLUSTER_KEY]

    num_clusters = (
        len(set(bc_to_cluster_dct.values())) + 1
    )  # an additional one for non-clustered cells

    if num_clusters <= 20:
        clrmap = COLORS_20
    elif num_clusters <= 40:
        clrmap = COLORS_40
    else:
        turbo = matplotlib.colormaps["turbo"].resampled(256)
        clrmap = [
            (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255))
            for x in turbo(np.linspace(0, 1, num_clusters))
        ]

    with open(args.nucleus_segmentations) as f:
        gjsn = geojson.load(f)

    for gjsn_feature in gjsn.features:
        gjsn_feature_properties = gjsn_feature.properties
        gjsn_feature_properties[GEOJSON_CLASSIFICATION_KEY] = {}
        cell_id = gjsn_feature.properties[GEOJSON_CELL_ID_KEY]
        clster_id = int(bc_to_cluster_dct.get(cell_id, "0"))
        clstr_clr = clrmap[clster_id]
        clstr_name = f"Cluster-{clster_id}" if clster_id != 0 else "Unclustered"
        gjsn_feature_properties[GEOJSON_CLASSIFICATION_KEY][
            GEOJSON_CLASSIFICATION_NAME_KEY
        ] = clstr_name
        gjsn_feature_properties[GEOJSON_CLASSIFICATION_KEY][
            GEOJSON_CLASSIFICATION_COLOR_KEY
        ] = clstr_clr

    with open(outs.annotated_segmentations, "w") as f:
        geojson.dump(gjsn, f)
