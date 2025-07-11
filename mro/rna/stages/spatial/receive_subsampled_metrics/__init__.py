#!/usr/bin/env python
#
# Copyright (c) 2024 10X Genomics, Inc. All rights reserved.
#
"""Generate saturation and median genes plot."""

import json

import plotly.express as px
import polars as pl

import cellranger.websummary.plotly_tools as pltly
from cellranger.websummary.analysis_tab_aux import (
    SPATIAL_HD_MEAN_GENE_PLOT_HELP_JSON_STRING,
    VISIUM_HD_RTL_SEQ_SATURATION_PLOT_HELP,
)

PLOTLY_EXPRESS_TEMPLATE = "none"
BIN_SIZE_UM_KEY = "bin_size_um"
TOTAL_RAW_READS_KEY = "total_raw_reads_after_subsampling"
SEQUENCING_SATURATION_KEY = "sequencing_saturation"
MEAN_READS_PER_BIN = "mean_reads_per_bin"
MEAN_GENES_PER_BIN = "mean_genes_per_bin"

__MRO__ = """
stage RECEIVE_SUBSAMPLED_METRICS(
    in  csv  subsampled_metrics,
    out json saturation_plots,
    src py   "stages/spatial/receive_subsampled_metrics",
)
"""


def main(args, outs):
    if not args.subsampled_metrics:
        outs.saturation_plots = None
        return

    df = pl.read_csv(args.subsampled_metrics)
    if df.is_empty() or BIN_SIZE_UM_KEY not in df.columns:
        outs.saturation_plots = None
        return

    bin_size = df[BIN_SIZE_UM_KEY][0]

    if TOTAL_RAW_READS_KEY in df.columns and SEQUENCING_SATURATION_KEY in df.columns:
        sat_plot = px.line(
            data_frame=df,
            x=TOTAL_RAW_READS_KEY,
            y=SEQUENCING_SATURATION_KEY,
            labels={
                TOTAL_RAW_READS_KEY: "Total Reads",
                SEQUENCING_SATURATION_KEY: "Sequencing Saturation",
            },
            template=PLOTLY_EXPRESS_TEMPLATE,
        )
        sat_plot.update_layout(yaxis_range=[0, 1])

        sat_plot_dict = json.loads(sat_plot.to_json())
        sat_plot_dict.update({"config": pltly.PLOT_CONFIG})
        titled_saturation_plot = {
            "title": VISIUM_HD_RTL_SEQ_SATURATION_PLOT_HELP,
            "inner": sat_plot_dict,
        }
    else:
        titled_saturation_plot = None

    if MEAN_READS_PER_BIN in df.columns and MEAN_GENES_PER_BIN in df.columns:
        genes_plot = px.line(
            data_frame=df,
            x=MEAN_READS_PER_BIN,
            y=MEAN_GENES_PER_BIN,
            labels={
                MEAN_READS_PER_BIN: f"Mean Reads per {bin_size} µm bin",
                MEAN_GENES_PER_BIN: f"Mean Genes per {bin_size} µm bin",
            },
            template=PLOTLY_EXPRESS_TEMPLATE,
        )

        genes_plot_dict = json.loads(genes_plot.to_json())
        genes_plot_dict.update({"config": pltly.PLOT_CONFIG})
        titled_spatial_genes_plot = {
            "title": json.loads(
                "{" + SPATIAL_HD_MEAN_GENE_PLOT_HELP_JSON_STRING.format(bin_size=bin_size) + "}"
            ),
            "inner": genes_plot_dict,
        }
    else:
        titled_spatial_genes_plot = None

    full_saturation_plots = {
        "seq_saturation_plot": titled_saturation_plot,
        "genes_plot": titled_spatial_genes_plot,
    }

    if titled_spatial_genes_plot is None and titled_saturation_plot is None:
        outs.saturation_plots = None
        return

    with open(outs.saturation_plots, "w") as f:
        json.dump(full_saturation_plots, f, indent=4)
