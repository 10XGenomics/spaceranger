#![allow(missing_docs)]
use super::CellSegmentationMetrics;
use crate::common_websummary_components::generate_multilayer_chart_from_json;
use anyhow::Result;
use martian_filetypes::json_file::JsonFile;
use martian_filetypes::FileTypeRead;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tenx_websummary::components::{
    Card, DifferentialExpressionTable, MultiLayerImages, TableMetric, Title, TitleWithHelp,
    TwoColumn, VegaLitePlot, WithTitle,
};
use tenx_websummary::HtmlTemplate;

const DIFF_EXPRESSION_HELP_TEXT: &str = "The differential expression analysis seeks to find, \
for each cluster, features that are more highly expressed in that cluster relative to the rest \
of the sample. Here a differential expression test was performed between each cluster and the \
rest of the sample for each feature. The Log2 fold-change (L2FC) is an estimate of the log2 \
ratio of expression in a cluster to that in all other cells. A value of 1.0 indicates 2-fold \
greater expression in the cluster of interest. The p-value is a measure of the statistical \
significance of the expression difference and is based on a negative binomial test. The p-value \
reported here has been adjusted for multiple testing via the Benjamini-Hochberg procedure. In \
this table you can click on a column to sort by that value. Also, in this table features were \
filtered by (Mean object counts > 1.0) and the top N features by L2FC for each cluster were \
retained. Features with L2FC < 0 or adjusted p-value >= 0.10 were grayed out. The number of top \
features shown per cluster, N, is set to limit the number of table entries shown to 10,000; \
N=10,000/K^2 where K is the number of clusters. N can range from 1 to 50. For the full table, \
please refer to the 'differential_expression.csv' files produced by the pipeline.";

#[derive(Serialize, HtmlTemplate)]
pub(crate) struct NucleusSegmentationTabRight {
    pub(crate) segmentation_umap_chart: Option<Card<WithTitle<VegaLitePlot>>>,
    pub(crate) counts_per_bc_chart: Option<Card<WithTitle<VegaLitePlot>>>,
}

#[derive(Serialize, HtmlTemplate)]
pub(crate) struct NucleusSegmentationTabLeft {
    pub(crate) cell_area_chart: Option<Card<WithTitle<VegaLitePlot>>>,
    pub(crate) features_per_bc_chart: Option<Card<WithTitle<VegaLitePlot>>>,
}
#[derive(Serialize, HtmlTemplate)]
pub(crate) struct NucleusSegmentationTab {
    pub(crate) segmentation_metrics_table: Option<CellMetricsCard>,
    pub(crate) spatial_segmentation_chart: Option<Card<WithTitle<MultiLayerImages>>>,
    pub(crate) middle: TwoColumn<NucleusSegmentationTabLeft, NucleusSegmentationTabRight>,
    pub(crate) graphclust_diffexp_table: Option<Card<WithTitle<DifferentialExpressionTable>>>,
}

#[derive(Deserialize, Serialize, HtmlTemplate)]
pub(crate) struct CellMetricsCard {
    pub card: Card<WithTitle<TableMetric>>,
}

#[allow(dead_code)]
impl CellMetricsCard {
    pub(crate) fn new(segmentation_summary: TableMetric) -> Self {
        Self {
            card: Card::full_width(WithTitle::new(
                Title::new("Cell Segmentation"),
                segmentation_summary,
            )),
        }
    }
}

pub(crate) fn generate_chart_from_json(
    json_file: &JsonFile<Value>,
    title: &str,
    help: &str,
) -> Result<Card<WithTitle<VegaLitePlot>>> {
    Ok(Card::half_width(WithTitle {
        title: TitleWithHelp {
            title: title.to_string(),
            help: help.to_string(),
        }
        .into(),
        inner: VegaLitePlot {
            spec: json_file.read()?,
            actions: Some(Value::Bool(false)),
            renderer: None,
        },
    }))
}

pub(crate) fn generate_cell_area_chart_from_json(
    json_file: &JsonFile<Value>,
) -> Result<Card<WithTitle<VegaLitePlot>>> {
    generate_chart_from_json(
        json_file,
        "Cell Size Distribution",
        "For all cells with transcripts, a histogram of areas in μm². \
        The area is computed from the cell segmentation mask by summing the number of 2μm² bins.",
    )
}

pub(crate) fn generate_features_per_bc_chart_from_json(
    json_file: &JsonFile<Value>,
) -> Result<Card<WithTitle<VegaLitePlot>>> {
    generate_chart_from_json(
        json_file,
        "Genes per Cell",
        "For all cells with UMIs, a histogram of the total number of unique genes found in each cell.",
    )
}

pub(crate) fn generate_counts_per_bc_chart_from_json(
    json_file: &JsonFile<Value>,
) -> Result<Card<WithTitle<VegaLitePlot>>> {
    generate_chart_from_json(
        json_file,
        "UMIs per Cell",
        "For all cells with UMIs, a histogram of the total number of UMIs found in each cell over all genes.",
    )
}

pub(crate) fn generate_spatial_segmentation_chart_from_json(
    json_file: &JsonFile<Value>,
) -> Result<Card<WithTitle<MultiLayerImages>>> {
    generate_multilayer_chart_from_json(
        json_file,
        "Cell Segmentation",
        "Cells segmented by nucleus expansion. Cell boundaries are colored and filled by \
        graph based clustering assignment. Black box represents the Visium HD capture area. \
        The tissue image is cropped to the capture area + 30 pixels.",
    )
}

pub(crate) fn generate_segmentation_umap_chart_from_json(
    json_file: &JsonFile<Value>,
) -> Result<Card<WithTitle<VegaLitePlot>>> {
    generate_chart_from_json(
        json_file,
        "UMAP of Segmented Cells",
        "UMAP representation of segmented cells labeled with graph based clustering. \
              Plot is sampled to 20,000 cells for visualization purposes.",
    )
}

pub(crate) fn generate_graphclust_diffexp_table_from_json(
    json_file: &JsonFile<DifferentialExpressionTable>,
) -> Result<Card<WithTitle<DifferentialExpressionTable>>> {
    Ok(Card::full_width(WithTitle {
        title: TitleWithHelp {
            title: "Top Features by Cluster (Log2 fold-change, p-value)".to_string(),
            help: DIFF_EXPRESSION_HELP_TEXT.to_string(),
        }
        .into(),
        inner: json_file.read()?,
    }))
}

pub(crate) fn generate_cell_segmentation_table_from_json(
    json_file: &JsonFile<CellSegmentationMetrics>,
) -> Result<CellMetricsCard> {
    Ok(CellMetricsCard {
        card: json_file.read()?.card(false),
    })
}

impl NucleusSegmentationTab {
    pub(crate) fn new(
        segmentation_metrics: Option<JsonFile<CellSegmentationMetrics>>,
        cell_area_chart: Option<JsonFile<Value>>,
        features_per_bc_chart: Option<JsonFile<Value>>,
        spatial_segmentation_chart: Option<JsonFile<Value>>,
        counts_per_bc_chart: Option<JsonFile<Value>>,
        segmentation_umap_chart: Option<JsonFile<Value>>,
        graphclust_diffexp_table: Option<JsonFile<DifferentialExpressionTable>>,
    ) -> Result<Option<Self>> {
        if segmentation_metrics.is_none()
            && cell_area_chart.is_none()
            && features_per_bc_chart.is_none()
            && spatial_segmentation_chart.is_none()
            && counts_per_bc_chart.is_none()
            && segmentation_umap_chart.is_none()
            && graphclust_diffexp_table.is_none()
        {
            Ok(None)
        } else {
            Ok(Some(NucleusSegmentationTab {
                segmentation_metrics_table: segmentation_metrics
                    .as_ref()
                    .map(generate_cell_segmentation_table_from_json)
                    .transpose()?,
                spatial_segmentation_chart: spatial_segmentation_chart
                    .as_ref()
                    .map(generate_spatial_segmentation_chart_from_json)
                    .transpose()?,
                middle: TwoColumn {
                    left: NucleusSegmentationTabLeft {
                        cell_area_chart: cell_area_chart
                            .as_ref()
                            .map(generate_cell_area_chart_from_json)
                            .transpose()?,
                        features_per_bc_chart: features_per_bc_chart
                            .as_ref()
                            .map(generate_features_per_bc_chart_from_json)
                            .transpose()?,
                    },
                    right: NucleusSegmentationTabRight {
                        counts_per_bc_chart: counts_per_bc_chart
                            .as_ref()
                            .map(generate_counts_per_bc_chart_from_json)
                            .transpose()?,
                        segmentation_umap_chart: segmentation_umap_chart
                            .as_ref()
                            .map(generate_segmentation_umap_chart_from_json)
                            .transpose()?,
                    },
                },
                graphclust_diffexp_table: graphclust_diffexp_table
                    .as_ref()
                    .map(generate_graphclust_diffexp_table_from_json)
                    .transpose()?,
            }))
        }
    }
}
