use self::clustering::HdClusteringCard;
use crate::square_bin_name::SquareBinName;
use crate::stages::compute_bin_metrics::BinMetrics;
use anyhow::Result;
use cr_websummary::PrettyMetric;
use itertools::Itertools;
use martian_filetypes::json_file::JsonFile;
use martian_filetypes::FileTypeRead;
use serde::Serialize;
use serde_json::Value;
use std::collections::HashMap;
use tenx_websummary::components::{Card, GenericTable, ReactComponent, Title, WithTitle};
use tenx_websummary::HtmlTemplate;

pub mod clustering;

pub fn compute_bin_metrics_table(
    per_bin_metrics: HashMap<SquareBinName, BinMetrics>,
) -> Result<GenericTable> {
    let mut columns = vec![vec![
        "Bin Size (µm)".to_string(),
        "Number of Bins Under Tissue".to_string(),
        "Mean UMI Counts per Bin".to_string(),
        "Mean Genes per Bin".to_string(),
    ]];
    for (bin_name, bin_metrics) in per_bin_metrics
        .into_iter()
        .filter(|(bin_name, _)| bin_name.size_um() != 2)
        .sorted_by_key(|(bin_name, _)| *bin_name)
    {
        columns.push(vec![
            format!("{} µm", bin_name.size_um()),
            PrettyMetric::integer(bin_metrics.bins_under_tissue).to_string(),
            format!("{:.1}", bin_metrics.mean_umis_per_bin),
            format!("{:.1}", bin_metrics.mean_genes_per_bin),
        ]);
    }
    Ok(GenericTable::from_columns(columns, None))
}

#[derive(Serialize, HtmlTemplate)]
pub struct BinLevelMetricsTab {
    pub(crate) bin_metrics: BinMetricsCard,
    pub(crate) clustering: Option<HdClusteringCard>,
}

#[derive(Serialize, HtmlTemplate)]
pub struct BinMetricsCard {
    pub(crate) card: Card<WithTitle<GenericTable>>,
}

impl BinMetricsCard {
    const CARD_TITLE: &'static str = "Bin Metrics Overview";
    pub fn new(bin_metrics: GenericTable) -> Self {
        Self {
            card: Card::full_width(WithTitle::new(Title::new(Self::CARD_TITLE), bin_metrics)),
        }
    }
}

#[derive(Serialize)]
pub struct ClusteringSelector(pub Value);

impl ReactComponent for ClusteringSelector {
    fn component_name() -> &'static str {
        "ClusteringSelector"
    }
}

impl ClusteringSelector {
    pub fn new(json_file: JsonFile<Value>) -> Result<Self> {
        Ok(Self(json_file.read()?))
    }
}
