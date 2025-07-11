#![allow(missing_docs)]
use super::summary_tab::{CommandLineCard, GenomicDnaCard, GenomicDnaTemplate};
use anyhow::Result;
use cr_websummary::WsSample;
use serde::Deserialize;
use serde_json::Value;
use std::collections::HashMap;
use std::path::Path;
use tenx_websummary::components::{
    BlendedImage, Card, ImageProps, PlotlyChart, RawImage, TableMetric, TitleWithTermDesc,
    WithTitle, WsNavBar, ZoomViewer,
};
use tenx_websummary::scrape_json::scrape_json_str_from_html;
use tenx_websummary::{Alert, Alerts, SharedResources};

const RESOURCES_KEY_PREFIX: &str = "_resources_";

#[derive(Deserialize)]
pub struct SdWebSummaryJson {
    summary: SpatialReporterSummaryContent,
    #[serde(rename = "_resources", default)]
    resources: HashMap<String, Value>,
}

impl SdWebSummaryJson {
    pub fn load(ws_path: &Path) -> Result<Self> {
        let ws_json_str = scrape_json_str_from_html(std::fs::read(ws_path)?.as_slice())?;
        Ok(serde_json::from_str(&ws_json_str)?)
    }

    pub fn alerts(&self) -> Vec<Alert> {
        self.summary.alerts.alerts.clone()
    }

    pub fn sequencing_metrics_card(&self) -> Card<WithTitle<TableMetric>> {
        self.summary.summary_tab.sequencing.card()
    }
    pub fn mapping_metrics_card(&self) -> Card<WithTitle<TableMetric>> {
        self.summary.summary_tab.mapping.card()
    }
    pub fn genomic_dna_card(&self) -> Option<GenomicDnaCard> {
        self.summary.analysis_tab.gdna.as_ref().map(Gdna::card)
    }
    pub fn command_line_card(&self) -> CommandLineCard {
        self.summary.summary_tab.cmdline.card()
    }
    pub fn nav_bar(&self) -> WsNavBar {
        WsNavBar {
            pipeline: format!(
                "{} • {}",
                self.summary.sample.command, self.summary.sample.subcommand
            ),
            id: self.summary.sample.id.clone(),
            description: self.summary.sample.description.clone(),
        }
    }
    pub fn resources(&self) -> SharedResources {
        if let Some(gdna) = &self.summary.analysis_tab.gdna {
            SharedResources(
                gdna.resource_keys()
                    .map(|key| (key.to_string(), self.resources[key].clone()))
                    .collect(),
            )
        } else {
            SharedResources::new()
        }
    }
    pub fn run_summary(&self) -> TableMetric {
        self.summary.summary_tab.pipeline_info_table.clone()
    }
    pub fn tissue_fiducial_image(&self) -> Option<RawImage> {
        self.summary
            .summary_tab
            .image
            .zoom_images
            .as_ref()
            .map(|x| {
                RawImage::new(x.big_image.clone())
                    .zoomable(0.5, 10.0)
                    .props(ImageProps::new().pixelated().container_width())
            })
    }
    pub fn regist_image(&self) -> Option<BlendedImage> {
        self.summary
            .summary_tab
            .image
            .regist_images
            .as_ref()
            .map(|x| {
                let mut blended_image = x.clone();
                blended_image.plot_title = None;
                blended_image
            })
    }
    pub fn diagnostics(&self) -> Value {
        self.summary.diagnostics.clone().unwrap_or_default()
    }
}

#[derive(Deserialize)]
struct SpatialReporterSummaryContent {
    summary_tab: SummaryTab,
    #[serde(default)]
    analysis_tab: AnalysisTab,
    #[serde(rename = "alarms", default)]
    alerts: Alerts,
    sample: WsSample,
    diagnostics: Option<Value>,
}

#[derive(Deserialize, Default)]
struct AnalysisTab {
    gdna: Option<Gdna>,
}

#[derive(Deserialize)]
struct Gdna {
    gems: TitledMetricTable,
    plot: PlotlyChart,
}

impl Gdna {
    fn card(&self) -> GenomicDnaCard {
        GenomicDnaCard {
            card: Card::full_width(WithTitle::new(
                self.gems.help.clone().into(),
                GenomicDnaTemplate {
                    table: self.gems.table.clone(),
                    plot: self.plot.clone(),
                },
            )),
        }
    }
    fn resource_keys(&self) -> impl Iterator<Item = &str> {
        self.plot.data.iter().flat_map(|d| {
            ["x", "y"].into_iter().filter_map(|key| {
                d.get(key)
                    .and_then(Value::as_str)
                    .and_then(|key| key.strip_prefix(RESOURCES_KEY_PREFIX))
            })
        })
    }
}

#[derive(Deserialize)]
struct SummaryTab {
    sequencing: TitledMetricTable,
    mapping: TitledMetricTable,
    pipeline_info_table: TableMetric,
    #[serde(default)]
    image: QcImages,
    cmdline: CommandLine,
}

#[derive(Deserialize, Default)]
struct QcImages {
    zoom_images: Option<ZoomViewer>,
    regist_images: Option<BlendedImage>,
}

#[derive(Deserialize)]
struct TitledMetricTable {
    help: TitleWithTermDesc,
    table: TableMetric,
}

impl TitledMetricTable {
    fn card(&self) -> Card<WithTitle<TableMetric>> {
        Card::half_width(WithTitle::new(self.help.clone().into(), self.table.clone()))
    }
}

#[derive(Deserialize)]
struct CommandLine {
    help: TitleWithTermDesc,
}

impl CommandLine {
    fn card(&self) -> CommandLineCard {
        CommandLineCard::new(self.help.title.clone(), self.help.data.clone())
    }
}
