//!
//! stage GENERATE_SEGMENT_WEBSUMMARY
//!
#![allow(missing_docs)]

use crate::common_websummary_components::generate_multilayer_chart_from_json;
use crate::HtmlFile;
use anyhow::Result;
use cr_types::constants::{COMMAND_LINE_ENV_DEFAULT_VALUE, COMMAND_LINE_ENV_VARIABLE_NAME};
use martian::{MartianMain, MartianRover};
use martian_derive::{make_mro, MartianStruct};
use martian_filetypes::json_file::JsonFile;
use martian_filetypes::FileTypeRead;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::env;
use tenx_websummary::components::{
    Card, CommandLine, MultiLayerImages, TableMetric, TermDesc, TitleWithTermDesc, WithTitle,
    WsNavBar,
};
use tenx_websummary::{Alert, AlertLevel, HtmlTemplate, SinglePageHtml};
use thousands::Separable;
use websummary_build::build_files;

fn generate_spatial_segmentation_chart_from_json(
    json_file: &JsonFile<Value>,
) -> Result<Card<WithTitle<MultiLayerImages>>> {
    generate_multilayer_chart_from_json(
        json_file,
        "Nucleus Segmentation",
        "Nuclei segmented by the Space Ranger algorithm.",
    )
}

fn create_metric_table_card(
    title: String,
    metric: String,
    description: String,
    rows: Vec<(String, String)>,
) -> Card<WithTitle<TableMetric>> {
    let metric_table = TableMetric { rows };
    let metric_card = WithTitle {
        title: TitleWithTermDesc {
            title: title.clone(),
            data: vec![TermDesc::with_one_desc(&metric, &description)],
        }
        .into(),
        inner: metric_table,
    };
    Card::full_width(metric_card)
}

#[derive(Deserialize)]
struct SegmentNucleiMetrics {
    num_nuclei_too_large: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize, MartianStruct)]
pub struct GenerateSegmentWebsummaryStageInputs {
    sample_id: Option<String>,
    sample_desc: Option<String>,
    spatial_segmentation_chart: Option<JsonFile<Value>>,
    num_nuclei_detected: i64,
    max_nucleus_diameter_px_used: i64,
    segment_nuclei_metrics: JsonFile<SegmentNucleiMetrics>,
}

#[derive(Debug, Clone, Serialize, Deserialize, MartianStruct)]
pub struct GenerateSegmentWebsummaryStageOutputs {
    summary: Option<HtmlFile>,
}

pub struct GenerateSegmentWebsummary;

#[derive(Serialize, HtmlTemplate)]
pub(crate) struct WebSummaryContent {
    pub(crate) segmentation_metrics_table: Card<WithTitle<TableMetric>>,
    pub(crate) segmentation_parameter_table: Card<WithTitle<TableMetric>>,
    pub(crate) spatial_segmentation_plot: Option<Card<WithTitle<MultiLayerImages>>>,
    pub(crate) command_line_card: Card<CommandLine>,
}

#[make_mro]
impl MartianMain for GenerateSegmentWebsummary {
    type StageInputs = GenerateSegmentWebsummaryStageInputs;
    type StageOutputs = GenerateSegmentWebsummaryStageOutputs;
    fn main(&self, args: Self::StageInputs, rover: MartianRover) -> Result<Self::StageOutputs> {
        let nav_bar = WsNavBar {
            pipeline: "Nucleus Segmentation".to_string(),
            id: args.sample_id.unwrap_or_default(),
            description: args.sample_desc.unwrap_or_default(),
        };

        let segmentation_metrics_table = create_metric_table_card(
            "Nucleus Segmentation Metrics".to_string(),
            "Number of Nuclei Detected".to_string(),
            "The total number of nuclei detected by the Space Ranger model.".to_string(),
            vec![(
                "Number of Nuclei Detected".to_string(),
                args.num_nuclei_detected.separate_with_commas(),
            )],
        );

        let segmentation_parameter_table = create_metric_table_card(
            "Nucleus Segmentation Parameters".to_string(),
            "Maximum Nucleus Diameter (pixels)".to_string(),
            "The maximum possible diameter of a nucleus detected in pixels. \
            The segmentation model ensures that nuclei overlapping tile boundaries are detected without being truncated."
                .to_string(),
            vec![(
                "Maximum Nucleus Diameter (pixels)".to_string(),
                format!("{:.1}", args.max_nucleus_diameter_px_used),
            )],
        );

        let spatial_segmentation_plot = args
            .spatial_segmentation_chart
            .map(|x| generate_spatial_segmentation_chart_from_json(&x))
            .transpose()?;

        let cmdline = env::var(COMMAND_LINE_ENV_VARIABLE_NAME)
            .unwrap_or_else(|_| COMMAND_LINE_ENV_DEFAULT_VALUE.to_string());
        let command_line_card = Card::full_width(CommandLine::new(&cmdline)?);

        let web_summary_contents = WebSummaryContent {
            segmentation_metrics_table,
            spatial_segmentation_plot,
            segmentation_parameter_table,
            command_line_card,
        };

        let num_nuclei_too_large = args.segment_nuclei_metrics.read()?.num_nuclei_too_large;
        let mut alerts = vec![];
        if args.num_nuclei_detected == 0 {
            alerts.push(Alert {
                level: AlertLevel::Error,
                title: "No Nuclei Detected".to_string(),
                formatted_value: None,
                message: "No nuclei were detected in the tissue image. \
                Please review your image."
                    .to_string(),
            });
        }
        if num_nuclei_too_large > 0 {
            alerts.push(Alert {
                level: AlertLevel::Warn,
                title: "Large Nuclei Detected".to_string(),
                formatted_value: None,
                message: "At least one nucleus detected is larger than the maximum diameter \
                of nuclei in pixels. Consider increasing the maximum nucleus diameter using the \
                CLI parameter max_nucleus_diameter_px"
                    .to_string(),
            });
        }

        let html = SinglePageHtml::new(
            nav_bar,
            web_summary_contents,
            if alerts.is_empty() {
                None
            } else {
                Some(alerts)
            },
        );
        let summary_html: HtmlFile = rover.make_path("segment_websummary");
        html.generate_html_file_with_build_files(&summary_html, build_files()?)?;
        Ok(GenerateSegmentWebsummaryStageOutputs {
            summary: Some(summary_html),
        })
    }
}
