//! spatial::stages::generate_hd_websummary_cs
#![allow(missing_docs)]

use self::bin_level_tab::clustering::AllBinLevelsData;
use self::bin_level_tab::{compute_bin_metrics_table, BinLevelMetricsTab, BinMetricsCard};
use self::end_to_end::{EndToEndAlignmentCard, EndToEndLayout};
use self::image_alignment_tab::{ImageAlignmentTab, RegistrationCard, TissueFidCard};
use self::spatial_reporter_summary::SdWebSummaryJson;
use self::summary_tab::key_metrics::KeyMetricsCard;
use self::summary_tab::{
    RunSummaryCard, SaturationPlots, SegmentationCard, SequencingSaturationCard, SummaryTab,
    SummaryTabLeft,
};
use crate::square_bin_name::SquareBinName;
use crate::stages::compute_bin_metrics::BinMetrics;
use crate::HtmlFile;
use anyhow::Result;
use martian::{MartianMain, MartianRover};
use martian_derive::{make_mro, martian_filetype, MartianStruct};
use martian_filetypes::json_file::JsonFile;
use martian_filetypes::FileTypeRead;
use nucleus_segmentation_tab::NucleusSegmentationTab;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use summary_tab::CardWithTableMetric;
use tenx_websummary::components::{
    BlendedImageZoomable, DifferentialExpressionTable, HdEndToEndAlignment, ImageProps,
    TableMetric, Tabs, TermDesc, TitleWithTermDesc, TwoColumn, WithTitle,
};
use tenx_websummary::{HtmlTemplate, SinglePageHtml};
use thousands::Separable;

pub mod bin_level_tab;
pub mod end_to_end;
pub mod image_alignment_tab;
pub mod nucleus_segmentation_tab;
pub mod spatial_reporter_summary;
pub mod summary_tab;

martian_filetype! {PngFile, "png"}

#[derive(Serialize)]
struct Diagnostics(Value);

impl HtmlTemplate for Diagnostics {
    fn template(&self, _data_key: Option<String>) -> String {
        String::new()
    }
}

#[derive(Serialize, HtmlTemplate)]
struct HdWebSummaryContent {
    tabs: Tabs,
    diagnostics: Diagnostics,
}

struct HdWebSummaryData {
    summary: SummaryTab,
    image_alignment: ImageAlignmentTab,
    bin_level_metrics: BinLevelMetricsTab,
    nucleus_segmentation_tab: Option<NucleusSegmentationTab>,
    diagnostics: Diagnostics,
}

impl HdWebSummaryData {
    const SUMMARY_TAB_TITLE: &'static str = "Summary";
    const IMAGE_ALIGNMENT_TAB_TITLE: &'static str = "Image Alignment";
    const BIN_LEVEL_METRICS_TAB_TITLE: &'static str = "Bin-Level Metrics";
    const CELL_SEGMENTATION_TAB_TITLE: &'static str = "Cell Segmentation";
    fn content(self) -> HdWebSummaryContent {
        let HdWebSummaryData {
            summary,
            image_alignment,
            bin_level_metrics,
            diagnostics,
            nucleus_segmentation_tab,
        } = self;
        let mut tabs = Tabs::new();
        tabs.push(Self::SUMMARY_TAB_TITLE, summary);
        tabs.push(Self::IMAGE_ALIGNMENT_TAB_TITLE, image_alignment);
        tabs.push(Self::BIN_LEVEL_METRICS_TAB_TITLE, bin_level_metrics);
        if let Some(nucleus_segmentation_tab) = nucleus_segmentation_tab {
            tabs.push(Self::CELL_SEGMENTATION_TAB_TITLE, nucleus_segmentation_tab);
        }
        HdWebSummaryContent { tabs, diagnostics }
    }
}

#[derive(Deserialize)]
pub struct CellSegmentationMetrics {
    filtered_cells: u32,
    fraction_counts_per_cell: f64,
    fraction_reads_in_cells: f64,
    mean_reads_per_cell: f64,
    median_genes_per_cell: f64,
    median_counts_per_cell: f64,
    median_cell_area: f64,
    median_nucleus_area: f64,
    max_nucleus_diameter_px: Option<u32>,
}

impl CellSegmentationMetrics {
    pub fn card(&self, half_width: bool) -> CardWithTableMetric {
        let table = TableMetric {
            rows: vec![
                (
                    "Number of Cells".to_string(),
                    self.filtered_cells.separate_with_commas(),
                ),
                (
                    "Reads in Cells".to_string(),
                    format!("{:.1}%", 100.00 * self.fraction_reads_in_cells),
                ),
                (
                    "UMIs in Cells".to_string(),
                    format!("{:.1}%", 100.00 * self.fraction_counts_per_cell),
                ),
                (
                    "Mean Reads per Cell".to_string(),
                    self.mean_reads_per_cell.separate_with_commas(),
                ),
                (
                    "Median Genes per Cell".to_string(),
                    format!("{:.1}", self.median_genes_per_cell),
                ),
                (
                    "Median UMIs per Cell".to_string(),
                    format!("{:.1}", self.median_counts_per_cell),
                ),
                (
                    "Median Cell Area (μm²)".to_string(),
                    format!("{:.1}", self.median_cell_area),
                ),
                (
                    "Median Nucleus Area (μm²)".to_string(),
                    format!("{:.1}", self.median_nucleus_area),
                ),
                (
                    "Maximum Nucleus Diameter (pixels)".to_string(),
                    if let Some(max_nucleus_diameter_px) = self.max_nucleus_diameter_px {
                        format!("{max_nucleus_diameter_px:.1}")
                    } else {
                        "N/A".to_string()
                    },
                ),
            ],
        };
        let card = WithTitle {
            title: TitleWithTermDesc {
                title: "Cell Segmentation Metrics".into(),
                data: vec![
                    TermDesc::with_one_desc(
                        "Number of Cells Detected",
                        "The total number of cells with >= 1 unique molecular identifier (UMI).",
                    ),
                    TermDesc::with_one_desc(
                        "Reads in Cells",
                        "The total number of reads assigned to cells divided by the total number of reads in the experiment.",
                    ),
                    TermDesc::with_one_desc(
                        "UMIs in Cells",
                        "The percentage of UMIs under tissue within cells.",
                    ),
                    TermDesc::with_one_desc(
                        "Mean Reads per Cell",
                        "The total number of reads assigned to cells divided by the number of cells.",
                    ),
                    TermDesc::with_one_desc(
                        "Median Genes per Cell",
                        "Median number of genes detected per cell.
                        Cells with zero genes detected are excluded from the calculation.",
                    ),
                    TermDesc::with_one_desc(
                        "Median UMIs per Cell",
                        "Median number of unique molecular identifiers (UMIs) detected per cell. \
                        Cells with zero UMIs are excluded from the calculation",
                    ),
                    TermDesc::with_one_desc(
                        "Median Cell Area (μm²)",
                        "Each cell area is calculated by the sum area of 2 μm² squares within the segmented cell.",
                    ),
                    TermDesc::with_one_desc(
                        "Median Nucleus Area (μm²)",
                        "A square is assigned to a nucleus based on the centroid of the square being under the nucleus. \
                        Each nucleus area is calculated by the sum area of 2 μm² squares within the segmented nucleus.",
                    ),
                    TermDesc::with_one_desc(
                        "Maximum Nucleus Diameter (pixels)",
                        "The maximum possible diameter of a nucleus detected in pixels. \
                        The segmentation model ensures that nuclei overlapping tile boundaries are detected without being truncated.",
                    ),
                ],
            }
            .into(),
            inner: table,
        };
        if half_width {
            CardWithTableMetric::half_width(card)
        } else {
            CardWithTableMetric::full_width(card)
        }
    }
}

#[derive(MartianStruct, Deserialize, Clone)]
pub struct StageInputs {
    sd_web_summary_json: JsonFile<SdWebSummaryJson>,
    end_to_end_alignment_data: JsonFile<HdEndToEndAlignment>,
    bin_level_metrics: HashMap<SquareBinName, JsonFile<BinMetrics>>,
    cluster_plot: Option<JsonFile<AllBinLevelsData>>,
    saturation_plots: Option<JsonFile<SaturationPlots>>,
    nucleus_segmentation_graphclust_diffexp: Option<JsonFile<DifferentialExpressionTable>>,
    segmentation_metrics: Option<JsonFile<CellSegmentationMetrics>>,
    cell_area_chart: Option<JsonFile<Value>>,
    features_per_bc_chart: Option<JsonFile<Value>>,
    counts_per_bc_chart: Option<JsonFile<Value>>,
    segmentation_umap_chart: Option<JsonFile<Value>>,
    spatial_segmentation_chart: Option<JsonFile<Value>>,
}

#[derive(MartianStruct, Serialize, Deserialize)]
pub struct StageOutputs {
    web_summary: HtmlFile,
}

pub struct GenerateHdWebsummaryCs;

#[make_mro(volatile = strict)]
impl MartianMain for GenerateHdWebsummaryCs {
    type StageInputs = StageInputs;
    type StageOutputs = StageOutputs;

    fn main(
        &self,
        StageInputs {
            sd_web_summary_json,
            end_to_end_alignment_data,
            bin_level_metrics,
            cluster_plot,
            saturation_plots,
            nucleus_segmentation_graphclust_diffexp,
            segmentation_metrics,
            cell_area_chart,
            features_per_bc_chart,
            counts_per_bc_chart,
            segmentation_umap_chart,
            spatial_segmentation_chart,
        }: Self::StageInputs,
        rover: MartianRover,
    ) -> Result<Self::StageOutputs> {
        let spatial_reporter_summary = sd_web_summary_json.read()?;
        let bin_level_metrics = bin_level_metrics
            .into_iter()
            .map(|(bin_name, json_file)| Ok((bin_name, json_file.read()?)))
            .collect::<Result<HashMap<_, _>>>()?;

        let web_summary_html: HtmlFile = rover.make_path("web_summary.html");

        let mut shared_resource = spatial_reporter_summary.resources();
        let end_to_end_alignment_data = end_to_end_alignment_data.read()?;
        let web_summary_data = HdWebSummaryData {
            summary: SummaryTab {
                top: TwoColumn {
                    left: SummaryTabLeft {
                        key_metrics: KeyMetricsCard::from_bin_metrics(
                            &bin_level_metrics[&SquareBinName::new(8)?],
                        ),
                        mapping_metrics: spatial_reporter_summary.mapping_metrics_card(),
                        sequencing_metrics: spatial_reporter_summary.sequencing_metrics_card(),
                        segmentation_summary: segmentation_metrics.as_ref().map(|x| {
                            let data = x.read().unwrap();
                            SegmentationCard {
                                card: data.card(true),
                            }
                        }),
                    },
                    right: summary_tab::SummaryTabRight {
                        end_to_end_alignment: EndToEndAlignmentCard::new(
                            end_to_end_alignment_data.clone(),
                            EndToEndLayout::SummaryTab,
                            &mut shared_resource,
                        ),
                    },
                },
                run_summary: RunSummaryCard::new(spatial_reporter_summary.run_summary()),
                sequencing_saturation: saturation_plots
                    .map(|x| x.read())
                    .transpose()?
                    .map(SequencingSaturationCard::from),
                genomic_dna: spatial_reporter_summary.genomic_dna_card(),
                command_line: spatial_reporter_summary.command_line_card(),
            },
            image_alignment: ImageAlignmentTab {
                tissue_fid_card: spatial_reporter_summary
                    .tissue_fiducial_image()
                    .map(TissueFidCard::new),
                registration_card: spatial_reporter_summary.regist_image().map(|x| {
                    RegistrationCard::new(
                        BlendedImageZoomable::new(x, 0.5, 10.0)
                            .img_props(
                                ImageProps::new()
                                    .pixelated()
                                    .width(end_to_end_alignment_data.display_width),
                            )
                            .slider_on_bottom(),
                    )
                }),
                end_to_end_alignment: EndToEndAlignmentCard::new(
                    end_to_end_alignment_data.clone(),
                    EndToEndLayout::ImageAlignmentTab,
                    &mut shared_resource,
                ),
            },
            bin_level_metrics: BinLevelMetricsTab {
                bin_metrics: BinMetricsCard::new(compute_bin_metrics_table(bin_level_metrics)?),
                clustering: cluster_plot
                    .map(|json_file| json_file.read().map(|data| data.card(&mut shared_resource)))
                    .transpose()?,
            },
            nucleus_segmentation_tab: NucleusSegmentationTab::new(
                segmentation_metrics,
                cell_area_chart,
                features_per_bc_chart,
                spatial_segmentation_chart,
                counts_per_bc_chart,
                segmentation_umap_chart,
                nucleus_segmentation_graphclust_diffexp,
            )?,
            diagnostics: Diagnostics(spatial_reporter_summary.diagnostics()),
        };

        SinglePageHtml::from_content(web_summary_data.content())
            .nav_bar(spatial_reporter_summary.nav_bar())
            .alerts(spatial_reporter_summary.alerts())
            .resources(shared_resource)
            .generate_html_file_with_build_files(
                &web_summary_html,
                websummary_build::build_files()?,
            )?;

        Ok(StageOutputs {
            web_summary: web_summary_html,
        })
    }
}
