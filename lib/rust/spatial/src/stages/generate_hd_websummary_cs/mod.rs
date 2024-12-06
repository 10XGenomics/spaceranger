use self::bin_level_tab::clustering::AllBinLevelsData;
use self::bin_level_tab::{compute_bin_metrics_table, BinLevelMetricsTab, BinMetricsCard};
use self::end_to_end::{EndToEndAlignmentCard, EndToEndLayout};
use self::image_alignment_tab::{ImageAlignmentTab, RegistrationCard, TissueFidCard};
use self::spatial_reporter_summary::SdWebSummaryJson;
use self::summary_tab::key_metrics::KeyMetricsCard;
use self::summary_tab::{
    RunSummaryCard, SaturationPlots, SequencingSaturationCard, SummaryTab, SummaryTabLeft,
};
use crate::square_bin_name::SquareBinName;
use crate::stages::compute_bin_metrics::BinMetrics;
use anyhow::Result;
use martian::{MartianMain, MartianRover};
use martian_derive::{make_mro, martian_filetype, MartianStruct};
use martian_filetypes::json_file::JsonFile;
use martian_filetypes::FileTypeRead;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tenx_websummary::components::{
    BlendedImageZoomable, HdEndToEndAlignment, ImageProps, Tabs, TwoColumn,
};
use tenx_websummary::SinglePageHtml;

pub mod bin_level_tab;
pub mod end_to_end;
pub mod image_alignment_tab;
pub mod spatial_reporter_summary;
pub mod summary_tab;

martian_filetype! {HtmlFile, "html"}
martian_filetype! {PngFile, "png"}

struct HdWebSummary {
    summary: SummaryTab,
    image_alignment: ImageAlignmentTab,
    bin_level_metrics: BinLevelMetricsTab,
}

impl HdWebSummary {
    const SUMMARY_TAB_TITLE: &'static str = "Summary";
    const IMAGE_ALIGNMENT_TAB_TITLE: &'static str = "Image Alignment";
    const BIN_LEVEL_METRICS_TAB_TITLE: &'static str = "Bin-Level Metrics";
    fn tabs(self) -> Tabs {
        let HdWebSummary {
            summary,
            image_alignment,
            bin_level_metrics,
        } = self;
        let mut tabs = Tabs::new();
        tabs.push(Self::SUMMARY_TAB_TITLE, summary);
        tabs.push(Self::IMAGE_ALIGNMENT_TAB_TITLE, image_alignment);
        tabs.push(Self::BIN_LEVEL_METRICS_TAB_TITLE, bin_level_metrics);
        tabs
    }
}

#[derive(MartianStruct, Deserialize, Clone)]
pub struct StageInputs {
    sd_web_summary_json: JsonFile<SdWebSummaryJson>,
    end_to_end_alignment_data: JsonFile<HdEndToEndAlignment>,
    bin_level_metrics: HashMap<SquareBinName, JsonFile<BinMetrics>>,
    cluster_plot: Option<JsonFile<AllBinLevelsData>>,
    saturation_plots: Option<JsonFile<SaturationPlots>>,
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
        let web_summary_content = HdWebSummary {
            summary: SummaryTab {
                top: TwoColumn {
                    left: SummaryTabLeft {
                        key_metrics: KeyMetricsCard::from_bin_metrics(
                            &bin_level_metrics[&SquareBinName::new(8)?],
                        ),
                        mapping_metrics: spatial_reporter_summary.mapping_metrics_card(),
                        sequencing_metrics: spatial_reporter_summary.sequencing_metrics_card(),
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
        };

        SinglePageHtml::from_content(web_summary_content.tabs())
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
