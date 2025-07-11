//! spatial::stages::generate_hd_websummary_cs::summary_tab
#![allow(missing_docs)]

use self::key_metrics::KeyMetricsCard;
use super::end_to_end::EndToEndAlignmentCard;
use serde::{Deserialize, Serialize};
use tenx_websummary::components::{
    Card, CommandLine, PlotlyChart, TableMetric, TermDesc, Title, TwoColumn, WithTitle,
};
use tenx_websummary::HtmlTemplate;

pub mod key_metrics;

pub type CardWithTableMetric = Card<WithTitle<TableMetric>>;

#[derive(Serialize, HtmlTemplate)]
pub struct SummaryTab {
    pub(crate) top: TwoColumn<SummaryTabLeft, SummaryTabRight>,
    pub(crate) run_summary: RunSummaryCard,
    pub(crate) sequencing_saturation: Option<SequencingSaturationCard>,
    pub(crate) genomic_dna: Option<GenomicDnaCard>,
    pub(crate) command_line: CommandLineCard,
}

#[derive(Serialize, HtmlTemplate)]
pub struct SummaryTabLeft {
    pub(crate) key_metrics: KeyMetricsCard,
    pub(crate) mapping_metrics: CardWithTableMetric,
    pub(crate) sequencing_metrics: CardWithTableMetric,
    pub(crate) segmentation_summary: Option<SegmentationCard>,
}

#[derive(Serialize, HtmlTemplate)]
pub struct GenomicDnaCard {
    pub card: Card<WithTitle<GenomicDnaTemplate>>,
}

#[derive(Serialize, HtmlTemplate)]
pub struct GenomicDnaTemplate {
    #[html(row = "1")]
    pub(crate) table: TableMetric,
    #[html(row = "1")]
    pub(crate) plot: PlotlyChart,
}

#[derive(Serialize, HtmlTemplate)]
pub struct SummaryTabRight {
    pub(crate) end_to_end_alignment: EndToEndAlignmentCard,
}
#[derive(Serialize, HtmlTemplate)]
pub struct SegmentationCard {
    pub card: Card<WithTitle<TableMetric>>,
}

impl SegmentationCard {
    pub fn new(segmentation_summary: TableMetric) -> Self {
        Self {
            card: Card::half_width(WithTitle::new(
                Title::new("Cell Segmentation"),
                segmentation_summary,
            )),
        }
    }
}

#[derive(Serialize, HtmlTemplate)]
pub struct RunSummaryCard {
    card: Card<WithTitle<TableMetric>>,
}

impl RunSummaryCard {
    const CARD_TITLE: &'static str = "Sample Metadata";
    pub fn new(run_summary: TableMetric) -> Self {
        Self {
            card: Card::full_width(WithTitle::new(Title::new(Self::CARD_TITLE), run_summary)),
        }
    }
}

#[derive(Serialize, HtmlTemplate)]
pub struct SequencingSaturationCard {
    pub(crate) sequencing_depth: Card<WithTitle<SaturationPlots>>,
}

impl SequencingSaturationCard {
    const CARD_TITLE: &'static str = "Sequencing Depth Evaluation";
}

impl From<SaturationPlots> for SequencingSaturationCard {
    fn from(sequencing_depth: SaturationPlots) -> Self {
        Self {
            sequencing_depth: Card::full_width(WithTitle::new(
                Title::new(Self::CARD_TITLE),
                sequencing_depth,
            )),
        }
    }
}

#[derive(Serialize, HtmlTemplate, Deserialize, Debug)]
pub struct SaturationPlots {
    #[html(row = "1")]
    pub(crate) seq_saturation_plot: Option<WithTitle<PlotlyChart>>,
    #[html(row = "1")]
    pub(crate) genes_plot: Option<WithTitle<PlotlyChart>>,
}

#[derive(Serialize, HtmlTemplate)]
pub struct CommandLineCard {
    card: Card<CommandLine>,
}

impl CommandLineCard {
    pub fn new(title: String, data: Vec<TermDesc>) -> Self {
        Self {
            card: Card::full_width(CommandLine {
                title,
                data,
                show_dark_button_icon: true,
            }),
        }
    }
}
