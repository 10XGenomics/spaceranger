#![allow(missing_docs)]
use serde::{Deserialize, Serialize};
use tenx_websummary::components::{
    ButtonSelector, ButtonSelectorOption, Card, DifferentialExpressionTable, HdClusteringPlot,
    Title, WithTitle,
};
use tenx_websummary::{AddToSharedResource, HtmlTemplate, SharedResources};

#[derive(Serialize, HtmlTemplate)]
pub struct HdClusteringCard {
    card: Card<WithTitle<HdClusteringTemplate>>,
}

impl HdClusteringCard {
    const CARD_TITLE: &'static str = "Clustering";
    pub fn new(template: HdClusteringTemplate) -> Self {
        Self {
            card: Card::full_width(WithTitle::new(Title::new(Self::CARD_TITLE), template)),
        }
    }
}

#[derive(Serialize, HtmlTemplate)]
pub struct HdClusteringTemplate {
    per_bin_level: ButtonSelector<SingleClusteringTemplate>,
}

#[derive(Serialize, Deserialize, HtmlTemplate)]
pub struct DifferentialExpression {
    title: Title,
    table: DifferentialExpressionTable,
}

#[derive(Serialize, Deserialize, HtmlTemplate)]
pub struct SingleClusteringTemplate {
    plot: HdClusteringPlot,
    differential_expression: DifferentialExpression,
}

#[derive(Deserialize)]
struct SingleClusteringData {
    hd_clustering_plot: HdClusteringPlot,
    differential_expression: DifferentialExpression,
}

impl SingleClusteringData {
    fn template(self, shared_resource: &mut SharedResources) -> SingleClusteringTemplate {
        SingleClusteringTemplate {
            plot: self
                .hd_clustering_plot
                .with_shared_resource(shared_resource),
            differential_expression: self.differential_expression,
        }
    }
}

#[derive(Deserialize)]
struct BinLevelClusterData {
    bin_name: String,
    clustering_data: SingleClusteringData,
}

impl BinLevelClusterData {
    fn template(self, shared_resource: &mut SharedResources) -> SingleClusteringTemplate {
        self.clustering_data.template(shared_resource)
    }
}

#[derive(Deserialize)]
pub struct AllBinLevelsData {
    bin_plots: Vec<BinLevelClusterData>,
}

impl AllBinLevelsData {
    pub fn card(self, shared_resource: &mut SharedResources) -> HdClusteringCard {
        let per_bin_level = ButtonSelector {
            props: Default::default(),
            options: self
                .bin_plots
                .into_iter()
                .map(|data| ButtonSelectorOption {
                    name: data.bin_name.clone(),
                    component: data.template(shared_resource),
                })
                .collect(),
        };
        HdClusteringCard::new(HdClusteringTemplate { per_bin_level })
    }
}
