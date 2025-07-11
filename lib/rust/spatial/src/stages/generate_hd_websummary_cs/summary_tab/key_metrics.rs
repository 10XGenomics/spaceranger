#![allow(missing_docs)]
use crate::stages::compute_bin_metrics::BinMetrics;
use cr_websummary::PrettyMetric;
use serde::Serialize;
use tenx_websummary::components::{Card, HeroMetric, Threshold, Title, WithTitle};
use tenx_websummary::HtmlTemplate;

#[derive(Serialize, HtmlTemplate)]
pub struct KeyMetricsCard {
    card: Card<WithTitle<KeyMetricsTemplate>>,
}

impl KeyMetricsCard {
    const CARD_TITLE: &'static str = "Key Metrics";

    pub fn from_bin_metrics(bin_metrics: &BinMetrics) -> Self {
        Self {
            card: Card::half_width(WithTitle::new(
                Title::new(Self::CARD_TITLE),
                KeyMetricsTemplate::from_bin_metrics(bin_metrics),
            )),
        }
    }
}

#[derive(Serialize, HtmlTemplate)]
pub struct KeyMetricsTemplate {
    #[html(row = "1")]
    bin_8um_spots_under_tissue: HeroMetric,
    #[html(row = "1")]
    bin_8um_mean_read_per_bin: HeroMetric,
    #[html(row = "2")]
    bin_8um_mean_umis_per_bin: HeroMetric,
    #[html(row = "2")]
    total_genes_detected: HeroMetric,
}

impl KeyMetricsTemplate {
    const BIN_8UM_SPOTS_UNDER_TISSUE: &'static str = "Number of 8 µm binned squares under tissue";
    const BIN_8UM_MEAN_READ_PER_BIN: &'static str = "Mean reads per 8 µm bin";
    const BIN_8UM_MEAN_UMIS_PER_BIN: &'static str = "Mean UMIs per 8 µm bin";
    const TOTAL_GENES_DETECTED: &'static str = "Total genes detected";

    pub fn from_bin_metrics(bin_metrics: &BinMetrics) -> Self {
        assert_eq!(bin_metrics.bin_size_um, 8);
        Self {
            bin_8um_spots_under_tissue: HeroMetric::with_threshold(
                Self::BIN_8UM_SPOTS_UNDER_TISSUE,
                PrettyMetric::integer(bin_metrics.bins_under_tissue),
                Threshold::Pass,
            ),
            bin_8um_mean_read_per_bin: HeroMetric::with_threshold(
                Self::BIN_8UM_MEAN_READ_PER_BIN,
                format!("{:.1}", bin_metrics.mean_reads_per_bin),
                Threshold::Pass,
            ),
            bin_8um_mean_umis_per_bin: HeroMetric::with_threshold(
                Self::BIN_8UM_MEAN_UMIS_PER_BIN,
                format!("{:.1}", bin_metrics.mean_umis_per_bin),
                Threshold::Pass,
            ),
            total_genes_detected: HeroMetric::with_threshold(
                Self::TOTAL_GENES_DETECTED,
                PrettyMetric::integer(bin_metrics.total_genes_detected_under_tissue),
                Threshold::Pass,
            ),
        }
    }
}
