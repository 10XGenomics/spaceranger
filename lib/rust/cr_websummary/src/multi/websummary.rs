// There are a **lot** of metrics which need to be imported here.
#![allow(missing_docs)]
#![allow(clippy::wildcard_imports)]
use super::metrics::MetricTier;
use super::websummary_vdj::*;
use crate::alert::AlertLevel;
use crate::multi::svg::SvgGraph;
use crate::{
    Alert, AlertContext, AlertSpec, ChartWithHelp, GenericTable, MakePretty, RawChartWithHelp, Tab,
    TableRow, WsSample,
};
use anyhow::Result;
use cr_types::websummary::{AlertConfig, AlertIfMetricIs, MetricEtlConfig};
use cr_types::{
    AlignerParam, BarcodeMultiplexingType, FeatureBarcodeType, LibraryType, TargetingMethod,
};
use csv::Writer;
use itertools::Itertools;
use metric::TxHashMap;
use serde::{Deserialize, Serialize};
use serde_json::value::Value;
use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::fmt::Display;
use std::path::Path;
use tenx_websummary::components::{
    DifferentialExpressionTable, TableMetric, VegaLitePlot, WithTitle,
};
use websummary_derive::Alert;

const CELL_ANNOTATION_FAILURE_TITLE: &str = "No cell type annotations produced!";

const CELL_ANNOTATION_FAILURE_MESSAGE: &str = r#"<p>Please check your cellranger logs.
If you wish to attempt cell type annotation again please use
<a href="https://www.10xgenomics.com/support/software/cloud-analysis/latest/tutorials/CA-cell-annotation-pipeline">cellranger annotate</a>.
</p>"#;
const CELL_ANNOTATION_DE_WARN_TITLE: &str = "Cell type differential expression not run";
const CELL_ANNOTATION_DE_WARN_MESSAGE: &str = "Too few cell types to run differential expression.";
pub const CELL_ANNOTATION_ADVERTISEMENT_STRING: &str = r"<p><b>Automated cell type annotation is now available for Cell Ranger!</b><br>
For details on how to run cell type annotation and species we have it available for, visit our
<a href='https://www.10xgenomics.com/support/software/cell-ranger/latest/getting-started/cr-what-is-cell-ranger'
target='_blank' title='Cell Type Annotation' rel='noopener noreferrer'>support page</a>.</p>";

/// The threshold to trigger a web summary alert when an unexpected probe barcode is observed or
/// an expected probe barcode is not observed.
pub const UMI_PER_PROBE_BARCODE_BACKGROUND_THRESHOLD: f64 = 0.005;

const METRICS_SUMMARY_CSV_HEADER: [&str; 6] = [
    "Category",
    "Library Type",
    "Grouped By",
    "Group Name",
    "Metric Name",
    "Metric Value",
];

#[derive(Clone, Serialize, Deserialize)]
pub struct JsonMetricSummary {
    pub key: String,
    pub value: Value,
    pub pretty_value: String,
    pub category: MetricTier,
    /// FIXME: rename this field to section, it isn't really the same as library type.
    #[serde(rename = "library_type")]
    pub section: Section,
    pub config: MetricEtlConfig,
    /// For metrics of the same key that may have multiple values,
    /// this optional grouping key can be used to distinguish the different
    /// instances from each other, and associate them with related data.
    /// For example, in a table of probe barcode metrics, this would be
    /// the probe barcode associated with a particular metric.
    ///
    /// This is a stopgap solution - improving the namespacing of metrics in the
    /// first place would be a better long-term solution.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub grouping_key: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// If this metric has a grouping key, this is the header to use in
    /// the output metrics CSV table. This is the same as the display name of
    /// the metric from which grouping_key originated.
    pub grouping_header: Option<String>,
    /// The collection of alerts triggered for this metric.
    pub alerts: Vec<AlertSpec>,
}

impl JsonMetricSummary {
    /// Construct a JSON metric summary from input data.
    ///
    /// This includes computing any alerts that were triggered.
    /// We pre-format the value for later printing.
    #[allow(clippy::too_many_arguments)]
    pub fn new<T: Serialize + MakePretty>(
        key: String,
        value: Option<&T>,
        category: MetricTier,
        section: Section,
        config: MetricEtlConfig,
        grouping_key: Option<String>,
        grouping_header: Option<String>,
        ctx: &AlertContext,
    ) -> Self {
        let pretty_value = value.map_or_else(|| "---".to_string(), MakePretty::make_pretty);
        Self {
            alerts: Self::construct_alerts(&key, value, &config.alerts, ctx),
            value: serde_json::to_value(value).unwrap(),
            pretty_value,
            key,
            category,
            section,
            config,
            grouping_key,
            grouping_header,
        }
    }

    /// Construct alerts for the provided data.
    ///
    /// This oddball static method is used as a shim to avoid code duplication
    /// in the make_tables macro until it goes away.
    pub fn construct_alerts<T: MakePretty>(
        key: &str,
        val: Option<&T>,
        alerts: &[AlertConfig],
        ctx: &AlertContext,
    ) -> Vec<AlertSpec> {
        if alerts.is_empty() {
            return vec![];
        }
        let Some(val) = val else {
            return vec![];
        };
        let (as_f64, pretty) = (val.as_f64(), val.make_pretty());
        let triggered = |thresh: Option<f64>, cond: AlertIfMetricIs| {
            let Some(thresh) = thresh else {
                return false;
            };
            match cond {
                AlertIfMetricIs::GreaterThanOrEqual => as_f64 >= thresh,
                AlertIfMetricIs::LessThanOrEqual => as_f64 <= thresh,
            }
        };
        alerts
            .iter()
            .filter(|config| {
                let mut matches_conditions = true;
                if let Some(required_include_introns_setting) = config.conditions.include_introns {
                    matches_conditions = ctx.include_introns == required_include_introns_setting;
                }
                if let Some(required_is_rtl_setting) = config.conditions.is_rtl {
                    matches_conditions = ctx.is_rtl == required_is_rtl_setting;
                }
                matches_conditions
            })
            .filter_map(|config| {
                let cond = config.alert_if().unwrap();
                if triggered(config.error_threshold, cond) {
                    return Some(AlertSpec {
                        level: AlertLevel::Error,
                        title: config.error_title(key).to_string(),
                        formatted_value: pretty.clone(),
                        message: config.detail.clone(),
                    });
                }
                if triggered(config.warn_threshold, cond) {
                    return Some(AlertSpec {
                        level: AlertLevel::Warn,
                        title: config.warn_title(key).to_string(),
                        formatted_value: pretty.clone(),
                        message: config.detail.clone(),
                    });
                }
                None
            })
            .collect()
    }
}

/// Wrap a collection of metrics to adapt into the Alert/ToCsvRows/ToJsonSummary traits.
#[derive(Clone, Default)]
pub struct MetricsTraitWrapper(pub Vec<JsonMetricSummary>);

impl MetricsTraitWrapper {
    pub fn to_csv_rows(&self) -> Vec<Vec<String>>
    where
        Self: Sized,
    {
        self.0
            .iter()
            .filter_map(|metric| {
                // Skip rows for which the grouping header matches the metric header,
                // since the value in this row is included as a column in the
                // rest of the metrics from this group.
                if let Some(grouping_header) = &metric.grouping_header {
                    if grouping_header == &metric.config.header {
                        return None;
                    }
                }
                Some(vec![
                    metric.category.to_string(),
                    metric.section.to_string(),
                    metric.grouping_header.clone().unwrap_or_default(),
                    metric.grouping_key.clone().unwrap_or_default(),
                    metric.config.header.clone(),
                    metric.pretty_value.clone(),
                ])
            })
            .collect()
    }
}

impl Alert for MetricsTraitWrapper {
    fn alerts(&self, _ctx: &AlertContext) -> Vec<AlertSpec> {
        self.0
            .iter()
            .flat_map(|metric| &metric.alerts)
            .cloned()
            .collect()
    }
}

#[derive(Debug, Serialize, PartialEq, Eq, Clone, Default)]
pub struct SampleHeaderInfo {
    #[serde(rename = "Sample ID")]
    pub sample_id: String,
    #[serde(rename = "Sample Description")]
    pub sample_desc: String,
    #[serde(rename = "Pipeline Version")]
    pub pipeline_version: String,
}

#[derive(Debug, Serialize, PartialEq, Eq, Clone, Default)]
pub struct LibraryHeaderInfo {
    #[serde(rename = "Run ID")]
    pub run_id: String,
    #[serde(rename = "Run Description")]
    pub run_desc: String,
    #[serde(rename = "Pipeline Version")]
    pub pipeline_version: String,
}

#[derive(Debug, Serialize, PartialEq, Eq, Clone)]
#[serde(into = "GenericTable")]
pub struct CountParametersTable {
    pub chemistry: String,
    pub introns_included: bool,
    pub reference_path: Option<String>,
    pub transcriptome: String,
    pub feature_ref_path: Option<String>,
    pub cmo_set_path: Option<String>,
    pub target_set_name: Option<String>,
    pub targeting_method: Option<TargetingMethod>,
    pub filter_probes: Option<bool>,
    pub disable_ab_aggregate_detection: bool,
    pub disable_high_occupancy_gem_detection: bool,
    pub num_genes_on_target: Option<usize>,
    pub library_type: LibraryType,
    pub throughput: Option<String>,
    pub tenx_cmos: Option<bool>,
    pub aligner: AlignerParam,
    pub antigen_negative_control: bool,
    // Things we may need to alert on.
    pub dropped_tags: Vec<String>,
    pub probe_barcodes_high_gem_overlap: Vec<String>,
    pub unspecified_probe_barcodes_detected: Vec<String>,
    pub specified_probe_barcodes_missing: Vec<String>,
    pub mismatched_probe_barcode_pairings: Option<MismatchedProbeBarcodePairings>,
}

impl From<CountParametersTable> for GenericTable {
    fn from(info: CountParametersTable) -> GenericTable {
        let CountParametersTable {
            chemistry,
            introns_included,
            reference_path,
            transcriptome,
            feature_ref_path,
            cmo_set_path,
            target_set_name,
            targeting_method,
            filter_probes,
            disable_ab_aggregate_detection,
            disable_high_occupancy_gem_detection,
            num_genes_on_target,
            library_type,
            throughput,
            aligner,
            antigen_negative_control,
            // Supporting data that should not appear in the table itself.
            tenx_cmos: _,
            dropped_tags: _,
            probe_barcodes_high_gem_overlap: _,
            unspecified_probe_barcodes_detected: _,
            specified_probe_barcodes_missing: _,
            mismatched_probe_barcode_pairings: _,
        } = info;

        let chemistry_with_throughput = match throughput {
            Some(ref tp) => {
                if tp == "HT" && !["HT", "MT", "LT"].iter().any(|x| chemistry.contains(x)) {
                    format!("{chemistry} HT")
                } else {
                    chemistry
                }
            }
            None => chemistry,
        };

        let mut rows = vec![TableRow::two_col("Chemistry", chemistry_with_throughput)];

        // GEX tab
        if library_type.is_gex() {
            rows.extend([
                TableRow::two_col(
                    "Reference Path",
                    reference_path.unwrap_or_else(|| "None".to_string()),
                ),
                TableRow::two_col("Transcriptome", transcriptome),
            ]);
            if aligner != AlignerParam::Hurtle {
                rows.push(TableRow::two_col("Include Introns", introns_included));
            }
            if targeting_method.is_some() {
                rows.push(TableRow::two_col(
                    "Probe Set Name",
                    target_set_name.unwrap(),
                ));
            }
            if let Some(n_genes) = num_genes_on_target {
                rows.push(TableRow::two_col(
                    "Number of Genes Targeted",
                    n_genes.make_pretty(),
                ));
            }
            if filter_probes == Some(false) {
                rows.push(TableRow::two_col("Filter Probes", "Disabled"));
            }
        }
        // Feature Tabs
        if let (LibraryType::FeatureBarcodes(_), Some(feature_ref)) =
            (library_type, feature_ref_path)
        {
            rows.push(TableRow::two_col("Feature Reference", feature_ref));
        }

        let fb_type = library_type.feature_barcode_type();
        if let (Some(cmo_set_path), Some(FeatureBarcodeType::Multiplexing)) =
            (cmo_set_path, fb_type)
        {
            rows.push(TableRow::two_col("CMO Set", cmo_set_path));
        }

        if fb_type == Some(FeatureBarcodeType::Antigen) {
            rows.push(TableRow::two_col(
                "Control Specified",
                antigen_negative_control,
            ));
        }
        if disable_ab_aggregate_detection {
            rows.push(TableRow::two_col(
                "Antibody Aggregate Filtering",
                "Disabled",
            ));
        }
        if disable_high_occupancy_gem_detection {
            rows.push(TableRow::two_col(
                "High-occupancy GEM Filtering",
                "Disabled",
            ));
        }

        GenericTable {
            header: None,
            rows,
            grouping_header: None,
        }
    }
}

impl Alert for CountParametersTable {
    fn alerts(&self, ctx: &AlertContext) -> Vec<AlertSpec> {
        let mut alerts = vec![];

        if self.library_type.is_fb_type(FeatureBarcodeType::Antigen)
            && !self.antigen_negative_control
        {
            alerts.push(AlertSpec {
                level: AlertLevel::Info,
                title: "No antigen negative control specified".to_string(),
                formatted_value: String::default(),
                message: r"Your data has been analyzed without a negative control. Antigen specificity scores and related outputs will not be generated.".into(),
            });
        }
        if let Some(tp) = &self.throughput {
            if self.chemistry.ends_with("HT") && tp != "HT" {
                alerts.push(AlertSpec {
                    level: AlertLevel::Warn,
                    title: "Inconsistent throughput detected".to_string(),
                    formatted_value: "HT".to_string(),
                    message: "High throughput (HT) chemistry was specified, but Cell Ranger detected that data was not HT. This may be due to low sequencing depth, sample quality issues, or incorrect chemistry input to Cell Ranger. If applicable, cell multiplexing results may not be accurate.".to_string(),
                });
            }
        }
        if self.tenx_cmos == Some(false) {
            alerts.push(AlertSpec {
                level: AlertLevel::Warn,
                title: "Unsupported multiplexing tag used".to_string(),
                formatted_value: String::default(),
                message: "Multiplexing performance cannot be guaranteed".to_string(),
            });
        }
        if ctx.is_fiveprime
            && ctx.multiplexing_method
                == Some(BarcodeMultiplexingType::CellLevel(cr_types::CellLevel::CMO))
        {
            alerts.push(AlertSpec {
                level: AlertLevel::Warn,
                title: "Unsupported combination of 5' chemistry with multiplexing".to_string(),
                formatted_value: String::default(),
                message: "Multiplexing performance cannot be guaranteed".to_string(),
            });
        }
        if ctx.multiplexing_method
            == Some(BarcodeMultiplexingType::CellLevel(
                cr_types::CellLevel::Hashtag,
            ))
        {
            alerts.push(AlertSpec {
                level: AlertLevel::Warn,
                title: "Unsupported multiplexing tag used".to_string(),
                formatted_value: String::default(),
                message: "Multiplexing performance cannot be guaranteed".to_string(),
            });
        }
        if ctx.is_arc_chemistry {
            alerts.push(AlertSpec {
                level: AlertLevel::Warn,
                title: "Unsupported workflow used".to_string(),
                formatted_value: String::default(),
                message: "Multiome Gene Expression only analysis is not a supported workflow. Results may vary.".to_string(),
            });
        }

        if !self.dropped_tags.is_empty() {
            let filtered_tags = self.dropped_tags.join(", ");
            let message = format!(
                "Tag(s) {filtered_tags} were identified as contaminants.  This will happen if \
            any of the following conditions are met.  The total UMI count for that tag is < 1,000, \
            the sum of UMIs for a tag is less than 2% of the maximum UMI count for all other tags, \
            or the pearson correlation with a tag and another was >0.5 and it had fewer UMIs than the \
            correlated tag. \
            You should check if your input CSV contained the right list of tags used."
            );
            alerts.push(AlertSpec {
                level: AlertLevel::Warn,
                title: "Contaminant tags detected".to_string(),
                formatted_value: String::default(),
                message,
            });
        }
        if !self.probe_barcodes_high_gem_overlap.is_empty() {
            let total_high_overlap_pairs = self.probe_barcodes_high_gem_overlap.len();

            const MAX_DISPLAY_PAIRS: usize = 5;

            let message = if total_high_overlap_pairs <= MAX_DISPLAY_PAIRS {
                let probe_pairs = self.probe_barcodes_high_gem_overlap.join(", ");
                format!(
                    "The following pair(s) of probe barcode IDs were identified as having much \
                     higher than expected GEM barcode overlap: {probe_pairs}. This may be due to \
                     cell clumping or indicate that probes with two or more different probe \
                     barcodes were used in the same hybridization reaction. \
                     See frp_gem_barcode_overlap.csv for a complete set of overlap metrics."
                )
            } else {
                // Provide a modified alert if the number of pairs is high
                let probe_pairs =
                    self.probe_barcodes_high_gem_overlap[0..MAX_DISPLAY_PAIRS].join(", ");
                let remaining_pairs = total_high_overlap_pairs - MAX_DISPLAY_PAIRS;
                format!(
                    "The following pair(s) of probe barcode IDs were identified as having much \
                     higher than expected GEM barcode overlap: {probe_pairs}... [{remaining_pairs} \
                     additional pair(s)]. \
                     This may be due to cell clumping or indicate that probes with two or more \
                     different barcodes were used in the same hybridization reaction. \
                     See frp_gem_barcode_overlap.csv for a complete set of overlap metrics."
                )
            };

            alerts.push(AlertSpec {
                level: AlertLevel::Error,
                title: "Probe barcodes with high GEM barcode overlap detected".to_string(),
                formatted_value: format!("{total_high_overlap_pairs} probe barcode pair(s)"),
                message,
            });
        }

        if !self.unspecified_probe_barcodes_detected.is_empty() {
            let barcodes_string = self.unspecified_probe_barcodes_detected.join(", ");
            let message = format!(
                "The following probe barcodes ID(s) were not specified in config CSV but account \
                 for at least {:.2}% of UMIs: {}. This could result from omitting one or more \
                 probe barcode IDs used in the experiment or from accidental contamination coming \
                 from a barcode that was not used in this experiment. \
                 Please check your config CSV file.",
                UMI_PER_PROBE_BARCODE_BACKGROUND_THRESHOLD * 100.0,
                barcodes_string
            );
            alerts.push(AlertSpec {
                level: AlertLevel::Error,
                title: "Probe barcode ID(s) missing from config CSV".to_string(),
                formatted_value: String::default(),
                message,
            });
        }

        if !self.specified_probe_barcodes_missing.is_empty() {
            let barcodes_string = self.specified_probe_barcodes_missing.join(", ");
            let message = format!(
                "The following probe barcodes ID(s) were specified in config CSV but account for \
                 less than {:.2}% of UMIs: {}. This could result from adding a probe barcode to \
                 your config CSV when it was not actually used in the experiment. \
                 Please check your config CSV file.",
                UMI_PER_PROBE_BARCODE_BACKGROUND_THRESHOLD * 100.0,
                barcodes_string
            );
            alerts.push(AlertSpec {
                level: AlertLevel::Warn,
                title: "Specified probe barcode ID(s) not detected".to_string(),
                formatted_value: String::default(),
                message,
            });
        }

        if let Some(mismatched_probe_bcs) = &self.mismatched_probe_barcode_pairings {
            alerts.push(AlertSpec {
                level: AlertLevel::Warn,
                title: "Mismatch between declared and detected probe barcode pairings".to_string(),
                formatted_value: mismatched_probe_bcs.formatted_value(),
                message: mismatched_probe_bcs.to_string(),
            });
        }

        alerts
    }
}

// Websummary data structures may have shared _resources that they access by key
// If a websummary struct has any _resources they are emptied and bubbled up and stored in the top-level _resources
pub type MultiSharedResource = TxHashMap<String, Value>;

impl Alert for MultiSharedResource {}

#[derive(Serialize, Clone)]
pub struct MultiWebSummary {
    // FIXME delete this CELLRANGER-8423
    pub sample: WsSample,
    pub library: MultiWebSummaryLibraryData,
    pub per_sample: Vec<MultiWebSummarySampleData>,
    pub experimental_design: ExperimentalDesign,
    pub diagnostics: MultiDiagnostics,
    pub sample_diagnostics: Vec<SampleDiagnostics>,
    // FIXME delete this CELLRANGER-8423
    #[serde(rename = "_resources")]
    pub resources: MultiSharedResource,
}

#[derive(Serialize, Clone, Default)]
#[allow(non_snake_case)]
pub struct MultiDiagnostics {
    pub corrected_bc_frac: Option<Value>,
    pub corrected_bc_in_gel_bead_frac: Option<Value>,
    pub corrected_bc_in_probe_frac: Option<Value>,
    pub ANTIBODY_corrected_bc_frac: Option<Value>,
    pub ANTIBODY_corrected_bc_in_gel_bead_frac: Option<Value>,
    pub ANTIBODY_corrected_bc_in_probe_frac: Option<Value>,
    pub i1_bases_with_q30_frac: Option<Value>,
    pub i2_bases_with_q30_frac: Option<Value>,
    pub low_support_umi_reads_frac: Option<Value>,
    pub tag_contaminant_info: Option<Value>,
    pub tso_frac: Option<Value>,
    pub probe_barcode_overlap_coefficients: Option<Value>,
    pub fraction_reads_high_occupancy_gems: Option<Value>,
    pub high_occupancy_probe_barcode_count_threshold: Option<Value>,
    pub unknown_feature_barcode_seqs: HashMap<String, Value>,
}

#[derive(Serialize, Clone, Default)]
pub struct SampleDiagnostics {
    pub vdj_t: Option<VdjDiagnostics>,
    pub vdj_b: Option<VdjDiagnostics>,
    pub vdj_t_gd: Option<VdjDiagnostics>,
}

impl MultiWebSummary {
    pub fn to_csv(&self, filename: &Path, sample_index: usize) -> Result<()> {
        let mut rows = self.library.data.to_csv_rows();
        rows.extend(self.per_sample[sample_index].data.to_csv_rows());
        rows.sort();
        rows.dedup();

        let mut writer = Writer::from_path(filename)?;
        writer.write_record(METRICS_SUMMARY_CSV_HEADER)?;
        for row in rows {
            writer.write_record(row)?;
        }

        writer.flush()?;
        Ok(())
    }
}

#[derive(Serialize, Clone)]
pub struct MultiWebSummaryLibraryData {
    pub data: LibraryWebSummary,
    /// All unique library types used in the analysis.
    pub types: Vec<LibraryType>,
    pub metrics: Vec<JsonMetricSummary>,
}

#[derive(Serialize, Clone)]
pub struct MultiWebSummarySampleData {
    pub data: SampleWebSummary,
    pub metrics: Vec<JsonMetricSummary>,
}

#[derive(Serialize, Clone)]
pub struct ExperimentalDesign {
    pub svg: SvgGraph,
    pub csv: String,
    pub multiplexing_method: Option<BarcodeMultiplexingType>,
    /// True if this experiment is using a Flex kit.
    pub is_rtl: bool,
    /// A flag indicating if this is a barnyard analysis.
    pub is_barnyard: bool,
}

#[derive(
    Default, PartialEq, Eq, Debug, Clone, Copy, Serialize, Deserialize, strum_macros::Display,
)]
pub enum Section {
    #[default]
    #[serde(rename = "Gene Expression")]
    #[strum(serialize = "Gene Expression")]
    Gex,
    #[serde(rename = "VDJ T")]
    #[strum(serialize = "VDJ T")]
    VdjT,
    #[serde(rename = "VDJ T GD")]
    #[strum(serialize = "VDJ T GD")]
    VdjTGd,
    #[serde(rename = "VDJ B")]
    #[strum(serialize = "VDJ B")]
    VdjB,
    #[serde(rename = "Antibody Capture")]
    #[strum(serialize = "Antibody Capture")]
    Antibody,
    #[serde(rename = "Antigen Capture")]
    #[strum(serialize = "Antigen Capture")]
    Antigen,
    #[serde(rename = "CRISPR Guide Capture")]
    #[strum(serialize = "CRISPR Guide Capture")]
    Crispr,
    #[serde(rename = "Custom Feature")]
    #[strum(serialize = "Custom Feature")]
    Custom,
    #[serde(rename = "Multiplexing Capture")]
    #[strum(serialize = "Multiplexing Capture")]
    Cmo,
    #[serde(rename = "Cell Annotation")]
    #[strum(serialize = "Cell Annotation")]
    CellAnnotation,
    #[serde(rename = "Hashtag")]
    #[strum(serialize = "Hashtag")]
    Hashtag,
}

#[derive(Serialize, Clone, Default)]
pub struct LibraryWebSummary {
    pub header_info: LibraryHeaderInfo,
    pub gex_tab: Option<Tab<LibraryGexWebSummary>>,
    pub vdj_t_tab: Option<Tab<LibraryVdjWebSummary>>,
    pub vdj_t_gd_tab: Option<Tab<LibraryVdjWebSummary>>,
    pub vdj_b_tab: Option<Tab<LibraryVdjWebSummary>>,
    pub antibody_tab: Option<Tab<LibraryAntibodyOrAntigenWebSummary>>,
    pub antigen_tab: Option<Tab<LibraryAntibodyOrAntigenWebSummary>>,
    pub crispr_tab: Option<Tab<LibraryCrisprWebSummary>>,
    pub custom_feature_tab: Option<Tab<LibraryCustomFeatureWebSummary>>,
    pub cmo_tab: Option<Tab<LibraryCmoWebSummary>>,
    pub hashtag_tab: Option<Tab<LibraryHashtagWebSummary>>,
    #[serde(rename = "_resources")]
    pub resources: MultiSharedResource,
}

fn tab_apply<T: Alert, F: FnOnce(&T) -> U, U: Default>(tab: &Option<Tab<T>>, f: F) -> U {
    tab.as_ref().map(|t| f(&t.content)).unwrap_or_default()
}

impl LibraryWebSummary {
    pub fn to_csv_rows(&self) -> Vec<Vec<String>> {
        [
            tab_apply(&self.gex_tab, |t| t.metrics.to_csv_rows()),
            tab_apply(&self.vdj_t_tab, |t| t.metrics.to_csv_rows()),
            tab_apply(&self.vdj_t_gd_tab, |t| t.metrics.to_csv_rows()),
            tab_apply(&self.vdj_b_tab, |t| t.metrics.to_csv_rows()),
            tab_apply(&self.antibody_tab, |t| t.metrics.to_csv_rows()),
            tab_apply(&self.antigen_tab, |t| t.metrics.to_csv_rows()),
            tab_apply(&self.crispr_tab, |t| t.metrics.to_csv_rows()),
            tab_apply(&self.custom_feature_tab, |t| t.metrics.to_csv_rows()),
            tab_apply(&self.cmo_tab, |t| t.metrics.to_csv_rows()),
            tab_apply(&self.hashtag_tab, |t| t.metrics.to_csv_rows()),
        ]
        .concat()
    }

    pub fn to_json_summary(&self) -> Vec<JsonMetricSummary> {
        [
            tab_apply(&self.gex_tab, |t| t.metrics.0.clone()),
            tab_apply(&self.vdj_t_tab, |t| t.metrics.0.clone()),
            tab_apply(&self.vdj_t_gd_tab, |t| t.metrics.0.clone()),
            tab_apply(&self.vdj_b_tab, |t| t.metrics.0.clone()),
            tab_apply(&self.antibody_tab, |t| t.metrics.0.clone()),
            tab_apply(&self.antigen_tab, |t| t.metrics.0.clone()),
            tab_apply(&self.crispr_tab, |t| t.metrics.0.clone()),
            tab_apply(&self.custom_feature_tab, |t| t.metrics.0.clone()),
            tab_apply(&self.cmo_tab, |t| t.metrics.0.clone()),
            tab_apply(&self.hashtag_tab, |t| t.metrics.0.clone()),
        ]
        .concat()
    }
}

#[derive(Serialize, Clone, Default)]
pub struct SampleWebSummary {
    pub header_info: SampleHeaderInfo,
    pub gex_tab: Option<Tab<SampleGexWebSummary>>,
    pub vdj_t_tab: Option<Tab<SampleVdjWebSummary>>,
    pub vdj_t_gd_tab: Option<Tab<SampleVdjWebSummary>>,
    pub vdj_b_tab: Option<Tab<SampleVdjWebSummary>>,
    pub antibody_tab: Option<Tab<SampleAntibodyWebSummary>>,
    pub antigen_tab: Option<Tab<SampleAntigenWebSummary>>,
    pub crispr_tab: Option<Tab<SampleCrisprWebSummary>>,
    pub custom_feature_tab: Option<Tab<SampleCustomFeatureWebSummary>>,
    pub cell_annotation_tab: Option<Tab<SampleCellAnnotationWebSummary>>,
}

impl SampleWebSummary {
    pub fn to_csv_rows(&self) -> Vec<Vec<String>> {
        [
            tab_apply(&self.gex_tab, |t| t.metrics.to_csv_rows()),
            tab_apply(&self.vdj_t_tab, |t| t.metrics.to_csv_rows()),
            tab_apply(&self.vdj_t_gd_tab, |t| t.metrics.to_csv_rows()),
            tab_apply(&self.vdj_b_tab, |t| t.metrics.to_csv_rows()),
            tab_apply(&self.antibody_tab, |t| t.metrics.to_csv_rows()),
            tab_apply(&self.antigen_tab, |t| t.metrics.to_csv_rows()),
            tab_apply(&self.crispr_tab, |t| t.metrics.to_csv_rows()),
            tab_apply(&self.custom_feature_tab, |t| t.metrics.to_csv_rows()),
        ]
        .concat()
    }

    pub fn to_json_summary(&self) -> Vec<JsonMetricSummary> {
        [
            tab_apply(&self.gex_tab, |t| t.metrics.0.clone()),
            tab_apply(&self.vdj_t_tab, |t| t.metrics.0.clone()),
            tab_apply(&self.vdj_t_gd_tab, |t| t.metrics.0.clone()),
            tab_apply(&self.vdj_b_tab, |t| t.metrics.0.clone()),
            tab_apply(&self.antibody_tab, |t| t.metrics.0.clone()),
            tab_apply(&self.antigen_tab, |t| t.metrics.0.clone()),
            tab_apply(&self.crispr_tab, |t| t.metrics.0.clone()),
            tab_apply(&self.custom_feature_tab, |t| t.metrics.0.clone()),
        ]
        .concat()
    }
}

impl<T: Alert> Alert for Option<T> {
    fn alerts(&self, ctx: &AlertContext) -> Vec<AlertSpec> {
        match self {
            Some(t) => t.alerts(ctx),
            None => vec![],
        }
    }
}

#[derive(Serialize, Clone, Alert)]
pub struct LibraryGexWebSummary {
    pub parameters_table: CountParametersTable,
    pub sequencing_saturation_plot: ChartWithHelp,
    pub median_genes_per_cell_plot: ChartWithHelp,
    pub barcode_rank_plot: ChartWithHelp,
    #[serde(skip)]
    pub metrics: MetricsTraitWrapper,
}

#[derive(Serialize, Clone, Alert)]
pub struct LibraryAntibodyOrAntigenWebSummary {
    pub parameters_table: CountParametersTable,
    pub barcode_rank_plot: ChartWithHelp,
    pub feature_histogram: Option<RawChartWithHelp>,
    #[serde(skip)]
    pub metrics: MetricsTraitWrapper,
}

#[derive(Serialize, Clone, Alert)]
pub struct LibraryCrisprWebSummary {
    pub parameters_table: CountParametersTable,
    pub barcode_rank_plot: ChartWithHelp,
    #[serde(skip)]
    pub metrics: MetricsTraitWrapper,
}

#[derive(Serialize, Clone, Alert)]
pub struct LibraryCustomFeatureWebSummary {
    pub parameters_table: CountParametersTable,
    pub barcode_rank_plot: ChartWithHelp,
    #[serde(skip)]
    pub metrics: MetricsTraitWrapper,
}

#[derive(Serialize, Clone, Alert)]
pub struct LibraryCmoWebSummary {
    pub parameters_table: CountParametersTable,
    pub barcode_rank_plot: ChartWithHelp,
    pub jibes_biplot: Option<RawChartWithHelp>,
    pub jibes_histogram: Option<RawChartWithHelp>,
    pub cmo_umi_projection_plot: Option<RawChartWithHelp>,
    pub cmo_tags_projection_plot: Option<RawChartWithHelp>,
    #[serde(rename = "_resources")]
    pub resources: MultiSharedResource,
    #[serde(skip)]
    pub metrics: MetricsTraitWrapper,
}

#[derive(Serialize, Clone, Alert)]
pub struct LibraryHashtagWebSummary {
    pub parameters_table: CountParametersTable,
    pub jibes_biplot: Option<RawChartWithHelp>,
    pub jibes_histogram: Option<RawChartWithHelp>,
    pub hashtag_umi_projection_plot: Option<RawChartWithHelp>,
    pub hashtag_tags_projection_plot: Option<RawChartWithHelp>,
    #[serde(rename = "_resources")]
    pub resources: MultiSharedResource,
    #[serde(skip)]
    pub metrics: MetricsTraitWrapper,
}

impl Alert for Option<String> {}
#[derive(Serialize, Clone, Alert)]
pub struct SampleGexWebSummary {
    pub parameters_table: CountParametersTable,
    pub barcode_rank_plot: Option<ChartWithHelp>,
    pub median_genes_per_cell_plot: Option<ChartWithHelp>,
    pub clustering_and_diffexp_plots: Value,
    pub disclaimer: Option<String>,
    #[serde(skip)]
    pub metrics: MetricsTraitWrapper,
}

impl Alert for Value {}

#[derive(Serialize, Clone, Alert)]
pub struct SampleAntibodyWebSummary {
    pub parameters_table: CountParametersTable,
    pub antibody_treemap: Option<RawChartWithHelp>,
    pub barcode_rank_plot: Option<ChartWithHelp>,
    pub clustering_and_diffexp_plots: Option<Value>,
    pub projection_plot: Option<RawChartWithHelp>,
    pub feature_histogram: Option<RawChartWithHelp>,
    #[serde(skip)]
    pub metrics: MetricsTraitWrapper,
}

#[derive(Serialize, Clone, Alert)]
pub struct SampleAntigenWebSummary {
    pub parameters_table: CountParametersTable,
    pub antigen_treemap: Option<RawChartWithHelp>,
    // Heatmap of clonotypes x antigen specificity hierarchically clustered
    pub clonotype_clustermap: Option<RawChartWithHelp>,
    #[serde(skip)]
    pub metrics: MetricsTraitWrapper,
}

#[derive(Serialize, Clone, Alert)]
pub struct SampleCrisprWebSummary {
    pub parameters_table: CountParametersTable,
    pub barcode_rank_plot: Option<ChartWithHelp>,
    pub projection_plot: Option<RawChartWithHelp>,
    #[serde(skip)]
    pub metrics: MetricsTraitWrapper,
}

#[derive(Serialize, Clone, Alert)]
pub struct SampleCustomFeatureWebSummary {
    pub parameters_table: CountParametersTable,
    pub barcode_rank_plot: Option<ChartWithHelp>,
    pub projection_plot: Option<RawChartWithHelp>,
    #[serde(skip)]
    pub metrics: MetricsTraitWrapper,
}

#[derive(Serialize, Clone)]
pub struct SampleCellAnnotationWebSummary {
    pub parameters_table: Option<TableMetric>,
    pub disclaimer: Option<String>,
    pub cas_success: Option<bool>,
    pub cell_annotation_disable_differential_expression: Option<bool>,
    pub cell_annotation_cell_types_chart: Option<WithTitle<VegaLitePlot>>,
    pub cell_annotation_violin_plot_chart: Option<WithTitle<VegaLitePlot>>,
    pub cell_annotation_umap_plot_chart: Option<WithTitle<VegaLitePlot>>,
    pub cell_annotation_diffexp_table: Option<WithTitle<DifferentialExpressionTable>>,
}

impl Alert for SampleCellAnnotationWebSummary {
    fn alerts(&self, _ctx: &AlertContext) -> Vec<AlertSpec> {
        match (
            self.cas_success,
            self.cell_annotation_disable_differential_expression,
        ) {
            (Some(false), _) => vec![AlertSpec {
                level: AlertLevel::Error,
                title: CELL_ANNOTATION_FAILURE_TITLE.to_string(),
                formatted_value: String::default(),
                message: CELL_ANNOTATION_FAILURE_MESSAGE.to_string(),
            }],
            (_, Some(true)) => vec![AlertSpec {
                level: AlertLevel::Warn,
                title: CELL_ANNOTATION_DE_WARN_TITLE.to_string(),
                formatted_value: String::default(),
                message: CELL_ANNOTATION_DE_WARN_MESSAGE.to_string(),
            }],
            _ => Vec::new(),
        }
    }
}

#[derive(Default, Debug, Clone, PartialEq, Eq, Serialize)]
pub struct MismatchedProbeBarcodePairings {
    /// Probe barcode pairings declared in the config but not detected.
    configured_not_detected: Vec<String>,
    /// Probe barcode pairings detected during detect chemistry but not declared
    /// in the config.
    detected_not_configured: Vec<String>,
}

impl MismatchedProbeBarcodePairings {
    pub fn new(
        configured_probe_barcode_pairing: &HashSet<String>,
        detected_probe_barcode_pairing: &HashSet<String>,
    ) -> Self {
        Self {
            configured_not_detected: configured_probe_barcode_pairing
                .difference(detected_probe_barcode_pairing)
                .cloned()
                .sorted()
                .collect(),
            detected_not_configured: detected_probe_barcode_pairing
                .difference(configured_probe_barcode_pairing)
                .cloned()
                .sorted()
                .collect(),
        }
    }

    fn formatted_value(&self) -> String {
        let count = self
            .configured_not_detected
            .len()
            .max(self.detected_not_configured.len());
        format!("{count} probe barcode(s)")
    }
}

impl Display for MismatchedProbeBarcodePairings {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let format_pairings = |pairings: &[String]| {
            if pairings.is_empty() {
                return Cow::Borrowed("none");
            }
            Cow::Owned(pairings.join(", "))
        };
        write!(
            f,
            "Mismatch found between probe barcode pairing specified in config \
             CSV file and chemistry detection. \
             Pairings specified in the config CSV but not detected: {}. \
             Pairings detected but not specified in the config CSV: {}. \
             This may be due to an error in the config CSV, or \
             a workflow error resulting in cross-contamination between samples or \
             some hybridization reactions having the wrong probe barcodes.",
            format_pairings(&self.configured_not_detected),
            format_pairings(&self.detected_not_configured),
        )
    }
}
