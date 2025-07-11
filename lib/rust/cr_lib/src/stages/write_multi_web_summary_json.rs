//! Martian stage WRITE_MULTI_WEB_SUMMARY_JSON
//! Write a JSON file with the metrics and plots data for the websummary.
#![allow(missing_docs)]

use super::parse_multi_config::{CellCalling, VdjGenInputs};
use crate::cell_annotation_ws_parameters::{
    generate_cell_type_barchart_from_value, generate_cell_type_diffexp_from_value,
    generate_cell_type_parameter_table, generate_cell_type_umap_plot_from_value,
    generate_cell_type_violin_plot_from_value, CellAnnotationMetrics,
};
use crate::stages::build_per_sample_vdj_ws_contents::{VdjWsContents, VdjWsContentsFormat};
use crate::stages::compute_antigen_vdj_metrics::{AntigenVdjMetrics, AntigenVdjMetricsFormat};
use crate::stages::detect_chemistry::DetectedProbeBarcodePairingFile;
use crate::stages::parse_multi_config::{CommonInputs, CountInputs};
use crate::{PerLibrarySequencingMetrics, SequencingMetricsFormat, SvgFile};
use anyhow::{bail, Result};
use barcode::whitelist::{categorize_rtl_multiplexing_barcode_id, RTLMultiplexingBarcodeType};
use cr_types::chemistry::{ChemistryDefs, ChemistryDefsExt, ChemistryName};
use cr_types::reference::feature_reference::{FeatureConfig, SpecificityControls};
use cr_types::websummary::{AlertConfig, MetricEtlConfig};
use cr_types::{
    AlignerParam, BarcodeMultiplexingType, CellLevel, CrMultiGraph, Fingerprint, GenomeName,
    LibraryType, ReadLevel, Sample, SampleAssignment, TargetingMethod,
};
use cr_websummary::alert::AlertContext;
use cr_websummary::multi::antigen::{clonotype_specificity_heatmap, AntigenSpecificityRow};
use cr_websummary::multi::metrics::{
    load_metrics_etl, load_metrics_etl_special, ActiveConditions, CountAndPercentTransformer,
    MetricTier, MetricsProcessor,
};
use cr_websummary::multi::plots::{
    format_barcode_rank_plot, format_histogram, format_jibes_biplots, format_tags_on_umap_plot,
    format_umi_on_umap_plot, library_median_genes_plot_from_metrics,
    library_sequencing_saturation_plot_from_metrics, sample_median_genes_plot_from_metrics,
    PlotType,
};
use cr_websummary::multi::svg::SvgGraph;
use cr_websummary::multi::websummary::{
    CountParametersTable, ExperimentalDesign, JsonMetricSummary,
    LibraryAntibodyOrAntigenWebSummary, LibraryCmoWebSummary, LibraryCrisprWebSummary,
    LibraryCustomFeatureWebSummary, LibraryGexWebSummary, LibraryHashtagWebSummary,
    LibraryHeaderInfo, LibraryWebSummary, MetricsTraitWrapper, MismatchedProbeBarcodePairings,
    MultiDiagnostics, MultiSharedResource, MultiWebSummary, MultiWebSummaryLibraryData,
    MultiWebSummarySampleData, SampleAntibodyWebSummary, SampleAntigenWebSummary,
    SampleCellAnnotationWebSummary, SampleCrisprWebSummary, SampleCustomFeatureWebSummary,
    SampleDiagnostics, SampleGexWebSummary, SampleHeaderInfo, SampleWebSummary, Section,
    CELL_ANNOTATION_ADVERTISEMENT_STRING, UMI_PER_PROBE_BARCODE_BACKGROUND_THRESHOLD,
};
use cr_websummary::{PlotlyChart, RawChartWithHelp, Tab, WsSample};
use fastq_set::WhichEnd;
use itertools::Itertools;
use json_report_derive::JsonReport;
use martian::prelude::*;
use martian_derive::{make_mro, MartianStruct};
use martian_filetypes::json_file::JsonFile;
use martian_filetypes::tabular_file::CsvFile;
use martian_filetypes::FileTypeRead;
use metric::{join_metric_name, TxHashMap};
use multi::config::{
    MultiConfigCsv, MultiConfigCsvFile, ProbeBarcodeIterationMode, PROBE_BARCODE_ID_GROUPING,
};
use ordered_float::NotNan;
use serde::{Deserialize, Serialize};
use serde_json::value::Value;
use serde_json::{self, json, Number};
use std::collections::{HashMap, HashSet};
use std::hash::Hash;
use std::string::ToString;
use tenx_websummary::components::DifferentialExpressionTable;

pub const GEM_BARCODE_OVERLAP_ALERT_THRESHOLD: f64 = 0.6;

pub struct WriteMultiWebSummaryJson;

#[derive(Clone, Serialize, Deserialize)]
pub struct JibesBiplotHistogramData {
    pub biplot: Value,
    pub histogram: Value,
    #[serde(rename = "_resources")]
    pub resources: MultiSharedResource,
}

#[allow(clippy::type_complexity)]
#[derive(Clone, Deserialize, MartianStruct)]
pub struct StageInputs {
    pub per_sample_metrics: TxHashMap<SampleAssignment, Option<JsonFile<TxHashMap<String, Value>>>>,
    pub library_metrics: Option<JsonFile<TxHashMap<String, Value>>>,
    pub sequencing_metrics: Option<SequencingMetricsFormat>,
    pub multi_config: MultiConfigCsvFile,
    pub multi_graph: JsonFile<CrMultiGraph>,
    pub multi_graph_svg: SvgFile,
    pub common_inputs: CommonInputs,
    pub count_inputs: Option<CountInputs>,
    pub count_cell_calling_config: Option<CellCalling>,
    pub vdj_gen_inputs: Option<VdjGenInputs>,
    pub tag_contaminant_info: Option<JsonFile<Value>>,
    pub sample_projection_plots: TxHashMap<SampleAssignment, Option<JsonFile<SampleUmapPlots>>>,
    pub sample_barcode_rank_plots:
        TxHashMap<SampleAssignment, Option<JsonFile<TxHashMap<LibraryType, PlotlyChart>>>>,
    pub sample_treemap_plots: Option<
        TxHashMap<SampleAssignment, Option<JsonFile<TxHashMap<LibraryType, RawChartWithHelp>>>>,
    >,
    pub barcode_rank_plots: Option<JsonFile<TxHashMap<LibraryType, PlotlyChart>>>,
    pub jibes_biplot_histogram: Option<JsonFile<Value>>,
    pub antibody_histograms: Option<JsonFile<RawChartWithHelp>>,
    pub sample_antibody_histograms:
        Option<TxHashMap<SampleAssignment, Option<JsonFile<RawChartWithHelp>>>>,
    pub antigen_histograms: Option<JsonFile<RawChartWithHelp>>,
    pub cmo_projection_plot: Option<JsonFile<MultiplexingUmapPlots>>,
    pub vdj_t_contents: Option<TxHashMap<SampleAssignment, VdjWsContentsFormat>>,
    pub vdj_t_gd_contents: Option<TxHashMap<SampleAssignment, VdjWsContentsFormat>>,
    pub vdj_b_contents: Option<TxHashMap<SampleAssignment, VdjWsContentsFormat>>,
    pub target_set_name: Option<String>,
    pub antigen_vdj_metrics: Option<TxHashMap<SampleAssignment, Option<AntigenVdjMetricsFormat>>>,
    pub antigen_specificity:
        Option<TxHashMap<SampleAssignment, Option<CsvFile<AntigenSpecificityRow>>>>,
    pub cell_annotation_barcharts: Option<TxHashMap<SampleAssignment, Option<JsonFile<Value>>>>,
    pub cell_annotation_box_plots: Option<TxHashMap<SampleAssignment, Option<JsonFile<Value>>>>,
    pub cell_annotation_umap_plots: Option<TxHashMap<SampleAssignment, Option<JsonFile<Value>>>>,
    pub cell_annotation_diffexp_tables:
        Option<TxHashMap<SampleAssignment, Option<JsonFile<DifferentialExpressionTable>>>>,
    pub cell_annotation_metrics_jsons:
        Option<TxHashMap<SampleAssignment, Option<JsonFile<CellAnnotationMetrics>>>>,
    pub cell_annotation_viable_but_not_requested: Option<TxHashMap<SampleAssignment, Option<bool>>>,
    pub feature_config: Option<FeatureConfig>,
    /// chemistry_defs is None for a VDJ-only analysis.
    pub chemistry_defs: Option<ChemistryDefs>,
    pub detected_probe_barcode_pairing: Option<DetectedProbeBarcodePairingFile>,
    pub no_preflight: bool,
}

#[derive(Clone, Serialize, Deserialize, MartianStruct)]
pub struct StageOutputs {
    #[mro_retain]
    pub web_summary_json: TxHashMap<SampleAssignment, JsonFile<MultiWebSummary>>,
    pub metrics_summary_csv: TxHashMap<SampleAssignment, CsvFile<()>>,
}

struct LibWsBuilder {
    alert_context: AlertContext,
    common_inputs: CommonInputs,
    multi_graph: CrMultiGraph,
    multi_config: MultiConfigCsv,
    chemistry_defs: Option<ChemistryDefs>,
    lib_metrics: TxHashMap<String, Value>,
    lib_metrics_proc: MetricsProcessor,
    special_metrics_proc: MetricsProcessor,
    barcode_rank_plots: TxHashMap<LibraryType, PlotlyChart>,
    sequencing_metrics: PerLibrarySequencingMetrics,
    count_inputs: Option<CountInputs>,
    count_cell_calling_config: Option<CellCalling>,
    jibes_biplot_histogram: Option<Value>,
    antibody_histograms: Option<RawChartWithHelp>,
    antigen_histograms: Option<RawChartWithHelp>,
    cmo_projection_plot: Option<MultiplexingUmapPlots>,
    target_set_name: Option<String>,
    vdj_t_contents: Option<VdjWsContents>,
    vdj_t_gd_contents: Option<VdjWsContents>,
    vdj_b_contents: Option<VdjWsContents>,
    /// True if any form of multiplexing is used in this analysis.
    multiplexing_method: Option<BarcodeMultiplexingType>,
    specificity_controls: Option<SpecificityControls>,
    dropped_tags: Vec<String>,
    probe_barcodes_high_gem_overlap: Vec<String>,
    mismatched_probe_barcode_pairings: Option<MismatchedProbeBarcodePairings>,
    /// Derived parameters.
    is_rtl: bool,
    targeting_method: Option<TargetingMethod>,
}

impl LibWsBuilder {
    fn vdj_tab_names(&self) -> [Option<&'static str>; 3] {
        let mut tab_names = [None; 3];
        for (i, (name, content)) in [
            ("VDJ-T", &self.vdj_t_contents),
            ("VDJ-B", &self.vdj_b_contents),
            ("VDJ-T-GD", &self.vdj_t_gd_contents),
        ]
        .into_iter()
        .enumerate()
        {
            if content.is_some() {
                tab_names[i] = Some(name);
            }
        }
        tab_names
    }

    /// Return true if multiplexed using CMO.
    fn is_cmo_multiplexed(&self) -> bool {
        self.multiplexing_method == Some(BarcodeMultiplexingType::CellLevel(CellLevel::CMO))
    }

    /// Return true if multiplexed using Hashtag.
    fn is_hashtag_multiplexed(&self) -> bool {
        self.multiplexing_method == Some(BarcodeMultiplexingType::CellLevel(CellLevel::Hashtag))
    }

    /// Return true if multiplexed using OH
    fn is_oh_multiplexed(&self) -> bool {
        self.multiplexing_method == Some(BarcodeMultiplexingType::ReadLevel(ReadLevel::OH))
    }

    /// Return true if multiplexed using RTL .
    fn is_rtl_multiplexed(&self) -> bool {
        self.multiplexing_method == Some(BarcodeMultiplexingType::ReadLevel(ReadLevel::RTL))
    }

    /// Return true if antigen specificity controls are specified.
    fn has_antigen_controls(&self) -> bool {
        self.specificity_controls.is_some()
    }

    /// Return the aligner.
    fn aligner(&self) -> AlignerParam {
        get_metric_string(&self.lib_metrics, "alignment_aligner")
            .unwrap()
            .unwrap()
            .parse()
            .unwrap()
    }

    /// Return the feature ref path input by the user
    fn get_feature_ref(&self) -> Option<String> {
        self.multi_config
            .feature
            .as_ref()
            .and_then(|x| x.reference_path.as_ref().map(|r| r.display().to_string()))
    }

    /// Return the cmo-set input by the user
    fn get_cmo_set(&self) -> Option<String> {
        self.multi_config
            .gene_expression
            .as_ref()
            .and_then(|x| x.cmo_set.as_ref().map(|r| r.display().to_string()))
    }

    fn build(&self, pipeline_version: String, context: &AlertContext) -> Result<LibraryWebSummary> {
        let mut lib_ws = LibraryWebSummary {
            header_info: LibraryHeaderInfo {
                run_id: self.common_inputs.sample_id.clone(),
                run_desc: self.common_inputs.sample_desc.clone(),
                pipeline_version,
            },
            vdj_t_tab: self
                .vdj_t_contents
                .clone()
                .map(|content| Tab::new(content.to_library_ws(), context)),
            vdj_t_gd_tab: self
                .vdj_t_gd_contents
                .clone()
                .map(|content| Tab::new(content.to_library_ws(), context)),
            vdj_b_tab: self
                .vdj_b_contents
                .clone()
                .map(|content| Tab::new(content.to_library_ws(), context)),
            ..Default::default()
        };
        for lib in &self.multi_graph.libraries {
            match lib.library_type {
                LibraryType::Gex => {
                    assert!(lib_ws.gex_tab.is_none());
                    lib_ws.gex_tab = Some(Tab::new(
                        self.build_gex_ws(&lib.physical_library_id)?,
                        context,
                    ));
                }
                LibraryType::Antibody => {
                    assert!(lib_ws.antibody_tab.is_none());
                    lib_ws.antibody_tab = Some(Tab::new(
                        self.build_antibody_or_antigen_ws(&lib.physical_library_id, true)?,
                        context,
                    ));
                    if self.is_hashtag_multiplexed() {
                        let mut hashtag_tab: LibraryHashtagWebSummary = self.build_hashtag_ws()?;
                        // bubble up shared resources, currently just cmo counts
                        lib_ws.resources.extend(hashtag_tab.resources.drain());
                        lib_ws.hashtag_tab = Some(Tab::new(hashtag_tab, context));
                    }
                }
                LibraryType::Antigen => {
                    assert!(lib_ws.antigen_tab.is_none());
                    lib_ws.antigen_tab = Some(Tab::new(
                        self.build_antibody_or_antigen_ws(&lib.physical_library_id, false)?,
                        context,
                    ));
                }
                LibraryType::Crispr => {
                    assert!(lib_ws.crispr_tab.is_none());
                    lib_ws.crispr_tab = Some(Tab::new(
                        self.build_crispr_ws(&lib.physical_library_id)?,
                        context,
                    ));
                }
                LibraryType::Vdj(_) => {}
                LibraryType::Custom => {
                    assert!(lib_ws.custom_feature_tab.is_none());
                    lib_ws.custom_feature_tab = Some(Tab::new(
                        self.build_custom_ws(&lib.physical_library_id)?,
                        context,
                    ));
                }
                LibraryType::Cellplex => {
                    assert!(lib_ws.cmo_tab.is_none());
                    let mut cmo_tab: LibraryCmoWebSummary =
                        self.build_cmo_ws(&lib.physical_library_id)?;
                    // bubble up shared resources, currently just cmo counts
                    lib_ws.resources.extend(cmo_tab.resources.drain());
                    lib_ws.cmo_tab = Some(Tab::new(cmo_tab, context));
                }
                LibraryType::Atac => unreachable!(),
            }
        }
        Ok(lib_ws)
    }

    fn get_mean_reads_per_cell_associated_partitions_metric(
        &self,
        section: Section,
        library_type: LibraryType,
        physical_library_id: &str,
    ) -> Result<JsonMetricSummary> {
        let key = match library_type {
            LibraryType::GeneExpression => {
                "multi_transcriptome_total_raw_reads_per_filtered_bc".to_string()
            }
            LibraryType::FeatureBarcodes(fb) => join_metric_name(fb, "reads_per_cell"),
            _ => panic!(
                "cannot get mean reads per cell associated partitions metric for library type {library_type}"
            ),
        };

        let config = MetricEtlConfig {
            header: "Mean reads per cell".to_string(),
            json_key: Some(key),
            ty: "FloatAsInt".to_string(),
            ..Default::default()
        };

        let mut metric = self.lib_metrics_proc.process_one(
            &self.alert_context,
            &self.lib_metrics,
            &config,
            MetricTier::Library,
            section,
        )?;

        metric.key = "mean_reads_per_cell_associated_partition".to_string();
        metric.grouping_key = Some(physical_library_id.to_string());
        metric.grouping_header = Some("Physical library ID".to_string());

        Ok(metric)
    }

    fn get_cell_associated_partitions_metric(
        &self,
        section: Section,
        physical_library_id: &str,
    ) -> Result<JsonMetricSummary> {
        let has_gex = self.multi_graph.has_library_type(LibraryType::Gex);
        let has_antibody = self.multi_graph.has_library_type(LibraryType::Antibody);
        let key = if has_antibody && !has_gex {
            "ANTIBODY_filtered_bcs_transcriptome_union"
        } else {
            "filtered_bcs_transcriptome_union"
        };

        let alert = AlertConfig {
            error_threshold: Some(0.),
            warn_threshold: Some(100.),
            warn_title: Some("Low Number of Cells Detected".to_string()),
            error_title: Some("No Cells Detected".to_string()),
            detail: "Estimated number of cells is expected to be > 100. This usually indicates poor cell handling, poor library quality, or poor sequencing quality. Application performance is likely to be affected.".to_string(),
            ..Default::default()
        };

        let config = MetricEtlConfig {
            header: "Cells".to_string(),
            json_key: Some(key.to_string()),
            ty: "usize".to_string(),
            alerts: vec![alert],
            ..Default::default()
        };

        let mut metric = self.lib_metrics_proc.process_one(
            &self.alert_context,
            &self.lib_metrics,
            &config,
            MetricTier::Library,
            section,
        )?;

        metric.key = "cell_associated_partitions".to_string();
        metric.grouping_key = Some(physical_library_id.to_string());
        metric.grouping_header = Some("Physical library ID".to_string());

        Ok(metric)
    }

    /// Conditionally get the high-occupancy GEM metric.
    ///
    /// FIXME CELLRANGER-8444 refactor to always get this metric for RTL multiplexing,
    /// but consider moving the display logic to the frontend. This would impact
    /// whether the metric appears in the CSV output.
    fn get_high_occupancy_gem_metric(
        &self,
        section: Section,
        library_type: LibraryType,
        physical_library_id: &str,
    ) -> Result<Option<JsonMetricSummary>> {
        if !self.is_rtl_multiplexed() {
            return Ok(None);
        }
        let has_gex = self.multi_graph.has_library_type(LibraryType::Gex);
        if !(library_type == LibraryType::Gex || !has_gex && library_type == LibraryType::Antibody)
        {
            return Ok(None);
        }
        let alert = AlertConfig {
                error_threshold: Some(0.0),
                warn_threshold: Some(0.9),
                warn_title: Some("Low fraction of initial cell calls pass high occupancy GEM filtering.".to_string()),
                detail: "Numbers under 90% could be due to partial clogs, wetting failures, cell clumping, or significant deviations from the recommended chip loading protocol.".to_string(),
                ..Default::default()
            };
        let config = MetricEtlConfig {
            json_key: Some("rtl_multiplexing_fraction_cells_in_high_occupancy_gems".to_string()),
            ty: "Percent".to_string(),
            transformer: Some("ComplementPercent".to_string()),
            header: "Fraction of initial cell barcodes passing high occupancy GEM filtering"
                .to_string(),
            alerts: vec![alert],
            ..Default::default()
        };
        let mut metric = self.lib_metrics_proc.process_one(
            &self.alert_context,
            &self.lib_metrics,
            &config,
            MetricTier::Library,
            section,
        )?;

        metric.grouping_key = Some(physical_library_id.to_string());
        metric.grouping_header = Some("Physical library ID".to_string());
        Ok(Some(metric))
    }

    /// Return the chemistry description and append "(manual)" if a manual chemistry is specified.
    fn chemistry_description_with_manual(&self, library_type: LibraryType) -> String {
        let chemistry_defs = self.chemistry_defs.as_ref().unwrap();
        let chemistry_description = &chemistry_defs[&library_type].description;
        let chemistry_spec = self.multi_config.chemistry_specs().unwrap()[&library_type];

        if chemistry_spec.is_auto() {
            chemistry_description.to_string()
        } else {
            format!("{chemistry_description} (manual)")
        }
    }

    fn count_param_table(
        &self,
        library_type: LibraryType,
        probe_barcode_data: PerProbeBarcodeData,
    ) -> Result<CountParametersTable> {
        let count_inputs = self.count_inputs.as_ref().unwrap();
        let cell_calling_config = self.count_cell_calling_config.as_ref().unwrap();

        // TODO: Jira: CELLRANGER-9147: Consider using reference_genomes list instead
        let transcriptome = if let Some(reference_genomes) =
            get_metric_string(&self.lib_metrics, "reference_genomes")?
        {
            format!(
                "{}-{}",
                reference_genomes,
                get_metric_string(&self.lib_metrics, "reference_version")?.unwrap()
            )
        } else {
            String::new()
        };

        let reference_path = count_inputs
            .reference_info
            .as_ref()
            .and_then(|x| x.get_reference_path());
        Ok(CountParametersTable {
            chemistry: self.chemistry_description_with_manual(library_type),
            introns_included: count_inputs.include_introns,
            reference_path: reference_path.map(|x| x.display().to_string()),
            transcriptome,
            feature_ref_path: self.get_feature_ref(),
            cmo_set_path: self.get_cmo_set(),
            target_set_name: self.target_set_name.clone(),
            targeting_method: self.targeting_method,
            filter_probes: count_inputs.filter_probes,
            disable_ab_aggregate_detection: cell_calling_config.disable_ab_aggregate_detection,
            disable_high_occupancy_gem_detection: cell_calling_config
                .disable_high_occupancy_gem_detection,
            num_genes_on_target: self
                .lib_metrics
                .get("num_genes_on_target")
                .map(|x| x.as_u64().unwrap() as usize),
            library_type,
            throughput: get_metric_string(&self.lib_metrics, "throughput_inferred")?,
            tenx_cmos: count_inputs.tenx_cmos,
            aligner: self.aligner(),
            antigen_negative_control: self.has_antigen_controls(),
            dropped_tags: self.dropped_tags.clone(),
            probe_barcodes_high_gem_overlap: self.probe_barcodes_high_gem_overlap.clone(),
            mismatched_probe_barcode_pairings: self.mismatched_probe_barcode_pairings.clone(),
            unspecified_probe_barcodes_detected: probe_barcode_data.unexpected,
            specified_probe_barcodes_missing: probe_barcode_data.missing,
        })
    }

    fn genomes(&self) -> &[GenomeName] {
        self.count_inputs
            .as_ref()
            .expect("count_inputs is None")
            .get_genomes()
            .unwrap_or_default()
    }

    fn build_ocm_per_overhang_metrics(
        &self,
        library_type: LibraryType,
        section: Section,
        active_conditions: &ActiveConditions,
    ) -> Result<Vec<JsonMetricSummary>> {
        if !self.is_oh_multiplexed() {
            return Ok(Default::default());
        }

        let ocm_barcode_id_to_sample_id = self.multi_graph.get_tag_name_to_sample_id_map().unwrap();

        let umi_per_ocm_barcode: Vec<(&str, usize)> = if let Some(umi_per_ocm_barcode) = self
            .lib_metrics
            .get(&join_metric_name(library_type, "umi_per_overhang"))
        {
            umi_per_ocm_barcode
                .as_object()
                .unwrap()
                .iter()
                .map(|(id, n)| (id.as_str(), n.as_u64().unwrap() as usize))
                .sorted()
                .collect()
        } else {
            Vec::default()
        };

        let umi_sum = umi_per_ocm_barcode.iter().map(|(_id, n)| n).sum();

        let filtered_barcodes_per_ocm_barcode: TxHashMap<&str, usize> = self.lib_metrics
            ["filtered_barcodes_per_overhang"]
            .as_object()
            .unwrap()
            .iter()
            .map(|(id, n)| (id.as_str(), n.as_u64().unwrap() as usize))
            .collect();

        let filtered_barcodes_sum: usize = filtered_barcodes_per_ocm_barcode
            .iter()
            .map(|(_id, &n)| n)
            .sum();

        let mut special_metrics_proc = self.special_metrics_proc.with_default_transformers();
        special_metrics_proc
            .add_transformer("UmiFraction", CountAndPercentTransformer::new(umi_sum));
        special_metrics_proc.add_transformer(
            "CellsFraction",
            CountAndPercentTransformer::new(filtered_barcodes_sum),
        );

        let mut metrics = vec![];
        for (ocm_barcode_id, umi_count) in umi_per_ocm_barcode {
            let sample_id = ocm_barcode_id_to_sample_id
                .get(ocm_barcode_id)
                .map(|(sample_id, _)| *sample_id);
            let cells = *filtered_barcodes_per_ocm_barcode
                .get(ocm_barcode_id)
                .unwrap_or(&0);

            let cells_per_ocm_barcode = if sample_id.is_some() {
                Some(cells)
            } else {
                assert_eq!(cells, 0);
                None
            };

            #[derive(JsonReport)]
            struct Metrics<'a> {
                ocm_barcode_id: &'a str,
                sample_id: Option<&'a str>,
                umi_per_ocm_barcode: usize,
                cells_per_ocm_barcode: Option<usize>,
            }

            metrics.extend(special_metrics_proc.process_group(
                "ocm_per_overhang_metrics",
                section,
                active_conditions,
                &self.alert_context,
                &Metrics {
                    ocm_barcode_id,
                    sample_id,
                    umi_per_ocm_barcode: umi_count,
                    cells_per_ocm_barcode,
                },
            )?);
        }

        Ok(metrics)
    }
    /// Return per probe barcode metrics for the specified library type.
    /// Also compile missing and unexpected probe barcodes.
    fn build_rtl_probe_barcode_metrics(
        &self,
        library_type: LibraryType,
        section: Section,
        active_conditions: &ActiveConditions,
    ) -> Result<PerProbeBarcodeData> {
        if !self.is_rtl_multiplexed() {
            return Ok(Default::default());
        }

        let mut special_metrics_proc = self.special_metrics_proc.with_default_transformers();

        let probe_barcode_id_to_sample_id_and_mapped_probe_barcode_ids =
            self.multi_graph.get_tag_name_to_sample_id_map().unwrap();

        let umi_per_probe_barcode: Vec<(&str, usize)> = if let Some(umi_per_probe_barcode) = self
            .lib_metrics
            .get(&join_metric_name(library_type, "umi_per_probe_barcode"))
        {
            umi_per_probe_barcode
                .as_object()
                .unwrap()
                .iter()
                .map(|(id, n)| (id.as_str(), n.as_u64().unwrap() as usize))
                .sorted()
                .collect()
        } else {
            Vec::default()
        };

        let umi_sum = umi_per_probe_barcode.iter().map(|(_id, n)| n).sum();

        let filtered_barcodes_per_probe_barcode: TxHashMap<&str, usize> = self.lib_metrics
            ["filtered_barcodes_per_probe_barcode"]
            .as_object()
            .unwrap()
            .iter()
            .map(|(id, n)| (id.as_str(), n.as_u64().unwrap() as usize))
            .collect();

        let filtered_barcodes_sum = filtered_barcodes_per_probe_barcode
            .iter()
            .map(|(_id, &n)| n)
            .sum();

        special_metrics_proc
            .add_transformer("UmiFraction", CountAndPercentTransformer::new(umi_sum));
        special_metrics_proc.add_transformer(
            "CellsFraction",
            CountAndPercentTransformer::new(filtered_barcodes_sum),
        );

        let mut metrics = vec![];
        let mut unspecified_probe_barcodes_detected = vec![];
        let mut specified_probe_barcodes_missing = vec![];

        for (probe_barcode_id, umi_count) in umi_per_probe_barcode {
            let (sample_id, mapped_barcode_ids) =
                probe_barcode_id_to_sample_id_and_mapped_probe_barcode_ids
                    .get(probe_barcode_id)
                    .copied()
                    .unzip();

            // Elide probe barcodes not assigned to a sample and with few UMI.
            let umi_frac = umi_count as f64 / umi_sum as f64;
            if sample_id.is_none() && umi_frac < UMI_PER_PROBE_BARCODE_BACKGROUND_THRESHOLD {
                continue;
            }

            let barcode_ids: Vec<_> = std::iter::once(probe_barcode_id)
                .chain(mapped_barcode_ids.into_iter().flatten().map(String::as_str))
                .collect();

            let cells = *filtered_barcodes_per_probe_barcode
                .get(probe_barcode_id)
                .unwrap_or(&0);
            let cells_per_probe_barcode = if sample_id.is_some() {
                Some(cells)
            } else {
                assert_eq!(cells, 0);
                None
            };

            #[derive(JsonReport)]
            struct Metrics {
                probe_barcode_id: String,
                sample_id: Option<String>,
                umi_per_probe_barcode: usize,
                cells_per_probe_barcode: Option<usize>,
            }

            // Unpack probe barcode and sample IDs, and collect missing or unexpected
            // probe barcodes.

            let (probe_barcode_id, sample_id) = match sample_id {
                Some(sample_id) => {
                    let probe_barcode_ids_str = barcode_ids.join(PROBE_BARCODE_ID_GROUPING);
                    if umi_frac < UMI_PER_PROBE_BARCODE_BACKGROUND_THRESHOLD {
                        specified_probe_barcodes_missing.push(probe_barcode_ids_str.clone());
                    }
                    (probe_barcode_ids_str, Some(sample_id.to_string()))
                }
                None => {
                    unspecified_probe_barcodes_detected.push(probe_barcode_id.to_string());
                    (probe_barcode_id.to_string(), None)
                }
            };

            metrics.extend(special_metrics_proc.process_group(
                "rtl_probe_barcode_metrics",
                section,
                active_conditions,
                &self.alert_context,
                &Metrics {
                    probe_barcode_id,
                    sample_id,
                    umi_per_probe_barcode: umi_count,
                    cells_per_probe_barcode,
                },
            )?);
        }
        Ok(PerProbeBarcodeData {
            metrics,
            unexpected: unspecified_probe_barcodes_detected,
            missing: specified_probe_barcodes_missing,
        })
    }

    fn active_metric_conditions(&self, section: Section) -> ActiveConditions {
        let has_gdna = !matches!(
            self.lib_metrics.get("estimated_gdna_content"),
            None | Some(Value::Null)
        );
        if has_gdna {
            assert!(self.is_rtl);
        }

        ActiveConditions {
            section,
            vdj_receptor: None,
            is_multiplexed: self.multiplexing_method.is_some(),
            is_cell_multiplexed: self.is_cmo_multiplexed() || self.is_hashtag_multiplexed(),
            is_read_multiplexed: self.is_oh_multiplexed() || self.is_rtl_multiplexed(),
            is_rtl: self.is_rtl,
            has_gdna,
            include_introns: self.alert_context.include_introns,
            has_vdj_reference: false,
        }
    }

    fn add_cell_metrics(
        &self,
        section: Section,
        library_type: LibraryType,
        physical_library_id: &str,
        metrics_out: &mut Vec<JsonMetricSummary>,
    ) -> Result<()> {
        metrics_out.push(self.get_mean_reads_per_cell_associated_partitions_metric(
            section,
            library_type,
            physical_library_id,
        )?);

        metrics_out.push(self.get_cell_associated_partitions_metric(section, physical_library_id)?);
        if let Some(metric) =
            self.get_high_occupancy_gem_metric(section, library_type, physical_library_id)?
        {
            metrics_out.push(metric);
        }
        Ok(())
    }

    /// Return a clone of the library metrics, with the provided physical library ID injected.
    fn get_metrics_with_physical_library_id(
        &self,
        physical_library_id: &str,
    ) -> TxHashMap<String, Value> {
        // FIXME CELLRANGER-8444 pick a better way to inject this than copying
        let mut metrics = self.lib_metrics.clone();
        metrics.insert(
            "physical_library_id".to_string(),
            json!(physical_library_id),
        );
        metrics
    }

    fn build_gex_ws(&self, physical_library_id: &str) -> Result<LibraryGexWebSummary> {
        let library_type = LibraryType::Gex;
        let section = Section::Gex;
        let lib_metrics = self.get_metrics_with_physical_library_id(physical_library_id);

        let active_conditions = self.active_metric_conditions(section);

        let mut metrics = self.lib_metrics_proc.process(
            section,
            &active_conditions,
            &self.alert_context,
            &lib_metrics,
        )?;

        self.add_cell_metrics(section, library_type, physical_library_id, &mut metrics)?;

        let mut per_probe_barcode_data =
            self.build_rtl_probe_barcode_metrics(library_type, section, &active_conditions)?;
        metrics.append(&mut per_probe_barcode_data.metrics);

        metrics.extend(self.build_ocm_per_overhang_metrics(
            library_type,
            section,
            &active_conditions,
        )?);

        metrics.extend(self.get_sequencing_metrics(section, library_type, &active_conditions)?);

        let barcode_rank_plot = format_barcode_rank_plot(
            self.barcode_rank_plots
                .get(&LibraryType::Gex)
                .expect("Gene expression barcode rank plot was missing."),
            "GEX",
        );

        // Construct final websummary
        Ok(LibraryGexWebSummary {
            parameters_table: self.count_param_table(library_type, per_probe_barcode_data)?,
            sequencing_saturation_plot: library_sequencing_saturation_plot_from_metrics(
                &self.lib_metrics,
            ),
            median_genes_per_cell_plot: library_median_genes_plot_from_metrics(
                &self.lib_metrics,
                self.genomes().iter().cloned().collect(),
                PlotType::LibraryPlot,
                self.targeting_method,
            ),
            barcode_rank_plot,
            metrics: MetricsTraitWrapper(metrics),
        })
    }

    fn build_antibody_or_antigen_ws(
        &self,
        physical_library_id: &str,
        // true for antibody, false for antigen
        is_antibody: bool,
    ) -> Result<LibraryAntibodyOrAntigenWebSummary> {
        let (section, library_type) = if is_antibody {
            (Section::Antibody, LibraryType::Antibody)
        } else {
            (Section::Antigen, LibraryType::Antigen)
        };

        let lib_metrics = self.get_metrics_with_physical_library_id(physical_library_id);

        let active_conditions = self.active_metric_conditions(section);

        let mut metrics = self.lib_metrics_proc.process(
            section,
            &active_conditions,
            &self.alert_context,
            &lib_metrics,
        )?;

        self.add_cell_metrics(section, library_type, physical_library_id, &mut metrics)?;

        let mut per_probe_barcode_data =
            self.build_rtl_probe_barcode_metrics(library_type, section, &active_conditions)?;
        metrics.append(&mut per_probe_barcode_data.metrics);

        metrics.extend(self.get_sequencing_metrics(section, library_type, &active_conditions)?);

        let barcode_rank_plot = format_barcode_rank_plot(
            self.barcode_rank_plots
                .get(&library_type)
                .unwrap_or_else(|| panic!("{library_type} barcode rank plot was missing.")),
            if is_antibody { "Antibody" } else { "Antigen" },
        );

        // Only show the antibody histogram at the library level for CMO and HASHTAG.
        let feature_histogram = match (
            is_antibody,
            (self.is_cmo_multiplexed() || self.is_hashtag_multiplexed()),
        ) {
            (true, true) => self.antibody_histograms.clone(),
            (true, false) => None,
            (false, _) => self.antigen_histograms.clone(),
        };

        Ok(LibraryAntibodyOrAntigenWebSummary {
            parameters_table: self.count_param_table(library_type, per_probe_barcode_data)?,
            barcode_rank_plot,
            feature_histogram,
            metrics: MetricsTraitWrapper(metrics),
        })
    }

    fn build_crispr_ws(&self, physical_library_id: &str) -> Result<LibraryCrisprWebSummary> {
        let library_type = LibraryType::Crispr;
        let section = Section::Crispr;
        let lib_metrics = self.get_metrics_with_physical_library_id(physical_library_id);

        let active_conditions = self.active_metric_conditions(section);

        let mut metrics = self.lib_metrics_proc.process(
            section,
            &active_conditions,
            &self.alert_context,
            &lib_metrics,
        )?;

        self.add_cell_metrics(section, library_type, physical_library_id, &mut metrics)?;

        let mut per_probe_barcode_data =
            self.build_rtl_probe_barcode_metrics(library_type, section, &active_conditions)?;
        metrics.append(&mut per_probe_barcode_data.metrics);

        metrics.extend(self.get_sequencing_metrics(section, library_type, &active_conditions)?);

        Ok(LibraryCrisprWebSummary {
            parameters_table: self.count_param_table(library_type, per_probe_barcode_data)?,
            barcode_rank_plot: format_barcode_rank_plot(
                self.barcode_rank_plots
                    .get(&LibraryType::Crispr)
                    .expect("CRISPR Guide Capture barcode rank plot was missing."),
                "CRISPR",
            ),
            metrics: MetricsTraitWrapper(metrics),
        })
    }

    fn build_custom_ws(&self, physical_library_id: &str) -> Result<LibraryCustomFeatureWebSummary> {
        let library_type = LibraryType::Custom;
        let section = Section::Custom;
        let lib_metrics = self.get_metrics_with_physical_library_id(physical_library_id);

        let active_conditions = self.active_metric_conditions(section);

        let mut metrics = self.lib_metrics_proc.process(
            section,
            &active_conditions,
            &self.alert_context,
            &lib_metrics,
        )?;

        self.add_cell_metrics(section, library_type, physical_library_id, &mut metrics)?;

        let per_probe_barcode_data =
            self.build_rtl_probe_barcode_metrics(library_type, section, &active_conditions)?;
        // NOTE: we do not currently show these metrics for Custom, is this an oversight?
        // metrics.extend(per_probe_barcode_data.metrics.drain(..));

        metrics.extend(self.get_sequencing_metrics(section, library_type, &active_conditions)?);

        Ok(LibraryCustomFeatureWebSummary {
            parameters_table: self.count_param_table(library_type, per_probe_barcode_data)?,
            barcode_rank_plot: format_barcode_rank_plot(
                self.barcode_rank_plots
                    .get(&LibraryType::Custom)
                    .expect("Custom feature barcode rank plot was missing."),
                "Custom",
            ),
            metrics: MetricsTraitWrapper(metrics),
        })
    }

    fn build_cmo_ws(&self, physical_library_id: &str) -> Result<LibraryCmoWebSummary> {
        let library_type = LibraryType::Cellplex;
        let section = Section::Cmo;
        let lib_metrics = self.get_metrics_with_physical_library_id(physical_library_id);

        let active_conditions = self.active_metric_conditions(section);

        let mut metrics = self.lib_metrics_proc.process(
            section,
            &active_conditions,
            &self.alert_context,
            &lib_metrics,
        )?;

        metrics.extend(self.get_sequencing_metrics(section, library_type, &active_conditions)?);

        metrics.extend(self.build_per_tag_metrics(section, &active_conditions)?);

        let (jibes_biplot, jibes_histogram, resources): (
            Option<RawChartWithHelp>,
            Option<RawChartWithHelp>,
            TxHashMap<String, Value>,
        ) = match self.jibes_biplot_histogram.clone() {
            Some(jibes_biplot) => {
                let jibes_biplot_histogram: JibesBiplotHistogramData =
                    serde_json::from_value(jibes_biplot)?;
                (
                    Some(format_jibes_biplots(&jibes_biplot_histogram.biplot, "CMO")),
                    Some(format_histogram(&jibes_biplot_histogram.histogram, "CMO")),
                    jibes_biplot_histogram.resources.clone(),
                )
            }
            None => (None, None, TxHashMap::default()),
        };

        Ok(LibraryCmoWebSummary {
            parameters_table: self.count_param_table(LibraryType::Cellplex, Default::default())?,
            barcode_rank_plot: format_barcode_rank_plot(
                self.barcode_rank_plots
                    .get(&LibraryType::Cellplex)
                    .expect("CMO barcode rank plot was missing."),
                "CMO",
            ),
            jibes_biplot,
            jibes_histogram,
            cmo_umi_projection_plot: self.cmo_projection_plot.as_ref().map(|x| {
                format_umi_on_umap_plot(
                    &x.cmo_umi_projection_plot,
                    "Multiplexing Capture",
                    "UMAP Projection of Cells Colored by UMI Counts",
                )
            }),
            cmo_tags_projection_plot: self.cmo_projection_plot.as_ref().map(|x| {
                format_tags_on_umap_plot(&x.cmo_tags_projection_plot, "Multiplexing Capture")
            }),
            resources,
            metrics: MetricsTraitWrapper(metrics),
        })
    }

    fn build_hashtag_ws(&self) -> Result<LibraryHashtagWebSummary> {
        let library_type = LibraryType::Antibody;
        let section = Section::Hashtag;
        let active_conditions = self.active_metric_conditions(section);

        let mut metrics = self.lib_metrics_proc.process(
            section,
            &active_conditions,
            &self.alert_context,
            &self.lib_metrics,
        )?;

        metrics.extend(self.build_per_tag_metrics(section, &active_conditions)?);

        let (jibes_biplot, jibes_histogram, resources): (
            Option<RawChartWithHelp>,
            Option<RawChartWithHelp>,
            TxHashMap<String, Value>,
        ) = match self.jibes_biplot_histogram.clone() {
            Some(jibes_biplot) => {
                let jibes_biplot_histogram: JibesBiplotHistogramData =
                    serde_json::from_value(jibes_biplot)?;
                (
                    Some(format_jibes_biplots(
                        &jibes_biplot_histogram.biplot,
                        "Hashtag",
                    )),
                    Some(format_histogram(
                        &jibes_biplot_histogram.histogram,
                        "Hashtag",
                    )),
                    jibes_biplot_histogram.resources.clone(),
                )
            }
            None => (None, None, TxHashMap::default()),
        };

        Ok(LibraryHashtagWebSummary {
            parameters_table: self.count_param_table(library_type, Default::default())?,
            jibes_biplot,
            jibes_histogram,
            hashtag_umi_projection_plot: self.cmo_projection_plot.as_ref().map(|x| {
                format_umi_on_umap_plot(
                    &x.cmo_umi_projection_plot,
                    "Antibody Capture",
                    "UMAP Projection of Cells Colored by Hashtag UMI Counts",
                )
            }),
            hashtag_tags_projection_plot: self
                .cmo_projection_plot
                .as_ref()
                .map(|x| format_tags_on_umap_plot(&x.cmo_tags_projection_plot, "Antibody Capture")),
            resources,
            metrics: MetricsTraitWrapper(metrics),
        })
    }

    fn build_per_tag_metrics(
        &self,
        section: Section,
        active_conditions: &ActiveConditions,
    ) -> Result<Vec<JsonMetricSummary>> {
        let mut special_metric_proc = self.special_metrics_proc.with_default_transformers();
        special_metric_proc.add_transformer(
            "CellsFraction",
            CountAndPercentTransformer::new(
                get_metric_usize(&self.lib_metrics, "total_singlets")?.unwrap(),
            ),
        );

        let group = match section {
            Section::Cmo => "cmo_per_tag_metrics",
            Section::Hashtag => "hashtag_per_tag_metrics",
            _ => panic!("invalid section to create per tag metrics: {section}"),
        };

        #[derive(JsonReport)]
        struct Metrics<'a> {
            tag_id: &'a str,
            sample_id: &'a str,
            /// tag_{tag_name}_frac_reads_in_cells
            /// absent if no cells assigned to tag
            tag_reads_in_cell_associated_partitions: Option<f64>,
            /// tag_{tag_name}_number_of_singlets
            singlets_assigned_to_tag: usize,
            /// snr_{tag_name}_jibes
            /// absent if no cells assigned to tag
            tag_signal_to_background_ratio: Option<f64>,
        }

        let mut metrics = vec![];
        for sample in &self.multi_graph.samples {
            for fingerprint in &sample.fingerprints {
                let Fingerprint::Tagged { tag_name, .. } = fingerprint else {
                    panic!("Unable to process Tag metrics group involving untagged fingerprint.");
                };

                let singlets_assigned_to_tag_key = format!("tag_{tag_name}_number_of_singlets");

                metrics.extend(
                    special_metric_proc.process_group(
                        group,
                        section,
                        active_conditions,
                        &self.alert_context,
                        &Metrics {
                            tag_id: tag_name.as_str(),
                            sample_id: sample.sample_id.as_str(),
                            tag_reads_in_cell_associated_partitions: get_metric_f64(
                                &self.lib_metrics,
                                &format!("tag_{tag_name}_frac_reads_in_cells"),
                            )?,
                            singlets_assigned_to_tag: get_metric_usize(
                                &self.lib_metrics,
                                &singlets_assigned_to_tag_key,
                            )?
                            .expect(&singlets_assigned_to_tag_key),
                            tag_signal_to_background_ratio: get_metric_f64(
                                &self.lib_metrics,
                                &format!("snr_{tag_name}_jibes"),
                            )?,
                        },
                    )?,
                );
            }
        }
        Ok(metrics)
    }

    /// Build sequencing metrics for a particular library type.
    fn get_sequencing_metrics(
        &self,
        section: Section,
        library_type: LibraryType,
        active_conditions: &ActiveConditions,
    ) -> Result<Vec<JsonMetricSummary>> {
        let mut metrics = vec![];
        let Some(all_seq_metrics) = self.sequencing_metrics.get(&library_type) else {
            panic!("sequencing metrics for {library_type} not found")
        };
        for seq_metrics in all_seq_metrics {
            metrics.extend(self.special_metrics_proc.process_group(
                "sequencing_metrics",
                section,
                active_conditions,
                &self.alert_context,
                seq_metrics,
            )?);
        }
        Ok(metrics)
    }
}

#[derive(Default)]
struct PerProbeBarcodeData {
    metrics: Vec<JsonMetricSummary>,
    unexpected: Vec<String>,
    missing: Vec<String>,
}

// describes the various UMAP plots a single sample may have.
#[derive(Serialize, Deserialize, Clone, Default)]
pub struct SampleUmapPlots {
    // full UMAP/clustering/diffexp plots for gene expression run
    gex_diffexp_clustering_plots: Value,
    // full UMAP/clustering/diffexp plots for antibody-only case only
    antibody_diffexp_clustering_plots: Value,
    // the normal colored-umi UMAPs that feature barcode libraries get otherwise
    crispr_umi_on_umap: Value,
    antibody_umi_on_umap: Value,
    custom_umi_on_umap: Value,
}

// describes the various UMAP plots a multiplexing experiment may have.
#[derive(Serialize, Deserialize, Clone, Default)]
pub struct MultiplexingUmapPlots {
    cmo_umi_projection_plot: Value,
    cmo_tags_projection_plot: Value,
}

struct MultiWsBuilder {
    lib_ws_builder: LibWsBuilder,
    per_sample_metrics: TxHashMap<SampleAssignment, TxHashMap<String, Value>>,
    sample_metrics_proc: MetricsProcessor,
    sample_barcode_rank_plots: TxHashMap<SampleAssignment, TxHashMap<LibraryType, PlotlyChart>>,
    sample_treemap_plots:
        Option<TxHashMap<SampleAssignment, TxHashMap<LibraryType, RawChartWithHelp>>>,
    sample_projection_plots: TxHashMap<SampleAssignment, SampleUmapPlots>,
    sample_antibody_histograms: Option<TxHashMap<SampleAssignment, RawChartWithHelp>>,
    svg_str: String,
    csv_str: String,
    is_barnyard: bool,
    diagnostics: MultiDiagnostics,
    pipeline_version: String,
    alert_context: AlertContext,
    multiplexing_method: Option<BarcodeMultiplexingType>,
    targeting_method: Option<TargetingMethod>,
    antigen_vdj_metrics: Option<TxHashMap<SampleAssignment, AntigenVdjMetrics>>,
    clonotype_clustermap: Option<TxHashMap<SampleAssignment, RawChartWithHelp>>,
    vdj_t_contents: Option<TxHashMap<SampleAssignment, VdjWsContents>>,
    vdj_t_gd_contents: Option<TxHashMap<SampleAssignment, VdjWsContents>>,
    vdj_b_contents: Option<TxHashMap<SampleAssignment, VdjWsContents>>,
    cell_annotation_barcharts: Option<TxHashMap<SampleAssignment, Value>>,
    cell_annotation_box_plots: Option<TxHashMap<SampleAssignment, Value>>,
    cell_annotation_umap_plots: Option<TxHashMap<SampleAssignment, Value>>,
    cell_annotation_diffexp_tables:
        Option<TxHashMap<SampleAssignment, DifferentialExpressionTable>>,
    cell_annotation_metrics: Option<TxHashMap<SampleAssignment, CellAnnotationMetrics>>,
    cell_annotation_viable_but_not_requested: Option<TxHashMap<SampleAssignment, Option<bool>>>,
}

impl MultiWsBuilder {
    /// Return true if multiplexed using CMO.
    fn is_cmo_multiplexed(&self) -> bool {
        self.multiplexing_method == Some(BarcodeMultiplexingType::CellLevel(CellLevel::CMO))
    }

    /// Return true if multiplexed using Hashtag.
    fn is_hashtag_multiplexed(&self) -> bool {
        self.multiplexing_method == Some(BarcodeMultiplexingType::CellLevel(CellLevel::Hashtag))
    }

    /// Return true if multiplexed using RTL .
    fn is_rtl_multiplexed(&self) -> bool {
        self.multiplexing_method == Some(BarcodeMultiplexingType::ReadLevel(ReadLevel::RTL))
    }

    /// Return true if multiplexed using OH.
    fn is_overhang_multiplexed(&self) -> bool {
        self.multiplexing_method == Some(BarcodeMultiplexingType::ReadLevel(ReadLevel::OH))
    }

    /// Return the multi experimental design graph.
    fn multi_graph(&self) -> &CrMultiGraph {
        &self.lib_ws_builder.multi_graph
    }

    fn build(self) -> Result<TxHashMap<SampleAssignment, MultiWebSummary>> {
        let mut result = TxHashMap::default();

        let mut library_websummary = self
            .lib_ws_builder
            .build(self.pipeline_version.clone(), &self.alert_context)?;

        let multi_graph = self.multi_graph();

        let mut resources = TxHashMap::default();
        resources.extend(library_websummary.resources.drain());
        // iterate over samples
        // read in their respective metrics JSONs
        // populate the sample web summary JSON for each one
        for (i, sample) in multi_graph.samples.iter().enumerate() {
            let mut sample_ws = SampleWebSummary {
                header_info: SampleHeaderInfo {
                    sample_id: sample.sample_id.clone(),
                    sample_desc: sample.description.clone(),
                    pipeline_version: self.pipeline_version.clone(),
                },
                vdj_t_tab: self.vdj_t_contents.as_ref().map(|content| {
                    Tab::new(
                        content[&SampleAssignment::Assigned(sample.sample_id.clone())]
                            .clone()
                            .to_sample_ws(),
                        &self.alert_context,
                    )
                }),
                vdj_b_tab: self.vdj_b_contents.clone().map(|content| {
                    Tab::new(
                        content[&SampleAssignment::Assigned(sample.sample_id.clone())]
                            .clone()
                            .to_sample_ws(),
                        &self.alert_context,
                    )
                }),
                vdj_t_gd_tab: self.vdj_t_gd_contents.clone().map(|content| {
                    Tab::new(
                        content[&SampleAssignment::Assigned(sample.sample_id.clone())]
                            .clone()
                            .to_sample_ws(),
                        &self.alert_context,
                    )
                }),
                ..Default::default()
            };
            let sample_diagnostics = SampleDiagnostics {
                vdj_t: self.vdj_t_contents.as_ref().map(|content| {
                    VdjWsContents::diagnostics(
                        &content[&SampleAssignment::Assigned(sample.sample_id.clone())],
                    )
                }),
                vdj_t_gd: self.vdj_t_gd_contents.as_ref().map(|content| {
                    VdjWsContents::diagnostics(
                        &content[&SampleAssignment::Assigned(sample.sample_id.clone())],
                    )
                }),
                vdj_b: self.vdj_b_contents.as_ref().map(|content| {
                    VdjWsContents::diagnostics(
                        &content[&SampleAssignment::Assigned(sample.sample_id.clone())],
                    )
                }),
            };
            let sample_assignment = SampleAssignment::Assigned(sample.sample_id.clone());
            for lib in &multi_graph.libraries {
                match lib.library_type {
                    LibraryType::Gex => {
                        assert!(sample_ws.gex_tab.is_none());

                        sample_ws.gex_tab = Some(Tab::new(
                            self.build_gex_ws(&sample_assignment, &lib.physical_library_id)?,
                            &self.alert_context,
                        ));
                    }
                    LibraryType::Antibody => {
                        assert!(sample_ws.antibody_tab.is_none());
                        // For Flex, show the antibody tab if this sample has antibody reads
                        // or it has an antibody multiplexing barcode assigned to it.
                        let antibody_reads = self.per_sample_metrics[&sample_assignment]
                            ["ANTIBODY_total_read_pairs"]
                            .as_u64()
                            .unwrap();
                        if sample.barcode_multiplexing_type()
                            == Some(BarcodeMultiplexingType::ReadLevel(ReadLevel::RTL))
                            && antibody_reads == 0
                            && !sample_is_associated_with_flex_library_type(
                                sample,
                                LibraryType::Antibody,
                            )
                        {
                            continue;
                        }
                        sample_ws.antibody_tab = Some(Tab::new(
                            self.build_antibody_ws(&sample_assignment, &lib.physical_library_id)?,
                            &self.alert_context,
                        ));
                    }
                    LibraryType::Antigen => {
                        assert!(sample_ws.antigen_tab.is_none());
                        sample_ws.antigen_tab = Some(Tab::new(
                            self.build_antigen_ws(&sample_assignment, &lib.physical_library_id)?,
                            &self.alert_context,
                        ));
                    }
                    LibraryType::Crispr => {
                        assert!(sample_ws.crispr_tab.is_none());
                        sample_ws.crispr_tab = Some(Tab::new(
                            self.build_crispr_ws(&sample_assignment, &lib.physical_library_id)?,
                            &self.alert_context,
                        ));
                    }
                    LibraryType::Vdj(_) => {}
                    LibraryType::Custom => {
                        assert!(sample_ws.custom_feature_tab.is_none());
                        sample_ws.custom_feature_tab = Some(Tab::new(
                            self.build_custom_ws(&sample_assignment, &lib.physical_library_id)?,
                            &self.alert_context,
                        ));
                    }
                    LibraryType::Cellplex => {}
                    LibraryType::Atac => unreachable!(),
                }
            }

            let put_in_cell_annotation_tab = [
                self.cell_annotation_barcharts
                    .as_ref()
                    .is_some_and(|x| x.contains_key(&sample_assignment)),
                self.cell_annotation_box_plots
                    .as_ref()
                    .is_some_and(|x| x.contains_key(&sample_assignment)),
                self.cell_annotation_umap_plots
                    .as_ref()
                    .is_some_and(|x| x.contains_key(&sample_assignment)),
                self.cell_annotation_diffexp_tables
                    .as_ref()
                    .is_some_and(|x| x.contains_key(&sample_assignment)),
                self.cell_annotation_metrics
                    .as_ref()
                    .is_some_and(|x| x.contains_key(&sample_assignment)),
            ]
            .iter()
            .any(|x| *x);
            if put_in_cell_annotation_tab {
                sample_ws.cell_annotation_tab = Some(Tab::new(
                    self.build_cell_annotation_ws(&sample_assignment)?,
                    &self.alert_context,
                ));
            }

            result.insert(
                sample_assignment,
                MultiWebSummary {
                    sample: WsSample::multi(sample.sample_id.clone(), sample.description.clone()),
                    library: MultiWebSummaryLibraryData {
                        metrics: library_websummary.to_json_summary(),
                        types: multi_graph
                            .libraries
                            .iter()
                            .map(|lib| lib.library_type)
                            .sorted()
                            .dedup()
                            .collect(),
                        data: library_websummary.clone(),
                    },
                    per_sample: vec![MultiWebSummarySampleData {
                        metrics: sample_ws.to_json_summary(),
                        data: sample_ws,
                    }],
                    experimental_design: ExperimentalDesign {
                        svg: SvgGraph::new(
                            self.svg_str.clone(),
                            format!("sample_{}", i + 1),
                            self.multiplexing_method,
                        ),
                        csv: self.csv_str.clone(),
                        multiplexing_method: self.multiplexing_method,
                        is_rtl: self.lib_ws_builder.is_rtl,
                        is_barnyard: self.is_barnyard,
                    },
                    diagnostics: self.diagnostics.clone(),
                    sample_diagnostics: vec![sample_diagnostics],
                    resources: resources.clone(),
                },
            );
        }
        Ok(result)
    }

    /// Return a clone of the sample-level metrics, with the provided physical library ID injected.
    fn get_metrics_for_sample_with_physical_library_id(
        &self,
        sample_assignment: &SampleAssignment,
        physical_library_id: &str,
    ) -> TxHashMap<String, Value> {
        let mut metrics = self
            .per_sample_metrics
            .get(sample_assignment)
            .expect("Sample metrics file not found.")
            .clone();
        metrics.insert(
            "physical_library_id".to_string(),
            json!(physical_library_id),
        );
        metrics
    }

    fn build_gex_ws(
        &self,
        sample_assignment: &SampleAssignment,
        physical_library_id: &str,
    ) -> Result<SampleGexWebSummary> {
        let library_type = LibraryType::Gex;
        let section = Section::Gex;

        let sample_metrics = self.get_metrics_for_sample_with_physical_library_id(
            sample_assignment,
            physical_library_id,
        );

        let active_conditions = self.lib_ws_builder.active_metric_conditions(section);

        let mut metrics = self.sample_metrics_proc.process(
            section,
            &active_conditions,
            &self.alert_context,
            &sample_metrics,
        )?;

        let genomes = self.lib_ws_builder.genomes();

        let disclaimer_banner = if self
            .cell_annotation_viable_but_not_requested
            .as_ref()
            .and_then(|x| x.get(sample_assignment).copied())
            .flatten()
            .unwrap_or(false)
        {
            Some(CELL_ANNOTATION_ADVERTISEMENT_STRING.to_string())
        } else {
            None
        };

        #[derive(JsonReport)]
        struct HeroMetrics<'a> {
            genome: Option<&'a str>,
            total_singlets: Option<usize>,
            mean_reads_per_cell: Option<f64>,
            median_genes_per_singlet: Option<f64>,
            total_genes_detected: Option<usize>,
            median_umi_per_singlet: Option<f64>,
            confidently_mapped_reads_in_cells: Option<f64>,
        }

        for genome in genomes {
            let mean_reads_per_cell = if genomes.len() >= 2 {
                // Barnyard metrics are prefixed by genome.
                get_metric_f64(
                    &sample_metrics,
                    &format!("{genome}_filtered_bcs_conf_mapped_barcoded_reads_per_filtered_bc"),
                )?
            } else {
                // Non-barynard metrics are not prefixed by genome.
                get_metric_f64(&sample_metrics, "filtered_reads_per_filtered_bc")?
            };
            let (median_genes_per_singlet, total_genes_detected, median_umi_per_singlet) = {
                let (keys, prefix) = match self.targeting_method {
                    Some(TargetingMethod::TemplatedLigation) => {
                        // Use the filtered probe set metrics.
                        // Targeted metrics are prefixed by genome only if there are >=2 genomes.
                        (
                            [
                                "median_genes_per_cell_on_target",
                                "num_genes_detected_on_target",
                                "median_umis_per_cell_on_target",
                            ],
                            (genomes.len() >= 2).then_some(genome.as_str()),
                        )
                    }
                    // Non-targeted metrics are always prefixed by genome, even if there is only one genome.
                    _ => (
                        [
                            "filtered_bcs_median_unique_genes_detected",
                            "filtered_bcs_total_unique_genes_detected",
                            "filtered_bcs_median_counts",
                        ],
                        Some(genome.as_str()),
                    ),
                };
                let keys = keys.map(|key| join_metric_name(prefix, key));
                (
                    get_metric_f64(&sample_metrics, &keys[0]).unwrap(),
                    get_metric_usize(&sample_metrics, &keys[1]).unwrap(),
                    get_metric_f64(&sample_metrics, &keys[2]).unwrap(),
                )
            };

            let confidently_mapped_reads_in_cells =
                if self.is_cmo_multiplexed() || self.is_hashtag_multiplexed() {
                    // CMO and HASHTAG multiplexing does not demultiplex reads outside of cells.
                    None
                } else {
                    get_metric_f64(
                        &sample_metrics,
                        &format!("{genome}_filtered_bcs_conf_mapped_barcoded_reads_cum_frac"),
                    )?
                };

            metrics.extend(self.lib_ws_builder.special_metrics_proc.process_group(
                "gex_sample_hero_metrics",
                section,
                &active_conditions,
                &self.alert_context,
                &HeroMetrics {
                    genome: (genomes.len() >= 2).then_some(genome),
                    total_singlets: get_metric_usize(
                        &sample_metrics,
                        &format!("{genome}_singlets_assigned_to_this_sample"),
                    )?,
                    mean_reads_per_cell,
                    median_genes_per_singlet,
                    total_genes_detected,
                    median_umi_per_singlet,
                    confidently_mapped_reads_in_cells,
                },
            )?);
        }

        let barcode_rank_plot = if self.is_rtl_multiplexed() | self.is_overhang_multiplexed() {
            Some(format_barcode_rank_plot(
                self.sample_barcode_rank_plots
                    .get(sample_assignment)
                    .unwrap()
                    .get(&LibraryType::Gex)
                    .expect("GEX sample barcode rank plot not found"),
                "GEX",
            ))
        } else {
            None
        };

        let per_probe_barcode_data = self.lib_ws_builder.build_rtl_probe_barcode_metrics(
            library_type,
            section,
            &active_conditions,
        )?;
        Ok(SampleGexWebSummary {
            parameters_table: self
                .lib_ws_builder
                .count_param_table(library_type, per_probe_barcode_data)?,
            median_genes_per_cell_plot: if self.multiplexing_method.is_some() {
                Some(sample_median_genes_plot_from_metrics(
                    &sample_metrics,
                    genomes.iter().cloned().collect(),
                    PlotType::SamplePlot,
                    self.targeting_method,
                ))
            } else {
                None
            },
            clustering_and_diffexp_plots: self
                .sample_projection_plots
                .get(sample_assignment)
                .cloned()
                .unwrap_or_default()
                .gex_diffexp_clustering_plots,
            barcode_rank_plot,
            disclaimer: disclaimer_banner,
            metrics: MetricsTraitWrapper(metrics),
        })
    }

    fn build_antibody_ws(
        &self,
        sample_assignment: &SampleAssignment,
        physical_library_id: &str,
    ) -> Result<SampleAntibodyWebSummary> {
        let library_type = LibraryType::Antibody;
        let section = Section::Antibody;

        let sample_metrics = self.get_metrics_for_sample_with_physical_library_id(
            sample_assignment,
            physical_library_id,
        );

        let active_conditions = self.lib_ws_builder.active_metric_conditions(section);

        let metrics = self.sample_metrics_proc.process(
            section,
            &active_conditions,
            &self.alert_context,
            &sample_metrics,
        )?;

        let projection_plots = self
            .sample_projection_plots
            .get(sample_assignment)
            .cloned()
            .unwrap_or_default();

        // if this is antibody_only case, we don't show the single umi-on-UMAP plot.
        let umi_on_projection_plot = match projection_plots.antibody_diffexp_clustering_plots {
            Value::Null => Some(format_umi_on_umap_plot(
                &projection_plots.antibody_umi_on_umap,
                "Antibody Capture",
                "UMAP Projection",
            )),
            _ => None,
        };

        let antibody_treemap: Option<RawChartWithHelp> =
            if let Some(ref sample_treemap) = self.sample_treemap_plots {
                sample_treemap[sample_assignment]
                    .get(&LibraryType::Antibody)
                    .cloned()
            } else {
                None
            };

        let barcode_rank_plot = if self.is_rtl_multiplexed() {
            Some(format_barcode_rank_plot(
                &self.sample_barcode_rank_plots[sample_assignment][&LibraryType::Antibody],
                "Antibody",
            ))
        } else {
            None
        };

        let per_probe_barcode_data = self.lib_ws_builder.build_rtl_probe_barcode_metrics(
            library_type,
            section,
            &active_conditions,
        )?;

        Ok(SampleAntibodyWebSummary {
            parameters_table: self
                .lib_ws_builder
                .count_param_table(library_type, per_probe_barcode_data)?,
            antibody_treemap,
            clustering_and_diffexp_plots: Some(projection_plots.antibody_diffexp_clustering_plots),
            projection_plot: umi_on_projection_plot,
            barcode_rank_plot,
            feature_histogram: self
                .sample_antibody_histograms
                .as_ref()
                .map(|histos_per_sample| histos_per_sample[sample_assignment].clone()),
            metrics: MetricsTraitWrapper(metrics),
        })
    }

    fn build_antigen_ws(
        &self,
        sample_assignment: &SampleAssignment,
        physical_library_id: &str,
    ) -> Result<SampleAntigenWebSummary> {
        let library_type = LibraryType::Antigen;
        let section = Section::Antigen;

        let mut sample_metrics = self.get_metrics_for_sample_with_physical_library_id(
            sample_assignment,
            physical_library_id,
        );
        sample_metrics.insert(
            "feature_type".to_string(),
            Value::from(LibraryType::Gex.to_string()),
        );

        let active_conditions = self.lib_ws_builder.active_metric_conditions(section);

        let mut metrics = self.sample_metrics_proc.process(
            section,
            &active_conditions,
            &self.alert_context,
            &sample_metrics,
        )?;

        if let Some(antigen_vdj_metrics) = self.antigen_vdj_metrics.as_ref() {
            #[allow(non_snake_case)]
            #[derive(JsonReport)]
            struct Metrics<'a> {
                feature_type: &'a str,
                ANTIGEN_multi_filtered_bcs: usize,
                ANTIGEN_multi_filtered_bcs_median_counts: f64,
                ANTIGEN_multi_usable_reads_per_filtered_bc: f64,
            }

            let antigen_vdj_metrics =
                antigen_vdj_metrics
                    .get(sample_assignment)
                    .unwrap_or_else(|| {
                        panic!("Sample antigen metrics file not found for {sample_assignment}")
                    });

            metrics.extend(
                self.sample_metrics_proc.process_group(
                    "antigen_sample_hero_metrics",
                    section,
                    &active_conditions,
                    &self.alert_context,
                    &Metrics {
                        feature_type: self
                            .lib_ws_builder
                            .vdj_tab_names()
                            .into_iter()
                            .flatten()
                            .exactly_one() // Exactly 1 VDJ library guaranteed for antigen runs
                            .unwrap(),
                        ANTIGEN_multi_filtered_bcs: antigen_vdj_metrics.num_cells as usize,
                        ANTIGEN_multi_filtered_bcs_median_counts: antigen_vdj_metrics
                            .median_umis_per_cell,
                        ANTIGEN_multi_usable_reads_per_filtered_bc: antigen_vdj_metrics
                            .mean_usable_reads_per_cell,
                    },
                )?,
            );
        }

        let antigen_treemap: Option<RawChartWithHelp> =
            if let Some(ref sample_treemap) = self.sample_treemap_plots {
                sample_treemap
                    .get(sample_assignment)
                    .unwrap()
                    .get(&LibraryType::Antigen)
                    .cloned()
            } else {
                None
            };

        let per_probe_barcode_data = self.lib_ws_builder.build_rtl_probe_barcode_metrics(
            library_type,
            section,
            &active_conditions,
        )?;

        Ok(SampleAntigenWebSummary {
            parameters_table: self
                .lib_ws_builder
                .count_param_table(library_type, per_probe_barcode_data)?,
            antigen_treemap,
            clonotype_clustermap: self
                .clonotype_clustermap
                .as_ref()
                .and_then(|c| c.get(sample_assignment).cloned()),
            metrics: MetricsTraitWrapper(metrics),
        })
    }

    fn build_crispr_ws(
        &self,
        sample_assignment: &SampleAssignment,
        physical_library_id: &str,
    ) -> Result<SampleCrisprWebSummary> {
        let library_type = LibraryType::Crispr;
        let section = Section::Crispr;

        let sample_metrics = self.get_metrics_for_sample_with_physical_library_id(
            sample_assignment,
            physical_library_id,
        );

        let active_conditions = self.lib_ws_builder.active_metric_conditions(section);

        let metrics = self.sample_metrics_proc.process(
            section,
            &active_conditions,
            &self.alert_context,
            &sample_metrics,
        )?;

        let per_probe_barcode_data = self.lib_ws_builder.build_rtl_probe_barcode_metrics(
            library_type,
            section,
            &active_conditions,
        )?;

        Ok(SampleCrisprWebSummary {
            parameters_table: self
                .lib_ws_builder
                .count_param_table(library_type, per_probe_barcode_data)?,
            projection_plot: self
                .sample_projection_plots
                .get(sample_assignment)
                .map(|x| {
                    format_umi_on_umap_plot(
                        &x.crispr_umi_on_umap,
                        "CRISPR Guide Capture",
                        "UMAP Projection",
                    )
                }),
            barcode_rank_plot: None,
            metrics: MetricsTraitWrapper(metrics),
        })
    }

    fn build_custom_ws(
        &self,
        sample_assignment: &SampleAssignment,
        physical_library_id: &str,
    ) -> Result<SampleCustomFeatureWebSummary> {
        let library_type = LibraryType::Custom;
        let section = Section::Custom;

        let sample_metrics = self.get_metrics_for_sample_with_physical_library_id(
            sample_assignment,
            physical_library_id,
        );

        let active_conditions = self.lib_ws_builder.active_metric_conditions(section);

        let metrics = self.sample_metrics_proc.process(
            section,
            &active_conditions,
            &self.alert_context,
            &sample_metrics,
        )?;

        let per_probe_barcode_data = self.lib_ws_builder.build_rtl_probe_barcode_metrics(
            library_type,
            section,
            &active_conditions,
        )?;

        Ok(SampleCustomFeatureWebSummary {
            parameters_table: self
                .lib_ws_builder
                .count_param_table(library_type, per_probe_barcode_data)?,
            projection_plot: Some(format_umi_on_umap_plot(
                &self
                    .sample_projection_plots
                    .get(sample_assignment)
                    .cloned()
                    .unwrap_or_default()
                    .custom_umi_on_umap,
                "Custom Feature",
                "UMAP Projection",
            )),
            barcode_rank_plot: None,
            metrics: MetricsTraitWrapper(metrics),
        })
    }

    fn build_cell_annotation_ws(
        &self,
        sample_assignment: &SampleAssignment,
    ) -> Result<SampleCellAnnotationWebSummary> {
        let cell_annotation_barchart = self
            .cell_annotation_barcharts
            .as_ref()
            .and_then(|c| c.get(sample_assignment).cloned());
        let cell_annotation_violin_plot = self
            .cell_annotation_box_plots
            .as_ref()
            .and_then(|c| c.get(sample_assignment).cloned());
        let cell_annotation_umap_plot = self
            .cell_annotation_umap_plots
            .as_ref()
            .and_then(|c| c.get(sample_assignment).cloned());
        let cell_annotation_diffexp_table = self
            .cell_annotation_diffexp_tables
            .as_ref()
            .and_then(|c| c.get(sample_assignment).cloned());
        let cell_annotation_metric_struct = self
            .cell_annotation_metrics
            .as_ref()
            .and_then(|c| c.get(sample_assignment).cloned());
        Ok(SampleCellAnnotationWebSummary {
            cas_success: cell_annotation_metric_struct
                .as_ref()
                .and_then(|x| x.cell_annotation_success),
            cell_annotation_disable_differential_expression: cell_annotation_metric_struct
                .as_ref()
                .and_then(|x| x.cell_annotation_differential_expression),
            disclaimer: cell_annotation_metric_struct
                .as_ref()
                .and_then(CellAnnotationMetrics::generate_disclaimer_html_fragment),
            parameters_table: cell_annotation_metric_struct
                .map(generate_cell_type_parameter_table)
                .transpose()?,
            cell_annotation_cell_types_chart: cell_annotation_barchart
                .map(generate_cell_type_barchart_from_value)
                .transpose()?,
            cell_annotation_violin_plot_chart: cell_annotation_violin_plot
                .map(generate_cell_type_violin_plot_from_value)
                .transpose()?,
            cell_annotation_umap_plot_chart: cell_annotation_umap_plot
                .map(generate_cell_type_umap_plot_from_value)
                .transpose()?,
            cell_annotation_diffexp_table: cell_annotation_diffexp_table
                .map(generate_cell_type_diffexp_from_value)
                .transpose()?,
        })
    }
}

fn get_metric(metrics: &TxHashMap<String, Value>, key: &str) -> Result<Option<Number>> {
    match metrics.get(key) {
        Some(val) => match val {
            Value::Number(n) => Ok(Some(n.clone())),
            Value::String(s) => {
                if s.to_lowercase() == "nan" {
                    Ok(None)
                } else {
                    bail!("JSON metric {} had unexpected type {:?}", key, val);
                }
            }
            _ => {
                bail!("JSON metric {} had unexpected type {:?}", key, val);
            }
        },
        None => Ok(None),
    }
}

fn get_metric_usize(metrics: &TxHashMap<String, Value>, key: &str) -> Result<Option<usize>> {
    Ok(get_metric(metrics, key)?.map(|x| {
        x.as_f64()
            .expect("Error converting metric to float in get_metric_usize.")
            .round() as usize
    }))
}

fn get_metric_f64(metrics: &TxHashMap<String, Value>, key: &str) -> Result<Option<f64>> {
    Ok(get_metric(metrics, key)?.map(|x| {
        x.as_f64()
            .expect("Error converting metric to float in get_metric_f64.")
    }))
}

fn get_metric_string(metrics: &TxHashMap<String, Value>, key: &str) -> Result<Option<String>> {
    match metrics.get(key) {
        Some(val) => match val {
            Value::String(n) => Ok(Some(n.clone())),
            Value::Bool(b) => Ok(Some(b.to_string())),
            Value::Number(n) => Ok(Some(n.to_string())),
            Value::Null => Ok(None),
            Value::Array(_) | Value::Object(_) => {
                bail!(
                    // probably makes sense to have these error out
                    "JSON metric {} was accessed as a String but was an Array or an Object.",
                    key
                );
            }
        },
        None => Ok(None),
    }
}

fn identify_dropped_tags(
    multi_cfg_file: &MultiConfigCsvFile,
    tag_contaminant_info: &Option<Value>,
) -> Vec<String> {
    let Some(contaminants) = tag_contaminant_info.as_ref() else {
        return Default::default();
    };

    // Load expected tags
    let cfg = multi_cfg_file
        .read()
        .expect("Could not load configuration CSV.");
    let used_tags = cfg.sample_barcode_ids_used_in_experiment(ProbeBarcodeIterationMode::Mapped);

    contaminants
        .as_object()
        .expect("Contaminant info not a map.")
        .iter()
        .filter(|t| {
            t.1["is_contaminant"]
                .as_bool()
                .expect("Contaminant flag missing from JSON")
                & used_tags.contains(t.0)
        })
        .map(|z| z.0.to_string())
        .collect()
}

/// Return true if this sample is associated with the provided FLEX-compatible library type.
/// Return false if not multiplexed, or if the multiplexing type is not RTL.
fn sample_is_associated_with_flex_library_type(sample: &Sample, library_type: LibraryType) -> bool {
    if sample.barcode_multiplexing_type()
        != Some(BarcodeMultiplexingType::ReadLevel(ReadLevel::RTL))
    {
        return false;
    }
    let expected_multi_bc_type = match library_type {
        LibraryType::Gex => RTLMultiplexingBarcodeType::Gene,
        LibraryType::Antibody => RTLMultiplexingBarcodeType::Antibody,
        LibraryType::Crispr => RTLMultiplexingBarcodeType::Crispr,
        _ => {
            return false;
        }
    };
    sample.tag_names().any(|tag| {
        categorize_rtl_multiplexing_barcode_id(tag).expect("Cannot determine multiplexing type!")
            == expected_multi_bc_type
    })
}

#[make_mro(volatile = strict, mem_gb = 22)]
impl MartianMain for WriteMultiWebSummaryJson {
    type StageInputs = StageInputs;
    type StageOutputs = StageOutputs;

    fn main(&self, args: Self::StageInputs, rover: MartianRover) -> Result<Self::StageOutputs> {
        let multi_graph = args.multi_graph.read()?;

        let lib_metrics = args
            .library_metrics
            .as_ref()
            .map(FileTypeRead::read)
            .transpose()?
            .unwrap_or_default();

        // maps sample assignment to websummary
        let mut sample_to_web_summary = TxHashMap::default();
        let mut sample_to_metrics_csv = TxHashMap::default();

        // Determine if any tags specified in the CSV config were dropped as contaminants.
        let contaminant_tags = args.tag_contaminant_info.map(|f| f.read()).transpose()?;
        let dropped_tags = identify_dropped_tags(&args.multi_config, &contaminant_tags);

        // For RTL samples alert on any probe barcodes that have very high GEM barcode overlap
        let probe_barcode_overlap_coefficients = lib_metrics
            .get("probe_barcode_overlap_coefficients")
            .cloned();
        let probe_barcodes_high_gem_overlap = probe_barcode_overlap_coefficients
            .as_ref()
            .map(|coefficients| {
                coefficients
                    .as_object()
                    .unwrap()
                    .into_iter()
                    .sorted_by_key(|&(_, x)| NotNan::new(x.as_f64().unwrap()).unwrap())
                    .rev()
                    .filter_map(|(k, v)| {
                        let v_f64 = v.as_f64().unwrap();
                        if v_f64 >= GEM_BARCODE_OVERLAP_ALERT_THRESHOLD {
                            Some(format!("{k} ({:.2}%)", v_f64 * 100.0))
                        } else {
                            None
                        }
                    })
                    .collect()
            })
            .unwrap_or_default();

        let chemistries = if let Some(chemistry_defs) = &args.chemistry_defs {
            chemistry_defs.values().map(|x| x.name).collect()
        } else {
            HashSet::new()
        };

        // chemistry_defs is None for a VDJ-only analysis, which is a 5' chemistry.
        assert!(
            args.chemistry_defs.is_some()
                || args.vdj_b_contents.is_some()
                || args.vdj_t_contents.is_some()
                || args.vdj_t_gd_contents.is_some()
        );
        let is_fiveprime = args
            .chemistry_defs
            .as_ref()
            .is_none_or(|x| x.endedness() == Some(WhichEnd::FivePrime));

        let targeting_method = args.count_inputs.as_ref().and_then(|x| x.targeting_method);
        let is_rtl = targeting_method == Some(TargetingMethod::TemplatedLigation);

        let alert_context = AlertContext {
            is_rtl,
            is_arc_chemistry: chemistries.contains(&ChemistryName::ArcV1),
            library_types: multi_graph.library_types().collect(),
            is_fiveprime,
            multiplexing_method: multi_graph.barcode_multiplexing_type(),
            include_introns: args
                .count_inputs
                .as_ref()
                .is_some_and(|inputs| inputs.include_introns),
            no_preflight: args.no_preflight,
        };

        //******************************************************************************************
        // POPULATE LIBRARY WEBSUMMARIES
        //******************************************************************************************
        let (mut lib_metrics_proc, mut sample_metrics_proc) = load_metrics_etl()?;

        let cells_fraction_transformer = CountAndPercentTransformer::new(
            get_metric_usize(&lib_metrics, "total_cell_associated_partitions")?.unwrap_or_default(),
        );
        lib_metrics_proc.add_transformer("CellsFraction", cells_fraction_transformer.clone());
        sample_metrics_proc.add_transformer("CellsFraction", cells_fraction_transformer);

        let lib_ws_builder = LibWsBuilder {
            common_inputs: args.common_inputs.clone(),
            multi_graph: multi_graph.clone(),
            multi_config: args.multi_config.read()?,
            chemistry_defs: args.chemistry_defs.clone(),
            lib_metrics,
            lib_metrics_proc,
            alert_context: alert_context.clone(),
            special_metrics_proc: load_metrics_etl_special()?,
            barcode_rank_plots: match args.barcode_rank_plots {
                Some(ref plots) => plots.read()?,
                None => TxHashMap::default(),
            },
            sequencing_metrics: match args.sequencing_metrics {
                Some(f) => f.read()?,
                None => TxHashMap::default(),
            },
            count_inputs: args.count_inputs.clone(),
            count_cell_calling_config: args.count_cell_calling_config.clone(),
            jibes_biplot_histogram: args.jibes_biplot_histogram.map(|j| j.read()).transpose()?,
            antibody_histograms: args.antibody_histograms.map(|j| j.read()).transpose()?,
            antigen_histograms: args.antigen_histograms.map(|j| j.read()).transpose()?,
            cmo_projection_plot: args.cmo_projection_plot.map(|j| j.read()).transpose()?,
            vdj_t_contents: args
                .vdj_t_contents
                .clone()
                .map(|f| {
                    f.values()
                        .next()
                        .expect("Missing VDJ-T web-summary contents")
                        .read()
                })
                .transpose()?,
            vdj_t_gd_contents: args
                .vdj_t_gd_contents
                .clone()
                .map(|f| {
                    f.values()
                        .next()
                        .expect("Missing VDJ-T-GD web-summary contents")
                        .read()
                })
                .transpose()?,
            vdj_b_contents: args
                .vdj_b_contents
                .clone()
                .map(|f| {
                    f.values()
                        .next()
                        .expect("Missing VDJ-B web-summary contents")
                        .read()
                })
                .transpose()?,
            target_set_name: args.target_set_name,
            multiplexing_method: multi_graph.barcode_multiplexing_type(),
            specificity_controls: args
                .feature_config
                .unwrap_or(FeatureConfig {
                    beam_mode: None,
                    specificity_controls: None,
                    functional_map: None,
                    hashtag_ids: None,
                })
                .specificity_controls,
            dropped_tags,
            probe_barcodes_high_gem_overlap,
            mismatched_probe_barcode_pairings: get_mismatched_probe_barcode_pairings(
                &args.multi_config,
                args.detected_probe_barcode_pairing.as_ref(),
            )?,
            is_rtl,
            targeting_method,
        };

        //******************************************************************************************
        // POPULATE DIAGNOSTIC DATA
        //******************************************************************************************
        let get_metric = |metric| lib_ws_builder.lib_metrics.get(metric).cloned();
        let unknown_feature_bcs: HashMap<String, Value> = lib_ws_builder
            .lib_metrics
            .iter()
            .filter(|(key, _)| key.contains("unknown_feature_bcs"))
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect();

        let diagnostics = MultiDiagnostics {
            corrected_bc_frac: get_metric("corrected_bc_frac"),
            corrected_bc_in_gel_bead_frac: get_metric("corrected_bc_in_gel_bead_frac"),
            corrected_bc_in_probe_frac: get_metric("corrected_bc_in_probe_frac"),
            ANTIBODY_corrected_bc_frac: get_metric("ANTIBODY_corrected_bc_frac"),
            ANTIBODY_corrected_bc_in_gel_bead_frac: get_metric(
                "ANTIBODY_corrected_bc_in_gel_bead_frac",
            ),
            ANTIBODY_corrected_bc_in_probe_frac: get_metric("ANTIBODY_corrected_bc_in_probe_frac"),
            i1_bases_with_q30_frac: get_metric("i1_bases_with_q30_frac"),
            i2_bases_with_q30_frac: get_metric("i2_bases_with_q30_frac"),
            low_support_umi_reads_frac: get_metric("low_support_umi_reads_frac"),
            tag_contaminant_info: contaminant_tags,
            tso_frac: get_metric("tso_frac"),
            probe_barcode_overlap_coefficients,
            fraction_reads_high_occupancy_gems: get_metric(
                "rtl_multiplexing_fraction_reads_high_occupancy_gems",
            ),
            high_occupancy_probe_barcode_count_threshold: get_metric(
                "rtl_multiplexing_high_occupancy_probe_barcode_count_threshold",
            ),
            unknown_feature_barcode_seqs: unknown_feature_bcs,
        };

        //******************************************************************************************
        // POPULATE SAMPLE WEBSUMMARIES
        //******************************************************************************************
        let multi_ws_builder = MultiWsBuilder {
            lib_ws_builder,
            per_sample_metrics: read_optional_file_map(&args.per_sample_metrics)?,
            sample_metrics_proc,
            sample_projection_plots: read_optional_file_map(&args.sample_projection_plots)?,
            sample_barcode_rank_plots: read_optional_file_map(&args.sample_barcode_rank_plots)?,
            sample_treemap_plots: args
                .sample_treemap_plots
                .as_ref()
                .map(read_optional_file_map)
                .transpose()?,
            sample_antibody_histograms: args
                .sample_antibody_histograms
                .as_ref()
                .map(read_optional_file_map)
                .transpose()?,
            svg_str: std::fs::read_to_string(args.multi_graph_svg)
                .expect("Error reading  multi graph svg"),
            csv_str: std::fs::read_to_string(&args.multi_config)?
                .replace("\r\n", "\n")
                .replace('\r', "\n"),
            is_barnyard: is_barnyard(args.count_inputs.as_ref()),
            diagnostics,
            pipeline_version: rover.pipelines_version(),
            alert_context,
            multiplexing_method: multi_graph.barcode_multiplexing_type(),
            targeting_method: args
                .count_inputs
                .as_ref()
                .and_then(|inputs| inputs.targeting_method),
            antigen_vdj_metrics: args
                .antigen_vdj_metrics
                .as_ref()
                .map(read_optional_file_map)
                .transpose()?,
            clonotype_clustermap: args.antigen_specificity.as_ref().map(|files| {
                files
                    .iter()
                    .filter_map(|(k, opt)| {
                        opt.as_ref()
                            .map(|v| (k.clone(), clonotype_specificity_heatmap(v.clone()).unwrap()))
                    })
                    .fold(TxHashMap::default(), |mut acc, (k, v)| {
                        if let Some(v) = v {
                            acc.insert(k, v);
                        }
                        acc
                    })
            }),
            vdj_t_contents: args
                .vdj_t_contents
                .as_ref()
                .map(read_files_into_map)
                .transpose()?,
            vdj_t_gd_contents: args
                .vdj_t_gd_contents
                .as_ref()
                .map(read_files_into_map)
                .transpose()?,
            vdj_b_contents: args
                .vdj_b_contents
                .as_ref()
                .map(read_files_into_map)
                .transpose()?,
            cell_annotation_barcharts: args
                .cell_annotation_barcharts
                .as_ref()
                .map(read_optional_file_map)
                .transpose()?,
            cell_annotation_box_plots: args
                .cell_annotation_box_plots
                .as_ref()
                .map(read_optional_file_map)
                .transpose()?,
            cell_annotation_umap_plots: args
                .cell_annotation_umap_plots
                .as_ref()
                .map(read_optional_file_map)
                .transpose()?,
            cell_annotation_diffexp_tables: args
                .cell_annotation_diffexp_tables
                .as_ref()
                .map(read_optional_file_map)
                .transpose()?,
            cell_annotation_metrics: args
                .cell_annotation_metrics_jsons
                .as_ref()
                .map(read_optional_file_map)
                .transpose()?,
            cell_annotation_viable_but_not_requested: args.cell_annotation_viable_but_not_requested,
        };

        for (sample, sample_ws) in multi_ws_builder.build()? {
            // Write the web summary data to JSON
            // put the path to the JSON into sample_to_web_summary, with key being sample ID
            let json_file: JsonFile<MultiWebSummary> =
                rover.make_path(format!("{sample}_web_summary_data.json"));

            serde_json::to_writer_pretty(json_file.buf_writer()?, &sample_ws)?;
            sample_to_web_summary.insert(sample.clone(), json_file);

            let csv_file: CsvFile<()> = rover.make_path(format!("{sample}_metric_summary_csv"));
            sample_ws.to_csv(&csv_file, 0)?;
            sample_to_metrics_csv.insert(sample.clone(), csv_file);
        }

        Ok(StageOutputs {
            web_summary_json: sample_to_web_summary,
            metrics_summary_csv: sample_to_metrics_csv,
        })
    }
}

/// Read a mapping of optional files.
fn read_optional_file_map<F: FileTypeRead<T>, T, K: Clone + Hash + Eq>(
    files: &TxHashMap<K, Option<F>>,
) -> Result<TxHashMap<K, T>> {
    read_files_into_map(
        files
            .iter()
            .filter_map(|(k, opt)| opt.as_ref().map(|v| (k, v))),
    )
}

/// Read an iterator of keys/martian files into a map containing the parsed file contents.
fn read_files_into_map<'a, F: FileTypeRead<T> + 'a, T, K: Clone + Hash + Eq + 'a>(
    files: impl IntoIterator<Item = (&'a K, &'a F)>,
) -> Result<TxHashMap<K, T>> {
    let mut out = TxHashMap::default();
    for (key, file) in files {
        out.insert(key.clone(), file.read()?);
    }
    Ok(out)
}

/// Return any differences between configured and detected probe barcode pairings.
/// Return None if there are no differences, or if there is no detected or
/// configured pairing.
fn get_mismatched_probe_barcode_pairings(
    multi_config: &MultiConfigCsvFile,
    detected_probe_barcode_pairing: Option<&DetectedProbeBarcodePairingFile>,
) -> Result<Option<MismatchedProbeBarcodePairings>> {
    let Some(samples) = multi_config.read()?.samples else {
        return Ok(None);
    };
    let Some(detected_probe_barcode_pairing) = detected_probe_barcode_pairing else {
        return Ok(None);
    };
    let detected_probe_barcode_pairing: HashSet<_> = detected_probe_barcode_pairing
        .read()?
        .into_iter()
        .filter(|(_, source_bc)| {
            // CRISPR not yet supported in pairing detection
            categorize_rtl_multiplexing_barcode_id(source_bc).unwrap()
                != RTLMultiplexingBarcodeType::Crispr
        })
        .map(|(target_bc, source_bc)| format!("{target_bc}{PROBE_BARCODE_ID_GROUPING}{source_bc}"))
        .collect();

    let configured_probe_barcode_pairing: HashSet<_> = samples
        .get_translated_probe_barcodes()
        .into_iter()
        .filter(|(source_bc, _)| {
            // CRISPR not yet supported in pairing detection
            categorize_rtl_multiplexing_barcode_id(source_bc).unwrap()
                != RTLMultiplexingBarcodeType::Crispr
        })
        .map(|(source_bc, target_bc)| format!("{target_bc}{PROBE_BARCODE_ID_GROUPING}{source_bc}"))
        .collect();
    if configured_probe_barcode_pairing.is_empty()
        || configured_probe_barcode_pairing == detected_probe_barcode_pairing
    {
        return Ok(None);
    }
    Ok(Some(MismatchedProbeBarcodePairings::new(
        &configured_probe_barcode_pairing,
        &detected_probe_barcode_pairing,
    )))
}

/// Determine if this is a barnyard analysis.
fn is_barnyard(count_inputs: Option<&CountInputs>) -> bool {
    count_inputs.is_some_and(|x| {
        x.reference_info
            .as_ref()
            .is_some_and(|x| x.genomes.len() >= 2)
    })
}
