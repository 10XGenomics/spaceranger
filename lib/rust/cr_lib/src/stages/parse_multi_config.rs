//! Martian stage PARSE_MULTI_CONFIG
#![allow(missing_docs)]

use super::setup_reference_info::get_reference_info;
use crate::preflight::hostname;
use anyhow::{anyhow, bail, Result};
use base64::prelude::BASE64_STANDARD;
use base64::Engine;
use cr_types::cell_annotation::CellAnnotationModel;
use cr_types::chemistry::{
    AutoChemistryName, AutoOrRefinedChemistry, ChemistryDef, ChemistryDefs, ChemistrySpecs,
    IndexScheme,
};
use cr_types::probe_set::{merge_probe_set_csvs, ProbeSetReferenceMetadata};
use cr_types::reference::feature_reference::{FeatureConfig, FeatureReferenceFile};
use cr_types::reference::probe_set_reference::TargetSetFile;
use cr_types::reference::reference_info::ReferenceInfo;
use cr_types::sample_def::SampleDef;
use cr_types::types::FileOrBytes;
use cr_types::{AlignerParam, GenomeName, LibraryType, TargetingMethod, VdjChainType};
use itertools::Itertools;
use martian::prelude::*;
use martian_derive::{make_mro, MartianStruct, MartianType};
use martian_filetypes::json_file::JsonFile;
use martian_filetypes::tabular_file::CsvFile;
use metric::TxHashMap;
use multi::barcode_sample_assignment::SampleAssignmentCsv;
use multi::config::preflight::build_feature_reference_with_cmos;
use multi::config::{create_feature_config, MultiConfigCsvFile};
use serde::{Deserialize, Serialize};
use serde_json::Value as JValue;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};

#[derive(Clone, Default, Serialize, Deserialize, MartianType)]
pub struct MultiParams {
    #[serde(default)]
    initial_reads: Option<usize>,
    #[serde(default)]
    subsample_rate: Option<f64>,
    #[serde(default)]
    primers: Vec<Primers>,
    #[serde(default)]
    index_schemes: HashMap<LibraryType, IndexScheme>,
    #[serde(default)]
    barcode_whitelist: Option<String>,
    #[serde(default)]
    special_genomic_regions: Option<Vec<String>>,
    #[serde(default)]
    cell_calling_mode: Option<String>,
}

#[derive(Clone, Deserialize, MartianStruct)]
pub struct ParseMultiConfigStageInputs {
    pub sample_id: String,
    pub sample_desc: String,
    pub config: FileOrBytes,
    pub config_hash: Option<String>,
    pub params: Option<MultiParams>,
    pub is_pd: bool,
}

/// The `per_gem_well` parameter is applied at the library-level (CMO-multiplexing, standard GEX).
/// The `per_sample` field is only valid in case of RTL multiplexed inputs.
/// The fields `per_gem_well` and `per_sample` are mutually exclusive.
#[derive(Clone, Serialize, Deserialize, MartianStruct)]
#[cfg_attr(test, derive(Debug))]
pub struct CellCallingParam {
    pub per_gem_well: Option<f64>,
    pub per_sample: Option<TxHashMap<String, Option<f64>>>,
}

impl CellCallingParam {
    // Return total values for parameters that can be input either at the per_gem_well or per_sample level
    pub fn sum(&self) -> Option<f64> {
        match (self.per_gem_well, self.per_sample.as_ref()) {
            (Some(x), Some(xs)) => {
                assert!(xs.values().all(Option::is_none));
                Some(x)
            }
            (Some(x), None) => Some(x),
            (None, Some(xs)) => xs.values().try_fold(0.0, |acc, x| match (acc, x) {
                (acc, Some(x)) => Some(acc + x),
                _ => None,
            }),
            (None, None) => None,
        }
    }
}

/// Carries options to customize the cell calling mode.
#[derive(Clone, Serialize, Deserialize, MartianStruct)]
#[cfg_attr(test, derive(Debug))]
pub struct CellCalling {
    pub recovered_cells: CellCallingParam,
    pub force_cells: CellCallingParam,
    pub emptydrops_minimum_umis: CellCallingParam,
    pub global_minimum_umis: CellCallingParam,
    pub max_mito_percent: CellCallingParam,
    pub cell_barcodes: Option<JsonFile<()>>,
    pub override_mode: Option<String>,
    pub override_library_types: Option<Vec<String>>,
    pub disable_ab_aggregate_detection: bool,
    pub disable_high_occupancy_gem_detection: bool,
}

#[derive(Clone, Serialize, Deserialize, MartianStruct)]
#[cfg_attr(test, derive(Debug))]
pub struct CommonInputs {
    pub sample_id: String,
    pub sample_desc: String,
    pub multi_config_sha: String,
}

#[derive(Clone, Serialize, Deserialize, MartianStruct)]
#[cfg_attr(test, derive(Debug))]
pub struct BarcodeAssignments {
    pub sample_barcodes: Option<JsonFile<()>>,
    pub non_singlet_barcodes: Option<JsonFile<()>>,
    pub cells_per_tag: Option<JsonFile<()>>,
}

pub type Primers = JValue;
pub type GeneticDemuxParams = JValue;

/// CountInputs struct
#[derive(Clone, Serialize, Deserialize, MartianStruct)]
#[cfg_attr(test, derive(Debug))]
pub struct CountInputs {
    pub sample_def: Vec<SampleDef>,
    pub target_set: Option<TargetSetFile>,
    pub target_set_name: Option<String>,
    pub reference_info: Option<ReferenceInfo>,
    pub chemistry_specs: ChemistrySpecs,
    pub custom_chemistry_defs: ChemistryDefs,
    pub gene_index: Option<JsonFile<()>>,
    #[mro_type = "map[]"]
    pub primers: Vec<Primers>,
    pub subsample_rate: Option<f64>,
    pub initial_reads: Option<usize>,
    pub primer_initial_reads: Option<usize>,
    pub special_genomic_regions: Option<Vec<String>>,
    pub r1_length: Option<usize>,
    pub r2_length: Option<usize>,
    pub trim_polya_min_score: Option<i64>,
    pub trim_tso_min_score: Option<i64>,
    pub no_secondary_analysis: bool,
    pub filter_probes: Option<bool>,
    pub feature_reference: Option<FeatureReferenceFile>,
    pub include_exons: bool,
    pub include_introns: bool,
    pub targeting_method: Option<TargetingMethod>,
    pub aligner: Option<AlignerParam>,
    #[mro_type = "map"]
    pub genetic_demux_params: Option<GeneticDemuxParams>,
    pub throughput: Option<String>,
    pub check_library_compatibility: bool,
    pub no_bam: bool,
    pub force_sample_barcodes: BarcodeAssignments,
    pub tenx_cmos: Option<bool>,
    pub min_assignment_confidence: Option<f64>,
    pub min_crispr_umi_threshold: Option<usize>,
    pub annotations: Option<Vec<CsvFile<()>>>,
    pub cell_annotation_model: Option<String>,
    pub skip_cell_annotation: bool,
    pub tenx_cloud_token_path: Option<String>,
    pub enable_tsne: bool,
}

impl CountInputs {
    /// Return the list of genomes
    pub fn get_genomes(&self) -> Option<&[GenomeName]> {
        if let Some(reference_info) = self.reference_info.as_ref() {
            Some(&reference_info.genomes)
        } else {
            None
        }
    }

    /// Return transcriptome reference_path
    pub fn get_reference_path(&self) -> Option<&Path> {
        self.reference_info
            .as_ref()
            .and_then(|x| x.get_reference_path())
    }
}

/// General VdjInputs which are not chain specific
#[derive(Clone, Serialize, Deserialize, MartianStruct)]
#[cfg_attr(test, derive(Debug))]
pub struct VdjGenInputs {
    pub reference_path: Option<PathBuf>,
    pub vdj_reference_path: Option<PathBuf>,
    pub min_contig_length: Option<usize>,
    pub filter_flags: VdjFilterFlags,
    pub skip_clonotyping: bool,
}

#[derive(Default, Clone, Serialize, Deserialize, MartianStruct)]
#[cfg_attr(test, derive(Debug))]
pub struct VdjFilterFlags {
    pub multiplet_filter: Option<bool>,
    pub shared_contig_filter: Option<bool>,
    pub umi_baseline_filter: Option<bool>,
}

/// VDJInputs struct
#[derive(Clone, Serialize, Deserialize, MartianStruct)]
#[cfg_attr(test, derive(Debug))]
pub struct VdjInputs {
    pub sample_def: Vec<SampleDef>,
    pub chemistry_spec: AutoOrRefinedChemistry,
    pub custom_chemistry_def: Option<ChemistryDef>,
    #[mro_type = "map[]"]
    pub primers: Vec<Primers>,
    pub subsample_rate: Option<f64>,
    pub initial_reads: Option<usize>,
    pub primer_initial_reads: Option<usize>,
    pub special_genomic_regions: Option<Vec<String>>,
    pub denovo: bool,
    pub r1_length: Option<usize>,
    pub r2_length: Option<usize>,
    pub inner_enrichment_primers: Option<PathBuf>,
    pub chain_type: Option<String>,
    pub physical_library_id: Option<String>,
}

#[derive(Clone, Serialize, Deserialize, MartianStruct)]
#[cfg_attr(test, derive(Debug))]
pub struct BasicPipelineConfig {
    pub disable_count: bool,
    pub disable_vdj: bool,
    /// boolean to disable stages that are only needed in the multi pipeline
    pub disable_multi: bool,
    /// boolean to disable stages that are only needed when count libraries are
    /// present in the multi pipeline
    pub disable_multi_count: bool,
    /// boolean to disable stages that are only needed when probes are present
    pub disable_rtl: bool,
    /// boolean to disable annotate stages
    pub disable_annotate: bool,
}

#[derive(Clone, Serialize, Deserialize, MartianStruct)]
#[cfg_attr(test, derive(Debug))]
pub struct ParseMultiConfigStageOutputs {
    pub common_input: CommonInputs,
    pub count_input: Option<CountInputs>,
    pub count_cell_calling_config: Option<CellCalling>,
    pub vdj_inputs: Vec<VdjInputs>, // or just JSON w/ outer array?
    pub vdj_gen_inputs: Option<VdjGenInputs>,
    pub basic_config: BasicPipelineConfig,
    pub config_file: MultiConfigCsvFile,
    pub feature_config: Option<FeatureConfig>,
    #[mro_retain]
    pub feature_ref: Option<FeatureReferenceFile>,
    #[mro_retain]
    pub target_set: Option<TargetSetFile>,
    #[mro_retain]
    pub cell_barcodes: Option<JsonFile<()>>,
    #[mro_retain]
    pub sample_barcodes: Option<JsonFile<()>>,
    #[mro_retain]
    pub non_singlet_barcodes: Option<JsonFile<()>>,
    #[mro_retain]
    pub cells_per_tag: Option<JsonFile<()>>,
    #[mro_retain]
    pub barcode_sample_assignments: Option<CsvFile<()>>,
}

// This is our stage struct
pub struct ParseMultiConfig;

#[make_mro(mem_gb = 6, volatile = strict)]
impl MartianMain for ParseMultiConfig {
    type StageInputs = ParseMultiConfigStageInputs;
    type StageOutputs = ParseMultiConfigStageOutputs;

    fn main(&self, args: Self::StageInputs, rover: MartianRover) -> Result<Self::StageOutputs> {
        // make a new chunk per sample_def to do chemistry detection
        let config_bytes = match args.config {
            FileOrBytes {
                file: None,
                bytes: Some(ref bytes),
            } => BASE64_STANDARD.decode(bytes)?,
            FileOrBytes {
                bytes: None,
                file: Some(ref file),
            } => {
                let mut bytes = vec![];
                let mut handle = File::open(file)?;
                let _ = handle.read_to_end(&mut bytes)?;
                bytes
            }
            _ => {
                bail!("exactly one of config file or config bytes must be provided");
            }
        };
        if !cfg!(test) {
            std::io::stdout().write_all(&config_bytes[..])?;
        }
        let config_file = {
            let file: MultiConfigCsvFile = rover.make_path("config");
            let mut writer = file.buf_writer()?;
            writer.write_all(&config_bytes[..])?;
            writer.flush()?;
            file
        };
        let cfg = config_file.read()?;

        let MultiParams {
            initial_reads,
            subsample_rate,
            primers,
            index_schemes,
            barcode_whitelist,
            special_genomic_regions,
            cell_calling_mode,
        } = args.params.unwrap_or_default();

        let (
            count_input,
            count_cell_calling_config,
            feature_ref,
            cell_barcodes,
            sample_barcodes,
            non_singlet_barcodes,
            cells_per_tag,
            barcode_sample_assignments,
        ) = {
            // TODO: need to march over distinct GemWells --
            //   so I need to provide an API for that over on MultiConfigCsv,
            //   which is good, because I probably want it for validating [gem-wells]
            //   misc thought, do deep validation of libraries and gem-wells settings,
            //   e.g. vdj_force_cells vs (gex_)force_cells
            let mut sample_def =
                cfg.libraries
                    .0
                    .iter()
                    .try_fold(Vec::new(), |mut acc, x| -> Result<_> {
                        if x.is_gex() || x.library_type().is_fb() {
                            acc.push(x.to_sample_def()?);
                        }
                        Ok(acc)
                    })?;
            if sample_def.is_empty() {
                (None, None, None, None, None, None, None, None)
            } else {
                let gex = cfg.gene_expression.as_ref().ok_or_else(
                    #[cold]
                    || {
                        anyhow!(
                            "[gene-expression] section with a path to the transcriptome reference \
                             is a required input (even if only antibody-data is present)."
                        )
                    },
                )?;
                let feature = cfg.feature.as_ref();
                for sample in &mut sample_def {
                    if let Some(LibraryType::Gex) = sample.library_type {
                        sample.r1_length = gex.r1_length;
                        sample.r2_length = gex.r2_length;
                    } else {
                        sample.r1_length = feature.and_then(|f| f.r1_length);
                        sample.r2_length = feature.and_then(|f| f.r2_length);
                    }
                }

                let gene_index = if let Some(reference_path) = &gex.reference_path {
                    let gene_index: JsonFile<()> = rover.make_path("gene_index");
                    transcriptome::python_gene_index::write_gene_index(
                        reference_path,
                        &gene_index,
                    )?;
                    Some(gene_index)
                } else {
                    None
                };

                let (
                    cell_barcodes,
                    sample_barcodes,
                    non_singlet_barcodes,
                    cells_per_tag,
                    barcode_sample_assignments,
                ) = if let Some(bsa) = gex.barcode_sample_assignment.as_ref() {
                    let sample_assignments = SampleAssignmentCsv::from_file(bsa, &cfg)?;
                    let cell_barcodes: JsonFile<()> = rover.make_path("cell_barcodes");
                    let sample_barcodes: JsonFile<()> = rover.make_path("sample_barcodes");
                    let non_singlet_barcodes: JsonFile<()> =
                        rover.make_path("non_singlet_barcodes");
                    let cells_per_tag: JsonFile<()> = rover.make_path("cells_per_tag");
                    let barcode_sample_assignment_csv: CsvFile<()> =
                        rover.make_path("barcode_sample_assignment");

                    std::fs::copy(bsa, &barcode_sample_assignment_csv)?;
                    sample_assignments.to_cell_barcodes_json(&cell_barcodes)?;
                    sample_assignments.to_sample_barcodes_json(&sample_barcodes)?;
                    sample_assignments.to_non_singlet_barcodes_json(&non_singlet_barcodes)?;
                    let cells_per_tag =
                        if sample_assignments.to_cells_per_tag_json(&cells_per_tag)? {
                            Some(cells_per_tag)
                        } else {
                            None
                        };
                    (
                        Some(cell_barcodes),
                        Some(sample_barcodes),
                        Some(non_singlet_barcodes),
                        cells_per_tag,
                        Some(barcode_sample_assignment_csv),
                    )
                } else {
                    (None, None, None, None, None)
                };

                let cell_calling_config = CellCalling {
                    force_cells: CellCallingParam {
                        per_gem_well: gex.force_cells.map(|x| x as f64),
                        per_sample: cfg
                            .samples
                            .as_ref()
                            .map(multi::config::SamplesCsv::get_force_cells),
                    },
                    recovered_cells: CellCallingParam {
                        per_gem_well: gex.expect_cells.map(|x| x as f64),
                        per_sample: cfg
                            .samples
                            .as_ref()
                            .map(multi::config::SamplesCsv::get_expect_cells),
                    },
                    emptydrops_minimum_umis: CellCallingParam {
                        per_gem_well: gex.emptydrops_minimum_umis.map(|x| x as f64),
                        per_sample: cfg
                            .samples
                            .as_ref()
                            .map(multi::config::SamplesCsv::get_emptydrops_minimum_umis),
                    },
                    global_minimum_umis: CellCallingParam {
                        per_gem_well: gex.global_minimum_umis.map(|x| x as f64),
                        per_sample: cfg
                            .samples
                            .as_ref()
                            .map(multi::config::SamplesCsv::get_global_minimum_umis),
                    },
                    max_mito_percent: CellCallingParam {
                        per_gem_well: gex.max_mito_percent,
                        per_sample: cfg
                            .samples
                            .as_ref()
                            .map(multi::config::SamplesCsv::get_max_mito_percent),
                    },
                    cell_barcodes: cell_barcodes.clone(),
                    override_mode: cell_calling_mode,
                    override_library_types: None,
                    disable_ab_aggregate_detection: feature
                        .is_some_and(|feat| !feat.filter_aggregates),
                    disable_high_occupancy_gem_detection: !gex.filter_high_occupancy_gems,
                };

                let (feature_reference_file, tenx_cmos) =
                    match build_feature_reference_with_cmos(&cfg, args.is_pd, &hostname())? {
                        (Some(_), None) => {
                            // Not using CMO multiplexing; use the original feature reference file.
                            let src = cfg
                                .feature
                                .as_ref()
                                .unwrap()
                                .reference_path
                                .as_ref()
                                .unwrap();
                            let dst: FeatureReferenceFile = rover.make_path("feature_ref");
                            if std::fs::hard_link(src, &dst).is_err() {
                                std::fs::copy(src, &dst)?;
                            }
                            (Some(dst), None)
                        }
                        (Some(feature_ref), Some(tenx_cmos)) => {
                            // Using CMO, need to write out the constructed feature reference file.
                            let dst: FeatureReferenceFile = rover.make_path("feature_ref");
                            let mut w = BufWriter::new(File::create(&dst)?);
                            feature_ref.to_csv(&mut w)?;
                            (Some(dst), Some(tenx_cmos))
                        }
                        (None, _) => (None, None),
                    };

                let custom_chemistry_defs = index_schemes
                    .into_iter()
                    .map(|(lib_type, index_scheme)| {
                        anyhow::Ok((
                            lib_type,
                            index_scheme.to_chemistry_def(
                                barcode_whitelist
                                    .as_ref()
                                    .expect("Barcode Set is required for custom chemistry"),
                            )?,
                        ))
                    })
                    .try_collect()?;

                let chemistry_specs = cfg.chemistry_specs()?;
                let has_gex_library = chemistry_specs
                    .iter()
                    .any(|(lib_type, _)| lib_type.is_gex());

                let (target_set, target_set_name) =
                    match (cfg.gene_expression.as_ref(), has_gex_library) {
                        (Some(gex_params), true) => match gex_params.probe_set() {
                            [] => (None, None),
                            [probe_set] => (
                                Some(probe_set.clone()),
                                Some(
                                    ProbeSetReferenceMetadata::load_from(probe_set)?
                                        .panel_name()
                                        .to_string(),
                                ),
                            ),
                            probe_sets => {
                                let target_set: TargetSetFile =
                                    rover.make_path("combined_probe_set");
                                let name = merge_probe_set_csvs(
                                    probe_sets,
                                    target_set.buf_writer()?,
                                    gex_params.reference_path.as_deref(),
                                )?;
                                (Some(target_set), Some(name))
                            }
                        },
                        _ => (None, None),
                    };
                let reference_info =
                    get_reference_info(gex.reference_path.as_deref(), target_set.as_ref())?;

                let count_input = Some(CountInputs {
                    reference_info: Some(reference_info),
                    // Filter the VDJ libraries out of this collection since VDJ
                    // chemistry is handled separately.
                    chemistry_specs: chemistry_specs
                        .into_iter()
                        .filter(|(lib_type, _)| !lib_type.is_vdj())
                        .collect(),
                    target_set,
                    target_set_name,
                    sample_def,
                    custom_chemistry_defs,
                    gene_index,
                    primers: primers.clone(),
                    subsample_rate,
                    initial_reads,
                    primer_initial_reads: Some(1000000),
                    special_genomic_regions: special_genomic_regions.clone(),
                    r1_length: None,
                    r2_length: None,
                    trim_polya_min_score: None,
                    trim_tso_min_score: None,
                    no_secondary_analysis: gex.no_secondary_analysis,
                    filter_probes: gex.filter_probes,
                    feature_reference: feature_reference_file.clone(),
                    include_exons: true,
                    include_introns: gex.include_introns,
                    targeting_method: gex.targeting_method(),
                    aligner: gex.aligner,
                    genetic_demux_params: None,
                    throughput: None,
                    check_library_compatibility: gex.check_library_compatibility,
                    no_bam: !gex.create_bam,
                    force_sample_barcodes: BarcodeAssignments {
                        sample_barcodes: sample_barcodes.clone(),
                        non_singlet_barcodes: non_singlet_barcodes.clone(),
                        cells_per_tag: cells_per_tag.clone(),
                    },
                    tenx_cmos,
                    min_assignment_confidence: gex.min_assignment_confidence,
                    min_crispr_umi_threshold: cfg.feature.as_ref().map(|x| x.min_crispr_umi),
                    annotations: None,
                    skip_cell_annotation: gex.skip_cell_annotation
                        || (gex.cell_annotation_model.is_none() && !args.is_pd),
                    cell_annotation_model: gex
                        .cell_annotation_model
                        .as_ref()
                        .and_then(CellAnnotationModel::to_pipeline_inputs),
                    // this will return a default token path if not supplied
                    tenx_cloud_token_path: gex.get_tenx_cloud_token_path(),
                    enable_tsne: true,
                });

                (
                    count_input,
                    Some(cell_calling_config),
                    feature_reference_file,
                    cell_barcodes,
                    sample_barcodes,
                    non_singlet_barcodes,
                    cells_per_tag,
                    barcode_sample_assignments,
                )
            }
        };
        let (vdj_inputs, vdj_gen_inputs) = {
            let mut per_lib_vdj_sample_def = BTreeMap::new();
            for lib in &cfg.libraries.0 {
                // TODO: Pass only a single VdjInput with all sample_defs and
                // vector-types for other per-chain configurables
                if lib.is_vdj() {
                    per_lib_vdj_sample_def
                        .entry((lib.physical_library_id(), lib.library_type()))
                        .or_insert_with(Vec::new)
                        .push(lib.to_sample_def()?);
                }
            }
            if per_lib_vdj_sample_def.len() > 3 {
                // TODO: This needs to be relaxed with multi gem well
                bail!(
                    "Found {} VDJ libraries, but we expect at most 3 libraries",
                    per_lib_vdj_sample_def.len()
                );
            }
            // TODO: we need to partition sample_def by gem-well and vdj_b or vdj_t,
            // but only know the first now...
            if per_lib_vdj_sample_def.is_empty() {
                (vec![], None)
            } else {
                let vdj = cfg.vdj.as_ref().ok_or_else(
                    #[cold]
                    || anyhow!("missing [vdj] table for VDJ libraries"),
                )?;
                let mut vdj_inputs = Vec::new();
                for ((physical_library_id, feature_type), sample_def) in per_lib_vdj_sample_def {
                    let chain_type = if let Some(ct) = feature_type.vdj_chain_type() {
                        match ct {
                            VdjChainType::VdjT => Some("TR"),
                            VdjChainType::VdjTGD => {
                                // In gamma/delta mode we need inner-enrichment-primers to be present in multi config
                                if vdj.inner_enrichment_primers.is_none() {
                                    bail!(
                                        "VDJ-T-GD library requires inner enrichment primers to be specified in the multi config file."
                                    );
                                }
                                Some("TR_GD")
                            }
                            VdjChainType::VdjB => Some("IG"),
                            VdjChainType::Auto => None,
                        }
                    } else {
                        None
                    }.map(ToString::to_string);

                    vdj_inputs.push(VdjInputs {
                        chemistry_spec: AutoOrRefinedChemistry::Auto(AutoChemistryName::Vdj),
                        sample_def,
                        custom_chemistry_def: None,
                        primers: primers.clone(),
                        subsample_rate,
                        initial_reads,
                        primer_initial_reads: Some(1000000),
                        special_genomic_regions: special_genomic_regions.clone(),
                        denovo: vdj.denovo.unwrap_or_default(),
                        r1_length: vdj.r1_length,
                        r2_length: vdj.r2_length,
                        inner_enrichment_primers: vdj.inner_enrichment_primers.clone(),
                        chain_type,
                        physical_library_id: Some(physical_library_id.to_string()),
                    });
                }
                let vdj_gen_inputs = VdjGenInputs {
                    reference_path: cfg
                        .gene_expression
                        .and_then(|gex| gex.reference_path.clone()),
                    vdj_reference_path: vdj.reference_path.clone(),
                    min_contig_length: vdj.min_contig_length,
                    filter_flags: VdjFilterFlags {
                        multiplet_filter: vdj.multiplet_filter,
                        shared_contig_filter: vdj.shared_contig_filter,
                        umi_baseline_filter: vdj.umi_baseline_filter,
                    },
                    skip_clonotyping: vdj.skip_clonotyping.unwrap_or_default(),
                };
                (vdj_inputs, Some(vdj_gen_inputs))
            }
        };
        let basic_config = BasicPipelineConfig {
            disable_count: count_input.is_none(),
            disable_vdj: vdj_inputs.is_empty(),
            disable_multi: false,
            disable_multi_count: count_input.is_none(),
            disable_rtl: count_input.as_ref().is_none_or(|c| c.target_set.is_none()),
            disable_annotate: count_input.as_ref().is_none_or(|c| c.skip_cell_annotation),
        };

        let common_input = CommonInputs {
            sample_id: args.sample_id,
            sample_desc: args.sample_desc,
            multi_config_sha: {
                let mut hasher = Sha256::new();
                hasher.update(&std::fs::read_to_string(&config_file)?);
                format!("{:x}", hasher.finalize())
            },
        };

        let feature_config = create_feature_config(
            cfg.antigen_specificity.as_ref(),
            cfg.functional_map.as_ref(),
            cfg.libraries.beam_mode(),
            cfg.samples.as_ref(),
        );

        // Create feature reference object to make sure features are continous
        // And validate feature ref if beam mode can be deciphered (VDJ chain type provided)
        if let Some(ref feature_ref) = feature_ref {
            let feature_reference = feature_ref.read(feature_config.as_ref())?;
            if let Some(beam_mode) = feature_config.as_ref().and_then(|c| c.beam_mode) {
                feature_reference.validate_beam_feature_ref(beam_mode)?;
            }
        }

        Ok(ParseMultiConfigStageOutputs {
            common_input,
            target_set: count_input.as_ref().and_then(|c| c.target_set.clone()),
            count_input,
            count_cell_calling_config,
            vdj_inputs,
            vdj_gen_inputs,
            basic_config,
            config_file,
            feature_config,
            feature_ref,
            cell_barcodes,
            sample_barcodes,
            non_singlet_barcodes,
            cells_per_tag,
            barcode_sample_assignments,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use insta::{assert_snapshot, assert_yaml_snapshot};
    use multi::config::ProbeBarcodeIterationMode;
    use std::path::Path;

    fn test_run_stage_is_pd(
        csv_file: impl AsRef<Path>,
        sample_id: &str,
        is_pd: bool,
    ) -> Result<ParseMultiConfigStageOutputs> {
        ParseMultiConfig.test_run_tmpdir(ParseMultiConfigStageInputs {
            sample_id: sample_id.into(),
            sample_desc: String::new(),
            config: FileOrBytes {
                bytes: None,
                file: Some(csv_file.as_ref().into()),
            },
            config_hash: None,
            params: None,
            is_pd,
        })
    }

    fn test_run_stage(
        csv_file: impl AsRef<Path>,
        sample_id: &str,
    ) -> Result<ParseMultiConfigStageOutputs> {
        test_run_stage_is_pd(csv_file, sample_id, false)
    }

    fn insta_settings(test_csv: &str) -> insta::Settings {
        use insta::dynamic_redaction;
        let replace_if_not_none = |value, key| {
            if matches!(value, insta::internals::Content::None) {
                value
            } else {
                format!("[{key}:redacted]").into()
            }
        };
        let mut settings = insta::Settings::clone_current();
        settings.add_redaction(".config_file", test_csv);
        settings.add_redaction(
            ".count_input.gene_index",
            dynamic_redaction(move |value, _| replace_if_not_none(value, "gene_index")),
        );
        settings.add_redaction(
            ".count_input.feature_reference",
            dynamic_redaction(move |value, _| replace_if_not_none(value, "feature_reference")),
        );
        settings.add_redaction(
            ".feature_ref",
            dynamic_redaction(move |value, _| replace_if_not_none(value, "feature_ref")),
        );
        settings.add_redaction(".multi_graph", "[multi_graph:redacted]");
        settings.set_sort_maps(true);
        settings
    }

    fn yaml_snapshot(sample_id: &str) {
        let test_csv = &format!("test/multi/{sample_id}.csv");
        insta_settings(test_csv).bind(|| {
            let outs = test_run_stage(test_csv, sample_id).unwrap();
            assert_yaml_snapshot!(sample_id, &outs);
        });
    }

    #[test]
    fn test_parse_cmos() {
        let cfg = MultiConfigCsvFile::new("test/multi", "gex_multi_cmos.csv")
            .read()
            .unwrap();
        let cmos = cfg.sample_barcode_ids_used_in_experiment(ProbeBarcodeIterationMode::All);
        assert_eq!(cmos.len(), 3);
        assert!(cmos.contains("CMO1"));
        assert!(cmos.contains("CMO2"));
        assert!(cmos.contains("CMO3"));

        assert!(cfg
            .samples
            .unwrap()
            .get_translated_probe_barcodes()
            .is_empty());
    }

    #[test]
    fn test_parse_probe_barcode_ids() {
        let cfg = MultiConfigCsvFile::new("test/multi", "mfrp_multi.csv")
            .read()
            .unwrap();
        let probe_barcode_ids =
            cfg.sample_barcode_ids_used_in_experiment(ProbeBarcodeIterationMode::All);
        assert_eq!(probe_barcode_ids.len(), 2);
        assert!(probe_barcode_ids.contains("BC001"));
        assert!(probe_barcode_ids.contains("BC002"));
    }

    #[test]
    fn test_parse_mapped_probe_barcode_ids() {
        let cfg = MultiConfigCsvFile::from("test/multi/mfrp_ab_multi.csv")
            .read()
            .unwrap();
        let probe_barcode_ids =
            cfg.sample_barcode_ids_used_in_experiment(ProbeBarcodeIterationMode::All);
        assert_eq!(probe_barcode_ids.len(), 4);
        assert!(probe_barcode_ids.contains("BC001"));
        assert!(probe_barcode_ids.contains("BC002"));
        assert!(probe_barcode_ids.contains("AB002"));
        assert!(probe_barcode_ids.contains("BC004"));
        let probe_barcode_ids =
            cfg.sample_barcode_ids_used_in_experiment(ProbeBarcodeIterationMode::Mapped);
        assert_eq!(probe_barcode_ids.len(), 3);
        assert!(probe_barcode_ids.contains("BC001"));
        assert!(probe_barcode_ids.contains("BC002"));
        assert!(probe_barcode_ids.contains("BC004"));
    }

    #[test]
    fn test_vdj_internal() {
        yaml_snapshot("vdj_micro");
    }

    #[test]
    fn test_vdj_gd_missing_primers() {
        let outs = test_run_stage(
            "test/multi/invalid_csvs/vdj_micro_gd_no_primer.csv",
            "vdj_micro_gd_noprimer",
        );
        assert_snapshot!(&outs.unwrap_err());
    }

    #[test]
    fn test_vdj_gex_internal() {
        yaml_snapshot("vdj_gex_micro");
    }

    #[test]
    fn test_gex_fbc_internal() {
        yaml_snapshot("gex_fbc_micro");
    }

    #[test]
    fn test_gex_fbc_dos_internal() {
        yaml_snapshot("gex_fbc_micro_dos");
    }

    #[test]
    fn test_gex_fbc_dos_utf8_internal() {
        yaml_snapshot("gex_fbc_micro_dos_utf8");
    }

    #[test]
    fn test_gex_fbc_mac_utf8_internal() {
        yaml_snapshot("gex_fbc_micro_mac_utf8");
    }

    #[test]
    fn test_fb_only_missing_gex_section() {
        let outs = test_run_stage(
            "test/multi/invalid_csvs/fb_only_missing_gex_section.csv",
            "fb",
        );
        assert_snapshot!(outs.unwrap_err());
    }

    #[test]
    fn test_gex_missing_gex_section() {
        let outs = test_run_stage("test/multi/invalid_csvs/gex_missing_gex_section.csv", "gex");
        assert_snapshot!(outs.unwrap_err());
    }

    #[test]
    fn test_gex_vdj_beamab_internal() {
        yaml_snapshot("beamab_vdj_gex");
    }

    #[test]
    fn test_gex_vdj_beamt_internal() {
        yaml_snapshot("beamt_vdj_gex");
    }

    #[test]
    fn test_gex_vdj_beamab_with_antigen_specificity_internal() {
        yaml_snapshot("beamab_vdj_gex_antigen_spec");
    }

    #[test]
    fn test_non_continous_feature_ref() {
        let outs = test_run_stage(
            "test/multi/non_continous_feature_ref.csv",
            "non_continous_feature_ref",
        );
        assert_snapshot!(outs.unwrap_err());
    }

    #[test]
    fn test_vdj_filters() {
        yaml_snapshot("vdj_micro_filters");
    }
}
