use anyhow::{bail, Result};
use clap::{self, ArgGroup, Parser};
use cr_types::chemistry::ChemistryName;
use cr_types::sample_def::SampleDef;
use cr_types::types::AlignerParam;
use cr_types::{FeatureBarcodeType, LibraryType, TargetingMethod};
use cr_wrap::create_bam_arg::CreateBam;
use cr_wrap::fastqs::{get_target_set_name, FastqArgs};
use cr_wrap::mrp_args::MrpArgs;
use cr_wrap::shared_cmd::{self, HiddenCmd, RnaSharedCmd, SharedCmd};
use cr_wrap::utils::{validate_id, AllArgs, CliPath};
use cr_wrap::{check_deprecated_os, env, execute, make_mro, mkfastq, set_env_vars};
use serde::Serialize;
use slide_id::{
    AreaId, SlideId, SlideInformation, SlideInformationFromDifferentSources, SlideRevision,
    SlideSerialCaptureArea, UnknownSlide,
};
use std::fmt::Debug;
use std::path::PathBuf;
use std::process::ExitCode;
use std::str::FromStr;

mod cytassist;
mod slide_id;

const MAX_IMAGE_SCALE: f64 = 10.0;

const CMD: &str = "spaceranger";

// TODO: make this a const fn
fn sr_version() -> &'static str {
    match option_env!("SPACERANGER_VERSION") {
        Some(s) => s,
        _ => "redacted",
    }
}

/// Process 10x Genomics Visium data
#[derive(Parser, Debug)]
#[clap(name = CMD, version = sr_version(), before_help = format!("{CMD} {}", sr_version()))]
struct SpaceRanger {
    #[clap(subcommand)]
    subcmd: SubCommand,

    /// Provide a path to the environment definition json for the build
    #[clap(long, hide = true)]
    env_json: Option<PathBuf>,
}

#[derive(Parser, Debug)]
#[allow(clippy::large_enum_variant)]
enum SubCommand {
    /// Count gene expression and protein expression reads from a single capture area.
    #[clap(name = "count")]
    Count(Count),

    /// Aggregate data from multiple 'spaceranger count' runs. Visium HD not supported in this release.
    #[clap(name = "aggr")]
    Aggr(Aggr),

    /// Run Illumina demultiplexer on sample sheets that contain 10x-specific
    /// sample index sets.
    #[clap(name = "mkfastq")]
    Mkfastq(AllArgs),

    /// Execute the 'count' pipeline on a small test dataset
    #[clap(name = "testrun")]
    Testrun(Testrun),

    /// Auxiliary commands
    #[clap(flatten)]
    RnaShared(RnaSharedCmd),

    /// Commands to transmit metadata to 10x Genomics
    #[clap(flatten)]
    Shared(SharedCmd),

    /// Not user facing
    #[clap(flatten)]
    Hidden(HiddenCmd),
}

/// Count gene and protein expression reads from a single capture area on a Visium slide.
#[derive(Parser, Debug, Clone)]
#[clap(group = ArgGroup::new("img").required(true).multiple(true))]
struct Count {
    /// A unique run id and output folder name [a-zA-Z0-9_-]+.
    #[clap(long, value_name = "ID", value_parser = validate_id, required = true)]
    id: String,

    /// Sample description to embed in output files.
    #[clap(long, value_name = "TEXT")]
    description: Option<String>,

    /// Single H&E brightfield image in either TIFF or JPG format. Optional if --cytaimage is
    /// specified.
    #[clap(long, value_name = "IMG", group = "img", conflicts_with_all(&["darkimage", "colorizedimage"]))]
    image: Option<CliPath>,

    /// Multi-channel, dark-background fluorescence image as either a single, multi-layer TIFF file
    /// or multiple TIFF or JPEG files. Optional if --cytaimage is specified.
    #[clap(
        long,
        value_delimiter = ',',
        value_name = "IMG",
        group = "img",
        conflicts_with_all(&["image", "colorizedimage"])
    )]
    darkimage: Option<Vec<CliPath>>,

    /// Color image representing pre-colored dark-background fluorescence images as either a
    /// single-layer TIFF or single JPEG file. Optional if --cytaimage is specified.
    #[clap(long, value_name = "IMG", group = "img", conflicts_with_all(&["image", "darkimage"]))]
    colorizedimage: Option<CliPath>,

    /// Brightfield image generated by the CytAssist instrument (the CytAssist image)
    #[clap(long, value_name = "IMG", group = "img")]
    cytaimage: Option<CliPath>,

    /// Index of DAPI channel (1-indexed) of fluorescence image
    #[clap(long, value_name = "NUM", requires_all = &["darkimage", "cytaimage"])]
    dapi_index: Option<usize>,

    /// Visium slide serial number, for example 'V10J25-015'. If unknown, use --unknown-slide instead.
    #[clap(
        long,
        required_unless_present_any = &["unknown_slide", "loupe_alignment", "cytaimage"],
        conflicts_with = "unknown_slide",
        value_name = "TEXT"
    )]
    slide: Option<SlideId>,

    /// Visium capture area identifier, for example 'A1' (or, e.g. 'A' for 11 mm capture area slides).
    /// Must be used along with --slide unless --unknown-slide is used.
    #[clap(long,
        required_unless_present_any = &["unknown_slide", "loupe_alignment", "cytaimage"],
        conflicts_with = "unknown_slide",
        value_name = "TEXT"
    )]
    area: Option<AreaId>,

    /// Overrides the slide serial number and capture area provided in the Cytassist
    /// image metadata. Use in combination with --slide and --area or with
    /// --unknown-slide.
    #[clap(long, requires = "cytaimage", conflicts_with = "loupe_alignment")]
    override_id: bool,

    /// Path of folder containing 10x-compatible reference.
    #[clap(long, value_name = "PATH")]
    transcriptome: CliPath,

    /// CSV file specifying the probe set used, if any.
    #[clap(long, value_name = "CSV")]
    probe_set: Option<CliPath>,

    /// Whether to filter the probe set using the "included" column of the probe set CSV.
    /// Filtering probes is recommended. See online documentation for details.
    #[clap(
        long,
        value_name = "true|false",
        default_value = "true",
        requires = "probe_set",
        conflicts_with = "no_probe_filter"
    )]
    filter_probes: Option<bool>,

    /// Equivalent to --filter-probes=false. Provided for backward compatibility.
    #[clap(
        long,
        hide = true,
        requires = "probe_set",
        conflicts_with = "filter_probes"
    )]
    no_probe_filter: bool,

    // Arguments to specify fastq data -- should be shared across most pipelines.
    #[clap(flatten)]
    fastqs: FastqArgs,

    /// CSV file declaring input library data sources.
    #[clap(
        long,
        value_name = "CSV",
        required_unless_present = "fastqs",
        conflicts_with = "no_libraries"
    )]
    libraries: Option<CliPath>,

    /// Feature reference CSV file corresponding to the antibody panel used to
    /// prepare the protein expression library.
    #[clap(long, value_name = "CSV")]
    feature_ref: Option<CliPath>,

    /// Use this option if the slide serial number and area were entered
    /// incorrectly on the CytAssist instrument and the correct values
    /// are unknown. When --unknown-slide is used, the data may not be well aligned
    /// to the images. Please see the online documentation for more information.
    /// Not compatible with --slide, --area, or --slide-file options.
    #[clap(
        long,
        ignore_case = true,
        value_name = "visium-1|visium-2|visium-2-large|visium-hd"
    )]
    unknown_slide: Option<UnknownSlide>,

    /// Whether the pipeline should rotate and mirror the image to align the
    /// “upright hourglass” fiducial pattern in the top left corner.
    /// Only set this to false if you are certain your image is already oriented
    /// with the “upright hourglass” in the top left corner. [default: true]
    #[clap(
        long = "reorient-images",
        value_name = "true|false",
        conflicts_with = "loupe_alignment"
    )]
    reorient_images: Option<bool>,

    /// Slide design file for your slide, based on serial number and downloaded from
    /// 10x Genomics. NOTE: this is only required if your
    /// machine doesn't have internet access. You must still
    /// specify --slide and --area when using this argument.
    #[clap(long, value_name = "GPR|VLF", conflicts_with = "unknown_slide")]
    slidefile: Option<SlideFile>,

    /// Microns per microscope image pixel. Only use if the microscope is well calibrated
    /// and the scale information is accurate. Scale is used to improve CytAssist
    /// to microscope image registration.
    #[clap(long, value_name = "FLOAT")]
    image_scale: Option<f64>,

    /// Alignment file produced by the Loupe Browser manual alignment step.
    #[clap(long, value_name = "PATH")]
    loupe_alignment: Option<CliPath>,

    #[clap(flatten)]
    create_bam: CreateBam,

    /// Disable secondary analysis, e.g. clustering. Optional.
    #[clap(long = "nosecondary")]
    no_secondary_analysis: bool,

    /// Hard trim the input Read 1 to this length before analysis.
    #[clap(long, value_name = "NUM")]
    r1_length: Option<usize>,

    /// Hard trim the input Read 2 to this length before analysis.
    #[clap(long, value_name = "NUM")]
    r2_length: Option<usize>,

    /// Include intronic reads in count (default=false).
    #[clap(
        long = "include-introns",
        hide = true,
        value_name = "true|false",
        conflicts_with = "probe_set"
    )]
    include_introns: Option<bool>,

    /// Specify the aligner.
    #[clap(long, hide = true, value_name = "hurtle|star", requires = "probe_set")]
    aligner: Option<AlignerParam>,

    /// Skip tissue segmentation and use all spots
    #[clap(long, hide = true)]
    use_all_spots: bool,

    /// Proceed with processing using a --feature-ref but no
    /// protein expression libraries specified with the
    /// --libraries flag.
    #[clap(long, requires = "feature_ref", conflicts_with = "libraries")]
    no_libraries: bool,

    /// Do not execute the pipeline.
    /// Generate an pipeline invocation (.mro) file and stop.
    #[clap(long)]
    dry: bool,

    #[clap(flatten)]
    mrp: MrpArgs,

    /// Include previous filtered feature barcode matrix for pattern fix
    #[clap(long = "v1-filtered-fbm", hide = true)]
    v1_filtered_fbm: Option<CliPath>,

    #[clap(long = "v1-pattern-type", hide = true, requires = "v1_filtered_fbm")]
    v1_pattern_type: Option<u32>,

    /// Bin Visium HD data to the specified bin size in addition to the standard binning size
    /// (2 um, 8 um, 16 um). Also produces a Loupe file and secondary analysis outputs
    /// at the specified bin size in addition to the standard analysis bin size (8 um) if it is
    /// larger than 8 um.
    ///
    /// Supply a value in microns. Only even integer values between 4 and 100
    /// are allowed.
    #[clap(long, value_name = "NUM")]
    custom_bin_size: Option<CustomBinSize>,
}

const GPR_EXTENSION: &str = "gpr";
const VLF_EXTENSION: &str = "vlf";

#[derive(Debug, Clone)]
enum SlideFile {
    Gpr(CliPath),
    Vlf(CliPath),
}

impl FromStr for SlideFile {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        let path: CliPath = s.parse()?;
        match path.extension().and_then(|ext| ext.to_str()) {
            Some(GPR_EXTENSION) => Ok(SlideFile::Gpr(path)),
            Some(VLF_EXTENSION) => Ok(SlideFile::Vlf(path)),
            _ => bail!("Slide file must be a '.{GPR_EXTENSION}' or '.{VLF_EXTENSION}' file"),
        }
    }
}

impl SlideFile {
    fn gpr_file(&self) -> Option<CliPath> {
        match self {
            SlideFile::Gpr(path) => Some(path.clone()),
            SlideFile::Vlf(_) => None,
        }
    }
    fn vlf_file(&self) -> Option<CliPath> {
        match self {
            SlideFile::Vlf(path) => Some(path.clone()),
            SlideFile::Gpr(_) => None,
        }
    }
}

#[derive(Clone, Debug)]
struct CustomBinSize(u32);

impl FromStr for CustomBinSize {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        let err = "Custom bin size must be an even integer between 4 and 100";
        // Allow um suffix
        let bin_size = s
            .trim_end_matches("um")
            .parse::<u32>()
            .map_err(|_| anyhow::anyhow!(err))?;
        if !(4..=100).contains(&bin_size) || bin_size % 2 != 0 {
            bail!(err);
        }
        Ok(CustomBinSize(bin_size))
    }
}

// selects three modes of input images - must match
// lib/python/cellranger/constants.py
const DARK_IMAGES_BRIGHTFIELD: u8 = 0;
const DARK_IMAGES_CHANNELS: u8 = 1;
const DARK_IMAGES_COLORIZED: u8 = 2;

impl Count {
    pub fn to_mro_args(&self) -> Result<CountCsMro> {
        let c = self.clone();
        let targeting_method = c
            .probe_set
            .as_ref()
            .map(|_| TargetingMethod::TemplatedLigation);

        let mut sample_defs = Vec::new();
        if let Some(libraries) = &c.libraries {
            // parse the libraries.csv & convert it into a set of SampleDefs.
            let libraries = cr_types::parse_legacy_libraries_csv(libraries)?;

            let has_antibody_library = libraries.iter().any(|library| {
                library
                    .library_type
                    .is_fb_type(FeatureBarcodeType::Antibody)
            });
            let has_gex_library = libraries
                .iter()
                .any(|library| library.library_type.is_gex());

            if has_antibody_library && !has_gex_library && c.probe_set.is_some() {
                bail!(
                    "No Gene Expression library found in the libraries csv. \
                When specifying --probe-set and --libraries, you must \
                also specify a Gene Expression library in the libraries \
                csv using the --libraries flag."
                );
            } else if has_gex_library && !has_antibody_library && c.feature_ref.is_some() {
                bail!(
                    "No Antibody Capture library found in the libraries csv. \
                When specifying --feature-ref, you must also specify \
                an Antibody Capture library in the libraries csv using \
                the --libraries flag or specify --no-libraries if no \
                Antibody Capture library is provided."
                );
            } else if has_gex_library && !has_antibody_library && c.feature_ref.is_none() {
                bail!(
                    "Only Gene Expression library was found in the libraries csv. \
                If you do not have an Antibody Capture library, you must \
                specify the Gene Expression library using --fastqs."
                );
            }

            for l in libraries {
                let fq = FastqArgs {
                    fastqs: vec![CliPath::from(l.fastqs)],
                    project: l.project,
                    sample: Some(vec![l.sample]),
                    lanes: None,
                };

                if l.library_type == LibraryType::Gex && c.probe_set.is_none() {
                    bail!(
                        "When specifying a gene expression library using --libraries, \
                    you must also specify --probe-set. If no probe set was used to \
                    create this library then you should use the --fastqs flag instead."
                    );
                } else if l.library_type != LibraryType::Gex
                    && l.library_type != LibraryType::Antibody
                {
                    bail!(
                        "Allowed library types include Gene Expression and Antibody \
                    Capture. Please check the libraries csv passed with the \
                    --libraries flag."
                    );
                }
                sample_defs.extend(fq.get_sample_defs(l.library_type)?);
            }
        } else {
            sample_defs.extend(c.fastqs.get_sample_defs(LibraryType::Gex)?);
        }

        // override-id needs something to override. Either slide and area, or unknown-slide.
        if c.override_id && !(c.unknown_slide.is_some() || c.slide.is_some()) {
            bail!("--override-id requires either --slide and --area or --unknown-slide.")
        }

        // image-scale needs some microscope image
        if c.image_scale.is_some()
            && !(c.image.is_some() || c.darkimage.is_some() || c.colorizedimage.is_some())
        {
            bail!("--image-scale requires a microscope image to scale. Please provide a microscope image using --image, --darkimage, or --colorizedimage.")
        }

        if let Some(image_scale) = c.image_scale {
            if image_scale <= 0.0 || image_scale > MAX_IMAGE_SCALE {
                bail!(
                    "--image-scale is the microns per pixel size and requires a positive value between 0 and {MAX_IMAGE_SCALE} ({MAX_IMAGE_SCALE} microns/pixel). Image scale provided : {} microns/pixel",
                    image_scale
                );
            }
        }

        let slide_serial_capture_area_from_all_sources = SlideInformationFromDifferentSources::new(
            c.cytaimage.as_ref(),
            c.loupe_alignment.as_ref(),
            c.slide,
            c.area,
            c.unknown_slide.as_ref(),
            c.override_id,
        )?;

        let consolidated_slide_serial_capture_area =
            slide_serial_capture_area_from_all_sources.consolidate_slide_id()?;

        let consolidated_override_id = c.override_id || c.loupe_alignment.is_some();

        let v1_pattern_fix = match (c.v1_filtered_fbm, c.v1_pattern_type) {
            (Some(fbm), None) => Some(V1PatternFixArgs {
                v1_filtered_fbm: fbm,
                v1_pattern_type: 1, // for backwards compat, default to 1
            }),
            (Some(fbm), Some(pat)) => Some(V1PatternFixArgs {
                v1_filtered_fbm: fbm,
                v1_pattern_type: pat,
            }),
            (None, _) => None,
        };

        v1_pattern_fix
            .as_ref()
            .map(|x| x.validate(&consolidated_slide_serial_capture_area))
            .transpose()?;

        let slide_serial_capture_area =
            consolidated_slide_serial_capture_area.get_slide_serial_capture_area()?;

        let slide_revision = consolidated_slide_serial_capture_area.revision()?;

        if c.custom_bin_size.is_some() && slide_revision != SlideRevision::H1 {
            bail!("--custom-bin-size should only be used for a Visium HD run. The slide version you have provided is {:?} (inferred from {}).", 
            &slide_revision, &consolidated_slide_serial_capture_area.slide_id_str());
        }

        let chemistry = slide_revision.chemistry();
        if !chemistry.is_cytassist_compatible() && c.cytaimage.is_some() {
            bail!(
                "The chemistry indicated by {}: {chemistry}, is incompatible with '--cytaimage <IMG>' input.", 
                &consolidated_slide_serial_capture_area.slide_id_str()
            );
        }

        if chemistry.is_cytassist_compatible() && c.cytaimage.is_none() {
            bail!(
                "The chemistry indicated by {}: {chemistry}, needs a '--cytaimage <IMG>' input, but none was provided.", 
                &consolidated_slide_serial_capture_area.slide_id_str()
            );
        }

        if chemistry.is_spatial_hd_rtl() && c.probe_set.is_none() {
            bail!(
                "The chemistry indicated by {}: {chemistry}, requires '--probe-set <PATH>' input.",
                &consolidated_slide_serial_capture_area.slide_id_str()
            );
        }

        if chemistry.is_spatial_fb() && c.probe_set.is_none() && c.libraries.is_none() {
            // and why does the error message mention probe-set but check --libraries?
            bail!(
                "The chemistry indicated by {}: {chemistry}, requires '--probe-set <PATH>' input.",
                &consolidated_slide_serial_capture_area.slide_id_str()
            );
        }

        if !chemistry.is_spatial_fb() && (c.libraries.is_some() || c.feature_ref.is_some()) {
            bail!(
                "The chemistry indicated by {}: {chemistry}, cannot be used with '--libraries <PATH>' or '--feature-ref <PATH>'.",
                &consolidated_slide_serial_capture_area.slide_id_str()
            );
        }

        let (tissue_image_paths, dark_images) = if c.darkimage.is_some() {
            (c.darkimage.unwrap(), DARK_IMAGES_CHANNELS)
        } else if c.colorizedimage.is_some() {
            (vec![c.colorizedimage.unwrap()], DARK_IMAGES_COLORIZED)
        } else if c.image.is_some() {
            (vec![c.image.unwrap()], DARK_IMAGES_BRIGHTFIELD)
        } else {
            (vec![], DARK_IMAGES_BRIGHTFIELD)
        };

        let cytassist_image_paths = c.cytaimage.map_or_else(Vec::new, |img| vec![img]);

        let reorientation_mode = if c.reorient_images == Some(false) {
            None
        } else {
            Some("rotation+mirror".to_string())
        };

        // customer facing RTL defaults to 50 base R2 trim if not set
        const DEFAULT_RTL_R2_LENGTH: usize = 50;
        let r1_length = c.r1_length;
        let r2_length = if targeting_method == Some(TargetingMethod::TemplatedLigation)
            && c.r2_length.is_none()
            && c.aligner != Some(AlignerParam::Star)
        {
            Some(DEFAULT_RTL_R2_LENGTH)
        } else {
            c.r2_length
        };

        Ok(CountCsMro {
            sample_id: c.id,
            sample_def: sample_defs,
            target_set_name: get_target_set_name(&c.probe_set),
            target_set: c.probe_set,
            sample_desc: c.description.unwrap_or_default(),
            reference_path: c.transcriptome,
            no_bam: !c.create_bam.validated()?,
            no_secondary_analysis: c.no_secondary_analysis,
            filter_probes: c.filter_probes.unwrap() && !c.no_probe_filter,
            r1_length,
            r2_length, // possible default used instead of command line
            targeting_method,
            aligner: c.aligner,
            chemistry,
            trim_polya_min_score: Some(20),
            trim_tso_min_score: Some(20),
            feature_reference: c.feature_ref,
            include_introns: c.include_introns.unwrap_or_default(),
            tissue_image_paths,
            cytassist_image_paths,
            dark_images,
            dapi_channel_index: c.dapi_index,
            reorientation_mode,
            image_scale: c.image_scale,
            override_id: consolidated_override_id,
            skip_tissue_detection: c.use_all_spots,
            slide_serial_capture_area,
            loupe_alignment_file: c.loupe_alignment,
            gpr_file: c.slidefile.as_ref().and_then(SlideFile::gpr_file),
            hd_layout_file: c.slidefile.as_ref().and_then(SlideFile::vlf_file),
            v1_pattern_fix,
            custom_bin_size: c.custom_bin_size.map(|x| x.0),
        })
    }
}

#[derive(Serialize, Clone)]
struct V1PatternFixArgs {
    v1_filtered_fbm: CliPath,
    v1_pattern_type: u32,
}

impl V1PatternFixArgs {
    fn is_valid_pattern_slide(&self, slide_id: &SlideId) -> bool {
        let valid: &[&str] = match &self.v1_pattern_type {
            1 => &[
                "V11D", "V12J", "V12F", "V12M", "V12A", "V12Y", "V12U", "V12L",
            ],
            2 => &["V42D", "V43J", "V43M", "V43A", "V43F"],
            _ => unreachable!(),
        };
        valid.iter().any(|&prefix| slide_id.starts_with(prefix))
    }

    fn is_valid_pattern_slide_revision(&self, slide_revision: &SlideRevision) -> bool {
        matches!(
            (self.v1_pattern_type, &slide_revision),
            (1, SlideRevision::V1) | (2, SlideRevision::V4)
        )
    }

    fn validate(&self, slide_information: &SlideInformation) -> Result<()> {
        if let SlideInformation::Known(SlideSerialCaptureArea { slide_id, area: _ }) =
            &slide_information
        {
            if !self.is_valid_pattern_slide(slide_id) {
                bail!(
                    "Slide ID {} cannot be run with UMI downsampling (--v1-filtered-fbm).",
                    slide_id
                );
            }
        }
        let slide_revision = slide_information.revision()?;
        if !self.is_valid_pattern_slide_revision(&slide_revision) {
            bail!(
                "{} cannot be run with UMI downsampling (--v1-filtered-fbm).",
                slide_information.slide_id_str()
            );
        }

        Ok(())
    }
}

#[derive(Serialize)]
struct CountCsMro {
    sample_id: String,
    sample_def: Vec<SampleDef>,
    target_set: Option<CliPath>,
    target_set_name: Option<String>,
    sample_desc: String,
    reference_path: CliPath,
    no_bam: bool,
    no_secondary_analysis: bool,
    filter_probes: bool,
    r1_length: Option<usize>,
    r2_length: Option<usize>,
    targeting_method: Option<TargetingMethod>,
    aligner: Option<AlignerParam>,
    chemistry: ChemistryName,
    trim_polya_min_score: Option<usize>,
    trim_tso_min_score: Option<usize>,
    feature_reference: Option<CliPath>,
    include_introns: bool,
    tissue_image_paths: Vec<CliPath>,
    cytassist_image_paths: Vec<CliPath>,
    override_id: bool,
    dark_images: u8,
    dapi_channel_index: Option<usize>,
    image_scale: Option<f64>,
    reorientation_mode: Option<String>,
    skip_tissue_detection: bool,
    slide_serial_capture_area: Option<String>,
    loupe_alignment_file: Option<CliPath>,
    gpr_file: Option<CliPath>,
    hd_layout_file: Option<CliPath>,
    v1_pattern_fix: Option<V1PatternFixArgs>,
    custom_bin_size: Option<u32>,
}

/// Aggregates the feature-barcode count data
/// generated from multiple runs of the 'spaceranger count' pipeline.
// To run this pipeline, supply a CSV that enumerates the paths to the
// filtered_matrix.h5 files produced by 'spaceranger count'.
#[derive(Parser, Debug, Clone, Serialize)]
struct Aggr {
    /// A unique run id and output folder name [a-zA-Z0-9_-]+.
    #[clap(long = "id", value_name = "ID", value_parser = validate_id, required = true )]
    sample_id: String,

    /// Sample description to embed in output files.
    #[clap(long = "description", default_value = "", value_name = "TEXT")]
    sample_desc: String,

    /// Path of CSV file enumerating 'spaceranger count' outputs.
    #[clap(long = "csv", value_name = "CSV")]
    aggregation_csv: CliPath,

    /// Library depth normalization mode.
    #[clap(
        long = "normalize",
        default_value = "mapped",
        value_name = "MODE",
        value_parser = ["mapped", "none"],
    )]
    normalization_mode: String,

    /// Do not execute the pipeline.
    /// Generate an pipeline invocation (.mro) file and stop.
    #[serde(skip)]
    #[clap(long)]
    dry: bool,

    // not a cmd-line arg -- should be filled in with the working dir
    #[clap(hide = true, default_value = ".")]
    pipestance_root: PathBuf,

    #[serde(skip)]
    #[clap(flatten)]
    mrp: MrpArgs,
}

#[derive(Parser, Debug, Clone)]
struct Testrun {
    /// A unique run id and output folder name [a-zA-Z0-9_-]+.
    #[clap(long, value_name = "ID", required = true)]
    id: String,

    /// Sample description to embed in output files.
    #[clap(long, value_name = "TEXT")]
    description: Option<String>,

    /// Do not execute the pipeline.
    /// Generate a pipeline invocation (.mro) file and stop.
    #[clap(long)]
    dry: bool,

    /// Set this if no internet connection is available on the processing computer/node/HPC.
    #[clap(long)]
    no_internet: bool,

    #[clap(flatten)]
    mrp: MrpArgs,
}

impl Testrun {
    fn to_mro_args(&self, pkg_env: &env::PkgEnv) -> Result<CountCsMro> {
        // determine paths relative to build
        let files_path = pkg_env.build_path.join("external");
        let inputs_path = files_path.join("spaceranger_tiny_inputs");
        let fastqs = vec![CliPath::from(inputs_path.join("fastqs"))];
        let reference_path = CliPath::from(files_path.join("spaceranger_tiny_ref/"));
        let tissue_image_paths = vec![CliPath::from(inputs_path.join("image/tinyimage.jpg"))];
        let cytassist_image_paths = Vec::new();

        // sample defs
        let mut sample_def = Vec::new();
        sample_def.extend(
            FastqArgs {
                fastqs,
                project: None,
                sample: Some(vec![String::from("tinytest")]),
                lanes: Some(vec![1]),
            }
            .get_sample_defs(LibraryType::Gex)?,
        );

        let slide_serial_capture_area = if self.no_internet {
            None
        } else {
            Some("V19L29-035-A1".to_string())
        };

        let t = self.clone();
        Ok(CountCsMro {
            sample_id: t.id,
            sample_def,
            target_set: None,
            target_set_name: None,
            sample_desc: t.description.unwrap_or_default(),
            reference_path,
            no_bam: false,
            no_secondary_analysis: false,
            filter_probes: true,
            r1_length: None,
            r2_length: None,
            targeting_method: None,
            aligner: None,
            chemistry: ChemistryName::from_str("SPATIAL3Pv1").unwrap(),
            trim_polya_min_score: None,
            trim_tso_min_score: None,
            feature_reference: None,
            include_introns: false,
            tissue_image_paths,
            cytassist_image_paths,
            dark_images: DARK_IMAGES_BRIGHTFIELD,
            dapi_channel_index: None,
            reorientation_mode: Some("rotation+mirror".to_string()),
            skip_tissue_detection: false,
            slide_serial_capture_area,
            loupe_alignment_file: None,
            image_scale: None,
            override_id: false,
            gpr_file: None,
            hd_layout_file: None,
            v1_pattern_fix: None,
            custom_bin_size: None,
        })
    }
}

fn inner_main() -> Result<ExitCode> {
    set_env_vars();

    // parse the cmd-line
    let opts = SpaceRanger::parse();

    // setup the environment
    let pkg_env = env::setup_env(opts.env_json, CMD, sr_version())?;

    // check deprecated os bits using a subprocess
    check_deprecated_os()?;

    // You can handle information about subcommands by requesting their matches by name
    // (as below), requesting just the name used, or both at the same time
    match opts.subcmd {
        SubCommand::Count(c) => {
            // Make sure an image is passed (clap seems to be missing this check)
            if c.image.is_none()
                && c.darkimage.is_none()
                && c.colorizedimage.is_none()
                && c.cytaimage.is_none()
            {
                bail!(
                    "You must supply either --image, --darkimage, --colorizedimage, or --cytaimage."
                );
            }

            if c.cytaimage.is_none()
                && (c.darkimage.is_some() || c.colorizedimage.is_some())
                && c.loupe_alignment.is_none()
            {
                bail!(
                    "A loupe alignment file is required when using a fluorescence image without a corresponding CytAssist image."
                );
            }

            if c.dapi_index.is_some_and(|index| index < 1) {
                bail!("The dapi_index is 1-indexed and should be larger than 0.");
            }

            // If --feature-ref is provided, require --libraries or --no-libraries.
            //
            // --no-libraries historically was used to allow a feature reference to be added to
            // pre-feature-barcoding GEX data in order to allow aggr of that data w/ GEX+FB
            // libraries by specifying an (unused, but compatible) feature reference for the
            // GEX-only run.
            //
            if c.feature_ref.is_some() && c.libraries.is_none() && !c.no_libraries {
                bail!(
"You specified --feature-ref, but not --libraries. Did you mean to input protein expression libraries?
If you have 1 or more protein expression libraries:
    Use --libraries to specify your input FASTQs and the associated library types.
If you want to proceed with a feature reference, but no protein expression data:
    Add the --no-libraries flag.");
            }

            // make sure --id isn't tool long

            let mro_args = c.to_mro_args()?;
            let mro = make_mro(
                "SPATIAL_RNA_COUNTER_CS",
                &mro_args,
                "rna/spatial_rna_counter_cs.mro",
            )?;
            execute(&c.id, &mro, &c.mrp, c.dry)
        }

        SubCommand::Aggr(mut aggr) => {
            // Custom validation

            // fill in pipestance_root
            aggr.pipestance_root = std::env::current_dir()?;

            let mro = make_mro(
                "SPATIAL_RNA_AGGREGATOR_CS",
                &aggr,
                "rna/spatial_rna_aggregator_cs.mro",
            )?;
            execute(&aggr.sample_id, &mro, &aggr.mrp, aggr.dry)
        }

        SubCommand::Mkfastq(m) => mkfastq::run_mkfastq(&m, "_spaceranger_internal"),
        SubCommand::Testrun(t) => {
            let mro_args = t.to_mro_args(&pkg_env)?;
            let mro = make_mro(
                "SPATIAL_RNA_COUNTER_CS",
                &mro_args,
                "rna/spatial_rna_counter_cs.mro",
            )?;
            execute(&t.id, &mro, &t.mrp, t.dry)
        }
        SubCommand::RnaShared(args) => shared_cmd::run_rna_shared(&pkg_env, args),
        SubCommand::Shared(args) => shared_cmd::run_shared(&pkg_env, args),
        SubCommand::Hidden(args) => shared_cmd::run_hidden(args),
    }
}

fn main() -> ExitCode {
    match inner_main() {
        Ok(exit_code) => exit_code,
        Err(err) => {
            cr_wrap::utils::print_error_chain(&err);
            ExitCode::FAILURE
        }
    }
}