#![deny(missing_docs)]
use crate::cytassist::CytassistMetadata;
use anyhow::{bail, Context, Result};
use cr_types::chemistry::ChemistryName;
use cr_wrap::utils::CliPath;
use serde::Deserialize;
use slide_design::validate_slide_id_name;
use std::fmt::{Debug, Display, Formatter};
use std::fs::File;
use std::io::BufReader;
use std::str::FromStr;

const AREA_USAGE: &str = r"
- Expecting A1, B1, C1, or D1 for 6.5 mm (V1-, V2-, V3-, V4-prefix) slides
- You can use A and B for 11 mm (V5-prefix) slides
- You can use A and D for HD v1 slides
Capture Area provided:";

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct SlideId(String);

impl FromStr for SlideId {
    type Err = anyhow::Error;

    fn from_str(slide_id: &str) -> Result<Self> {
        match validate_slide_id_name(slide_id) {
            Ok(s) => Ok(SlideId(s)),
            Err(e) => Err(e),
        }
    }
}

impl Display for SlideId {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl SlideId {
    pub(crate) fn starts_with(&self, prefix: &str) -> bool {
        self.0.starts_with(prefix)
    }

    // if we have a slide_serial_capture_area, we use the first two characters
    // (in lowercase) as the slide type (e.g. v1, v2, v3, etc...)
    pub(crate) fn revision(&self) -> SlideRevision {
        match &self.0[..2] {
            "V1" => SlideRevision::V1,
            "V2" => SlideRevision::V2,
            "V3" => SlideRevision::V3,
            "V4" => SlideRevision::V4,
            "V5" => SlideRevision::V5,
            "H1" => SlideRevision::H1,
            _ => unreachable!(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct AreaId(String);
impl FromStr for AreaId {
    type Err = anyhow::Error;
    fn from_str(area: &str) -> Result<Self> {
        let re = regex::Regex::new(r"^(?i)[ABCD](1?)$")?; // case insensitive and optional "1"
        if re.is_match(area) {
            Ok(AreaId(String::from(area)))
        } else {
            bail!(format!("{AREA_USAGE} {area}"));
        }
    }
}
impl Display for AreaId {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(PartialEq, Clone, Copy, Debug)]
pub(crate) enum SlideRevision {
    V1,
    V2,
    V3,
    V4,
    V5,
    H1,
}

impl SlideRevision {
    pub(crate) fn chemistry(self, has_probe_set: bool) -> ChemistryName {
        match self {
            SlideRevision::V1 => ChemistryName::SpatialThreePrimeV1,
            SlideRevision::V2 => ChemistryName::SpatialThreePrimeV2,
            SlideRevision::V3 => ChemistryName::SpatialThreePrimeV3,
            SlideRevision::V4 => ChemistryName::SpatialThreePrimeV4,
            SlideRevision::V5 => ChemistryName::SpatialThreePrimeV5,
            SlideRevision::H1 => {
                if has_probe_set {
                    ChemistryName::SpatialHdV1Rtl
                } else {
                    ChemistryName::SpatialHdV1ThreePrime
                }
            }
        }
    }
    pub(crate) fn validated_area(self, area: &AreaId) -> Result<String> {
        let mut area = area.0.to_uppercase();
        // Allow only A/D in HD
        if self == SlideRevision::H1 && !["A", "D", "A1", "D1"].contains(&area.as_str()) {
            bail!(format!(
                "invalid value for '--area <TEXT>': {AREA_USAGE} {area}"
            ));
        }
        // allow specification of single-letter names for V5 slides and HD
        // assume column 1 for everything that follows
        if area.len() == 1 {
            if matches!(self, SlideRevision::V5 | SlideRevision::H1) {
                area += "1";
            } else {
                bail!(format!(
                    "invalid value for '--area <TEXT>': {AREA_USAGE} {area}"
                )); // exit top level with Error
            }
        }
        Ok(area)
    }

    pub(crate) fn is_visium_hd(self) -> bool {
        match self {
            Self::H1 => true,
            Self::V1 | Self::V2 | Self::V3 | Self::V4 | Self::V5 => false,
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub(crate) enum SlideResolutionErrors {
    MissingFromAllSources,
    LoupeFileMismatch {
        cli_info: SlideInformation,
        loupe_info: SlideInformation,
    },
    ImageMetadataMismatch {
        cli_info: SlideInformation,
        image_metadata: SlideSerialCaptureArea,
    },
}

impl Display for SlideResolutionErrors {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let msg = match self {
            Self::MissingFromAllSources => String::from(
                "Slide ID and capture area information must be provided using either --slide and --area or --unknown-slide.",
            ),
            Self::LoupeFileMismatch {
                cli_info,
                loupe_info,
            } => {
                let cli_str = cli_info.cli_str();
                let loupe_str = loupe_info.loupe_str();
                format!(
                    "You specified {cli_str}, but during manual image alignment in Loupe you specified {loupe_str}. If the information given to Loupe was correct, remove {cli_str}. Otherwise, repeat manual image alignment in Loupe."
                )
            }
            Self::ImageMetadataMismatch {
                cli_info,
                image_metadata,
            } => {
                let cli_str = cli_info.cli_str();
                let image_str = image_metadata.description_string();
                format!(
                    "You specified {cli_str}, but the CytAssist run information present in the image says {image_str}. If the CytAssist run information is correct, remove {cli_str}. Otherwise, add --override-id to override the CytAssist run information."
                )
            }
        };
        write!(f, "{msg}")
    }
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct SlideSerialCaptureAreaReader {
    #[serde(alias = "serialNumber")]
    slide_id: Option<String>,
    area: Option<String>,
}

impl SlideSerialCaptureAreaReader {
    /// Converts SlideSerialCaptureAreaReader to SlideInformation
    /// This is UnknownSlide only if we see at least one empty string in among
    /// the slide_id and area
    fn into_slide_information(self) -> Result<Option<SlideInformation>> {
        if let Self {
            slide_id: Some(slide_id),
            area: Some(area),
        } = self
        {
            match (slide_id.is_empty(), area.is_empty()) {
                (false, false) => Ok(Some(SlideInformation::Known(
                    SlideSerialCaptureArea::from_non_empty_strings(slide_id, area)?,
                ))),
                (true, true) => Ok(Some(SlideInformation::NoInformationFound)),
                _ => Ok(None),
            }
        } else {
            Ok(None)
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct SlideSerialCaptureArea {
    pub(crate) slide_id: SlideId,
    pub(crate) area: AreaId,
}

impl SlideSerialCaptureArea {
    pub(crate) fn get_slide_serial_capture_area(&self) -> Result<String> {
        let slide_revision = self.slide_id.revision();
        Ok(format!(
            "{}-{}",
            self.slide_id,
            slide_revision.validated_area(&self.area)?
        ))
    }

    fn from_non_empty_strings(slide_id: String, area: String) -> Result<SlideSerialCaptureArea> {
        Ok(SlideSerialCaptureArea {
            slide_id: slide_id.parse().context("Error validating slide ID")?,
            area: area.parse().context("Error validating capture area")?,
        })
    }

    pub(crate) fn description_string(&self) -> String {
        format!("slide ID {} and capture area {}", self.slide_id, self.area)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum SlideInformation {
    Known(SlideSerialCaptureArea),
    NoInformationFound,
    UnknownSlideFound(UnknownSlide),
}

impl SlideInformation {
    fn cli_str(&self) -> String {
        match &self {
            Self::Known(SlideSerialCaptureArea { slide_id, area }) => {
                format!("--slide {slide_id} --area {area}")
            }
            Self::UnknownSlideFound(unknown_slide) => {
                format!("--unknown-slide={unknown_slide}")
            }
            Self::NoInformationFound => String::from("NOT PROVIDED"),
        }
    }

    pub(crate) fn slide_id_str(&self) -> String {
        match &self {
            Self::Known(SlideSerialCaptureArea { slide_id, area: _ }) => {
                format!("Slide ID {slide_id}")
            }
            Self::UnknownSlideFound(unknown_slide) => {
                format!("unknown-slide of type {unknown_slide}")
            }
            Self::NoInformationFound => String::from("Slide information not provided"),
        }
    }

    fn loupe_str(&self) -> String {
        match &self {
            Self::Known(SlideSerialCaptureArea { slide_id, area }) => {
                format!("slide ID {slide_id} and capture area {area}")
            }
            Self::UnknownSlideFound(unknown_slide) => {
                format!("unkown slide of type {unknown_slide}")
            }
            Self::NoInformationFound => String::from("slide ID is not known"),
        }
    }

    pub(crate) fn revision(&self) -> Result<SlideRevision> {
        match &self {
            Self::Known(SlideSerialCaptureArea { slide_id, area: _ }) => Ok(slide_id.revision()),
            Self::UnknownSlideFound(unknown_slide) => Ok(unknown_slide.revision()),
            // Should not occur. This should be handled by consolidation of slide ID.
            Self::NoInformationFound => bail!(
                "Either --slide and --area must be set or you should have slide and area in a loupe file or in the cytassist image or you must use --unknown-slide."
            ),
        }
    }

    pub(crate) fn get_slide_serial_capture_area(&self) -> Result<Option<String>> {
        if let SlideInformation::Known(slide_serial_capture_area) = &self {
            Some(slide_serial_capture_area.get_slide_serial_capture_area()).transpose()
        } else {
            Ok(None)
        }
    }
}

#[derive(Debug, PartialEq)]
pub(crate) struct SlideInformationFromDifferentSources {
    cli: SlideInformation,
    loupe: Option<SlideInformation>,
    cytassist_image: Option<SlideSerialCaptureArea>,
    override_id: bool,
}

impl SlideInformationFromDifferentSources {
    pub(crate) fn new(
        cytaimage: Option<&CliPath>,
        loupe_alignment: Option<&CliPath>,
        slide_in: Option<SlideId>,
        area_in: Option<AreaId>,
        unknown_slide: Option<&UnknownSlide>,
        override_id: bool,
    ) -> Result<SlideInformationFromDifferentSources> {
        let cytassist_image = if override_id {
            None
        } else {
            cytaimage.map(|cyta_img| {
            let cyta_metadata =  CytassistMetadata::new(cyta_img).context("Error while reading cytassist image metadata")?;
            if let CytassistMetadata { slide_serial:Some(slide_id), capture_area: Some(area)} = cyta_metadata {
                if slide_id.is_empty() || area.is_empty() {
                    Ok(None)
                } else {
                    Some(SlideSerialCaptureArea::from_non_empty_strings(slide_id, area).context("Error while reading slide ID and capture area from cytassist TIF metadata")).transpose()
                }
            } else {
                Ok(None)
            }
        }).transpose().context("Error while reading Slide ID and capture area from cytassist TIFF metadata.")?.flatten()
        };

        let loupe = loupe_alignment
            .map(|loupe_fl| {
                let file =
                    File::open(loupe_fl).context("Error while opening loupe alignment file.")?;
                let slide_info: SlideSerialCaptureAreaReader =
                    serde_json::from_reader(BufReader::new(file))
                        .context("Error while reading loupe alignment file.")?;
                slide_info.into_slide_information()
            })
            .transpose()
            .context("Error while reading Slide ID and capture area from loupe alignment file.")?
            .flatten();
        Ok(SlideInformationFromDifferentSources {
            cytassist_image,
            loupe,
            // The behaviour with the CLI is different from that with loupe. the slide information
            // is taken as UnknonwnSlide if we find nothing in the loupe file, but is considered
            // UnknownSlide only if we find --unknown-slide
            cli: match (slide_in, area_in, unknown_slide) {
                (Some(slide_id), Some(area), None) => {
                    SlideInformation::Known(SlideSerialCaptureArea { slide_id, area })
                }
                (None, None, Some(unkown_slide_name)) => {
                    SlideInformation::UnknownSlideFound(*unkown_slide_name)
                }
                (None, None, None) => SlideInformation::NoInformationFound,
                _ => unreachable!(
                    "--slide and --area must be provided together. --slide/--area cannot be used with--unknown-slide."
                ),
            },
            override_id,
        })
    }

    pub(crate) fn consolidate_slide_id(self) -> Result<SlideInformation, SlideResolutionErrors> {
        match self.cli {
            SlideInformation::Known(cli_slide_serial_capture_area) => {
                let cli_info = SlideInformation::Known(cli_slide_serial_capture_area);
                match (self.loupe, self.cytassist_image, self.override_id) {
                    (Some(loupe_info), _, _) => {
                        if cli_info == loupe_info {
                            Ok(loupe_info)
                        } else {
                            Err(SlideResolutionErrors::LoupeFileMismatch {
                                cli_info,
                                loupe_info,
                            })
                        }
                    }
                    (None, Some(image_metadata), false) => {
                        if cli_info == SlideInformation::Known(image_metadata.clone()) {
                            Ok(cli_info)
                        } else {
                            Err(SlideResolutionErrors::ImageMetadataMismatch {
                                cli_info,
                                image_metadata,
                            })
                        }
                    }
                    // either image metadata is absent or --override-id was given
                    (None, _, true) | (None, None, _) => Ok(cli_info),
                }
            }
            SlideInformation::UnknownSlideFound(unknown_slide) => {
                let cli_info = SlideInformation::UnknownSlideFound(unknown_slide);
                match (self.loupe, self.cytassist_image, self.override_id) {
                    (Some(loupe_info), _, _) => {
                        if loupe_info == cli_info
                            || loupe_info == SlideInformation::NoInformationFound
                        {
                            Ok(cli_info)
                        } else {
                            Err(SlideResolutionErrors::LoupeFileMismatch {
                                cli_info,
                                loupe_info,
                            })
                        }
                    }
                    (None, Some(image_metadata), false) => {
                        Err(SlideResolutionErrors::ImageMetadataMismatch {
                            cli_info,
                            image_metadata,
                        })
                    }
                    (None, _, true) | (None, None, _) => Ok(cli_info),
                }
            }
            SlideInformation::NoInformationFound => {
                match (self.loupe, self.cytassist_image, self.override_id) {
                    (Some(loupe_info), _, _) => Ok(loupe_info),
                    (None, Some(image_metadata), false) => {
                        Ok(SlideInformation::Known(image_metadata.clone()))
                    }
                    (None, _, true) => {
                        unreachable!(
                            "--override-id requires one of the following: slide and area, unknown-slide, or Loupe manual alignment file."
                        )
                    }
                    (None, None, _) => Err(SlideResolutionErrors::MissingFromAllSources),
                }
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum UnknownSlide {
    Visium1,
    Visium2,
    Visium2Large,
    VisiumHd,
}

impl FromStr for UnknownSlide {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "visium-1" => Ok(UnknownSlide::Visium1),
            "visium-2" => Ok(UnknownSlide::Visium2),
            "visium-2-large" => Ok(UnknownSlide::Visium2Large),
            "visium-hd" => Ok(UnknownSlide::VisiumHd),
            _ => bail!(
                "Allowed values for --unknown-slide: visium-1, visium-2, visium-2-large, or visium-hd"
            ),
        }
    }
}

impl UnknownSlide {
    // if we have --unknown_slide, then we map the terms:
    // visium-1 => "v1" original slides
    // visium-2 => "v4" new 6.5mm slides
    // visium-2-large => "v5" new XL slides
    // visium-hd => Visium HD v1 slides
    pub(crate) fn revision(&self) -> SlideRevision {
        match self {
            UnknownSlide::Visium1 => SlideRevision::V1,
            UnknownSlide::Visium2 => SlideRevision::V4,
            UnknownSlide::Visium2Large => SlideRevision::V5,
            UnknownSlide::VisiumHd => SlideRevision::H1,
        }
    }
}

impl Display for UnknownSlide {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let msg = match &self {
            UnknownSlide::Visium1 => "visium-1".to_string(),
            UnknownSlide::Visium2 => "visium-2".to_string(),
            UnknownSlide::Visium2Large => "visium-2-large".to_string(),
            UnknownSlide::VisiumHd => "visium-hd".to_string(),
        };
        write!(f, "{msg}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use itertools::{iproduct, Itertools};

    const VALID_HD_IDS: [&str; 2] = ["H1-PHFDVHK", "H1-VJQNJXM"];
    const VALID_AREAS: [&str; 2] = ["A1", "D1"];

    fn make_slide_serial_capture_areas() -> Vec<SlideSerialCaptureArea> {
        iproduct!(VALID_HD_IDS, VALID_AREAS)
            .map(|(s, a)| SlideSerialCaptureArea {
                slide_id: s.parse().unwrap(),
                area: a.parse().unwrap(),
            })
            .collect()
    }

    fn make_slide_infos() -> Vec<SlideInformation> {
        make_slide_serial_capture_areas()
            .into_iter()
            .map(SlideInformation::Known)
            .collect()
    }

    /// Arguments for running slide ID consolidation
    struct ConsolidationArgs(
        SlideInformation,
        Option<SlideInformation>,
        Option<SlideSerialCaptureArea>,
        bool,
    );

    fn consolidate_on_test_args(
        test_args: Vec<ConsolidationArgs>,
    ) -> Vec<Result<SlideInformation, SlideResolutionErrors>> {
        test_args
            .into_iter()
            .map(|args| SlideInformationFromDifferentSources {
                cli: args.0,
                loupe: args.1,
                cytassist_image: args.2,
                override_id: args.3,
            })
            .map(SlideInformationFromDifferentSources::consolidate_slide_id)
            .collect()
    }

    /// test slide ID consolidation when only a single slide is present, not from Loupe
    #[test]
    fn test_consolidation_single() {
        // TODO test
        let ids_areas = make_slide_serial_capture_areas();
        let infos = make_slide_infos();
        let i = 0;
        let test_args = vec![
            // (cli, loupe, cytassist_image, override_id)
            ConsolidationArgs(
                SlideInformation::NoInformationFound,
                None,
                Some(ids_areas[i].clone()),
                false,
            ),
            ConsolidationArgs(infos[i].clone(), None, Some(ids_areas[i].clone()), false),
            ConsolidationArgs(infos[i].clone(), None, Some(ids_areas[i].clone()), false),
            ConsolidationArgs(infos[i].clone(), None, None, false),
            ConsolidationArgs(infos[i].clone(), None, None, false),
        ];
        let results = consolidate_on_test_args(test_args);
        for res in results {
            assert!(res.is_ok());
            assert_eq!(res.unwrap(), SlideInformation::Known(ids_areas[i].clone()));
        }
    }

    /// Enum with four different things a source of truth can take
    #[derive(Debug, Deserialize)]
    #[serde(rename_all = "lowercase")]
    enum FourWay {
        Good,    // Means that the source had the correct slide ID
        Bad,     // Means that the source had an incorrect slide ID
        Absent,  // Implies that the source of truth was not found
        Unknown, // Implies the source said slide ID was unknown slide
    }

    const GOOD_SLIDE_ID: &str = "H1-K2BG7HN";
    const GOOD_SLIDE_AREA: &str = "A1";
    const BAD_SLIDE_ID_TIFF: &str = "H1-FZR38YM";
    const BAD_SLIDE_ID_LOUPE: &str = "H1-899GGTN";

    #[derive(Debug, Deserialize)]
    struct TestMatrixRow {
        cyta_image: FourWay,
        loupe_alignment: FourWay,
        override_id: bool,
        slide: FourWay,
        success: bool,
    }

    impl TestMatrixRow {
        fn into_test_input_output(self) -> Result<(SlideInformationFromDifferentSources, bool)> {
            let good_slide_id: SlideSerialCaptureArea =
                SlideSerialCaptureArea::from_non_empty_strings(
                    GOOD_SLIDE_ID.to_string(),
                    GOOD_SLIDE_AREA.to_string(),
                )?;
            let bad_slide_id_cyta_tiff: SlideSerialCaptureArea =
                SlideSerialCaptureArea::from_non_empty_strings(
                    BAD_SLIDE_ID_TIFF.to_string(),
                    GOOD_SLIDE_AREA.to_string(),
                )?;
            let bad_slide_id_loupe: SlideSerialCaptureArea =
                SlideSerialCaptureArea::from_non_empty_strings(
                    BAD_SLIDE_ID_LOUPE.to_string(),
                    GOOD_SLIDE_AREA.to_string(),
                )?;
            use FourWay::*;
            Ok((
                SlideInformationFromDifferentSources {
                    cytassist_image: match self.cyta_image {
                        Good => Some(good_slide_id.clone()),
                        Bad => Some(bad_slide_id_cyta_tiff.clone()),
                        _ => unimplemented!("Not implementing."),
                    },
                    cli: match self.slide {
                        Good => SlideInformation::Known(good_slide_id.clone()),
                        Unknown => SlideInformation::UnknownSlideFound(UnknownSlide::VisiumHd),
                        Absent => SlideInformation::NoInformationFound,
                        Bad => unimplemented!("Not implementing for now."),
                    },
                    loupe: match self.loupe_alignment {
                        Good => Some(SlideInformation::Known(good_slide_id.clone())),
                        Bad => Some(SlideInformation::Known(bad_slide_id_loupe.clone())),
                        Absent => None,
                        Unknown => unimplemented!("Not implementing for now."),
                    },
                    override_id: self.override_id,
                },
                self.success,
            ))
        }
    }

    #[test]
    fn run_slide_id_consolidation_matrix() -> Result<()> {
        let file = File::open("test/slide_id_cli_test_mtx.csv")
            .expect("Error while opening test matrix CSV file");
        let test_vec: Vec<TestMatrixRow> = csv::Reader::from_reader(file)
            .deserialize()
            .map(Result::unwrap)
            .collect_vec();
        for test_config in test_vec {
            let (test_struct, expected_result) = test_config.into_test_input_output()?;
            assert_eq!(test_struct.consolidate_slide_id().is_ok(), expected_result);
        }
        Ok(())
    }

    #[test]
    #[should_panic]
    fn run_slide_id_unreachable_config() {
        let test_config = TestMatrixRow {
            cyta_image: FourWay::Good,
            loupe_alignment: FourWay::Absent,
            override_id: true,
            slide: FourWay::Absent,
            success: false,
        };
        let (test_struct, _) = test_config.into_test_input_output().unwrap();
        test_struct.consolidate_slide_id().unwrap();
    }

    #[test]
    fn full_slide_id_unknown_slide_test() -> Result<()> {
        let slide_truth_sources = SlideInformationFromDifferentSources::new(
            Some(&CliPath::from_str("test/sd_cytassist_V43J24-025_D1.tiff")?),
            Some(&CliPath::from_str("test/unknown_slide_loupe.json")?),
            None,
            None,
            Some(UnknownSlide::VisiumHd).as_ref(),
            false,
        )?;
        assert_eq!(
            slide_truth_sources,
            SlideInformationFromDifferentSources {
                cli: SlideInformation::UnknownSlideFound(UnknownSlide::VisiumHd),
                loupe: Some(SlideInformation::NoInformationFound),
                cytassist_image: Some(SlideSerialCaptureArea {
                    slide_id: SlideId("V43J24-025".parse()?),
                    area: AreaId("D1".parse()?)
                }),
                override_id: false
            }
        );
        assert_eq!(
            slide_truth_sources.consolidate_slide_id()?,
            SlideInformation::UnknownSlideFound(UnknownSlide::VisiumHd)
        );
        Ok(())
    }

    #[test]
    fn full_slide_id_with_tiff_from_old_firmware() -> Result<()> {
        let slide_truth_sources = SlideInformationFromDifferentSources::new(
            Some(&CliPath::from_str(
                "test/sd_cytassist_V53M06-039_old_firmware.tiff",
            )?),
            Some(&CliPath::from_str(
                "test/sd_cytassist_V43J24-025_D1_loupe.json",
            )?),
            None,
            None,
            None,
            false,
        )?;
        assert_eq!(
            slide_truth_sources,
            SlideInformationFromDifferentSources {
                cli: SlideInformation::NoInformationFound,
                loupe: Some(SlideInformation::Known(SlideSerialCaptureArea {
                    slide_id: SlideId("V43J24-025".parse()?),
                    area: AreaId("D1".parse()?)
                })),
                cytassist_image: None,
                override_id: false
            }
        );
        assert_eq!(
            slide_truth_sources.consolidate_slide_id()?,
            SlideInformation::Known(SlideSerialCaptureArea {
                slide_id: SlideId("V43J24-025".parse()?),
                area: AreaId("D1".parse()?)
            })
        );
        Ok(())
    }

    #[test]
    fn full_slide_id_with_tiff_from_very_old_firmware() -> Result<()> {
        let slide_truth_sources = SlideInformationFromDifferentSources::new(
            Some(&CliPath::from_str(
                "test/sd_cytassist_V43J24-025_D1_no_tag.tiff",
            )?),
            Some(&CliPath::from_str(
                "test/sd_cytassist_V43J24-025_D1_loupe.json",
            )?),
            None,
            None,
            None,
            false,
        )?;
        assert_eq!(
            slide_truth_sources,
            SlideInformationFromDifferentSources {
                cli: SlideInformation::NoInformationFound,
                loupe: Some(SlideInformation::Known(SlideSerialCaptureArea {
                    slide_id: SlideId("V43J24-025".parse()?),
                    area: AreaId("D1".parse()?)
                })),
                cytassist_image: None,
                override_id: false
            }
        );
        assert_eq!(
            slide_truth_sources.consolidate_slide_id()?,
            SlideInformation::Known(SlideSerialCaptureArea {
                slide_id: SlideId("V43J24-025".parse()?),
                area: AreaId("D1".parse()?)
            })
        );
        Ok(())
    }

    #[test]
    fn expected_sop() -> Result<()> {
        let slide_truth_sources = SlideInformationFromDifferentSources::new(
            Some(&CliPath::from_str("test/sd_cytassist_V43J24-025_D1.tiff")?),
            None,
            None,
            None,
            None,
            false,
        )?;
        assert_eq!(
            slide_truth_sources,
            SlideInformationFromDifferentSources {
                cli: SlideInformation::NoInformationFound,
                loupe: None,
                cytassist_image: Some(SlideSerialCaptureArea {
                    slide_id: SlideId("V43J24-025".parse()?),
                    area: AreaId("D1".parse()?)
                }),
                override_id: false
            }
        );
        assert_eq!(
            slide_truth_sources.consolidate_slide_id()?,
            SlideInformation::Known(SlideSerialCaptureArea {
                slide_id: SlideId("V43J24-025".parse()?),
                area: AreaId("D1".parse()?)
            })
        );
        Ok(())
    }

    #[test]
    #[should_panic]
    fn loupe_with_bad_slide_id() {
        SlideInformationFromDifferentSources::new(
            None,
            Some(&CliPath::from_str("test/bad_slide_loupe.json").unwrap()),
            None,
            None,
            None,
            false,
        )
        .unwrap();
    }

    #[test]
    #[should_panic]
    fn cytaimage_with_bad_slide_id() {
        SlideInformationFromDifferentSources::new(
            Some(&CliPath::from_str("test/sd_cytassist_bad_slide_id_C05717-046_C1.tiff").unwrap()),
            None,
            None,
            None,
            None,
            false,
        )
        .unwrap();
    }
}
