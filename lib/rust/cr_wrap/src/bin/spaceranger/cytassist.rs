#![deny(missing_docs)]
use anyhow::{Context, Result};
use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use tiff::decoder::Decoder;
use tiff::tags::Tag;

// Custom tags written by the CytAssist software. The source of truth for this is
// the embedded apps repo.
const SLIDE_ID_TAG: Tag = Tag::Unknown(65002);
const CAPTURE_AREA_TAG: Tag = Tag::Unknown(65010);

const CYTA_ERROR: &str = r"
Invalid --cytaimage provided. File does not seem to be 
an unaltered image from the CytAssist instrument.";

#[derive(Debug, PartialEq)]
pub struct CytassistMetadata {
    pub slide_serial: Option<String>,
    pub capture_area: Option<String>,
}

impl CytassistMetadata {
    pub fn new(cytassist_image: &Path) -> Result<Self> {
        let mut decoder = Decoder::new(BufReader::new(
            File::open(cytassist_image)
                .with_context(|| format!("Error opening CytAssist image: {cytassist_image:?}"))?,
        ))
        .context(CYTA_ERROR)?;
        Ok(CytassistMetadata {
            slide_serial: decoder.get_tag_ascii_string(SLIDE_ID_TAG).ok(),
            capture_area: decoder.get_tag_ascii_string(CAPTURE_AREA_TAG).ok(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_read_tiff_tags() {
        assert_eq!(
            CytassistMetadata::new(Path::new("test/sd_cytassist_V43J24-025_D1.tiff")).unwrap(),
            CytassistMetadata {
                slide_serial: Some("V43J24-025".to_string()),
                capture_area: Some("D1".to_string()),
            }
        );
        assert_eq!(
            CytassistMetadata::new(Path::new(
                "test/sd_cytassist_bad_slide_id_C05717-046_C1.tiff"
            ))
            .unwrap(),
            CytassistMetadata {
                slide_serial: Some("C05717-046".to_string()),
                capture_area: Some("C1".to_string()),
            }
        );
    }

    #[test]
    fn test_read_tiff_tags_no_tag() {
        assert_eq!(
            CytassistMetadata::new(Path::new("test/sd_cytassist_V43J24-025_D1_no_tag.tiff"))
                .unwrap(),
            CytassistMetadata {
                slide_serial: None,
                capture_area: None,
            }
        );
    }

    #[test]
    fn test_read_tiff_tags_only_slide_tag() {
        assert_eq!(
            CytassistMetadata::new(Path::new("test/sd_cytassist_V53M06-039_old_firmware.tiff"))
                .unwrap(),
            CytassistMetadata {
                slide_serial: Some("V53M06-039".to_string()),
                capture_area: None,
            }
        );
    }
}
