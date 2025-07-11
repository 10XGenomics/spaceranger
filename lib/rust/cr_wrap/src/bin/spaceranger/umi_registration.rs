use crate::DARK_IMAGES_BRIGHTFIELD;
use anyhow::ensure;
use clap::Parser;
use cr_wrap::utils::CliPath;
use serde::Serialize;
use std::str::FromStr;

#[derive(Serialize, Clone, Debug)]
pub struct HdLayoutOffset {
    x_offset: f64,
    y_offset: f64,
}

const OFFSET_FORMAT_ERROR: &str = "Offset must be in the format 'dx=value,dy=value'";

impl FromStr for HdLayoutOffset {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let parts: Vec<&str> = s.split(',').collect();
        if parts.len() != 2 {
            return Err(OFFSET_FORMAT_ERROR.to_string());
        }

        let mut dx = None;
        let mut dy = None;

        for part in parts {
            let kv: Vec<&str> = part.split('=').collect();
            if kv.len() != 2 {
                return Err(OFFSET_FORMAT_ERROR.to_string());
            }
            let key = kv[0];
            let value = kv[1].parse::<f64>().map_err(|e| e.to_string())?;

            match key {
                "dx" => dx = Some(value),
                "dy" => dy = Some(value),
                _ => return Err(OFFSET_FORMAT_ERROR.to_string()),
            }
        }

        match (dx, dy) {
            (Some(dx_val), Some(dy_val)) => Ok(HdLayoutOffset {
                x_offset: dx_val,
                y_offset: dy_val,
            }),
            _ => Err(OFFSET_FORMAT_ERROR.to_string()),
        }
    }
}

#[derive(Serialize)]
pub struct UmiRegistrationInputs {
    disable: bool,
    offset: Option<HdLayoutOffset>,
}

impl Default for UmiRegistrationInputs {
    fn default() -> Self {
        Self {
            disable: true,
            offset: None,
        }
    }
}

#[derive(Parser, Debug, Clone)]
pub struct UmiRegistrationArgs {
    /// Boolean: true (default) | false: True enables the UMI-based registration algorithm
    /// when a Visium HD slide ID is provided as well as a brightfield (H&E) microscope image.
    /// An H&E tissue image must be specified (fluorescent images not supported).
    #[clap(long, value_name = "true|false")]
    pub umi_registration: Option<bool>,

    /// Custom offset for UMI registration.
    ///
    /// Provide custom offsets in microns [dx=X,dy=Y] to improve alignment between UMIs
    /// and the high-resolution microscopy image.
    ///
    /// The expected format is '--umi-to-image-offset dx=<float>,dy=<float>'
    ///
    /// Here, '<float>' should be a floating-point number representing the
    /// offset value.
    ///
    /// For example: '--umi-to-image-offset dx=1.5,dy=-2.0'
    ///
    /// Space Ranger will not run the UMI-based registration method if this is set.
    #[clap(long, value_name = "dx=<float>,dy=<float>")]
    pub umi_to_image_offset: Option<HdLayoutOffset>,
}

impl UmiRegistrationArgs {
    /// Assumptions:
    /// - `--umi-registration` is an optional flag. It is set to true if tissue image paths are provided,
    ///   input images are not IF and unknown slide is not set.
    pub fn validate(
        &self,
        tissue_image_paths: &[CliPath],
        dark_images: u8,
        unknown_slide: bool,
        is_visium_hd: bool,
    ) -> anyhow::Result<UmiRegistrationInputs> {
        if !is_visium_hd {
            ensure!(
                self.umi_registration.is_none() && self.umi_to_image_offset.is_none(),
                "UMI registration is only supported for Visium HD."
            );
        }
        let default_umi_registration = (dark_images == DARK_IMAGES_BRIGHTFIELD)
            && !unknown_slide
            && !tissue_image_paths.is_empty()
            && is_visium_hd;
        if self.umi_to_image_offset.is_some() {
            ensure!(
                self.umi_registration.is_none(),
                "--umi-to-image-offset cannot be set when --umi-registration is specified."
            );
        }

        if let Some(offset) = &self.umi_to_image_offset {
            ensure!(
                offset.x_offset.abs() < 6700.0 && offset.y_offset.abs() < 6700.0,
                "UMI registration offset values must each be less than 6700 microns in absolute value. Received dx: {}, dy: {}",
                offset.x_offset, offset.y_offset
            );
        }

        let enable_umi_registration = self.umi_registration.unwrap_or(default_umi_registration)
            || self.umi_to_image_offset.is_some();
        if enable_umi_registration {
            ensure!(
                !tissue_image_paths.is_empty(),
                "--umi-registration requires tissue image."
            );
            ensure!(
                !unknown_slide,
                "--umi-registration is not supported for unknown slide."
            );
            if dark_images != DARK_IMAGES_BRIGHTFIELD {
                ensure!(
                    self.umi_to_image_offset.is_some(),
                    "Automatic UMI registration is not supported for dark images. Please provide offset using --umi-to-image-offset."
                );
            }
        }

        Ok(UmiRegistrationInputs {
            disable: !enable_umi_registration,
            offset: self.umi_to_image_offset.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_comma_separated_dx_dy() {
        let result = "dx=2.5,dy=-1.0".parse::<HdLayoutOffset>();
        assert!(result.is_ok());
        let offset = result.unwrap();
        assert_eq!(offset.x_offset, 2.5);
        assert_eq!(offset.y_offset, -1.0);
    }

    #[test]
    fn test_parse_invalid_format_dx_dy() {
        assert!("dx=5.0 dy=3.16".parse::<HdLayoutOffset>().is_err());
        assert!("2.5,-1.0".parse::<HdLayoutOffset>().is_err());
        assert!("x=1.0,dy=2.0".parse::<HdLayoutOffset>().is_err());
        assert!("dx=1.0".parse::<HdLayoutOffset>().is_err());
        assert!("dx=abc,dy=1.0".parse::<HdLayoutOffset>().is_err());
    }
}
