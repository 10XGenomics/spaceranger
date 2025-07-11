//! Bin name used in visium HD
#![allow(missing_docs)]

use anyhow::{bail, Context, Result};
use martian::AsMartianPrimaryType;
use serde::{Deserialize, Serialize};

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, Hash)]
#[serde(try_from = "&str", into = "String")]
pub struct SquareBinName {
    size_um: u32,
}

const SQUARE_PREFIX: &str = "square_";
const UM_SUFFIX: &str = "um";

impl SquareBinName {
    pub fn new(size_um: u32) -> Result<Self> {
        if size_um == 0 || size_um > 999 {
            bail!("bin size should be between 1 and 999. Got {}", size_um);
        }
        Ok(SquareBinName { size_um })
    }
    pub fn size_um(&self) -> u32 {
        self.size_um
    }
    pub fn tab_title(self) -> String {
        format!("{} um", self.size_um)
    }
}

impl std::fmt::Display for SquareBinName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{SQUARE_PREFIX}{:03}{UM_SUFFIX}", self.size_um)
    }
}

impl From<SquareBinName> for String {
    fn from(value: SquareBinName) -> Self {
        value.to_string()
    }
}

impl TryFrom<&str> for SquareBinName {
    type Error = anyhow::Error;

    fn try_from(value: &str) -> Result<Self> {
        value.parse()
    }
}

impl std::str::FromStr for SquareBinName {
    type Err = anyhow::Error;

    fn from_str(bin_name: &str) -> Result<Self> {
        let size_um = bin_name
            .trim_start_matches(SQUARE_PREFIX)
            .trim_end_matches(UM_SUFFIX)
            .parse::<u32>()
            .with_context(|| format!("Unable to parse {bin_name} as a bin name."))?;
        SquareBinName::new(size_um)
    }
}

impl AsMartianPrimaryType for SquareBinName {
    fn as_martian_primary_type() -> martian::MartianPrimaryType {
        martian::MartianPrimaryType::Str
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_bin_name_display() {
        assert_eq!(
            format!("{}", SquareBinName::new(2).unwrap()),
            "square_002um"
        );
        assert_eq!(
            format!("{}", SquareBinName::new(16).unwrap()),
            "square_016um"
        );
    }

    #[test]
    fn test_bin_name_parse() {
        for size_um in 1..999 {
            let bin_name = SquareBinName::new(size_um).unwrap();
            assert_eq!(
                bin_name.to_string().parse::<SquareBinName>().unwrap(),
                bin_name
            );
        }
    }

    #[test]
    fn test_bin_name_serde() {
        assert_eq!(
            serde_json::to_string(&SquareBinName::new(2).unwrap()).unwrap(),
            serde_json::to_string("square_002um").unwrap()
        );

        println!("{:?}", serde_json::to_string("square_002um").unwrap());

        assert_eq!(
            SquareBinName::new(2).unwrap(),
            serde_json::from_str("\"square_002um\"").unwrap()
        );
    }
}
