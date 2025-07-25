//!
//! Barcodes which are defined based segmented spatial cells
//!
#![allow(missing_docs)]
use anyhow::ensure;
use pyo3::{pyclass, pymethods};
use serde::{Deserialize, Serialize};
use std::fmt::{Display, Formatter};
use std::str::FromStr;

pub(crate) const CELL_ID_PREFIX: &str = "cellid";

#[derive(Serialize, Deserialize, Clone, Copy, PartialOrd, Ord, PartialEq, Eq, Hash, Debug)]
#[pyclass(eq, str, get_all)]
pub struct CellId {
    pub id: u32,
}

impl Display for CellId {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{CELL_ID_PREFIX}_{:09}", self.id)
    }
}

impl FromStr for CellId {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        fn error(s: &str) -> anyhow::Error {
            anyhow::anyhow!("Unable to parse {s} as a segmented cell barcode.")
        }
        let s = match s.split_once('-') {
            Some((barcode, _gem_group)) => barcode,
            None => s,
        };
        let mut parts = s.split('_');
        let prefix = parts.next().ok_or_else(|| error(s))?;
        ensure!(
            prefix == CELL_ID_PREFIX,
            "Cell-IDs should have the prefix {}. Found cell-ID {} which has prefix {}",
            CELL_ID_PREFIX,
            s,
            prefix
        );

        let id = parts
            .next()
            .map(str::parse)
            .transpose()?
            .ok_or_else(|| error(s))?;

        Ok(CellId { id })
    }
}

#[pymethods]
impl CellId {
    #[new]
    #[pyo3(signature = (barcode=None, id=None))]
    pub fn new(barcode: Option<String>, id: Option<usize>) -> pyanyhow::Result<Self> {
        match (barcode, id) {
            (Some(barcode), None) => Ok(barcode.parse()?),
            (None, Some(id)) => Ok(CellId { id: id as u32 }),
            bad_args => Err(anyhow::anyhow!(
                "Cell-ID either has to \
             be a string with prefix {CELL_ID_PREFIX} or an integer. Found: {:?}",
                bad_args
            )
            .into()),
        }
    }

    pub fn with_default_gem_group(&self) -> String {
        format!("{self}-1")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Result;
    use proptest::proptest;

    #[test]
    fn test_cell_id() -> Result<()> {
        let b = CellId { id: 717 };
        println!("{b}");
        assert_eq!(b.to_string(), "cellid_000000717");
        assert_eq!(b.with_default_gem_group(), "cellid_000000717-1");
        assert_eq!(b, "cellid_000000717".parse()?);
        assert_eq!(b, "cellid_000000717-1".parse()?);
        Ok(())
    }

    proptest! {
        #[test]
        fn test_roundtrip(id in 0..999_999_999u32) {
            let barcode = CellId { id };
            assert_eq!(barcode, barcode.to_string().parse().unwrap());
        }
    }
}
