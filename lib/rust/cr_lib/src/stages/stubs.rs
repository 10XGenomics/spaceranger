#![deny(missing_docs)]
use anyhow::Result;
use barcode::Barcode;
use martian_derive::MartianStruct;
use martian_filetypes::json_file::JsonFile;
use metric::TxHashSet;
use serde::Deserialize;

#[derive(Clone, Deserialize, MartianStruct)]
pub struct V1PatternFixParams {
    pub affected_barcodes: JsonFile<Vec<String>>,
    pub correction_factor: f64,
}

impl V1PatternFixParams {
    pub fn barcode_subsampling(&self) -> Result<(TxHashSet<Barcode>, Option<f64>)> {
        Ok((TxHashSet::default(), None))
    }
}
