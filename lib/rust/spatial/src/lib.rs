//! spatial
#![deny(missing_docs)]
pub mod common_websummary_components;
pub mod square_bin_name;
pub mod stages;
mod types {
    #![allow(missing_docs)]

    use martian_derive::martian_filetype;
    use serde::{Deserialize, Serialize};

    martian_filetype! {GeoJsonFile, "geojson"}
    martian_filetype! {NpyFile, "npy"}
    martian_filetype! {H5File, "h5"}
    martian_filetype! {TiffFile, "tiff"}
    martian_filetype! {HtmlFile, "html"}
}

pub use types::*;
