//! Common components across HD and segment websummaries
use anyhow::Result;
use martian_filetypes::json_file::JsonFile;
use martian_filetypes::FileTypeRead;
use serde_json::Value;
use tenx_websummary::components::{Card, MultiLayerImages, TitleWithHelp, WithTitle};

pub(crate) fn generate_multilayer_chart_from_json(
    json_file: &JsonFile<Value>,
    title: &str,
    help: &str,
) -> Result<Card<WithTitle<MultiLayerImages>>> {
    Ok(Card::full_width(WithTitle {
        title: TitleWithHelp {
            title: title.to_string(),
            help: help.to_string(),
        }
        .into(),
        inner: serde_json::from_str(&json_file.read()?.to_string())?,
    }))
}
