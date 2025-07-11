//! Martian stage MergeBinMetrics
#![allow(missing_docs)]

use anyhow::Result;
use cr_types::MetricsFile;
use martian::prelude::*;
use martian_derive::{make_mro, MartianStruct};
use martian_filetypes::FileTypeRead;
use metric::{JsonReporter, Metric};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Clone, Deserialize, MartianStruct)]
pub struct MergeBinMetricsStageInputs {
    pub summaries: Vec<Option<MetricsFile>>,
    pub bin_summaries: Option<HashMap<String, MetricsFile>>,
}

/// The Martian stage outputs.
#[derive(Clone, Serialize, Deserialize, MartianStruct)]
pub struct MergeBinMetricsStageOutputs {
    pub summary: MetricsFile,
}

pub struct MergeBinMetrics;

/// Combine the summary.json files
#[make_mro(volatile = strict)]
impl MartianMain for MergeBinMetrics {
    type StageInputs = MergeBinMetricsStageInputs;
    type StageOutputs = MergeBinMetricsStageOutputs;

    fn main(&self, args: Self::StageInputs, rover: MartianRover) -> Result<Self::StageOutputs> {
        let mut metrics = JsonReporter::default();
        for summary_json in args.summaries.iter().flatten() {
            metrics.merge(summary_json.read()?);
        }
        for (bin_name, bin_summary) in args.bin_summaries.into_iter().flatten() {
            let bin_summary: serde_json::Value = bin_summary.read()?;
            metrics.insert(bin_name, bin_summary);
        }
        Ok(MergeBinMetricsStageOutputs {
            summary: MetricsFile::from_reporter(&rover, "summary", &metrics)?,
        })
    }
}
