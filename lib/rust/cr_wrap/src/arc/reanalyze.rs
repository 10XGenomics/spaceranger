//! Args for the reanalyze sub-command

use crate::arc::aggr::AggrDefs;
use crate::arc::types::{validate_distance, ForceCellsArgs, MinCounts, MAX_CLUSTERS_RANGE};
use crate::mrp_args::MrpArgs;
use crate::utils::{validate_id, CliPath};
use anyhow::{bail, Context, Result};
use clap::{self, Parser};
use csv::StringRecord;
use ordered_float::NotNan;
use serde::{self, Deserialize, Serialize};
use std::collections::HashMap;
use std::io::Read;

#[derive(Parser, Debug, Clone)]
pub struct ReanalyzeArgs {
    /// A unique run id and output folder name [a-zA-Z0-9_-]+ of maximum length
    /// 64 characters
    #[clap(long, value_name = "ID", required = true, value_parser = validate_id)]
    pub id: String,

    /// Sample description to embed in output files
    #[clap(long = "description", default_value = "", value_name = "TEXT")]
    sample_desc: String,

    /// Path to folder containing cellranger-arc-compatible reference. Reference
    /// packages can be downloaded from support.10xgenomics.com or constructed using
    /// the `cellranger-arc mkref` command. Note: this reference must match the
    /// reference used for the initial `cellranger-arc count` run.
    #[clap(long, value_name = "PATH", required = true)]
    reference: CliPath,

    /// Path to a feature barcode matrix H5 generated by cellranger-arc `count` or `aggr`. If you
    /// intend to subset to a set of barcodes then use the raw matrix, otherwise use the filtered
    /// feature barcode matrix.
    #[clap(long = "matrix", value_name = "H5", required = true)]
    feature_barcode_matrix: CliPath,

    /// Path to the atac_fragments.tsv.gz generated by cellranger-arc `count` or
    /// `aggr`. Note it is assumed that the tabix index file
    /// atac_fragments.tsv.gz.tbi is present in the same directory.
    #[clap(long, value_name = "TSV.GZ", required = true)]
    atac_fragments: CliPath,

    /// Specify key-value pairs in CSV format for analysis: any subset of `random_seed`,
    /// `k_means_max_clusters`, `feature_linkage_max_dist_mb`, `num_gex_pcs`, `num_atac_pcs`. For
    /// example, to override the number of GEX principal components used to 15 and the distance
    /// threshold for feature linkage computation to 2.5 megabases, the CSV would take the form
    /// (blank lines are ignored):
    ///
    /// num_gex_pcs,15
    ///
    /// feature_linkage_max_dist_mb,2.5
    #[clap(long = "params", value_name = "CSV")]
    parameters: Option<CliPath>,

    /// Specify barcodes to use in analysis. The barcodes could be specified in
    /// a text file that contains one barcode per line, like this (blank lines are ignored):
    ///
    /// ACGT-1
    ///
    /// TGCA-1
    ///
    /// Or you can supply a CSV (with/without a header) whose first column will be used - exports
    /// from Loupe Browser will have this format. For example,
    ///
    /// Barcode,Cluster
    ///
    /// ACGT-1,T cells
    ///
    /// TGCA-1,B cells
    #[clap(long, value_name = "CSV")]
    barcodes: Option<CliPath>,

    #[clap(flatten)]
    force_cells: ForceCellsArgs,

    /// Override peak caller: specify peaks to use in secondary analyses from
    /// supplied 3-column BED file.
    /// The supplied peaks file must be sorted by position and not contain overlapping peaks;
    /// comment lines beginning with `#` are allowed.
    #[clap(long, value_name = "BED")]
    peaks: Option<CliPath>,

    /// If the input matrix was produced by 'cellranger-arc aggr',
    /// it's possible to pass the same aggregation CSV in order to
    /// retain per-library tag information in the resulting
    /// .cloupe file.
    #[clap(long = "agg", value_name = "AGGREGATION_CSV")]
    aggregation_csv: Option<CliPath>,

    /// HIDDEN: Path to custom projection to use for the purpose of calculating feature linkage
    #[clap(long, value_name = "CSV", hide = true)]
    projection: Option<CliPath>,

    /// Do not execute the pipeline.
    /// Generate a pipeline invocation (.mro) file and stop.
    #[clap(long)]
    pub dry: bool,

    #[clap(flatten)]
    pub mrp: MrpArgs,
}

impl ReanalyzeArgs {
    pub fn to_mro_args(&self) -> Result<ReanalyzeMro> {
        // Validate input parameters.
        if self.peaks.is_none()
            && self.barcodes.is_none()
            && self.parameters.is_none()
            && self.force_cells.min_atac_count.is_none()
            && self.force_cells.min_gex_count.is_none()
        {
            bail!(
                "One of these arguments must be specified: --peaks, or --params, or --barcodes, or
                both --min-atac-count and --min-gex-count."
            );
        }

        if let Some(ref csv_path) = &self.aggregation_csv {
            AggrDefs::from_csv(csv_path).map(|_| ())?;
        }

        // Index path and check exists
        let index = CliPath::from(self.atac_fragments.as_ref().with_extension("gz.tbi"));
        if !index.as_ref().is_file() {
            bail!(
                "Expected fragments index file at path {:?}, but it does not exist or is not readable.
                    Make sure the argument to --fragments has extension .tsv.gz and that the index
                    file with extension .tsv.gz is present in the same dir.",
                index.as_ref()
            );
        }
        let params = self
            .parameters
            .as_ref()
            .map(ReanalyzeParams::from_csv)
            .transpose()?;
        Ok(ReanalyzeMro {
            sample_id: self.id.clone(),
            sample_desc: self.sample_desc.clone(),
            reference_path: self.reference.clone(),
            joint_matrix_h5: self.feature_barcode_matrix.clone(),
            peaks: self.peaks.clone(),
            parameters: params,
            cell_barcodes: self.barcodes.clone(),
            force_cells: self
                .force_cells
                .to_mro_arg()
                .context("Invalid value for cell caller override")?,
            projection: self.projection.clone(),
            fragments: self.atac_fragments.clone(),
            fragments_index: index,
            aggregation_csv: self.aggregation_csv.clone(),
        })
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, Eq, PartialEq)]
#[serde(transparent)]
pub struct NumPcs(pub usize);

impl NumPcs {
    pub fn validate(&self) -> Result<()> {
        let k = self.0;
        if !(2..=100).contains(&k) {
            bail!(
                "invalid value = {}: value must satisfy 2 <= value <= 100",
                k
            );
        }
        Ok(())
    }
}

#[derive(Debug, Deserialize, Serialize, Default, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
struct ReanalyzeParams {
    random_seed: Option<u64>,
    k_means_max_clusters: Option<u64>,
    feature_linkage_max_dist_mb: Option<NotNan<f64>>,
    num_gex_pcs: Option<NumPcs>,
    num_atac_pcs: Option<NumPcs>,
}

impl ReanalyzeParams {
    pub fn validate(&self) -> Result<()> {
        if let Some(s) = self.random_seed {
            if s > i64::MAX as u64 {
                bail!("Invalid value for `random_seed`: too large to fit into int64");
            }
        }
        if let Some(x) = self.feature_linkage_max_dist_mb {
            validate_distance(&x.to_string())
                .context("Invalid value for `feature_linkage_max_dist_mb`")?;
        }
        if let Some(x) = self.k_means_max_clusters {
            if !MAX_CLUSTERS_RANGE.contains(&x) {
                bail!("Invalid value for `k_means_max_clusters`: {x}");
            }
        }
        if let Some(x) = &self.num_atac_pcs {
            x.validate().context("Invalid value for `num_atac_pcs`")?;
        }
        if let Some(x) = &self.num_gex_pcs {
            x.validate().context("Invalid value for `num_gex_pcs`")?;
        }
        Ok(())
    }

    fn is_none(&self) -> bool {
        self.random_seed.is_none()
            && self.k_means_max_clusters.is_none()
            && self.num_atac_pcs.is_none()
            && self.num_gex_pcs.is_none()
            && self.feature_linkage_max_dist_mb.is_none()
    }

    fn from_reader<T: Read>(mut reader: csv::Reader<T>) -> Result<ReanalyzeParams> {
        let mut header = StringRecord::new();
        let mut values = StringRecord::new();
        for res in reader.records() {
            let record = res.context("Error processing --params")?;
            if record.len() != 2 {
                bail!(
                    "Row {record:?} of the parameters CSV has > 2 elements. \
                     Each row must be of the form 'parameter,value'"
                );
            }
            header.push_field(&record[0]);
            values.push_field(&record[1]);
        }
        let parameters: ReanalyzeParams = values
            .deserialize(Some(&header))
            .context("Error processing --params")?;
        parameters.validate().context("Error processing --params")?;
        if parameters.is_none() {
            bail!("Supplied parameters file is empty");
        }
        Ok(parameters)
    }

    /// Read algorithm parameters from a CSV
    fn from_csv(path: &CliPath) -> Result<ReanalyzeParams> {
        let reader = csv::ReaderBuilder::new()
            .has_headers(false)
            .from_path(path)?;
        ReanalyzeParams::from_reader(reader)
    }
}

#[derive(Serialize)]
pub struct ReanalyzeMro {
    sample_id: String,
    sample_desc: String,
    reference_path: CliPath,
    joint_matrix_h5: CliPath,
    peaks: Option<CliPath>,
    parameters: Option<ReanalyzeParams>,
    projection: Option<CliPath>,
    cell_barcodes: Option<CliPath>,
    force_cells: Option<HashMap<String, MinCounts>>,
    fragments: CliPath,
    fragments_index: CliPath,
    aggregation_csv: Option<CliPath>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_params() {
        let txt = "num_gex_pcs,10\n\nnum_atac_pcs,20\n".as_bytes();
        let p = ReanalyzeParams::from_reader(
            csv::ReaderBuilder::new()
                .has_headers(false)
                .from_reader(txt),
        )
        .unwrap();
        assert_eq!(
            p,
            ReanalyzeParams {
                num_atac_pcs: Some(NumPcs(20)),
                num_gex_pcs: Some(NumPcs(10)),
                random_seed: None,
                k_means_max_clusters: None,
                feature_linkage_max_dist_mb: None,
            }
        );
    }

    #[test]
    fn test_empty_params() {
        let txt = "".as_bytes();
        if ReanalyzeParams::from_reader(
            csv::ReaderBuilder::new()
                .has_headers(false)
                .from_reader(txt),
        )
        .is_ok()
        {
            panic!("Empty params file must fail")
        }
    }
}