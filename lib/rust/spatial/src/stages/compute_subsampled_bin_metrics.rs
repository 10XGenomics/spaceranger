//! Martian stage COMPUTE_SUBSAMPLED_BIN_METRICS
#![allow(missing_docs)]

use anyhow::{bail, Context, Result};
use barcode::BarcodeContent;
use cr_bam::constants::{ALN_BC_DISK_CHUNK_SZ, ALN_BC_ITEM_BUFFER_SZ, ALN_BC_SEND_BUFFER_SZ};
use cr_h5::molecule_info::{FullUmiCount, MoleculeInfoIterator, MoleculeInfoReader};
use cr_types::reference::feature_reference::FeatureType;
use cr_types::{CountShardFile, H5File};
use itertools::{izip, Itertools};
use martian::prelude::*;
use martian_derive::{make_mro, MartianStruct};
use martian_filetypes::tabular_file::CsvFile;
use martian_filetypes::FileTypeWrite;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use rand_distr::{Binomial, Distribution};
use serde::{Deserialize, Serialize};
use shardio::{ShardReader, ShardWriter};
use statrs::statistics::{Data, Distribution as _, Median};
use std::collections::{HashMap, HashSet};

/// Struct with Binned BC index and  UMI count.
/// The order below is important as we want to sort by binned bc index and are using
/// the default ord derived on the struct.
#[derive(PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, Debug, Clone)]
struct BinnedBcIndexWithFullUmiCount {
    binned_bc_idx: usize,
    full_umi_count: FullUmiCount,
}

#[derive(Debug, Clone, Deserialize, MartianStruct)]
pub struct ComputeSubsampledBinMetricsStageInputs {
    molecule_info: Option<H5File>,
    bin_scale: u32,
}

#[derive(Serialize, Deserialize, Debug, MartianStruct)]
pub struct SubsampledBinSeqSaturationMetrics {
    pub bin_scale: i32,
    pub bin_size_um: i32,
    pub total_raw_reads_after_subsampling: i64,
    pub number_bins_under_tissue: i32,
    pub subsampling_rate: f64,
    pub mean_reads_per_um_sq: f64,
    pub mean_reads_per_bin: f64,
    pub mean_umis_per_bin: f64,
    pub median_reads_per_bin: f64,
    pub median_umis_per_bin: f64,
    pub mean_genes_per_bin: f64,
    pub median_genes_per_bin: f64,
    pub sequencing_saturation: f64,
}

#[derive(Serialize, Clone, Deserialize, MartianStruct)]
pub struct ComputeSubsampledBinMetricsStageOutputs {
    subsampled_metrics: Option<CsvFile<SubsampledBinSeqSaturationMetrics>>,
}

pub struct ComputeSubsampledBinMetrics;

#[make_mro(mem_gb = 16, volatile = strict, stage_name = COMPUTE_SUBSAMPLED_BIN_METRICS)]
impl MartianMain for ComputeSubsampledBinMetrics {
    type StageInputs = ComputeSubsampledBinMetricsStageInputs;
    type StageOutputs = ComputeSubsampledBinMetricsStageOutputs;

    fn main(
        &self,
        args: Self::StageInputs,
        rover: MartianRover,
    ) -> Result<Self::StageOutputs, Error> {
        if args.molecule_info.is_none() {
            return Ok(ComputeSubsampledBinMetricsStageOutputs {
                subsampled_metrics: None,
            });
        }
        let molecule_info = args.molecule_info.unwrap();

        let total_raw_reads =
            MoleculeInfoReader::get_raw_reads_in_count_gex_libraries(&molecule_info)?;

        let barcode_coords: Vec<_> = MoleculeInfoReader::read_barcodes(&molecule_info)?
            .into_iter()
            .filter_map(|x| match x {
                BarcodeContent::SpatialIndex(sbi) => Some(sbi),
                BarcodeContent::Sequence(_) | BarcodeContent::CellName(_) => None,
            })
            .collect();
        if barcode_coords.is_empty() {
            bail!("Molecule info has no spatial index barcodes!!");
        }
        let area_of_spot_in_um_sq = barcode_coords[0].size_um.pow(2);
        let spot_pitch_um = barcode_coords[0].size_um * args.bin_scale;

        let mut bcs_seen_unbinned = HashSet::<_>::new();
        let mut binned_filtered_bcs = HashSet::<_>::new();

        for barcode_idx in MoleculeInfoIterator::new(&molecule_info)?
            .cell_barcodes_only(true)?
            .filter_features(|fdef| fdef.feature_type == FeatureType::Gene)?
            .map(|f| f.barcode_idx)
        {
            bcs_seen_unbinned.insert(barcode_idx);
            // this same logic is used in BIN_COUNT_MATRIX stage as well
            let binned_barcode = barcode_coords[barcode_idx as usize].binned(args.bin_scale);
            binned_filtered_bcs.insert(binned_barcode);
        }

        let num_unbinned_bcs_seen = bcs_seen_unbinned.len();
        let num_binned_bcs_under_tissue = binned_filtered_bcs.len();

        let binned_filtered_bcs_to_barcode_idx_map: HashMap<_, _> = binned_filtered_bcs
            .into_iter()
            .sorted()
            .enumerate()
            .map(|(idx, bc)| (bc, idx))
            .collect();

        // Writing sorted chunks of the molecule info to disk (sorted by default sort order of BinnedBcIndexWithFullUmiCount).
        // This helps us sorting the molecule info on disk.
        // This is done by sending chunks to shardio which
        // writes out sorted chunks to disk and then allows us to
        // The reason this was tricky is that we wanted to iterate the molecule info in
        // sort order of the binned barcode index to avoid storing sets of genes for each barcode.
        // Sorting the iterator directly basically allocates for the entire iterator blowing up memory.
        let sorted_mol_info: CountShardFile = rover.make_path("sorted_mol_info");
        let mut binned_bc_with_umi_cnts: ShardWriter<BinnedBcIndexWithFullUmiCount> =
            ShardWriter::new(
                &sorted_mol_info,
                ALN_BC_SEND_BUFFER_SZ,
                ALN_BC_DISK_CHUNK_SZ,
                ALN_BC_ITEM_BUFFER_SZ,
            )?;
        let mut sender = binned_bc_with_umi_cnts.get_sender();
        for (binned_bc_idx, full_umi_count) in MoleculeInfoIterator::new(&molecule_info)?
            .filter_features(|fdef| fdef.feature_type == FeatureType::Gene)?
            .filter_map(|f| {
                let binned_bc = barcode_coords[f.barcode_idx as usize].binned(args.bin_scale);
                binned_filtered_bcs_to_barcode_idx_map
                    .contains_key(&binned_bc)
                    .then(|| (binned_filtered_bcs_to_barcode_idx_map[&binned_bc], f))
            })
        {
            sender.send(BinnedBcIndexWithFullUmiCount {
                binned_bc_idx,
                full_umi_count,
            })?;
        }
        sender.finished()?;
        binned_bc_with_umi_cnts.finish()?;

        let achieved_reads_per_um_sq = (total_raw_reads as f64)
            / (num_unbinned_bcs_seen as f64 * area_of_spot_in_um_sq as f64);

        let mut fixed_subsampling_rate =
            if achieved_reads_per_um_sq.is_nan() || achieved_reads_per_um_sq <= 0.0 {
                vec![]
            } else {
                vec![
                    1.2 / achieved_reads_per_um_sq,
                    2.9 / achieved_reads_per_um_sq,
                    5.8 / achieved_reads_per_um_sq,
                    6.1261 / achieved_reads_per_um_sq,
                    8.7 / achieved_reads_per_um_sq,
                    12.3 / achieved_reads_per_um_sq,
                ]
            };

        let mut subsampling_grid: Vec<_> = (0..11).map(|x| x as f64 / 10.).collect();
        subsampling_grid.append(&mut fixed_subsampling_rate);
        subsampling_grid.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let subsampling_grid: Vec<_> = subsampling_grid
            .into_iter()
            .filter(|p| *p >= 0.0 && *p <= 1.0)
            .collect();

        let mut reads_per_bin = vec![vec![0; num_binned_bcs_under_tissue]; subsampling_grid.len()];
        let mut umis_per_bin = vec![vec![0; num_binned_bcs_under_tissue]; subsampling_grid.len()];
        let mut genes_per_bin = vec![vec![0; num_binned_bcs_under_tissue]; subsampling_grid.len()];
        let mut rng = SmallRng::seed_from_u64(0);

        // Reading in sorted chunks from disk
        let reader: ShardReader<BinnedBcIndexWithFullUmiCount> =
            ShardReader::open(&sorted_mol_info)?;
        // grouping by index of the binned barcode
        for (key_option, group) in &reader
            .iter()?
            .chunk_by(|count| count.as_ref().ok().map(|x| x.binned_bc_idx))
        {
            let umis_in_bin: Vec<_> = group.map_ok(|x| x).collect::<Result<Vec<_>>>()?;
            let key = key_option.with_context(|| {
                format!("key_option: {:?},\n group : {:?}", key_option, &umis_in_bin)
            })?;
            for (index, subsampling_rate) in subsampling_grid.iter().enumerate() {
                let mut set_of_genes_in_bin = HashSet::new();
                for (num_reads_for_umi, library_idx, gene_idx) in umis_in_bin.iter().map(|x| {
                    (
                        x.full_umi_count.umi_data.read_count,
                        x.full_umi_count.umi_data.library_idx,
                        x.full_umi_count.umi_data.feature_idx,
                    )
                }) {
                    if num_reads_for_umi >= 1 {
                        let binomial_rv =
                            Binomial::new(num_reads_for_umi as u64, *subsampling_rate)
                                .with_context(|| {
                                    format!("Failed on p={subsampling_rate}, n={num_reads_for_umi}")
                                })?;
                        let subsampled_reads = binomial_rv.sample(&mut rng);
                        if subsampled_reads >= 1 {
                            reads_per_bin[index][key] += subsampled_reads;
                            umis_per_bin[index][key] += 1;
                            set_of_genes_in_bin.insert((library_idx, gene_idx));
                        }
                    }
                }
                genes_per_bin[index][key] = set_of_genes_in_bin.len();
            }
        }

        let vec_of_results: Vec<_> = izip!(
            subsampling_grid,
            umis_per_bin.into_iter(),
            reads_per_bin.into_iter(),
            genes_per_bin.into_iter(),
        )
        .map(
            |(
                subsampling_rate,
                umis_per_bin_subsampled,
                reads_per_bin_subsampled,
                genes_per_bin_subsampled,
            )| {
                let total_reads: i64 = reads_per_bin_subsampled.iter().map(|x| *x as i64).sum();
                let sequencing_saturation = if total_reads > 0 {
                    let total_umis: i64 = umis_per_bin_subsampled.iter().sum();
                    1.0 - total_umis as f64 / total_reads as f64
                } else {
                    0.0
                };
                let subsampled_umis_per_bin = Data::new(
                    umis_per_bin_subsampled
                        .into_iter()
                        .map(|x| x as f64)
                        .collect_vec(),
                );
                let subsampled_reads_per_bin = Data::new(
                    reads_per_bin_subsampled
                        .into_iter()
                        .map(|x| x as f64)
                        .collect_vec(),
                );
                let subsampled_num_genes_per_bin = Data::new(
                    genes_per_bin_subsampled
                        .into_iter()
                        .map(|x| x as f64)
                        .collect_vec(),
                );
                SubsampledBinSeqSaturationMetrics {
                    bin_scale: args.bin_scale as i32,
                    bin_size_um: spot_pitch_um as i32,
                    total_raw_reads_after_subsampling: (total_raw_reads as f64 * subsampling_rate)
                        as i64,
                    number_bins_under_tissue: num_binned_bcs_under_tissue as i32,
                    subsampling_rate,
                    sequencing_saturation,
                    mean_reads_per_um_sq: subsampling_rate * achieved_reads_per_um_sq,
                    // The unwraps here are okay as the vector will never be empty.
                    mean_reads_per_bin: subsampled_reads_per_bin.mean().unwrap(),
                    median_reads_per_bin: subsampled_reads_per_bin.median(),
                    mean_umis_per_bin: subsampled_umis_per_bin.mean().unwrap(),
                    median_umis_per_bin: subsampled_umis_per_bin.median(),
                    mean_genes_per_bin: subsampled_num_genes_per_bin.mean().unwrap(),
                    median_genes_per_bin: subsampled_num_genes_per_bin.median(),
                }
            },
        )
        .collect();

        let subsampled_metrics = rover
            .make_path::<CsvFile<_>>("subsampled_metrics")
            .with_content(&vec_of_results)?;

        Ok(ComputeSubsampledBinMetricsStageOutputs {
            subsampled_metrics: Some(subsampled_metrics),
        })
    }
}
