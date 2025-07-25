//! FillClonotypeInfo stage code
#![allow(missing_docs)]

use crate::assigner::ProtoBinFile;
use anyhow::Result;
use cr_types::clonotype::ClonotypeId;
use enclone_proto::proto_io::ClonotypeIter;
use martian::prelude::*;
use martian_derive::{make_mro, MartianStruct};
use martian_filetypes::json_file::JsonFile;
use martian_filetypes::{LazyFileTypeIO, LazyWrite};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use vdj_ann::annotate::{ClonotypeInfo, ContigAnnotation, Region};
use vdj_reference::VdjRegion;

#[derive(Debug, Clone, Serialize, Deserialize, MartianStruct)]
pub struct FillClonotypeInfoStageInputs {
    pub sample_id: Option<String>,
    pub contig_annotations: JsonFile<Vec<ContigAnnotation>>,
    pub enclone_output: Option<ProtoBinFile>,
}

#[derive(Debug, Clone, Serialize, Deserialize, MartianStruct)]
pub struct FillClonotypeInfoStageOutputs {
    pub all_contig_annotations_json: JsonFile<Vec<ContigAnnotation>>,
}

#[derive(Clone)]
struct FieldsToFill {
    raw_clonotype_id: String,
    raw_consensus_id: String,
    exact_subclonotype_id: String,
    fwr1: Option<Region>,
    cdr1: Option<Region>,
    fwr2: Option<Region>,
    cdr2: Option<Region>,
    fwr3: Option<Region>,
    fwr4: Option<Region>,
    v_start: usize,
}

fn update_region(
    region: Option<Region>,
    consensus_start: usize,
    contig_start: usize,
    seq: &str,
) -> Option<Region> {
    region.map(|r| {
        let start = contig_start + (r.start - consensus_start);
        let stop = contig_start + (r.stop - consensus_start);
        assert!(stop <= seq.len(),
            "Out of bound index for seq = {}, region = {:?}, consensus_start = {}, contig_start = {}, seq_len = {}",
            seq,
            r,
            consensus_start,
            contig_start,
            seq.len()
        );
        assert!(
            &seq.as_bytes()[start..stop] == r.nt_seq.as_bytes(),
            "Incorrect index for seq = {seq}, region = {r:?}, consensus_start = {consensus_start}, contig_start = {contig_start}"
        );

        Region {
            start,
            stop,
            nt_seq: r.nt_seq,
            aa_seq: r.aa_seq,
        }
    })
}

pub struct FillClonotypeInfo;

#[make_mro(mem_gb = 2)]
impl MartianMain for FillClonotypeInfo {
    type StageInputs = FillClonotypeInfoStageInputs;
    type StageOutputs = FillClonotypeInfoStageOutputs;
    fn main(&self, args: Self::StageInputs, rover: MartianRover) -> Result<Self::StageOutputs> {
        let all_contig_ann_json_file = rover.make_path("all_contig_annotations.json");

        // Denovo
        if args.enclone_output.is_none() {
            std::fs::copy(&args.contig_annotations, &all_contig_ann_json_file)?;
            return Ok(FillClonotypeInfoStageOutputs {
                all_contig_annotations_json: all_contig_ann_json_file,
            });
        }

        let mut cell_barcodes = HashSet::new();
        // Create a map from contig ids to FieldsToFill.
        let mut contig_fields = HashMap::<String, FieldsToFill>::new();
        for (i, cl) in ClonotypeIter::from_file(args.enclone_output.unwrap())?.enumerate() {
            for (j, ex_cl) in cl.exact_clonotypes.into_iter().enumerate() {
                cell_barcodes.extend(ex_cl.cell_barcodes);
                for chain_info in ex_cl.chains {
                    let index = chain_info.index;
                    let chain = chain_info.chain; // now this is an ExactSubClonotypeChain
                    let clonotype_id = ClonotypeId {
                        id: i + 1,
                        sample_id: args.sample_id.as_deref(),
                    };
                    let fields = FieldsToFill {
                        raw_clonotype_id: clonotype_id.to_string(),
                        raw_consensus_id: clonotype_id.consensus_name((index + 1) as usize),
                        exact_subclonotype_id: format!("{}", j + 1),
                        fwr1: chain.fwr1_region(),
                        cdr1: chain.cdr1_region(),
                        fwr2: chain.fwr2_region(),
                        cdr2: chain.cdr2_region(),
                        fwr3: chain.fwr3_region(),
                        fwr4: chain.fwr4_region(),
                        v_start: chain.v_start as usize,
                    };
                    for contig_id in chain.contig_ids {
                        contig_fields.insert(contig_id, fields.clone());
                    }
                }
            }
        }

        // Define contig_ann reader
        let contig_reader = args.contig_annotations.lazy_reader()?;

        // Write the modified contig annotations json file.
        let mut contig_writer = all_contig_ann_json_file.lazy_writer()?;

        // Fill in the info entries, which have the clonotype information.
        for ann in contig_reader {
            let mut ann: ContigAnnotation = ann?;
            // Only barcodes assigned a clonotype are called as cells
            ann.is_cell = cell_barcodes.contains(&ann.barcode);
            match contig_fields.remove(&ann.contig_name) {
                Some(fields) => {
                    ann.info = ClonotypeInfo {
                        raw_clonotype_id: Some(fields.raw_clonotype_id),
                        raw_consensus_id: Some(fields.raw_consensus_id),
                        exact_subclonotype_id: Some(fields.exact_subclonotype_id),
                    };
                    let contig_v_start = ann.get_region(VdjRegion::V).unwrap().contig_match_start;

                    ann.fwr1 =
                        update_region(fields.fwr1, fields.v_start, contig_v_start, &ann.sequence);
                    ann.cdr1 =
                        update_region(fields.cdr1, fields.v_start, contig_v_start, &ann.sequence);
                    ann.fwr2 =
                        update_region(fields.fwr2, fields.v_start, contig_v_start, &ann.sequence);
                    ann.cdr2 =
                        update_region(fields.cdr2, fields.v_start, contig_v_start, &ann.sequence);
                    ann.fwr3 =
                        update_region(fields.fwr3, fields.v_start, contig_v_start, &ann.sequence);
                    ann.fwr4 =
                        update_region(fields.fwr4, fields.v_start, contig_v_start, &ann.sequence);
                }
                None => {
                    ann.info = ClonotypeInfo::default();
                }
            };
            contig_writer.write_item(&ann)?;
        }

        contig_writer.finish()?;

        Ok(FillClonotypeInfoStageOutputs {
            all_contig_annotations_json: all_contig_ann_json_file,
        })
    }
}
