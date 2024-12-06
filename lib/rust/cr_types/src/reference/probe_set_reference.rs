//! Probe set reference CSV

use crate::probe_set::{Probe, ProbeRegion, ProbeType};
use anyhow::{bail, ensure, Context, Result};
use itertools::Itertools;
use martian_derive::martian_filetype;
use metric::TxHashMap;
use serde::{Deserialize, Serialize};
use std::path::Path;
use transcriptome::{Gene, Transcriptome};

martian_filetype! { TargetSetFile, "csv" }

impl TargetSetFile {
    /// Read a probe set reference CSV.
    pub fn read(&self, transcriptome_reference_path: &Path) -> Result<Vec<Probe>> {
        // Read the transcriptome GTF to map gene IDs to gene names.
        let gene_id_to_name: TxHashMap<_, _> =
            Transcriptome::from_reference_path(transcriptome_reference_path)?
                .genes
                .into_iter()
                .map(|x| (x.id, x.name))
                .collect();

        // Read the probe set reference CSV file.
        let mut reader = csv::ReaderBuilder::new()
            .comment(Some(b'#'))
            .from_path(self)
            .with_context(|| self.display().to_string())?;

        // Ensure that the headers are correct.
        let header: Vec<_> = reader.headers().unwrap().iter().collect();
        assert_eq!(header[0], "gene_id");
        assert_eq!(header[1], "probe_seq");
        assert_eq!(header[2], "probe_id");
        if let Some(&included_header) = header.get(3) {
            assert_eq!(included_header, "included");
        }
        if let Some(&region_header) = header.get(4) {
            assert_eq!(region_header, "region");
        }
        if let Some(&ref_name_header) = header.get(5) {
            assert_eq!(ref_name_header, "ref_name");
        }
        if let Some(&ref_pos_header) = header.get(6) {
            assert_eq!(ref_pos_header, "ref_pos");
        }
        if let Some(&cigar_header) = header.get(7) {
            assert_eq!(cigar_header, "cigar");
        }

        reader
            .records()
            .map(|record| {
                let record = record?;
                let gene_id = record[0].to_string();
                let probe_seq_str = &record[1];
                let probe_seq = probe_seq_str.as_bytes();
                let probe_id = record[2].to_string();
                let included = record.get(3).map_or(Ok(true), |x| {
                    x.to_lowercase().parse().with_context(|| {
                        format!(r#"The column "included" must be "true" or "false" but saw "{x}""#)
                    })
                })?;
                let region = record
                    .get(4)
                    .map(str::to_string)
                    .map(|r| ProbeRegion::new(&r));

                let ref_sequence_name = record.get(5).unwrap_or("").to_string();
                let ref_sequence_pos: Option<usize> = record
                    .get(6)
                    .map(|pos| {
                        pos.parse().with_context(|| {
                            format!(r#"The column "ref_pos" must be an integer but saw "{pos}""#)
                        })
                    })
                    .transpose()?;
                let cigar_string = record.get(7).unwrap_or("").to_string();

                let gene_name = gene_id_to_name.get(&gene_id).unwrap_or(&gene_id).clone();

                let num_hyphens = probe_seq.iter().filter(|&&x| x == b'-').count();
                let probe_type = if probe_seq.starts_with(b"-") || probe_seq.ends_with(b"-") {
                    ensure!(
                        num_hyphens == 1,
                        "An unpaired probe must have exactly one hyphen \
                         for probe {probe_id}: {probe_seq_str}"
                    );
                    ProbeType::UnpairedGapAlign
                } else {
                    match num_hyphens {
                        0 | 1 => ProbeType::RTL,
                        2 => ProbeType::PairedGapAlign,
                        3.. => {
                            bail!(
                                "Too many hyphens in sequence of probe {probe_id}: {probe_seq_str}"
                            )
                        }
                    }
                };

                Ok(Probe {
                    probe_id,
                    gene: Gene {
                        id: gene_id,
                        name: gene_name,
                    },
                    included,
                    region,
                    probe_type,
                    ref_sequence_name,
                    ref_sequence_pos,
                    cigar_string,
                })
            })
            .try_collect()
            .with_context(|| self.display().to_string())
    }

    /// Return the gene IDs and their `included` status.
    pub fn read_gene_ids_and_included(
        &self,
        transcriptome_reference_path: &Path,
    ) -> Result<Vec<(String, bool)>> {
        let probes = self.read(transcriptome_reference_path)?;
        let gene_id_to_included: TxHashMap<_, _> = probes
            .iter()
            .map(|probe| (&probe.gene.id, probe.included))
            .into_group_map()
            .into_iter()
            .map(|(gene_id, included)| (gene_id.clone(), included.into_iter().all(|x| x)))
            .collect();
        Ok(probes
            .into_iter()
            .map(|x| x.gene.id)
            .unique()
            .map(|gene_id| {
                let included = gene_id_to_included[&gene_id];
                (gene_id, included)
            })
            .collect())
    }
}
