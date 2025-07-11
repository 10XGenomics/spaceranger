//! Write `ReadPair` objects to a set of FASTQ files.
#![allow(missing_docs)]

use crate::read_pair::{ReadPair, WhichRead};
use crate::read_pair_iter::InputFastqs;
use crate::utils;
use anyhow::{Context, Error};
use std::io::Write;
use std::path::{Path, PathBuf};

/// Read sequencing data from a parallel set of FASTQ files.
/// Illumina sequencers typically emit a parallel set of FASTQ files, with one file
/// for each read component taken by the sequencer. Up to 4 reads are possible (R1, R2, I1, and I2).
/// The reader supports any combination of R1/R2/I1/I2 read files,
/// as well as an interleaved R1/R2 file. Supports plain or gzipped FASTQ files, which
/// will be detected based on the filename extension.
pub struct ReadPairWriter {
    writers: [Option<Box<dyn Write>>; 4],
    paths: [Option<PathBuf>; 4],
    // Each input file can interleave up to 2 -- declare those here
    r1_interleaved: bool,
}

impl ReadPairWriter {
    /// Open a `ReadPairIter` given a `InputFastqs` describing a set of FASTQ files
    /// for the available parts of a read.
    pub fn from_fastq_files(input_fastqs: &InputFastqs) -> Result<ReadPairWriter, Error> {
        Self::new(
            Some(&input_fastqs.r1),
            input_fastqs.r2.as_ref(),
            input_fastqs.i1.as_ref(),
            input_fastqs.i2.as_ref(),
            input_fastqs.r1_interleaved,
        )
    }

    /// Open a `ReadPairIter` given of FASTQ files.
    /// For interleaved R1/R2 files, set `r2 = None`, and set
    /// `r1_interleaved = true`.
    pub fn new<P: AsRef<Path>>(
        r1: Option<P>,
        r2: Option<P>,
        i1: Option<P>,
        i2: Option<P>,
        r1_interleaved: bool,
    ) -> Result<ReadPairWriter, Error> {
        let mut writers = [None, None, None, None];
        let mut paths = [None, None, None, None];

        for (idx, r) in IntoIterator::into_iter([r1, r2, i1, i2]).enumerate() {
            if let Some(ref p) = r {
                let wtr = utils::write_with_gz(p.as_ref())?;
                writers[idx] = Some(wtr);
                paths[idx] = Some(p.as_ref().to_path_buf());
            }
        }

        Ok(ReadPairWriter {
            paths,
            writers,
            r1_interleaved,
        })
    }

    pub fn write(&mut self, rec: &ReadPair) -> Result<(), Error> {
        let paths = &self.paths;

        for (idx, writer_opt) in self.writers.iter_mut().enumerate() {
            if let Some(ref mut writer) = *writer_opt {
                let which = WhichRead::read_types()[idx];

                rec.write_fastq(which, writer).with_context(|| {
                    format!("error writing fastq record to file: {:?}", paths[idx])
                })?;

                if which == WhichRead::R1 && self.r1_interleaved {
                    rec.write_fastq(WhichRead::R2, writer).with_context(|| {
                        format!("error writing fastq record to file: {:?}", paths[idx])
                    })?;
                }
            }
        }

        Ok(())
    }
}
