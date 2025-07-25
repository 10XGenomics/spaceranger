#![allow(missing_docs)]
use crate::align_homopolymer::count_homopolymer_matches;
use crate::stages::align_and_count;
use crate::types::AnnSpillFormat;
use anyhow::Result;
use barcode::{Barcode, BarcodeContent};
use cr_types::chemistry::ChemistryDefsExt;
use cr_types::constants::ILLUMINA_QUAL_OFFSET;
use cr_types::probe_set::ProbeSetReference;
use cr_types::reference::feature_extraction::{FeatureData, FeatureExtractor};
use cr_types::reference::feature_reference::FeatureReference;
use cr_types::rna_read::{RnaChunk, RnaRead};
use cr_types::spill_vec::{SpillVec, SpillVecReader};
use cr_types::types::LibraryType;
use cr_types::FeatureBarcodeType;
use fastq_set::adapter_trimmer::{Adapter, AdapterTrimmer, TrimResult};
use fastq_set::WhichRead;
use martian::MartianFileType;
use metric::TxHasher;
use orbit::StarAligner;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rust_htslib::bam::record::{Aux, Cigar, CigarString, Record};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::ops::Range;
use std::path::PathBuf;
use std::sync::Arc;
use tx_annotation::mark_dups::{BarcodeDupMarker, DupBuilder, UmiCorrection};
use tx_annotation::read::{ReadAnnotations, ReadAnnotator, RecordAnnotation};
use umi::UmiInfo;

const PD_FRAC_FB_READS_TO_ALIGN: f64 = 0.1; // PD only so shouldn't go in parameters.toml
pub const MAX_ANNOTATIONS_IN_MEM: usize = 250_000; // With ~2KB per item, this would be ~500MB

#[derive(Serialize, Deserialize, Clone, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub struct BarcodeSummary {
    library_type: LibraryType,
    barcode: String,
    reads: u64,
    umis: u64,
    candidate_dup_reads: u64,
    umi_corrected_reads: u64,
}

impl BarcodeSummary {
    pub fn new(barcode: Barcode, library_type: LibraryType) -> Self {
        BarcodeSummary {
            library_type,
            barcode: barcode.to_string(),
            reads: 0,
            umis: 0,
            candidate_dup_reads: 0,
            umi_corrected_reads: 0,
        }
    }
    pub fn observe(&mut self, annotation: &ReadAnnotations) {
        self.reads += 1;
        if let Some(dup_info) = &annotation.dup_info {
            if !dup_info.is_low_support_umi {
                self.candidate_dup_reads += 1;
            }
            if dup_info.is_corrected {
                self.umi_corrected_reads += 1;
            }
            if dup_info.is_umi_count() {
                self.umis += 1;
            }
        }
    }
}

/// Adapters.
struct Adapters {
    polya: Adapter,
    tso: Adapter,
}

impl Adapters {
    /// Return new adapters.
    fn new() -> Self {
        use fastq_set::adapter_trimmer::AdapterLoc;
        use fastq_set::WhichEnd;

        /// PolyA sequence.
        const POLYA_SEQ: &str = "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA";

        /// Template-switching oligo (TSO) sequence.
        const TSO_SEQ: &str = "AAGCAGTGGTATCAACGCAGAGTACATGGG";

        Adapters {
            polya: Adapter::new(
                "polyA",
                WhichEnd::ThreePrime,
                AdapterLoc::NonInternal,
                POLYA_SEQ,
            ),
            tso: Adapter::new("tso", WhichEnd::FivePrime, AdapterLoc::Anywhere, TSO_SEQ),
        }
    }
}

/// Adapter trimmers.
struct Trimmers<'a> {
    /// PolyA trimmer.
    polya: AdapterTrimmer<'a>,

    /// TSO trimmer.
    tso: AdapterTrimmer<'a>,

    /// Trimming parameters.
    args: &'a align_and_count::StageInputs,
}

impl<'a> Trimmers<'a> {
    /// Return new adapter trimmers.
    fn new(adapters: &'a Adapters, args: &'a align_and_count::StageInputs) -> Self {
        Trimmers {
            polya: AdapterTrimmer::new(&adapters.polya),
            tso: AdapterTrimmer::new(&adapters.tso),
            args,
        }
    }

    /// Return a TrimResult that does not match.
    fn new_nonmatch(read_length: usize) -> TrimResult {
        TrimResult {
            adapter_range: 0..0,
            trim_range: 0..0,
            retain_range: 0..read_length,
            score: 0,
        }
    }

    /// Align adapter sequence. Return the alignment score if it exceeds the threshold, and None otherwise.
    fn align(&mut self, seq: &[u8]) -> TrimResults {
        let polya = if let Some(trim_polya_min_score) = self.args.trim_polya_min_score {
            match self.polya.find(seq) {
                Some(x) if x.score >= trim_polya_min_score as i32 => x,
                _ => Self::new_nonmatch(seq.len()),
            }
        } else {
            Self::new_nonmatch(seq.len())
        };

        // Align TSO sequence even when trimming is disabled to compute the metric tso_frac.
        let tso_alignment = self.tso.find(seq);
        let tso_score = match tso_alignment {
            Some(ref x) => x.score,
            None => 0,
        };
        let tso = if let Some(trim_tso_min_score) = self.args.trim_tso_min_score {
            match tso_alignment {
                Some(x) if x.score >= trim_tso_min_score as i32 => x,
                _ => Self::new_nonmatch(seq.len()),
            }
        } else {
            Self::new_nonmatch(seq.len())
        };

        TrimResults {
            polya,
            tso,
            tso_score,
        }
    }
}

/// The results of trimming a read.
struct TrimResults {
    /// Result of polyA sequence alignment.
    polya: TrimResult,
    /// Result of TSO sequence alignment.
    tso: TrimResult,
    /// Score of TSO sequence alignment, before thresholding.
    /// Used for computing the metric tso_frac independent of the trimming parameter trim_tso_min_score.
    tso_score: i32,
}

impl TrimResults {
    fn matched_tso(&self) -> bool {
        /// Minimum alignment score to match TSO sequence for the metric tso_frac.
        const MIN_TSO_SCORE: i32 = 20;

        self.tso_score >= MIN_TSO_SCORE
    }

    fn retain_range(&self) -> Range<usize> {
        use core::cmp::{max, min};
        let start = max(self.polya.retain_range.start, self.tso.retain_range.start);
        let end = max(
            start,
            min(self.polya.retain_range.end, self.tso.retain_range.end),
        );
        start..end
    }

    /// Add the TSO (ts:i) and polyA (pa:i) tags to the alignment records.
    fn add_tags(&self, records: &mut Vec<Record>) {
        if self.polya.score == 0 && self.tso.score == 0 {
            return;
        }
        for rec in records {
            if self.polya.score > 0 {
                rec.push_aux(b"pa", Aux::I32(self.polya.trim_range.len() as i32))
                    .unwrap();
            }
            if self.tso.score > 0 {
                rec.push_aux(b"ts", Aux::I32(self.tso.trim_range.len() as i32))
                    .unwrap();
            }
        }
    }
}

#[derive(Clone)]
pub struct Aligner {
    aligner: Option<StarAligner>,
    extractor: Arc<FeatureExtractor>,
    reference: Arc<FeatureReference>,
    annotator: Option<Arc<ReadAnnotator>>,
    target_panel_reference: Option<Arc<ProbeSetReference>>,
    read_chunks: Vec<RnaChunk>,
    filter_umis: bool,
    spill_folder: PathBuf,
}

/// Seeds an RNG using a barcode sequence
///
/// used for deterministic random sampling of feature bc reads for txome alignment for PD metric
fn barcode_seeded_rng(barcode: Barcode) -> SmallRng {
    SmallRng::seed_from_u64(match barcode.content() {
        BarcodeContent::Sequence(seq) => TxHasher::hash(seq),
        BarcodeContent::SpatialIndex(index) => ((index.row as u64) << 32) + index.col as u64,
        BarcodeContent::CellName(cell_id) => cell_id.id as u64,
    })
}

impl Aligner {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        aligner: Option<StarAligner>,
        reference: FeatureReference,
        feature_dist: Vec<f64>,
        annotator: Option<ReadAnnotator>,
        read_chunks: Vec<RnaChunk>,
        spill_folder: PathBuf,
        target_panel_reference: Option<ProbeSetReference>,
    ) -> Result<Aligner> {
        let reference = Arc::new(reference);
        let extractor = FeatureExtractor::new(reference.clone(), Some(feature_dist))?;
        Ok(Aligner {
            aligner,
            extractor: Arc::new(extractor),
            reference,
            annotator: annotator.map(Arc::new),
            read_chunks,
            filter_umis: true,
            spill_folder,
            target_panel_reference: target_panel_reference.map(Arc::new),
        })
    }

    /// Return a mutable referene to the aligner, which must exist.
    fn aligner(&mut self) -> &mut StarAligner {
        self.aligner.as_mut().unwrap()
    }

    /// Aligns, deduplicates and counts UMIs for all the reads in one barcode.
    pub fn process_barcode_se(
        &mut self,
        args: &align_and_count::StageInputs,
        barcode: Barcode,
        reads: impl IntoIterator<Item = Result<RnaRead>>,
        barcode_subsample_rate: f64,
    ) -> Result<AnnotationIter<'_>> {
        let spill_file = AnnSpillFormat::new(&self.spill_folder, barcode.to_string());
        let mut annotations = SpillVec::new(MAX_ANNOTATIONS_IN_MEM, spill_file);
        let mut dup_builder: HashMap<LibraryType, DupBuilder> = HashMap::new();
        let mut pd_rng = barcode_seeded_rng(barcode);

        // First pass through the reads
        for read in reads {
            let read = read?;
            let pd_align_feature_bc_read = args.is_pd
                && self.aligner.is_some()
                && pd_rng.random_bool(PD_FRAC_FB_READS_TO_ALIGN);
            let ann = self.annotate_read(args, read, pd_align_feature_bc_read);
            dup_builder
                .entry(ann.read.library_type)
                .or_default()
                .observe(&ann, &self.reference);
            annotations.push(ann)?;
        }

        let chemistry_name = args.chemistry_defs.primary().name;
        let dup_marker: HashMap<_, _> = dup_builder
            .into_iter()
            .map(|(library_type, builder)| {
                (
                    library_type,
                    builder.build(
                        self.filter_umis,
                        match library_type.is_fb_type(FeatureBarcodeType::Multiplexing)
                            && !(chemistry_name.is_sc_3p_v4() || chemistry_name.is_sc_5p_v3())
                        {
                            true => UmiCorrection::Disable,
                            false => UmiCorrection::Enable,
                        },
                    ),
                )
            })
            .collect();

        Ok(AnnotationIter::new(
            annotations.iter()?,
            dup_marker,
            &self.read_chunks,
            &self.reference,
            self.target_panel_reference.as_ref(),
            barcode_subsample_rate,
            barcode_seeded_rng(barcode),
        ))
    }

    /// Map an RTL read to the target panel CSV reference.
    fn align_probe_read(
        &mut self,
        args: &align_and_count::StageInputs,
        read: RnaRead,
    ) -> ReadAnnotations {
        let name = read.header().split(|c| *c == b' ').next().unwrap();
        let umi_info = UmiInfo::new(read.raw_umi_seq(), read.raw_umi_qual());

        let seq = read.r1_seq();
        let qual = read.r1_qual();
        let panel_ref = self.target_panel_reference.as_ref().unwrap();
        let mapped_probe = panel_ref.align_probe_read(seq);
        let recs = if args.no_bam || args.reference_path.is_none() {
            let mut record = Aligner::new_unmapped_records(&read).0;
            if let (true, Some(lhs), Some(rhs)) = (
                args.is_pd,
                mapped_probe.lhs_probe(),
                mapped_probe.rhs_probe(),
            ) {
                let lhs_index = panel_ref.probe_id_to_index(&lhs.probe_id) as i32;
                let rhs_index = panel_ref.probe_id_to_index(&rhs.probe_id) as i64;
                record.unset_unmapped();
                record.set_tid(lhs_index);
                record.set_pos(rhs_index);
                record.set_mapq(255);
            }
            vec![record]
        } else {
            self.aligner().align_read(name, seq, qual)
        };
        ReadAnnotations::from_mapped_probe(read, recs, mapped_probe, umi_info)
    }

    fn align_read(
        &mut self,
        args: &align_and_count::StageInputs,
        read: RnaRead,
    ) -> ReadAnnotations {
        let adapters = Adapters::new();
        let mut trimmers = Trimmers::new(&adapters, args);

        // Split the qname at the first space.
        let name = read.header().split(|c| *c == b' ').next().unwrap();
        let umi_info = UmiInfo::new(read.raw_umi_seq(), read.raw_umi_qual());

        // Align and trim adapter sequence.
        let trim_results = trimmers.align(read.r1_seq());
        let retain_range = trim_results.retain_range();
        let trimmed_seq = &read.r1_seq()[retain_range.clone()];
        let trimmed_qual = &read.r1_qual()[retain_range.clone()];

        match read.r2_seq() {
            Some(r2_seq) => {
                let r2_qual = read.r2_qual().unwrap();
                // normally we would do trimming here, but
                // for 5' chemistries, the usual trimmer doesn't apply.
                // TODO: let this comment serve as a placeholder for
                //   potential future 5' trimmer.

                let (mut recs1, recs2) = self.aligner().align_read_pair(
                    name,
                    trimmed_seq,
                    trimmed_qual,
                    r2_seq,
                    r2_qual,
                );

                Self::restore_trimmed_sequence(
                    &mut recs1,
                    name,
                    read.r1_seq(),
                    read.r1_qual(),
                    &retain_range,
                );
                trim_results.add_tags(&mut recs1);

                ReadAnnotations {
                    matched_tso: trim_results.matched_tso(),
                    ..self
                        .annotator
                        .as_ref()
                        .unwrap()
                        .annotate_read_pe(read, recs1, recs2, umi_info)
                }
            }
            None => {
                // Get alignment records for this read.
                let mut read_recs = self.aligner().align_read(name, trimmed_seq, trimmed_qual);

                // Restore the trimmed adapter sequence.
                Self::restore_trimmed_sequence(
                    &mut read_recs,
                    name,
                    read.r1_seq(),
                    read.r1_qual(),
                    &retain_range,
                );

                // Add adapter trimming tags.
                trim_results.add_tags(&mut read_recs);

                if args.is_pd {
                    // Add R1 polyT alignment matches BAM tag.
                    let r1_umi_end = read
                        .umi_ranges()
                        .filter(|r| r.read() == WhichRead::R1)
                        .map(|r| r.offset() + r.len().unwrap())
                        .max()
                        .unwrap_or_default();
                    if read.raw_illumina_read1_seq().len() > r1_umi_end {
                        let r1_polyt_matches =
                            count_homopolymer_matches(b'T', read.raw_illumina_read1_seq());
                        for rec in &mut read_recs {
                            rec.push_aux(b"t1", Aux::I32(r1_polyt_matches as i32))
                                .unwrap();
                        }
                    }
                }

                // Annotate the alignments.
                ReadAnnotations {
                    matched_tso: trim_results.matched_tso(),
                    ..self
                        .annotator
                        .as_ref()
                        .unwrap()
                        .annotate_read_se(read, read_recs, umi_info)
                }
            }
        }
    }

    fn annotate_read(
        &mut self,
        args: &align_and_count::StageInputs,
        read: RnaRead,
        pd_align_feature_bc_read: bool,
    ) -> ReadAnnotations {
        match read.library_type {
            LibraryType::Gex => {
                if self.target_panel_reference.is_some() {
                    self.align_probe_read(args, read)
                } else {
                    self.align_read(args, read)
                }
            }
            LibraryType::FeatureBarcodes(_) => {
                let umi_info = UmiInfo::new(read.raw_umi_seq(), read.raw_umi_qual());

                // extract the features for this read
                let primary = self.extract_features(&read);
                //let primary_ref = &primary;

                // if the read was sampled for PD alignment, align read to transcriptome
                // skip alignment if read has a valid feature barcode in feature reference
                let pd_alignment = match (pd_align_feature_bc_read, &primary) {
                    (
                        true,
                        RecordAnnotation::FeatureExtracted(
                            _,
                            FeatureData {
                                corrected_barcode: Some(_),
                                ..
                            },
                            _,
                        ),
                    ) => {
                        let (rec1, rec2) = Aligner::new_unmapped_records(&read);
                        Some(RecordAnnotation::Unmapped(rec1, rec2))
                    }
                    (true, _) => Some(self.align_read(args, read.clone()).primary),
                    _ => None,
                };

                // Annotate the extractions
                ReadAnnotations {
                    read,
                    matched_tso: false,
                    pair_improper: false,
                    primary,
                    umi_info,
                    dup_info: None,
                    pd_alignment,
                }
            }
            LibraryType::Vdj(_) | LibraryType::Atac => unreachable!("{}", read.library_type),
        }
    }

    /// Restore the trimmed sequence and quality, and add soft clippping to CIGAR.
    fn restore_trimmed_sequence(
        recs: &mut Vec<Record>,
        name: &[u8],
        seq: &[u8],
        qual_ascii: &[u8],
        retain_range: &Range<usize>,
    ) {
        if *retain_range == (0..seq.len()) {
            return;
        }
        let trim_left = retain_range.start as u32;
        assert!(seq.len() >= retain_range.end);
        let trim_right = (seq.len() - retain_range.end) as u32;

        // Convert the quality scores from ASCII to numeric.
        let qual: Vec<_> = qual_ascii.iter().map(|x| x - 33).collect();

        // Reverse complement the sequence and quality if any alignments are reversed.
        let (seq_reversed, qual_reversed) = if recs.iter().any(Record::is_reverse) {
            let qual_reversed: Vec<_> = qual.iter().rev().copied().collect();
            (Some(bio::alphabets::dna::revcomp(seq)), Some(qual_reversed))
        } else {
            (None, None)
        };

        let mut cigar: Vec<Cigar>;
        for rec in recs {
            let (clip_left, clip_right) = if rec.is_reverse() {
                (trim_right, trim_left)
            } else {
                (trim_left, trim_right)
            };
            cigar = rec.cigar().iter().copied().collect();
            if !cigar.is_empty() {
                if clip_left > 0 {
                    if let Some(Cigar::SoftClip(clip_left_orig)) = cigar.first() {
                        cigar[0] = Cigar::SoftClip(clip_left + clip_left_orig);
                    } else {
                        cigar.insert(0, Cigar::SoftClip(clip_left));
                    }
                }
                if clip_right > 0 {
                    if let Some(Cigar::SoftClip(clip_right_orig)) = cigar.last() {
                        *cigar.last_mut().unwrap() = Cigar::SoftClip(clip_right + clip_right_orig);
                    } else {
                        cigar.push(Cigar::SoftClip(clip_right));
                    };
                }
            }
            if rec.is_reverse() {
                rec.set(
                    name,
                    Some(&CigarString(cigar)),
                    seq_reversed.as_ref().unwrap(),
                    qual_reversed.as_ref().unwrap(),
                );
            } else {
                rec.set(name, Some(&CigarString(cigar)), seq, &qual);
            }
        }
    }

    fn new_unmapped_records(read: &RnaRead) -> (Record, Option<Record>) {
        fn new_record(name: &[u8], seq: &[u8], qual: &[u8]) -> Record {
            let mut record = Record::new();
            record.set(name, None, seq, qual);
            record.set_tid(-1);
            record.set_pos(-1);
            record.set_unmapped();
            record.set_mtid(-1);
            record.set_mpos(-1);
            record
        }
        // split the qname at the first space
        let name = read.header().split(|c| *c == b' ').next().unwrap();
        let qual: Vec<_> = read
            .r1_qual()
            .iter()
            .map(|&x| x - ILLUMINA_QUAL_OFFSET)
            .collect();
        let mut rec1 = new_record(name, read.r1_seq(), &qual);
        let rec2 = if let Some(r2_seq) = read.r2_seq() {
            let qual: Vec<_> = read
                .r2_qual()
                .unwrap()
                .iter()
                .map(|&x| x - ILLUMINA_QUAL_OFFSET)
                .collect();
            let mut rec2 = new_record(name, r2_seq, &qual);
            rec2.set_paired();
            rec2.set_mate_unmapped();
            rec2.set_last_in_template();
            // set the corresponding properties on R1
            rec1.set_paired();
            rec1.set_mate_unmapped();
            rec1.set_first_in_template();
            Some(rec2)
        } else {
            None
        };
        (rec1, rec2)
    }

    fn extract_features(&self, read: &RnaRead) -> RecordAnnotation {
        let (rec1, rec2) = Aligner::new_unmapped_records(read);
        match self.extractor.match_read(read) {
            Some(data) => RecordAnnotation::FeatureExtracted(rec1, data, rec2),
            None => RecordAnnotation::Unmapped(rec1, rec2),
        }
    }
}

pub struct AnnotationIter<'a> {
    store: SpillVecReader<ReadAnnotations, AnnSpillFormat>,
    dup_marker: HashMap<LibraryType, BarcodeDupMarker>,
    read_chunks: &'a [RnaChunk],
    feature_reference: &'a FeatureReference,
    probe_set_reference: Option<&'a Arc<ProbeSetReference>>,
    barcode_subsample_rate: f64,
    rng: SmallRng,
}

impl<'a> AnnotationIter<'a> {
    fn new(
        store: SpillVecReader<ReadAnnotations, AnnSpillFormat>,
        dup_marker: HashMap<LibraryType, BarcodeDupMarker>,
        read_chunks: &'a [RnaChunk],
        feature_reference: &'a FeatureReference,
        probe_set_reference: Option<&'a Arc<ProbeSetReference>>,
        barcode_subsample_rate: f64,
        rng: SmallRng,
    ) -> Self {
        AnnotationIter {
            store,
            dup_marker,
            read_chunks,
            feature_reference,
            probe_set_reference,
            barcode_subsample_rate,
            rng,
        }
    }
}

impl Iterator for AnnotationIter<'_> {
    type Item = Result<ReadAnnotations>;

    // next() is the only required method
    fn next(&mut self) -> Option<Self::Item> {
        match self.store.next() {
            Some(Ok(mut ann)) => {
                assert!(ann.dup_info.is_none());
                ann.dup_info = if ann.read.barcode_is_valid() {
                    self.dup_marker
                        .get_mut(&ann.read.library_type)
                        .unwrap()
                        .process(
                            &ann,
                            self.feature_reference,
                            self.probe_set_reference,
                            self.read_chunks,
                            self.barcode_subsample_rate,
                            &mut self.rng,
                        )
                } else {
                    None
                };
                ann.attach_tags(self.read_chunks);
                Some(Ok(ann))
            }
            next => next, // Some(Err) or None
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::aligner::{barcode_seeded_rng, Aligner};
    use barcode::{Barcode, BcSeq};
    use metric::TxHasher;
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};
    use rust_htslib::bam::record::{Aux, Cigar, CigarString, Record};

    #[test]
    #[allow(clippy::float_cmp)]
    fn test_barcode_seeded_rng() {
        // seed an RNG using a barcode sequence and get 3 random floats from it
        let mut bc_rng = barcode_seeded_rng(Barcode::with_seq(1, BcSeq::from_bytes(b"ACGT"), true));
        let res1: f64 = bc_rng.random();
        let res2: f64 = bc_rng.random();
        let res3: f64 = bc_rng.random();

        // build the expected RNG and get 3 floats from it
        let mut exp_rng = SmallRng::seed_from_u64(TxHasher::hash(b"ACGT"));
        let exp1: f64 = exp_rng.random();
        let exp2: f64 = exp_rng.random();
        let exp3: f64 = exp_rng.random();

        // check that the generated random floats match
        assert_eq!(res1, exp1);
        assert_eq!(res2, exp2);
        assert_eq!(res3, exp3);
    }

    #[test]
    fn test_restore_trimmed_sequence() {
        let name = b"123";
        let seq = b"AAACCCGGG";
        let qual = b"ABCDEFGHI";

        let mut recs = vec![Record::new()];
        recs[0].unset_flags();
        recs[0].set(
            b"123",
            Some(&CigarString(vec![Cigar::SoftClip(1), Cigar::Match(2)])),
            b"",
            b"",
        );
        recs[0].push_aux(b"AS", Aux::I32(12345)).unwrap();

        Aligner::restore_trimmed_sequence(&mut recs, name, seq, qual, &(2..5));
        assert_eq!(recs[0].qname(), b"123");
        assert_eq!(recs[0].seq().as_bytes(), b"AAACCCGGG");
        assert_eq!(recs[0].qual(), [32, 33, 34, 35, 36, 37, 38, 39, 40]);
        assert_eq!(recs[0].aux(b"AS").unwrap(), Aux::I32(12345));
        assert_eq!(
            recs[0].cigar(),
            CigarString(vec![
                Cigar::SoftClip(3),
                Cigar::Match(2),
                Cigar::SoftClip(4)
            ])
            .into_view(-1)
        );

        recs[0].set(
            b"123",
            Some(&CigarString(vec![Cigar::Match(2), Cigar::SoftClip(1)])),
            b"",
            b"",
        );
        recs[0].set_reverse();
        Aligner::restore_trimmed_sequence(&mut recs, name, seq, qual, &(2..5));
        assert_eq!(recs[0].qname(), b"123");
        assert_eq!(recs[0].seq().as_bytes(), b"CCCGGGTTT");
        assert_eq!(recs[0].qual(), [40, 39, 38, 37, 36, 35, 34, 33, 32]);
        assert_eq!(recs[0].aux(b"AS").unwrap(), Aux::I32(12345));
        assert_eq!(
            recs[0].cigar(),
            CigarString(vec![
                Cigar::SoftClip(4),
                Cigar::Match(2),
                Cigar::SoftClip(3)
            ])
            .into_view(-1)
        );
    }
}
