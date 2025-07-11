//! barcode_extensions::stubs
#![allow(missing_docs)]

use anyhow::Result;
use barcode::corrector::Posterior;
use barcode::{BarcodeConstruct, BarcodeCorrector, BcSegSeq, GelBeadAndProbeConstruct, Whitelist};
use cr_types::chemistry::BarcodeExtraction;
use metric::SimpleHistogram;
use std::ops::Range;

pub type CorrectionMap = ();
pub struct CorrectionMapBuilder;

impl CorrectionMapBuilder {
    pub fn build(&self, _: &SimpleHistogram<BcSegSeq>) -> Result<CorrectionMap> {
        unimplemented!();
    }

    pub fn new(_: &Whitelist) -> Self {
        unimplemented!();
    }
}

pub fn select_barcode_corrector(
    input: BarcodeConstruct<(Whitelist, SimpleHistogram<BcSegSeq>)>,
    barcode_extraction: Option<&BarcodeExtraction>,
    correction_map: Option<BarcodeConstruct<CorrectionMap>>,
) -> BarcodeConstruct<(BarcodeCorrector, Option<Range<usize>>)> {
    assert!(correction_map.is_none());

    match input {
        BarcodeConstruct::GelBeadOnly(gb) => {
            assert!(barcode_extraction.is_none());
            BarcodeConstruct::GelBeadOnly(basic_corrector(gb))
        }
        BarcodeConstruct::GelBeadAndProbe(GelBeadAndProbeConstruct { gel_bead, probe }) => {
            BarcodeConstruct::GelBeadAndProbe(match barcode_extraction {
                None => GelBeadAndProbeConstruct {
                    gel_bead: basic_corrector(gel_bead),
                    probe: basic_corrector(probe),
                },
                Some(BarcodeExtraction::VariableMultiplexingBarcode { .. }) => {
                    // let ssm = SymSpellMatch::with_whitelist(&probe.0, EditMetric::Levenshtein);
                    GelBeadAndProbeConstruct {
                        gel_bead: basic_corrector(gel_bead),
                        // probe: (BarcodeCorrector::new(probe.0, probe.1, ssm), None),
                        probe: basic_corrector(probe),
                    }
                }
                Some(BarcodeExtraction::JointBc1Bc2 { .. }) => {
                    panic!("joint barcode extraction is invalid for non-segmented barcodes");
                }
            })
        }
        BarcodeConstruct::Segmented(_) => {
            unimplemented!();
        }
    }
}

fn basic_corrector(
    (wl, bc_counts): (Whitelist, SimpleHistogram<BcSegSeq>),
) -> (BarcodeCorrector, Option<Range<usize>>) {
    (
        BarcodeCorrector::new(wl, bc_counts, Posterior::default()),
        None,
    )
}
