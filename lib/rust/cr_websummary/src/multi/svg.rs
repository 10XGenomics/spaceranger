#![allow(missing_docs)]
use crate::{TermDesc, TitleWithTermDesc};
use cr_types::{BarcodeMultiplexingType, CellLevel, ReadLevel};
use serde::Serialize;

#[derive(Serialize, Clone)]
pub struct SvgGraph {
    // Multi graph svg stored as a string
    svg_string: String,
    // id of the sample node to highlight in the svg
    sample_node: String,
    help: TitleWithTermDesc,
}

impl SvgGraph {
    pub fn new(
        svg_string: String,
        sample_node: String,
        multiplexing_method: Option<BarcodeMultiplexingType>,
    ) -> Self {
        let mut data = vec![TermDesc::with_one_desc(
            "Samples",
            "Shows all the samples that are present in \
			this analysis. There will be only one sample in a non-multiplexed analysis.",
        )];
        if let Some(multiplexing_method) = multiplexing_method {
            if multiplexing_method == BarcodeMultiplexingType::ReadLevel(ReadLevel::RTL) {
                data.push(TermDesc::with_one_desc(
                    "Probe Barcode IDs",
                    "The probe barcodes used in \
                    this experiment.",
                ));
            } else if multiplexing_method == BarcodeMultiplexingType::ReadLevel(ReadLevel::OH) {
                data.push(TermDesc::with_one_desc(
                    "OCM Barcode IDs",
                    "The OCM barcode IDs used in this experiment.",
                ));
            } else if multiplexing_method == BarcodeMultiplexingType::CellLevel(CellLevel::Hashtag)
            {
                data.push(TermDesc::with_one_desc(
                    "Hashtags",
                    "The antibody features used for cell hashing in this experiment.",
                ));
            } else {
                data.push(TermDesc::with_one_desc(
                    "CMO Tags",
                    "The cell multiplexing oligos used in \
                    this experiment.",
                ));
            }
        }
        data.push(TermDesc::with_one_desc(
            "GEM wells",
            "A single 10x Chromium Chip channel. One or more sequencing libraries can be derived \
            from a GEM well.",
        ));

        data.push(TermDesc::with_one_desc(
            "Physical Libraries",
            "Unique identifier for each library generated from a GEM well. This can be optionally \
            specified by the user under the libraries section in the input csv to multi. The \
            pipeline will assign an identifier automatically based on the library type if it is \
            not specified in the input file. Abbreviations for auto-generated identifiers: \
            GEX: Gene Expression, CMO: Multiplexing Capture, ABC: Antibody Capture, \
            CGC: CRISPR Guide Capture, CUST: Custom Capture.",
        ));

        data.push(TermDesc::with_one_desc(
            "Fastq IDs",
            "The Illumina sample name in the input fastqs. It is specified under the libraries \
            section in the input CSV.",
        ));

        SvgGraph {
            svg_string,
            sample_node,
            help: TitleWithTermDesc {
                title: "Experimental Design".into(),
                data,
            },
        }
    }
}
