use serde::Serialize;
use tenx_websummary::components::{
    Card, CardWidth, HdEndToEndAlignment, InlineHelp, Title, TwoColumn, WithTitle,
};
use tenx_websummary::{AddToSharedResource, HtmlTemplate, SharedResources};

#[derive(Clone, Copy)]
pub enum EndToEndLayout {
    SummaryTab,
    ImageAlignmentTab,
}

#[derive(Serialize, HtmlTemplate, Clone)]
pub struct EndToEndAlignmentCard {
    card: Card<WithTitle<EndToEndAlignment>>,
}

impl EndToEndAlignmentCard {
    const CARD_TITLE: &'static str = "Total UMI Count to Image Alignment ";

    pub fn new(
        alignment: HdEndToEndAlignment,
        layout: EndToEndLayout,
        shared_resource: &mut SharedResources,
    ) -> Self {
        let width = match layout {
            EndToEndLayout::SummaryTab => CardWidth::Half,
            EndToEndLayout::ImageAlignmentTab => CardWidth::Full,
        };
        Self {
            card: Card::with_width(
                WithTitle::new(
                    Title::new(Self::CARD_TITLE),
                    EndToEndAlignment::new(alignment.with_shared_resource(shared_resource), layout),
                ),
                width,
            ),
        }
    }
}

#[derive(Serialize, HtmlTemplate, Clone)]
pub struct EndToEndAlignment {
    // In single column layout we have help at top and the image shown below
    help: Option<InlineHelp>,
    alignment: Option<HdEndToEndAlignment>,
    // In a two column layout we have image on the left and the help on the right
    two_col: Option<TwoColumn<HdEndToEndAlignment, InlineHelp>>,
}

impl EndToEndAlignment {
    const SUMMARY_TAB_HELP_TEXT: &'static str = r#"
<p>
    The total UMI count in each 8 µm bin is overlaid onto the tissue image below to assess bin alignment and tissue detection.
    The highest value on the color scale corresponds to the 98th percentile of UMIs per 8 µm bin under tissue, excluding bins with no UMIs.
</p>
<p>
    Check that the overlay matches tissue morphology and covers all of the tissue of interest within the capture area.
</p>
<p>
<a>
    If the overlay is inaccurate, use the manual fiducial alignment and tissue detection workflow in <a href="https://www.10xgenomics.com/support/software/loupe-browser" target='_blank' rel='noopener noreferrer'>Loupe Browser</a>.
</a>
</p>
<p>
    Additional QC images are in the Image Alignment tab for review.
</p>
"#;

    const IMAGE_ALIGNMENT_TAB_HELP_TEXT: &'static str = r#"
<p>
    This image shows the 8 µm binned total UMI count overlaid on the high-resolution microscope image. If a high-resolution image is not supplied, the CytAssist image is used.
</p>
<p>
    To assess accuracy of alignment, click and drag the opacity slider between Microscope Image and 8 µm bin UMI counts.
</p>
<p>
    Check to confirm that the UMI counts match tissue morphology and expected expression patterns. If there is a mismatch, confirm that the correct FASTQ and image files were used when Space Ranger was run and that sample preparation and library generation guidelines were correctly followed.
</p>
"#;
    fn new(alignment: HdEndToEndAlignment, layout: EndToEndLayout) -> Self {
        match layout {
            EndToEndLayout::SummaryTab => Self {
                help: Some(InlineHelp::with_content(
                    Self::SUMMARY_TAB_HELP_TEXT.to_string(),
                )),
                alignment: Some(alignment),
                two_col: None,
            },
            EndToEndLayout::ImageAlignmentTab => Self {
                help: None,
                alignment: None,
                two_col: Some(TwoColumn {
                    left: alignment,
                    right: InlineHelp::with_content(
                        Self::IMAGE_ALIGNMENT_TAB_HELP_TEXT.to_string(),
                    ),
                }),
            },
        }
    }
}
