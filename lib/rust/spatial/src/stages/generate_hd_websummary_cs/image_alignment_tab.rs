#![allow(missing_docs)]
use super::end_to_end::EndToEndAlignmentCard;
use serde::Serialize;
use tenx_websummary::components::{
    BlendedImageZoomable, Card, InlineHelp, RawImage, Title, TwoColumn, WithTitle,
};
use tenx_websummary::HtmlTemplate;

#[derive(Serialize, HtmlTemplate)]
pub struct ImageAlignmentTab {
    pub tissue_fid_card: Option<TissueFidCard>,
    pub registration_card: Option<RegistrationCard>,
    pub end_to_end_alignment: EndToEndAlignmentCard,
}

#[derive(Serialize, HtmlTemplate)]
pub struct TissueFidCard {
    card: Card<WithTitle<TwoColumn<RawImage, InlineHelp>>>,
}

impl TissueFidCard {
    const CARD_TITLE: &'static str = "Tissue Detection and Fiducial Alignment";
    const HELP_TEXT: &'static str = r#"
<p>In this image, the CytAssist image is overlaid with the detected tissue to evaluate bin alignment and tissue detection.</p>

<p>The blue overlay represents the detected tissue and should cover the entire tissue of interest for the analysis.</p>

<p>The bins are accurately aligned when the black fiducials are uniformly surrounded by the red circles. The orientation of the fiducial corners should be as follows:</p>
<ul>
    <li>Square : top left</li>
    <li>Hexagon : top right</li>
    <li>Triangle : bottom left</li>
    <li>Circle : bottom right</li>
</ul>

<p>
    <a>If the tissue and fiducials are not accurately identified, proceed with the manual fiducial alignment and tissue detection workflow in </a>
    <a href="https://www.10xgenomics.com/support/software/loupe-browser" target='_blank' rel='noopener noreferrer'>Loupe Browser</a>.
</p>
"#;

    pub fn new(image: RawImage) -> Self {
        Self {
            card: Card::full_width(WithTitle::new(
                Title::new(Self::CARD_TITLE),
                TwoColumn {
                    left: image,
                    right: InlineHelp::with_content(Self::HELP_TEXT.to_string()),
                },
            )),
        }
    }
}

#[derive(Serialize, HtmlTemplate)]
pub struct RegistrationCard {
    card: Card<WithTitle<TwoColumn<BlendedImageZoomable, InlineHelp>>>,
}

impl RegistrationCard {
    const CARD_TITLE: &'static str = "CytAssist to Microscope Image Alignment";
    const HELP_TEXT: &'static str = r#"
<p>This image shows the registration of the high-resolution microscope image to the CytAssist image.</p>

<p>Click and drag the opacity slider between the microscope image and CytAssist image to confirm proper alignment. Zoom in to confirm that tissue boundaries and morphological features are aligned in the two images.</p>

<p>
    <a>If the results indicate poor alignment, use </a>
    <a href="https://www.10xgenomics.com/support/software/loupe-browser">Loupe Browser </a>
    <a>to perform manual fiducial alignment, tissue detection and high-resolution microscope image registration.</a>
</p>
"#;

    pub fn new(image: BlendedImageZoomable) -> Self {
        Self {
            card: Card::full_width(WithTitle::new(
                Title::new(Self::CARD_TITLE),
                TwoColumn {
                    left: image,
                    right: InlineHelp::with_content(Self::HELP_TEXT.to_string()),
                },
            )),
        }
    }
}
