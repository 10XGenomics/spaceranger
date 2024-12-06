# Copyright (c) 2023 10x Genomics, Inc. All rights reserved.
"""Generate data needed for the End-to-End Image Alignment Accuracy card."""

import json
from dataclasses import asdict, dataclass

# isort: off
# pylint: disable=wrong-import-position
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# pylint: enable=wrong-import-position
# isort: on


import martian
import numpy as np
import skimage

import cellranger.spatial.image_util as image_util
from cellranger.spatial.hd_feature_slice import HdFeatureSliceIo
from cellranger.spatial.image import base64_encode_image
from cellranger.spatial.transform import (
    css_transform_list,
    normalize_perspective_transform,
    scale_matrix,
    translation_matrix,
)
from cellranger.websummary.zoom import InitialZoomPan

__MRO__ = """
stage BUILD_HD_END_TO_END_ALIGNMENT(
    in  h5    hd_feature_slice_h5,
    in  jpg   websummary_tissue_image,
    in  float websummary_tissue_image_scale,
    out json  end_to_end_alignment_data,
    src py    "stages/spatial/build_hd_end_to_end_alignment",
) using (
    vmem_gb  = 8,
    volatile = strict,
)
"""

TISSUE_IMAGE_TITLE = "Tissue Image"
UMI_IMAGE_TITLE = "8 µm bin UMI Counts"

TISSUE_IMAGE_DISPLAY_WIDTH = 470
BIN_SCALE_8UM = 4
LINEAR_MAX_PERCENTILE = 98
# matplotlib cmap names. unfortunately they are case-sensitive and not consistent
UMI_IMAGE_COLORMAPS = ["jet", "Blues", "binary"]


def heatmap_legend(cmap, title, vmin, vmax, fname):
    """Generate a heatmap legend image for the UMI plot."""
    fig, ax = plt.subplots(figsize=(6, 1))
    fig.subplots_adjust(bottom=0.5)
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    colorbar = matplotlib.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, orientation="horizontal")
    colorbar.ax.set_title(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(fname, dpi=100, pad_inches=0)
    plt.close()
    return base64_encode_image(fname)


@dataclass
class UmiLegendImage:  # pylint: disable=invalid-name
    """Umi image data."""

    # camelCase to match javascript property names
    colormap: str
    legendImage: str


@dataclass
class EndToEndAlignmentData:  # pylint: disable=invalid-name
    """Data needed for the End-to-End Image Alignment card."""

    # camelCase to match javascript property names
    tissueImage: str
    tissueImageTitle: str
    tissueCssTransform: list[float]
    displayHeight: int
    displayWidth: int
    umiLegendImages: list[UmiLegendImage]  # different colormaps
    umiImageTitle: str
    umiCssTransform: list[float]
    tissueMaskImage: str
    grayscaleUmiImage: str  # grayscale images to be converted to colourscales
    initialZoomPan: InitialZoomPan


def main(args, outs):  # pylint: disable=too-many-locals
    websummary_tissue_image = image_util.cv_read_image_standard(args.websummary_tissue_image)
    image_width = websummary_tissue_image.shape[1]
    websummary_scale = TISSUE_IMAGE_DISPLAY_WIDTH / image_width

    feat_slice = HdFeatureSliceIo(args.hd_feature_slice_h5)

    under_tissue_mask = feat_slice.read_filtered_mask(BIN_SCALE_8UM)
    not_under_tissue_mask = np.logical_not(under_tissue_mask)

    tissue_mask_img = (255 * not_under_tissue_mask).astype(np.uint8)
    tissue_mask_img_path = martian.make_path("tissue_mask.png").decode()
    skimage.io.imsave(tissue_mask_img_path, tissue_mask_img)
    tissue_mask_img_encoded = base64_encode_image(tissue_mask_img_path)

    umis = feat_slice.total_umis(BIN_SCALE_8UM)
    umis_under_mask = umis[under_tissue_mask]
    nonzero_umis_under_mask = umis_under_mask[umis_under_mask > 0]
    percentile_max = max(
        int(
            (
                np.percentile(nonzero_umis_under_mask, LINEAR_MAX_PERCENTILE)
                if nonzero_umis_under_mask.size
                else 0
            ),
        ),
        1,
    )
    umis[umis > percentile_max] = percentile_max
    umis = ((umis / percentile_max) * 255).astype(np.uint8)

    umi_grayscale_img_path = martian.make_path("umi_grayscale.png").decode()
    skimage.io.imsave(umi_grayscale_img_path, umis)
    grayscale_img_encoded = base64_encode_image(umi_grayscale_img_path)

    umi_legend_images = []
    for colormap in UMI_IMAGE_COLORMAPS:
        legend_img_encoded = heatmap_legend(
            cmap=colormap,
            title="UMI counts",
            vmin=0,
            vmax=percentile_max,
            fname=martian.make_path(f"umis_{colormap}_legend.png").decode(),
        )
        umi_legend_images.append(
            UmiLegendImage(
                colormap=colormap.capitalize(),
                legendImage=legend_img_encoded,
            )
        )

    spot_colrow_to_tissue_image_colrow = (
        feat_slice.metadata.transform_matrices.get_spot_colrow_to_tissue_image_colrow_transform()
    )
    umi_image_transform = normalize_perspective_transform(
        scale_matrix(websummary_scale * args.websummary_tissue_image_scale)
        @ spot_colrow_to_tissue_image_colrow
        # spot_colrow is center-based and we need a corner-based transform for CSS
        @ translation_matrix(-0.5, -0.5)
        @ scale_matrix(BIN_SCALE_8UM)
    )

    with open(outs.end_to_end_alignment_data, "w") as f:
        json.dump(
            asdict(
                EndToEndAlignmentData(
                    tissueImage=base64_encode_image(args.websummary_tissue_image),
                    tissueImageTitle=TISSUE_IMAGE_TITLE,
                    tissueCssTransform=css_transform_list(scale_matrix(websummary_scale)),
                    displayHeight=TISSUE_IMAGE_DISPLAY_WIDTH,
                    displayWidth=TISSUE_IMAGE_DISPLAY_WIDTH,
                    umiLegendImages=umi_legend_images,
                    umiImageTitle=UMI_IMAGE_TITLE,
                    umiCssTransform=css_transform_list(umi_image_transform),
                    tissueMaskImage=tissue_mask_img_encoded,
                    grayscaleUmiImage=grayscale_img_encoded,
                    initialZoomPan=InitialZoomPan.compute(
                        umis, umi_image_transform, TISSUE_IMAGE_DISPLAY_WIDTH
                    ),
                )
            ),
            f,
        )
