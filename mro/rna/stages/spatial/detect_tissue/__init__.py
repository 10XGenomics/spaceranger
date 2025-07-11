#!/usr/bin/env python3
#
# Copyright (c) 2022 10X Genomics, Inc. All rights reserved.
#

"""Tissue detection stage."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2
import martian
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import PIL
import skimage.morphology as morphology

import cellranger.constants as cr_constants
import cellranger.spatial.image_util as image_util
import cellranger.spatial.tissue_detection as tissue_detection
from cellranger.spatial.loupe_util import LoupeParser
from cellranger.spatial.pipeline_mode import PipelineMode

if TYPE_CHECKING:
    from typing import Self

    from cellranger.spatial.slide_design_o3 import VisiumHdSlideWrapper

__MRO__ = """
stage DETECT_TISSUE(
    in  PipelineMode pipeline_mode,
    in  png          tissue_detection_grayscale_image,
    in  png          tissue_detection_saturation_image,
    in  bool         skip_tissue_detection,
    in  bool         ignore_loupe_tissue_detection,
    in  json         registered_spots_data_json,
    out json         registered_selected_spots_json,
    out json         tissue_mask_metrics,
    out jpg          qc_detected_tissue_image,
    out jpg          detected_tissue_mask,
    out jpg          initialisation_debug,
    out png          grabcut_markers,
    out bool         grabcut_failed,
    src py           "stages/spatial/detect_tissue",
) split (
) using (
    volatile = strict,
)
"""

HD_TISSUE_COLOR = [0, 113, 197, 168]  # "#0071C5"
HD_NON_TISSUE_COLOR = [0, 0, 0, 51]


@dataclass
class SpotDataMemo:
    """Memoised info from LoupeParser."""

    fiducials_img_xy: np.ndarray[tuple[int, int], np.dtype[np.float64]] | None
    oligos_img_xy: np.ndarray[tuple[int, int], np.dtype[np.float64]]
    fid_dia: float
    oligo_dia: float
    oligo_preselected: bool
    tissue_oligos_flag: np.ndarray[bool, np.dtype[np.bool_]]
    visium_hd_slide: VisiumHdSlideWrapper | None

    @classmethod
    def from_loupe_parser(cls, spots_data: LoupeParser) -> Self:
        """Load data from loupe parser."""
        fiducials_img_xy = spots_data.get_fiducials_imgxy()
        oligos_img_xy = spots_data.get_oligos_imgxy()
        fid_dia = spots_data.get_fiducials_diameter()
        oligo_dia = spots_data.get_oligos_diameter()
        oligo_preselected = spots_data.oligos_preselected()
        if oligo_preselected:
            tissue_oligos_flag = np.array(spots_data.tissue_oligos_flags())
        else:
            tissue_oligos_flag = np.zeros(len(oligos_img_xy)).astype(bool)
        return SpotDataMemo(
            fiducials_img_xy=fiducials_img_xy,
            oligos_img_xy=oligos_img_xy,
            fid_dia=fid_dia,
            oligo_dia=oligo_dia,
            oligo_preselected=oligo_preselected,
            tissue_oligos_flag=tissue_oligos_flag,
            visium_hd_slide=spots_data.hd_slide if spots_data.has_hd_slide() else None,
        )


def split(args):
    mem_gb = max(
        6.0, 1.0 + LoupeParser.estimate_mem_gb_from_json_file(args.registered_spots_data_json)
    )
    return {
        "chunks": [],
        "join": {
            "__mem_gb": mem_gb,
            "__vmem_gb": max(16.0, mem_gb + 3.0),
        },
    }


def join(args, outs, _chunk_defs, _chunk_outs):  # pylint: disable=too-many-locals
    pipeline_mode = PipelineMode(**args.pipeline_mode)
    cv2.setNumThreads(martian.get_threads_allocation())
    tissue_seg_img = image_util.cv_read_image_standard(args.tissue_detection_grayscale_image)

    spots_data = LoupeParser(args.registered_spots_data_json)

    if args.skip_tissue_detection:
        spots_data.set_all_tissue_oligos(True)
    elif args.ignore_loupe_tissue_detection:
        spots_data.set_all_tissue_oligos(False)
    outs.grabcut_failed = False

    spot_data_memo = SpotDataMemo.from_loupe_parser(spots_data)
    bounding_box = tissue_detection.get_bounding_box(
        spot_data_memo.oligos_img_xy, spot_data_memo.oligo_dia
    )

    # Test if the run is a HD run or an SD run with cytassist requesting new tissue det
    use_cytassist_image_processing = (
        pipeline_mode.is_hd_with_fiducials() or pipeline_mode.is_cytassist()
    )

    if not args.ignore_loupe_tissue_detection and spot_data_memo.oligo_preselected:
        qc_fig = cv2.cvtColor(tissue_seg_img.astype("uint8"), cv2.COLOR_GRAY2RGB)
        alpha_tissue_spots = cr_constants.TISSUE_SPOTS_ALPHA
        outs.detected_tissue_mask = None
    else:
        if use_cytassist_image_processing:
            clahe_tissue_seg_img = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(
                tissue_seg_img
            )
            tissue_sat_img = None
            if args.tissue_detection_saturation_image is not None:
                tissue_sat_img = cv2.imread(
                    args.tissue_detection_saturation_image, cv2.IMREAD_GRAYSCALE
                )

            mask, qc_fig, init_qc, gc_markers, outs.grabcut_failed = tissue_detection.get_mask_v2(
                clahe_tissue_seg_img,
                saturation_channel=tissue_sat_img,
                bounding_box=bounding_box,
                plot=True,
                use_full_fov=True,
            )
        else:
            mask, qc_fig, init_qc, gc_markers, outs.grabcut_failed = tissue_detection.get_mask(
                tissue_seg_img,
                bounding_box=bounding_box,
                plot=True,
            )

        binary_mask = mask > 0
        cv2.imwrite(outs.detected_tissue_mask, np.where(binary_mask, 255, 0))

        cv2.imwrite(outs.grabcut_markers, gc_markers)
        if init_qc is not None:
            init_qc.savefig(outs.initialisation_debug, bbox_inches="tight")
            plt.close(init_qc)
        else:
            outs.initialisation_debug = None

        # Test if the run is an SD run requesting new tissue det
        if not pipeline_mode.is_hd_with_fiducials() and use_cytassist_image_processing:
            mask = morphology.isotropic_dilation(
                mask,
                radius=int(np.ceil(spot_data_memo.oligo_dia / 2)),
            )

        for i, (imgx, imgy) in enumerate(spot_data_memo.oligos_img_xy):
            # using corner-based coordinates, so rounding down is correct
            row = int(imgy)
            col = int(imgx)
            if 0 <= row < mask.shape[0] and 0 <= col < mask.shape[1] and mask[row, col] > 0:
                spot_data_memo.tissue_oligos_flag[i] = True

        spots_data.update_tissue_oligos(spot_data_memo.tissue_oligos_flag)

        if cr_constants.TISSUE_SPOTS_ALPHA < cr_constants.TISSUE_NOSPOTS_ALPHA:
            raise ValueError(
                "TISSUE_SPOTS_ALPHA has to be strictly greater than TISSUE_NOSPOTS_ALPHA"
            )
        alpha_tissue_spots = (
            cr_constants.TISSUE_SPOTS_ALPHA - cr_constants.TISSUE_NOSPOTS_ALPHA
        ) / (1 - cr_constants.TISSUE_NOSPOTS_ALPHA)

    # save out spots data for upstream use
    # may or may not have been modified from what was loaded
    spots_data.save_to_json(outs.registered_selected_spots_json)

    if np.sum(spot_data_memo.tissue_oligos_flag.astype(int)) == 0:
        martian.exit(
            "No tissue is detected on the spots by automatic alignment. Please use manual alignment."
        )

    tissue_detection_metrics = {"grabcut_failed": outs.grabcut_failed}
    if pipeline_mode.is_hd():
        write_qc_figures_hd(
            qc_fig,
            spot_data_memo,
            outs.qc_detected_tissue_image,
        )
        if not spot_data_memo.visium_hd_slide:
            raise ValueError(
                "Pipeline mode is visium HD, but loupe file does not have visium HD slide."
            )
        tissue_detection_metrics[
            "tissue_mask_area_in_um_squared"
        ] = spot_data_memo.tissue_oligos_flag.sum() * (
            spot_data_memo.visium_hd_slide.spot_size() ** 2
        )
    else:
        write_qc_figures(
            qc_fig,
            spot_data_memo,
            outs.qc_detected_tissue_image,
            text_annotation=None,
            spots_alpha=alpha_tissue_spots,
            tissue_color=cr_constants.TISSUE_COLOR,
        )
        tissue_detection_metrics["tissue_mask_area_in_um_squared"] = (
            spot_data_memo.tissue_oligos_flag.sum() * image_util.SD_SPOT_TISSUE_AREA_UM2
        )

    json_path = martian.make_path(outs.tissue_mask_metrics)
    with open(json_path, "w") as json_out:
        json.dump(tissue_detection_metrics, json_out)


def write_qc_figures_hd(
    img: np.ndarray,
    spot_data_memo: SpotDataMemo,
    save_file_path: str,
) -> None:
    """Save qc figures for tissue detection including selected oligos for HD.

    Instead of plotting the spots explicitly in HD, we overlay a mask on top of
    the cytassist image.

    Args:
        img (np.ndarray): original image.
        spot_data_memo (SpotDataMemo): memoised version of the
            LoupeParser object that contains all the final spots
            information.
        save_file_path (str): path to save the qc figure
    """
    height, width = img.shape[:2]

    tissue_mask_layer = np.zeros((height, width, 4), dtype=np.uint8)

    for i, (imgx, imgy) in enumerate(spot_data_memo.oligos_img_xy):
        # using center-based coordinates, so (0.0, 0.0) is pixel center
        row = int(round(imgy))
        col = int(round(imgx))
        if 0 <= row < height and 0 <= col < width:
            if np.all(tissue_mask_layer[row, col, :] == HD_TISSUE_COLOR):
                continue
            tissue_mask_layer[row, col, :] = (
                HD_TISSUE_COLOR if spot_data_memo.tissue_oligos_flag[i] else HD_NON_TISSUE_COLOR
            )

    mixed_img = PIL.Image.alpha_composite(
        PIL.Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)),
        PIL.Image.fromarray(tissue_mask_layer),
    )

    fig, ax = plt.subplots(figsize=(width / 72.0, height / 72.0), dpi=72)
    plt.imshow(mixed_img)

    if spot_data_memo.fiducials_img_xy is not None and spot_data_memo.fiducials_img_xy.size > 0:
        ax.scatter(
            spot_data_memo.fiducials_img_xy[:, 0],
            spot_data_memo.fiducials_img_xy[:, 1],
            c="None",
            s=int(spot_data_memo.fid_dia**2),
            edgecolor="#fe2e2e",
            linewidth=3,
        )

    ax.set_axis_off()
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    fig.savefig(save_file_path, format="jpg", dpi=72, pad_inches=0)
    plt.close(fig)


def write_qc_figures(
    img: np.ndarray,
    spot_data_memo: SpotDataMemo,
    save_file_path: str,
    text_annotation: str | None = None,
    spots_alpha: float = 0.25,
    tissue_color: str = cr_constants.TISSUE_COLOR,
) -> None:
    """Save qc figures for tissue detection including selected oligos.

    Args:
        img (np.ndarray): original image.
        spot_data_memo (SpotDataMemo): memoised version of the
            LoupeParser object that contains all the final spots
            information.
        save_file_path (str): path to save the qc figure
        text_annotation (str, optional): text annotation in the figure. Defaults to None.
        spots_alpha (float, optional): transparency of the oligo spots. Defaults to 0.25.
        tissue_color (str, optional): color of the tissue. Defaults to cr_constants.TISSUE_COLOR.
    """
    height, width = img.shape[:2]
    fig, ax = plt.subplots(figsize=(width / 72.0, height / 72.0), dpi=72)
    ax.imshow(img)
    if spot_data_memo.oligo_dia == 1:
        spot_data_memo.oligo_dia = matplotlib.rcParams["lines.markersize"]
    if spot_data_memo.fid_dia == 1:
        spot_data_memo.fid_dia = matplotlib.rcParams["lines.markersize"]

    spot_linewidth = min(3, spot_data_memo.oligo_dia / 5)
    ax.scatter(
        spot_data_memo.oligos_img_xy[spot_data_memo.tissue_oligos_flag][:, 0],
        spot_data_memo.oligos_img_xy[spot_data_memo.tissue_oligos_flag][:, 1],
        c=tissue_color,
        s=spot_data_memo.oligo_dia**2,
        edgecolor="black",
        alpha=spots_alpha,
        linewidth=spot_linewidth,
    )
    ax.scatter(
        spot_data_memo.oligos_img_xy[np.logical_not(spot_data_memo.tissue_oligos_flag)][:, 0],
        spot_data_memo.oligos_img_xy[np.logical_not(spot_data_memo.tissue_oligos_flag)][:, 1],
        c="None",
        s=spot_data_memo.oligo_dia**2,
        edgecolor="black",
        alpha=0.25,
        linewidth=spot_linewidth,
    )

    if spot_data_memo.fiducials_img_xy is not None and spot_data_memo.fiducials_img_xy.size > 0:
        ax.scatter(
            spot_data_memo.fiducials_img_xy[:, 0],
            spot_data_memo.fiducials_img_xy[:, 1],
            c="None",
            s=int(spot_data_memo.fid_dia**2),
            edgecolor=cr_constants.FIDUCIAL_SPOT_COLOR,
            alpha=0.75,
            linewidth=3,
        )
    ax.set_axis_off()
    if text_annotation:
        ax.text(
            0.98,
            0.02,
            text_annotation,
            horizontalalignment="right",
            verticalalignment="center",
            transform=ax.transAxes,
            fontsize=36,
        )
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    fig.savefig(save_file_path, format="jpg", dpi=72, pad_inches=0)
    plt.close(fig)
