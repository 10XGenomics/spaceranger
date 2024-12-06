#
# Copyright (c) 2021 10X Genomics, Inc. All rights reserved.
#
"""Detect and align fiducials for all slide types.

These functions are
invoked by the ALIGN_FIDUCIALS and ALIGN_FIDUCIALS_TEST_WRAPPER stage code.
"""


from __future__ import annotations

import functools
from collections.abc import Iterable
from typing import TYPE_CHECKING

import cv2
import martian
import numpy as np
from matplotlib import pyplot as plt

import cellranger.metrics_names as metrics_names
from cellranger.spatial.fiducial_alignment import (
    point_cloud_fiducials_registration,
    register_visium_hd_fiducials,
)
from cellranger.spatial.fiducial_detect import (
    KeyPointDict,
    blob_detect_fiducials,
    detect_visium_hd_fiducials,
)
from cellranger.spatial.pipeline_mode import PipelineMode, Product, SlideType

if TYPE_CHECKING:
    from cellranger.spatial.loupe_util import LoupeParser
    from cellranger.spatial.slide_design_o3 import VisiumHdSlideWrapper

PIXELS_PER_UM = 4.625


def fiducial_registration(
    pipeline_mode: PipelineMode,
    spots_data: LoupeParser,
    detect_fid_dict: dict,
    reorientation_mode: str,
    img_shape: tuple[int, int],
) -> tuple[LoupeParser, dict, float, np.ndarray, np.ndarray[tuple[int, int], np.dtype[np.float64]]]:
    """Fiducial registration part of the pipeline.

    Args:
        pipeline_mode (PipelineMode): pipeline mode to determine the algorithm and parameter used
        spots_data (LoupeParser): LoupeParser object that contains all the spots data
        detect_fid_dict (dict): detected fiducial position in a dictionary format same
            as LoupeParser
        reorientation_mode (str): determine what initial registration matrix to use
            in registration
        img_shape (Tuple[int, int]): the shape of the original image. Used to determine
            if the fraction of registered oligos that are outside of the image

    Returns:
        Tuple[LoupeParser, dict, float]: updated spots data, registration metrics and fraction of
            spots outside the frame
    """
    algo_dict = {
        PipelineMode(Product.CYT, SlideType.VISIUM): functools.partial(
            point_cloud_fiducials_registration,
            reorientation_mode="rotation+mirror",
            init_hmat=None,
        ),
        PipelineMode(Product.CYT, SlideType.XL): functools.partial(
            point_cloud_fiducials_registration,
            reorientation_mode="rotation+mirror",
            init_hmat=None,
        ),
        PipelineMode(Product.VISIUM_HD_NOCYT_PD, SlideType.VISIUM_HD): register_visium_hd_fiducials,
        PipelineMode(Product.CYT, SlideType.VISIUM_HD): register_visium_hd_fiducials,
    }

    default_algo = functools.partial(
        point_cloud_fiducials_registration,
        reorientation_mode=reorientation_mode,
        outlier_dist=2e-6,
    )
    algo = algo_dict.get(pipeline_mode, default_algo)
    design_fid_dict = spots_data.get_fiducials_data()
    transform_mat, reg_metrics, overlapping_pts = algo(design_fid_dict, detect_fid_dict)
    spots_data.transform(transform_mat)

    # check whether oligos are rotated out of the frame by too much
    out_frac = outside_fraction(spots_data.get_oligos_imgxy(), img_shape[1] - 1, img_shape[0] - 1)

    msg = reg_metrics.get(metrics_names.SUSPECT_ALIGNMENT, None)
    if msg is not None:
        martian.log_info(msg)

    return spots_data, reg_metrics, out_frac, transform_mat, overlapping_pts


def fiducial_detection(
    pipeline_mode: PipelineMode,
    fid_detect_img: np.ndarray,
    roi_mask: np.ndarray,
    visium_hd_slide: VisiumHdSlideWrapper | None,
) -> tuple[list[dict], dict, list[str]]:
    """Fiducial detection for all product/slide combination.

    Args:
        pipeline_mode (PipelineMode): pipeline mode to determine algorithm and parameters
        fid_detect_img (np.ndarray): original image for fiducial detection
        roi_mask (np.ndarray): 2d array of 0 and 1 to mark the roi
        visium_hd_slide (VisiumHdSlideWrapper): Design of the HD slide

    Returns:
        Tuple[List[Dict], Dict]: The detected fiducial points and the metrics dict
    """
    algo_dict = {
        PipelineMode(Product.CYT, SlideType.VISIUM): functools.partial(
            blob_detect_fiducials, minsize=10, roi_mask=roi_mask
        ),
        PipelineMode(Product.CYT, SlideType.XL): functools.partial(
            blob_detect_fiducials,
            minsize=10,
            roi_mask=roi_mask,
        ),
        PipelineMode(Product.VISIUM_HD_NOCYT_PD, SlideType.VISIUM_HD): functools.partial(
            detect_visium_hd_fiducials,
            slide=visium_hd_slide,
        ),
        PipelineMode(Product.CYT, SlideType.VISIUM_HD): functools.partial(
            detect_visium_hd_fiducials,
            slide=visium_hd_slide,
        ),
    }
    default_algo = functools.partial(
        blob_detect_fiducials,
        minsize=10,
    )

    algo = algo_dict.get(pipeline_mode, default_algo)
    detected_fid_list, qc_img_dict, detection_metrics = algo(fid_detect_img)

    # save qc figures
    basename = "qc_detected_fiducials_images"
    qc_img_paths_dict = {}
    for fig_key, qc_fig in qc_img_dict.items():
        filename = f"{basename}_{fig_key}.jpg"
        cv2.imwrite(filename, qc_fig)
        qc_img_paths_dict[filename] = martian.make_path(filename)
    return detected_fid_list, detection_metrics, qc_img_paths_dict


def write_qc_fig(img: np.ndarray, spots_data: LoupeParser, save_path: str) -> None:
    """Save the registration qc figure.

    Args:
        img (np.ndarray): fiducial detection image
        spots_data (LoupeParser): spots data containing fiducial info.
        save_path (str): path to save the figure.
    """
    qc_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    fid_dia = spots_data.get_fiducials_diameter()
    embiggen = 1.2  # add 20% to spot size to be able to visualize underlying fiducial
    for x, y in spots_data.get_fiducials_imgxy():
        cv2.circle(qc_img, (int(x), int(y)), int(fid_dia / 2.0 * embiggen + 0.5), (0, 0, 255), 2)
    cv2.imwrite(save_path, qc_img)


def write_reg_err_fig(
    img: np.ndarray,
    design_fid_dict: Iterable[KeyPointDict],
    detect_fid_dict: Iterable[KeyPointDict],
    trans_mat,
    save_path: str,
) -> None:
    """Save the registration error qc figure.

    Args:
        img (np.ndarray): fiducial detection image
        design_fid_dict (Iterable[KeyPointDict]): dictionary of design fiducial coordinates
        detect_fid_dict (Iterable[KeyPointDict]): dictionary of detected fiducial coordinates
        trans_mat (_type_): homography matrix from design to detected coordinates
        save_path (str): path to save the figure
    """
    detect = {p["name"]: [p["x"], p["y"]] for p in detect_fid_dict}
    design = {p["name"]: [p["x"], p["y"]] for p in design_fid_dict}
    overlap = sorted(list(set(detect.keys()).intersection(set(design.keys()))))
    detect_xy = np.array([detect[i] for i in overlap])
    transform_xy = np.array(
        [np.matmul(trans_mat, np.array([design[i][0], design[i][1], 1])) for i in overlap]
    )
    transform_xy = np.array([i[:2] / i[2] for i in transform_xy])
    diff = transform_xy - detect_xy

    qc_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    scale_factor = 5
    calibration_bar = PIXELS_PER_UM / scale_factor
    plt.figure(dpi=300)
    plt.imshow(qc_img)
    quiver = plt.quiver(
        detect_xy[:, 0],
        detect_xy[:, 1],
        diff[:, 0],
        diff[:, 1],
        scale=scale_factor,
        linewidth=5,
        color="r",
    )
    plt.quiverkey(
        quiver,
        X=0.15,
        Y=0.05,
        U=calibration_bar,
        label=f"Quiver key, length = {calibration_bar} um",
        labelpos="E",
        labelcolor="r",
    )
    plt.savefig(save_path)


def outside_fraction(pts_xy: np.ndarray, x_up_bound: int, y_up_bound: int) -> float:
    """Check the fraction of the points that are outside of the boundaries.

    Args:
        pts_xy (np.ndarray): (n, 2) shape x, y coordinates of the points
        x_up_bound (int): upper bound of x coordinate
        y_up_bound (int): upper bound of y coordinate

    Returns:
        float: fraction of points that are outside of the boundary
    """
    x_is_out = np.logical_or(pts_xy[:, 0] < 0, pts_xy[:, 0] > x_up_bound)
    y_is_out = np.logical_or(pts_xy[:, 1] < 0, pts_xy[:, 1] > y_up_bound)
    pt_is_out = np.logical_or(x_is_out, y_is_out)
    out_fraction = np.sum(pt_is_out.astype(int)) / len(pt_is_out)
    return out_fraction


# TODO: use slide id to identify XL, gateway, or Visium HD to refine the ROI
def construct_roi(crop_info_dict: dict, shape: tuple[int, int]) -> np.ndarray:
    """Construct the region of interest for detection.

    The region of interest depends on the possible crop and the estimation
    of the position of fiducials.

    Args:
        crop_info_dict (Dict): the dictionary containing the crop information
        shape (Tuple[int, int]): the shape of the mask

    Returns:
        np.ndarray: the roi mask where 1 marks roi pixel
    """
    if len(crop_info_dict) == 0:
        return None
    mask = np.zeros(shape)
    row_min = crop_info_dict["row_min"]
    row_max = crop_info_dict["row_max"]
    col_min = crop_info_dict["col_min"]
    col_max = crop_info_dict["col_max"]
    mask[row_min:row_max, col_min:col_max] = 1
    return mask
