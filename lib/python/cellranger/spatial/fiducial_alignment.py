#
# Copyright (c) 2021 10X Genomics, Inc. All rights reserved.
#
"""Fiducial registration and utility functions."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, ClassVar, NamedTuple, TypedDict

import cv2
import numpy as np

import cellranger.metrics_names as metrics_names
from cellranger.spatial.hd_fiducial import fit_2d_similarity_transform
from cellranger.spatial.image_util import CYT_IMG_PIXEL_SIZE
from cellranger.spatial.pycpd.rigid_registration import rigid_registration
from cellranger.spatial.transform import transform_pts_2d

if TYPE_CHECKING:
    from cellranger.spatial.fiducial_detect import KeyPointDict


class RegMetrics(NamedTuple):
    """Fiducial registration metrics."""

    mse: float
    sigma_sq: float
    one2one: float
    scale: float
    rad: float
    logmaxp: float

    def extreme_transform_test(self):
        """Test whether the transform is too extreme to be correct."""
        if np.abs(self.rad) > np.pi / 4:
            return "The alignment implies an extreme rotation that is unlikely to be correct"
        elif self.scale < 0.25:
            return "The alignment implies an extreme scaling that is unlikely to be correct"
        return None


class RegResults(NamedTuple):
    """Named tuple for registration results."""

    rotation: np.ndarray
    scale: float
    translation: np.ndarray
    metric: RegMetrics


class FiducialDict(TypedDict):
    x: float
    y: float
    name: str


def construct_homogeneous_mat(extra_param):
    """Construct the 3 by 3 homogeneous transform matrix from legacy extra_param.

    Returns:
        np.ndarray: the 3x3 homogeneous transform matrix.
    """
    design_offset, best_init_transf, translation, rot, detect_offset = extra_param
    transform = rot.T @ best_init_transf
    offset = detect_offset + translation @ rot - design_offset @ transform.T
    return np.array(
        [
            [transform[0, 0], transform[0, 1], offset[0]],
            [transform[1, 0], transform[1, 1], offset[1]],
            [0.0, 0.0, 1.0],
        ]
    )


def _initialize_scale_from_data(dest_points, src_points):
    """Calculate the intial scale from the bounding box of points in each point set."""
    dest_x, dest_y = np.max(dest_points, axis=0) - np.min(dest_points, axis=0)
    src_x, src_y = np.max(src_points, axis=0) - np.min(src_points, axis=0)

    scale = min(float(dest_x) / src_x, float(dest_y) / src_y)  # min so one fits within the other

    return scale


def run_registration(
    detect_points: np.ndarray,
    design_points: np.ndarray,
    outlier_dist: float,
    transform_method: str = "rigid",
    fast: bool = False,
) -> RegResults:
    """Point cloud registration.

    The detect_points should be the "noisy" dataset, and design_points should be
    the template. The returned transform is to transform the design_points to
    the detect_points.

    Args:
        detect_points (np.ndarray): the detected points
        design_points (np.ndarray): the design points
        outlier_dist (float): parameter to determine outliers in GP
        transform_method (str, optional): transform method. Defaults to "rigid".
        fast (bool, optional): whether to use smaller maximum iteration. Defaults to False.

    Returns:
        RegResults: the final registration result
    """
    if fast:
        max_iterations = 80
        tolerance = 0.0001
    else:
        max_iterations = 300
        tolerance = 0.00000001

    if transform_method == "rigid":
        s_init = _initialize_scale_from_data(design_points, detect_points)
        reg = rigid_registration(
            X=design_points,
            Y=detect_points,
            max_iterations=max_iterations,
            tolerance=tolerance,
            s=s_init,
            w=outlier_dist,
        )
        _, (scale, rotation, translation), mse, correspondence_mat = reg.register()
        # the one2one metric measures quality of alignment after registration.
        one2one = np.amax(correspondence_mat, axis=1)
        # only consider the points when there is high enough correspondence
        one2one = np.sum(one2one[one2one > 0.5])
        # point_err = reg.get_point_err()
    else:
        raise ValueError(f"Unsupported transformation method in registration: {transform_method}")
    # TODO: the following parameters are NOT correct for homogeneous matrix construction
    # and are only to recover the original behavior. Convert back in a future PR
    inv_rotation = np.linalg.inv(rotation)
    inv_scale = 1.0 / scale
    # translation = -scale * np.matmul(rotation, translation)
    metric = RegMetrics(
        mse=mse,
        sigma_sq=reg.sigma2,
        one2one=one2one,
        scale=scale,
        rad=np.arccos(rotation[0][0]),
        logmaxp=np.sum(np.log(np.amax(correspondence_mat, axis=1) + 0.1)),
    )
    result = RegResults(
        rotation=inv_rotation, scale=inv_scale, translation=translation, metric=metric
    )
    return result


def compute_spot_sizes_from_gpr_data(gpr_data, trans):
    """Compute pixel diameter of fiducials.

    Given GPR data and a transform from GPR image space (microns) to user tissue image space, compute the pixel
    diameter of fiducials and spots in user tissue image space.
    transform should be a pre-multiply transform taking us from GPR space to user image space
    """
    fiducial_mean_gpr = np.mean([float(s["dia"]) for s in gpr_data["fiducial"]])
    spot_mean_gpr = np.mean([float(s["dia"]) for s in gpr_data["oligo"]])

    scale = np.mean(
        np.linalg.norm(trans, axis=1)
    )  # in theory, the norm of both rows should be the same

    return scale * fiducial_mean_gpr, scale * spot_mean_gpr


def point_cloud_fiducials_registration(  # pylint: disable=too-many-locals
    design_fid: Iterable[KeyPointDict],
    detected_fid: Iterable[KeyPointDict],
    outlier_dist: float = 2e-6,
    reorientation_mode: str | None = None,
    init_hmat: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any], np.ndarray[tuple[int, int], np.dtype[np.float64]]]:
    """Fiducial registration using point cloud registration.

    Args:
        design_fid (List[Dict]): list of information of each designed fiducial point
        detected_fid (List[Dict]): list of information of each detected fiducial point
        outlier_dist (float, optional): parameter to determine outliers in GP. Defaults to 1e-6.
        reorientation_mode (str, optional): different initial rotations to test. Defaults to None.
            options: "rotation", "rotation+mirror"
        init_hmat (np.ndarray): (3, 3) shape. 3x3 homogeneous transform matrix for initialization.
            Default to None

    Returns:
        The returned np.ndarray is the 3x3 homogeneous matrix that maps designed fiducials to
        the detected fiducials.

        The returned Dict is the registration metric

    Both the design_fid and detected_fid are list with structure like:
        [
            {"x": , "y": , ...},
            {"x": , "y": , ...},
            ...
        ]
    each dictionary must contain keys of "x" and "y". They can also contain other keys.
    """
    if init_hmat is not None:
        raise NotImplementedError("Passing initial transform matrix is not implemented")

    design_xy: np.ndarray[tuple[int, int], np.dtype[np.float64]] = np.array(
        [[pt["x"], pt["y"]] for pt in design_fid]
    ).astype(np.float64)
    detected_xy: np.ndarray[tuple[int, int], np.dtype[np.float64]] = np.array(
        [[pt["x"], pt["y"]] for pt in detected_fid]
    ).astype(np.float64)

    if len(detected_xy) < 10:
        raise RuntimeError(
            "[error] Fiducial alignment seems to have failed "
            "- too few keypoints detected. Please doublecheck the image command-line "
            "arguments entered and use the manual alignment tool in Loupe Browser if necessary. "
            "If you would like further assistance, please write to support@10xgenomics.com."
        )
    design_offset = np.median(design_xy, 0)
    design_centered = design_xy - design_offset
    detect_offset = np.median(detected_xy, 0)
    detect_centered = detected_xy - detect_offset

    # r0, r90, r180, r270, m-r0, m-r90, m-r180, m-r270
    transform_lst = [
        np.array([[1, 0], [0, 1]]),
        np.array([[0, -1], [1, 0]]),
        np.array([[-1, 0], [0, -1]]),
        np.array([[0, 1], [-1, 0]]),
        np.array([[1, 0], [0, -1]]),
        np.matmul(np.array([[0, -1], [1, 0]]), np.array([[1, 0], [0, -1]])),
        np.matmul(np.array([[-1, 0], [0, -1]]), np.array([[1, 0], [0, -1]])),
        np.matmul(np.array([[0, 1], [-1, 0]]), np.array([[1, 0], [0, -1]])),
    ]

    # transforms are ordered such that each reorientation_mode picks the first "n"
    if reorientation_mode is not None:
        if reorientation_mode == "rotation":
            limit_transforms = 4
        elif reorientation_mode == "rotation+mirror":
            limit_transforms = 8
        else:
            raise ValueError(f"Unsupported reorientation mode: {reorientation_mode}")
    else:
        limit_transforms = 1

    register_results: list[RegResults] = []
    for init_transf in transform_lst:
        design_init = np.matmul(design_centered, init_transf.T)
        result = run_registration(
            detect_centered,
            design_init,
            outlier_dist,
            transform_method="rigid",
            fast=True,
        )
        register_results.append(result)

    best_init_idx, metrics = parse_reg_results(register_results, limit_transforms)
    best_reg_result = register_results[best_init_idx]
    extra_param = (
        design_offset,
        transform_lst[best_init_idx],
        -best_reg_result.translation,
        best_reg_result.scale * best_reg_result.rotation,
        detect_offset,
    )

    h_mat = construct_homogeneous_mat(extra_param)
    return (h_mat, metrics, detected_xy)


class Visualize:  # pylint: disable=unused-argument, disable=too-few-public-methods
    """Helper method for generating HTML5 video of fiducial alignment."""

    def __init__(self):
        raise RuntimeError("Visualize() is just a container for a classmethod")

    allY: ClassVar[list[np.ndarray]] = []
    allX: ClassVar[list[np.ndarray]] = []

    @classmethod
    def plot_arrays(cls, iteration: int, error: float, X: np.ndarray, Y: np.ndarray):
        """X is the slide, Y is the fiducials."""
        if iteration == 0:
            cls.allY = []
            cls.allX = []
            return

        cls.allY.append(Y.T)
        cls.allX.append(X.T)


def parse_reg_results(
    register_results: list[RegResults],
    limit_transforms: int,
):
    """Return best results and metrics."""
    logmaxp_lst = [result.metric.logmaxp for result in register_results]
    if np.std(logmaxp_lst) == 0:
        scaled_logmaxp = logmaxp_lst - np.mean(logmaxp_lst)
    else:
        scaled_logmaxp = (logmaxp_lst - np.mean(logmaxp_lst)) / np.std(logmaxp_lst)
    overall_best_transf_idx = np.argmax(scaled_logmaxp)
    best_init_idx = np.argmax(scaled_logmaxp[:limit_transforms])
    found_best_ind = overall_best_transf_idx == best_init_idx

    # TODO: consider adjusting some of the metrics
    reg_metrics = register_results[best_init_idx].metric
    metrics = {}
    metrics["alignment_method"] = "Automatic Alignment"
    metrics["residual_spot_error"] = reg_metrics.mse if not np.isnan(reg_metrics.mse) else -1
    metrics["reg_sigma2"] = reg_metrics.sigma_sq
    metrics["reg_scale"] = reg_metrics.scale
    metrics["reg_rotation"] = reg_metrics.rad
    msg = reg_metrics.extreme_transform_test()
    metrics[metrics_names.SUSPECT_ALIGNMENT] = msg is not None
    metrics["logmaxp"] = logmaxp_lst
    metrics["best_transf_ind"] = best_init_idx
    metrics[metrics_names.REORIENTATION_NEEDED] = not found_best_ind
    if limit_transforms > 1:
        mask = np.ones(len(register_results), dtype=bool)
        mask[best_init_idx] = False
        mask[limit_transforms:] = False
        conf_score = scaled_logmaxp[best_init_idx] - np.max(scaled_logmaxp[mask])
    else:
        conf_score = 0
    metrics["best_transf_conf"] = conf_score
    return best_init_idx, metrics


def registration_error(detect_pts: np.ndarray, design_pts: np.ndarray, trans_mat: np.ndarray):
    """Returns registration error in pixels for each point."""
    error = transform_pts_2d(design_pts, trans_mat) - detect_pts  # type: ignore
    return np.linalg.norm(error, axis=1)


def ransac(
    detect: dict[str, tuple[float, float]],
    design: dict[str, tuple[float, float]],
    num_iter=10,
    num_pts=5,
    threshold=5,
):
    """RANSAC algorithm for detection incorrectly decoded fiducials.

    Randomly samples num_pts detected fiducials and uses them to generate a transformation matrix.
    Evaluates the output based on the number of design fiducials with significant registration error
    given that matrix. Repeats this num_iter times and picks the transformation matrix with the lowest
    number of high-error fiducials.

    Args:
        detect (dict): Dictionary of detected fiducials
        design (dict): Dictionary of design fiducials
        num_iter (int, optional): Number of random samples to try. Defaults to 10.
        num_pts (int, optional): Number of fiducials to use per sample. Defaults to 5.
        threshold (int, optional): Threshold (in pixels) of registration error to classify as inlier. Defaults to 5.

    Returns:
        np.ndarray: 3x3 homography matrix with the best fit among samples
    """
    np.random.seed(0)
    if num_pts < 4:
        raise ValueError("Cannot perform homography transform with less than 4 points")
    overlap = sorted(list(set(detect.keys()).intersection(set(design.keys()))))
    if len(overlap) < num_pts:
        raise ValueError(f"Not enough detected fiducials ({len(overlap)}) for {num_pts} samples")
    errors, matrices = [], []
    for _ in range(num_iter):
        sample = np.random.choice(overlap, num_pts, replace=False)
        design_train_pts = np.array([design[key] for key in sample])
        detect_train_pts = np.array([detect[key] for key in sample])
        design_test_pts = np.array([design[key] for key in overlap])
        detect_test_pts = np.array([detect[key] for key in overlap])
        trans_mat, _ = cv2.findHomography(
            design_train_pts,
            detect_train_pts,
        )
        if trans_mat is None:
            continue
        error_count = (
            registration_error(detect_test_pts, design_test_pts, trans_mat) > threshold
        ).sum()
        errors.append(error_count)
        matrices.append(trans_mat)

    if not errors:
        raise ValueError("Could not perform homography detection, fiducials likely collinear.")
    best_index = np.argmin(errors)
    return matrices[best_index]


def detect_outliers(detect, design, trans_mat, threshold=1):
    """Returns names of fiducials with registration error greater than 1 pixel."""
    overlap = np.array(sorted(list(set(detect.keys()).intersection(set(design.keys())))))
    detect_pts = np.array([detect[key] for key in overlap])
    design_pts = np.array([design[key] for key in overlap])
    return overlap[np.where(registration_error(detect_pts, design_pts, trans_mat) > threshold)[0]]


def register_visium_hd_fiducials(
    design_fiducials: Iterable[FiducialDict],
    detect_fiducials: Iterable[FiducialDict],
    use_ransac=True,
):
    """Fiducial registration for Visium HD slides."""
    metrics = {}
    detect = {p["name"]: (p["x"], p["y"]) for p in detect_fiducials}
    design = {p["name"]: (p["x"], p["y"]) for p in design_fiducials}

    overlap = set(detect.keys()).intersection(set(design.keys()))
    if use_ransac:
        try:
            trans_mat = ransac(detect, design)
            outliers = detect_outliers(detect, design, trans_mat)
            metrics[metrics_names.OUTLIERS_DETECTED] = len(outliers)
            ransac_overlap = overlap - set(outliers)
            # TODO (Chaitanya): Add a more sensible fallback
            if len(ransac_overlap) >= 4:
                overlap = ransac_overlap
        except ValueError:
            pass

    overlap = sorted(list(overlap))
    trans_mat, _ = cv2.findHomography(
        np.array([design[key] for key in overlap]),
        np.array([detect[key] for key in overlap]),
    )

    # Sanity check for registration matrix. For Visium HD, scale factor should
    # be approximately 1/CYT_IMG_PIXEL_SIZE, rotation/perspective correction should be minimal
    lower, upper = CYT_IMG_PIXEL_SIZE - 0.5, CYT_IMG_PIXEL_SIZE + 0.5
    scale = 1.0 / np.sqrt(trans_mat[0][0] ** 2 + trans_mat[0][1] ** 2)
    if scale < lower or scale > upper:
        raise ValueError(
            "Automatic registration is likely incorrect. Please manually register your Cytassist image with Loupe."
        )

    design_pts = np.array([design[i] for i in overlap])
    detect_pts = np.array([detect[i] for i in overlap])

    error_vec = registration_error(detect_pts, design_pts, trans_mat)
    error_mean = np.mean(error_vec)
    error_rms = np.sqrt(np.mean(error_vec**2))
    error_q95 = np.quantile(error_vec, 0.95)

    metrics["fiducial_frame_transform"] = trans_mat
    metrics[metrics_names.DECODED_FIDUCIALS] = len(overlap)
    metrics[metrics_names.FIDUCIAL_DECODING_RATE] = len(overlap) / len(detect)
    metrics["registered_fiducial_error_q95_pixels"] = error_q95
    metrics["registered_fiducial_error_mean_pixels"] = error_mean
    metrics["registered_fiducial_error_rms_pixels"] = error_rms
    metrics["registered_fiducial_errors_pixels"] = error_vec

    return trans_mat, metrics, overlap


def isolate_perspective_transform(
    design_fiducials: Iterable[FiducialDict],
    detect_fiducials: Iterable[FiducialDict],
    overlap,
) -> np.ndarray:
    """Given homography transform, approximately removes rotation and scaling components."""
    detect_dict = {i["name"]: [i["x"], i["y"]] for i in detect_fiducials}
    design_dict = {i["name"]: [i["x"], i["y"]] for i in design_fiducials}
    detect_xy = np.array([detect_dict[key] for key in overlap])
    design_xy = np.array([design_dict[key] for key in overlap])
    h_inv, _ = cv2.findHomography(detect_xy, design_xy)

    # Returns the rcos(theta), rsin(theta), t_x, and t_y components of a similarity transform
    components = fit_2d_similarity_transform(
        np.asarray([xy[0] for xy in detect_xy]),
        np.asarray([xy[1] for xy in detect_xy]),
        np.asarray([xy[0] for xy in design_xy]),
        np.asarray([xy[1] for xy in design_xy]),
    )

    similarity_mat = np.eye(3)

    similarity_mat[0, 0] = components[0]
    similarity_mat[0, 1] = -components[1]
    similarity_mat[0, 2] = components[2]
    similarity_mat[1, 0] = components[1]
    similarity_mat[1, 1] = components[0]
    similarity_mat[1, 2] = components[3]

    return similarity_mat @ h_inv
