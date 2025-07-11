# Copyright (c) 2021 10X Genomics, Inc. All rights reserved.
"""Algorithms and util functions for tissue registration."""

from __future__ import annotations

from typing import NamedTuple

import cv2
import numpy as np
import SimpleITK as sitk
from skimage import exposure, measure

from cellranger.spatial import tissue_regist_qc
from cellranger.spatial.bounding_box import BoundingBox
from cellranger.spatial.tissue_detection import get_mask
from cellranger.spatial.transform import (
    contains_reflection,
    convert_transform_corner_to_center,
    reflection_matrix,
)

INIT_TRANSFORM_LIST_KEY = "init_transform_list"
INIT_METHOD_USED_KEY = "init_method_used"
FEATURE_MATCHING = "feature_matching"
CENTROID_ALIGNMENT = "centroid_alignment"
TARGET_IMG_MIN_SIZE_UM = 4000  # Half width of the fiducial box
TARGET_IMG_MAX_SIZE_UM = 26000  # Recommended max glass slide width

ITK_ERROR_PREFIX = "ITK errored"


def downsample_image(img: np.ndarray, maxres: int):
    """Scale, resize, and return the downsampled image."""
    if len(img.shape) == 3:
        scale = 1.0 * maxres / max(img.shape[:2])
        scale = min(1.0, scale)
    else:
        scale = 1.0 * maxres / max(img.shape)
        scale = min(1.0, scale)
    rcvimg = cv2.resize(img, (0, 0), fx=scale, fy=scale)
    return rcvimg, scale


def adjust_image_contrast(
    img: np.ndarray,
    method: str = "adaptive",
    clip_limit: float = 0.01,
    invert: bool = True,
) -> np.ndarray:
    """Adjust contrast of the input grayscale image of type uint8."""
    if invert:
        img = 255 - img
    img = exposure.rescale_intensity(img, in_range="image", out_range=(0, 255)).astype(np.uint8)
    if method == "adaptive":
        # Adaptive Equalization
        img = exposure.equalize_adapthist(img, clip_limit=clip_limit)
    else:
        p_2, p_98 = np.percentile(img, (2, 98))
        img = exposure.rescale_intensity(img, in_range=(p_2, p_98))
    return (img * 255).astype(np.uint8)


def largest_region_prop(mask: np.ndarray) -> tuple[float, float, float, tuple[int, int, int, int]]:
    """Find the largest binary region and return its property.

    Args:
        mask (np.ndarray): binary mask.

    Returns:
        Tuple[float, float, float, tuple[int, int, int, int]]: area, centroid, orientation and bbox of the largest object in the binary mask.
    """
    labeled_mask = measure.label(mask)
    regions = measure.regionprops(labeled_mask)
    area_list = [[p.area, i] for i, p in enumerate(regions)]
    area_list = np.array(area_list)
    idx = np.argmax(area_list[:, 0])
    p = regions[int(area_list[idx, 1])]

    return (p.area, p.centroid, p.orientation, p.bbox)


def estimate_tissue_position(
    raw_img: np.ndarray,
    qc_image_path: str | bytes | None = None,
    maxsize: int = 2000,
) -> tuple[float, list[float], BoundingBox]:
    """Estimate tissue position with segmentation.

    Args:
        raw_img (np.ndarray): original image
        qc_image_path (str | bytes | None, optional) : path to write QC image to.
        maxsize (int, optional): downsample the image to this for segmentation. Defaults to 2000.

    Returns:
        Tuple[float, List[float], BoundingBox]: Estimated area, centroid, and bbox of the tissue.
    """
    down_im, down_im_scale = downsample_image(raw_img, maxres=maxsize)
    # Use a higher clip_limit for tissue detection
    down_im = adjust_image_contrast(down_im, method="adaptive", clip_limit=0.02, invert=False)
    down_im = cv2.convertScaleAbs(down_im)
    mask, _, qc_image, _, _ = get_mask(down_im, plot=True)
    area, centroid, _, bbox = largest_region_prop(mask)
    area = area / down_im_scale**2
    bbox = tuple((np.array(bbox) / down_im_scale).astype(int))
    bbox = BoundingBox(*bbox)
    if qc_image_path is not None and qc_image is not None:
        qc_image.savefig(qc_image_path, bbox_inches="tight")
    centroid = np.array(centroid) / down_im_scale
    return (area, centroid, bbox)


def get_centroid_alignment_transforms(
    target_img: np.ndarray,
    cyta_img: np.ndarray,
    pixel_size_target_to_cyta_ratio: float | None = None,
    target_tissue_detection_debug: str | bytes | None = None,
    cytassist_tissue_detection_debug: str | bytes | None = None,
) -> list[np.ndarray]:
    """Generate initial transforms by aligning the centroids of the largest detected tissue regions.

    Args:
        target_img (np.ndarray): registration target image.
        cyta_img (np.ndarray): registration source image.
        pixel_size_target_to_cyta_ratio (float | None, optional): the ratio of the physical pixel size of target to that of cyta. Defaults to None.
        target_tissue_detection_debug (str | bytes | None, optional): path to write QC image to. Defaults to None.
        cytassist_tissue_detection_debug (str | bytes | None, optional): path to write QC image to. Defaults to None.

    Returns:
        list[np.ndarray]: the initial transforms accounting for potential rotations and reflections.
    """
    cyta_area, cyta_centroid, cyta_bbox = estimate_tissue_position(
        cyta_img, cytassist_tissue_detection_debug
    )
    target_area, target_centroid, _ = estimate_tissue_position(
        target_img, target_tissue_detection_debug
    )
    init_transform_list = []
    num_rot_search = 4  # number of total different angles to search from 0 to 2*np.pi
    full_rot_angle = np.linspace(0, 2 * np.pi, num_rot_search + 1)[:-1]
    for theta in full_rot_angle:
        # From cyta to target
        scale_factor = (
            1 / pixel_size_target_to_cyta_ratio
            if pixel_size_target_to_cyta_ratio
            else np.sqrt(target_area / cyta_area)
        )
        init_transform = sitk.Similarity2DTransform(
            scale_factor,
            theta,
            (target_centroid[1] - cyta_centroid[1], target_centroid[0] - cyta_centroid[0]),
            (cyta_centroid[1], cyta_centroid[0]),
        )
        init_transform_mat = sitk_transform_to_matrix(init_transform)
        # search for flipped and non-flipped initial state separately
        init_transform_list.append(init_transform_mat.tolist())
        # reflection over the x-axis
        refl_mat = reflection_matrix(True, False, cyta_bbox.shape)
        refl_cyta_centroid = [cyta_centroid[0], cyta_bbox.shape[1] - cyta_centroid[1]]
        init_transform = sitk.Similarity2DTransform(
            scale_factor,
            theta,
            (
                target_centroid[1] - refl_cyta_centroid[1],
                target_centroid[0] - refl_cyta_centroid[0],
            ),
            (refl_cyta_centroid[1], refl_cyta_centroid[0]),
        )
        init_transform_mat = sitk_transform_to_matrix(init_transform)
        init_transform_list.append(init_transform_mat @ refl_mat)
    return init_transform_list


class RegistParameter(NamedTuple):
    """Parameters for image registration."""

    sampling_rate: float
    num_hist_bin: int
    learning_rate: float
    min_step: float
    num_iter: int
    grad_tol: float
    relax_factor: float
    shrink_factor: list[int]
    smooth_sigma: list[int]


def multires_registration(
    fixed_image,
    moving_image,
    initial_transform: sitk.Transform,
    params: RegistParameter,
) -> tuple[sitk.Transform, float, str]:
    """Multi-resolution registration between two images.

    Args:
        fixed_image (sitk.Image): fi image for registration
        moving_image (sitk.Image): mving image for registration
        initial_transform (sitk.Transform): initial itk transform
        params (RegistParameter): parameters for registration

    Returns:
        Tuple[sitk.Transform, float]: the final transform and the corresponding metric.
    """
    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsMattesMutualInformation(
        numberOfHistogramBins=params.num_hist_bin
    )
    registration_method.SetMetricSamplingStrategy(registration_method.REGULAR)
    registration_method.SetMetricSamplingPercentage(params.sampling_rate, seed=42)
    registration_method.SetInterpolator(sitk.sitkLinear)

    registration_method.SetOptimizerAsRegularStepGradientDescent(
        learningRate=params.learning_rate,
        minStep=params.min_step,
        numberOfIterations=params.num_iter,
        gradientMagnitudeTolerance=params.grad_tol,
        relaxationFactor=params.relax_factor,
    )
    registration_method.SetOptimizerScalesFromPhysicalShift()
    registration_method.SetInitialTransform(initial_transform, inPlace=False)
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=params.shrink_factor)
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=params.smooth_sigma)
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    return perform_registration(registration_method, fixed_image, moving_image, initial_transform)


def perform_registration(
    registration_method: sitk.ImageRegistrationMethod,
    fixed_image: sitk.Image,
    moving_image: sitk.Image,
    initial_transform: sitk.Transform,
) -> tuple[sitk.Transform, float, str]:
    """Perform the registration and handle exceptions.

    Args:
        registration_method (sitk.ImageRegistrationMethod): The registration method to use.
        fixed_image (sitk.Image): The fixed image for registration.
        moving_image (sitk.Image): The moving image for registration.
        initial_transform (sitk.Transform): The initial transform.

    Returns:
        tuple[sitk.Transform, float, str]: The final transform, metric value, and optimizer stop condition.
    """
    try:
        init_metric = registration_method.MetricEvaluate(
            sitk.Cast(fixed_image, sitk.sitkFloat32), sitk.Cast(moving_image, sitk.sitkFloat32)
        )
    except RuntimeError as err:
        print(f"ITK initialization failed with initial transform {initial_transform}\n")
        print(f"error message: {err}\n")
        optimizer_stop_condition = f"{ITK_ERROR_PREFIX} in initialization with initial transform {initial_transform}.\nError message: {err}"
        return (initial_transform, np.inf, optimizer_stop_condition)

    try:
        fixed_image_itk = sitk.Cast(fixed_image, sitk.sitkFloat32)
        fixed_image_itk.SetOrigin(np.array([0.5, 0.5]))
        moving_image_itk = sitk.Cast(moving_image, sitk.sitkFloat32)
        moving_image_itk.SetOrigin(np.array([0.5, 0.5]))
        final_transform = registration_method.Execute(fixed_image_itk, moving_image_itk)
        metric_value = registration_method.GetMetricValue()
        optimizer_stop_condition = registration_method.GetOptimizerStopConditionDescription()
    except RuntimeError as err:
        print(f"ITK registration failed with initial transform {initial_transform}\n")
        print(f"error message: {err}\n")
        final_transform = initial_transform
        metric_value = init_metric
        optimizer_stop_condition = f"{ITK_ERROR_PREFIX} in optimization with initial transform {initial_transform}.\nError message: {err}"

    return (final_transform, metric_value, optimizer_stop_condition)


def register_from_init_transform(
    microscope_img: np.ndarray,
    cyta_img: np.ndarray,
    init_transform_mat: np.ndarray,
    learning_rate: float = 10.0,
    init_image_debug_path: str | bytes | None = None,
) -> tuple[np.ndarray, float, str]:
    """Register with an initial estimate of the transform."""
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)

    # Prepare if initial transform contains reflection over either x or y axis
    # If no reflection exists, this code block has no effect
    refl_mat = reflection_matrix(contains_reflection(init_transform_mat), False, cyta_img.shape)
    inv_refl_mat = np.linalg.inv(refl_mat)
    # Reflect the reflected initial transform to remove reflection
    init_transform_mat = init_transform_mat @ refl_mat
    # Apply the opposite to the source image so the effect cancels out with the previous step
    cyta_img = cv2.warpPerspective(
        cyta_img,
        # Warping uses center-based coordinates, while transform calculations use corner-based
        convert_transform_corner_to_center(inv_refl_mat),
        cyta_img.shape[::-1],
    )

    # Here the initial transformation is strictly a similarity transform without reflection
    init_transform = matrix_to_sitk_similarity2d(init_transform_mat, cyta_img.shape).GetInverse()

    if init_image_debug_path is not None:
        tissue_regist_qc.save_init_image(
            sitk_transform_to_matrix(init_transform),
            microscope_img,
            cyta_img,
            init_image_debug_path,
        )

    fix_img_itk = sitk.GetImageFromArray(microscope_img)
    mv_img_itk = sitk.GetImageFromArray(cyta_img)

    final_transform, final_metric, stop_description = multires_registration(
        fix_img_itk,
        mv_img_itk,
        init_transform,
        RegistParameter(
            sampling_rate=0.3,
            num_hist_bin=60,
            learning_rate=learning_rate,
            min_step=1e-4,
            num_iter=100,
            grad_tol=1e-6,
            relax_factor=0.5,
            shrink_factor=[8, 4, 2, 1],
            smooth_sigma=[4, 2, 1, 0],
        ),
    )

    transform = sitk.CompositeTransform(final_transform).GetNthTransform(0)
    transform = sitk.Similarity2DTransform(transform).GetInverse()
    transform = sitk_transform_to_matrix(transform)
    # Finally, add back the reflection
    transform = transform @ inv_refl_mat
    return (transform, final_metric, stop_description)


def sitk_transform_to_matrix(transform) -> np.ndarray:
    """Convert an ITK transform to a matrix."""
    center = np.array(transform.GetCenter())
    trans = np.array(transform.GetTranslation())
    mat = np.array(transform.GetMatrix()).reshape(2, 2)
    offset = center + trans - np.matmul(mat, center)
    transform_mat = np.eye(3)
    transform_mat[:2, :2] = mat
    transform_mat[:2, 2] = offset
    return transform_mat


def matrix_to_sitk_similarity2d(
    transform_mat: np.ndarray, src_img_shape_rc: tuple[int, int]
) -> sitk.Transform:
    """Convert a matrix to an ITK similarity 2D transform."""
    assert not contains_reflection(transform_mat)
    mat = transform_mat[:2, :2]
    center = [src_img_shape_rc[1] / 2, src_img_shape_rc[0] / 2]
    offset = transform_mat[:2, 2]
    trans = offset - center + np.matmul(mat, center)
    sitk_transform = sitk.Similarity2DTransform()
    sitk_transform.SetMatrix(mat.flatten())
    sitk_transform.SetCenter(center)
    sitk_transform.SetTranslation(trans)
    return sitk_transform
