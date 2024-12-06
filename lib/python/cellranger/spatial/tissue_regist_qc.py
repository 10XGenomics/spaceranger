# Copyright (c) 2021 10X Genomics, Inc. All rights reserved.
"""QC plots for tissue registration."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import skimage
from skimage import img_as_ubyte

from cellranger.spatial.transform import convert_transform_corner_to_center


def create_tissue_regist_qc_img(
    target_img: np.ndarray,
    source_img: np.ndarray,
    transform_matrix: np.ndarray,
    qc_img_type: str = "color",
) -> tuple[np.ndarray, np.ndarray]:
    """Create qc image for tissue registration.

    For the qc image itself to be in the target image space, the transform matrix maps
    from the target_image to source_img.

    Args:
        target_img (np.ndarray): target image of registration
        source_img (np.ndarray): source image of registration
        transform_matrix (np.ndarray): transform matrix that maps from target_img to source_img
        qc_img_type (str): type of qc image.

    Returns:
        np.ndarray: the qc image
    """
    center_based_transform = convert_transform_corner_to_center(transform_matrix)
    transform = skimage.transform.ProjectiveTransform(matrix=center_based_transform)

    resampled_src_img = skimage.transform.warp(
        source_img, transform, output_shape=target_img.shape, preserve_range=True
    ).astype(np.uint8)
    if qc_img_type == "checkerboard":
        qc_fig = skimage.util.compare_images(
            target_img.astype(np.uint8), resampled_src_img, method="checkerboard", n_tiles=(10, 10)
        )
    elif qc_img_type == "color":
        row, col = target_img.shape
        qc_fig = np.zeros((row, col, 3), dtype=np.uint8)
        rescale_target = skimage.exposure.rescale_intensity(target_img.astype(np.uint8))
        rescale_src = skimage.exposure.rescale_intensity(resampled_src_img)
        qc_fig[:, :, 0] = rescale_target
        qc_fig[:, :, 1] = rescale_src
        qc_fig[:, :, 2] = rescale_target // 2 + rescale_src // 2  # avoid overflow
    else:
        raise ValueError(f"Support color and checkerboard for QC image but get {qc_img_type}.")
    return qc_fig, resampled_src_img


def float_image_to_ubyte(img) -> np.ndarray:
    """Convert float image to ubyte."""
    return img_as_ubyte(
        np.interp(
            img,
            (img.min(), img.max()),
            (0.0, 1.0),
        )
    )


def save_tissue_regist_qc_img(qc_img, save_path):
    """Save tissue registration QC image.

    Args:
        qc_img: Image
        save_path: Path
    """
    fig, ax = plt.subplots(figsize=(qc_img.shape[1] / 72.0, qc_img.shape[0] / 72.0), dpi=72)
    ax.imshow(qc_img, cmap=plt.cm.gray)
    fig.savefig(save_path, format="jpg", dpi=72, pad_inches=0)
    plt.close(fig)


def save_init_image(
    transform_matrix,
    fix_img,
    mv_img,
    save_path: str | bytes,
):
    """Save the QC initialization image.

    Args:
        transform_matrix: Transformation matrix
        fix_img (np.ndarray): fix image for registration
        mv_img (np.ndarray): moving image for registration
        save_path (str | bytes | None, optional) : path to write QC of initialization image to.
    """
    qc_img, _ = create_tissue_regist_qc_img(
        float_image_to_ubyte(fix_img),
        float_image_to_ubyte(mv_img),
        transform_matrix,
    )

    fig, ax = plt.subplots(figsize=(qc_img.shape[1] / 72.0, qc_img.shape[0] / 72.0), dpi=72)
    ax.imshow(qc_img, cmap=plt.cm.gray)
    fig.savefig(save_path, format="jpg", dpi=72, pad_inches=0)
    plt.close(fig)
