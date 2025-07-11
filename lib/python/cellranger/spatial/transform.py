# Copyright (c) 2024 10X Genomics, Inc. All rights reserved.
"""Functions to transform 2d coordinates."""

from __future__ import annotations

import numpy as np
from skimage import transform as sktr


def rotation_matrix(rad):
    """Generate a rotation transformation matrix."""
    return np.array(
        [
            [np.cos(rad), -np.sin(rad), 0.0],
            [np.sin(rad), np.cos(rad), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def translation_matrix(dx, dy, cv2_compliant=False):  # pylint: disable=invalid-name
    """Generate a translation transformation matrix."""
    if cv2_compliant:
        return np.array(
            [
                [1.0, 0.0, dx],
                [0.0, 1.0, dy],
            ]
        )
    else:
        return np.array(
            [
                [1.0, 0.0, dx],
                [0.0, 1.0, dy],
                [0.0, 0.0, 1.0],
            ]
        )


def scale_matrix(scale):
    """Generate a scaling transformation matrix given the scale."""
    assert scale > 0.0
    return np.array(
        [
            [scale, 0.0, 0.0],
            [0.0, scale, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def reflection_matrix(reflect_x: bool, reflect_y: bool, shape_rc: tuple[int, int]) -> np.ndarray:
    """Get transformation matrix to reflect an image.

    Args:
        reflect_x (bool): whether or not to reflect over the x-axis (columns).
        reflect_y (bool): whether or not to reflect over the y-axis (rows).
        shape_rc (tuple[int, int]): shape of the image being reflected.

    Returns:
        np.ndarray: transformation matrix that will reflect an image along the specified axes.
    """
    reflection_mat = np.eye(3)
    if reflect_x:
        reflection_mat[0, 0] = -1
        reflection_mat[0, 2] = shape_rc[1] - 1
    if reflect_y:
        reflection_mat[1, 1] = -1
        reflection_mat[1, 2] = shape_rc[0] - 1
    return reflection_mat


def padding_matrix(src_shape_rc: tuple[int, int], out_shape_rc: tuple[int, int]) -> np.ndarray:
    """Compute the padding transformation matrix."""
    img_h, img_w = src_shape_rc
    out_h, out_w = out_shape_rc
    d_h = out_h - img_h
    d_w = out_w - img_w
    h_pad = d_h / 2
    w_pad = d_w / 2
    mat = np.identity(3).astype(np.float64)
    mat[0, 2] = w_pad
    mat[1, 2] = h_pad
    return mat


def contains_reflection(mat: np.ndarray) -> bool:
    """Check if a transformation contains reflection over either x or y axis."""
    assert mat.shape == (3, 3)
    # Perform SVD on the transform
    u_mat, _, vt_mat = np.linalg.svd(mat[:2, :2])
    r_mat = u_mat @ vt_mat
    # Reflection over either x or y axis exists if determinant of the rotation is <0
    return np.linalg.det(r_mat) < 0


def convert_transform_corner_to_center(transform: np.ndarray) -> np.ndarray:
    """Convert a corner-based transform to center-based.

    This function converts the domain and range of a 2D image transform from
    corner-based coordinates [image corner at (0.0, 0.0)] to center-based coordinates
    [image corner at (-0.5, -0.5)].

    Conversion is required for cv2.warpPerspective and skimage.transform.warp to work
    with transforms designed for corner-based coordinates. For example, providing those
    warping functions with a diagonal scaling matrix will counter-intuitively result in
    a shifted image. See link below for an example
    https://answers.opencv.org/question/33516/cv2warpaffine-results-in-an-image-shifted-by-05-pixel/
    """
    # The (x, y) position of the top-left corner of pixel at column 0, row 0 is:
    #   (0.0, 0.0) in corner-based coordinates
    #   (-0.5, -0.5) in center-based coordinates
    # Thus converting center -> corner requires translating x/y by 0.5
    center_to_corner = translation_matrix(0.5, 0.5)
    corner_to_center = translation_matrix(-0.5, -0.5)
    return corner_to_center @ transform @ center_to_corner


def normalize_perspective_transform(transform: np.ndarray) -> np.ndarray:
    """Normalise a perspective transform to have 1.0 in the (2,2) coordinate.

    The normalizes a perspective transform to match the convention for a homogeneous transform matrix.
    """
    assert transform[2, 2] != 0.0, "Got a zero in perspective transform."
    return transform / transform[2, 2]


def compose_perspective_transforms(first_transform: np.ndarray, second_transform: np.ndarray):
    """Compose two perspective transforms.

    This function multiplies two transform matrices and normalizes the result
    to match the convention for a homogeneous transform matrix.
    """
    composed_transform = first_transform @ second_transform
    assert composed_transform[2, 2] != 0.0, "Got a zero while composing perspective matrices."
    return normalize_perspective_transform(composed_transform)


def scale_from_transform_matrix(mat: np.ndarray) -> float:
    return np.sqrt(np.abs(np.linalg.det(mat[:2, :2])))


def transform_pts_2d(pts_xy: np.ndarray, projective_mat: np.ndarray) -> np.ndarray:
    """Transform 2d points array with a projective transform matrix.

    Args:
        pts_xy (np.ndarray): (n, 2) shape. recoord the x, y coordinate of each point
        projective_mat (np.ndarray): (3, 3) shape. 3 x 3 projective transform matrix

    Returns:
        np.ndarray: (n, 2) shape. coordinates after the transform
    """
    n = pts_xy.shape[0]
    if n == 0:
        return pts_xy
    h_coord = np.concatenate((pts_xy, np.ones((n, 1))), axis=1)
    transformed_xyz = np.matmul(h_coord, projective_mat.T)

    z = transformed_xyz[:, -1]  # zt of each coordinate
    transformed_xyz = transformed_xyz / z[:, None]  # Normalize each coordinate

    return transformed_xyz[:, :2]


def css_transform_list(mat: np.ndarray) -> list[float]:
    """Convert a 3x3 transform matrix to a list of 16 numbers for CSS transform."""
    return (
        np.array(
            [
                [mat[0, 0], mat[0, 1], 0, mat[0, 2]],
                [mat[1, 0], mat[1, 1], 0, mat[1, 2]],
                [0, 0, 1, 0],
                [mat[2, 0], mat[2, 1], 0, mat[2, 2]],
            ],
        )
        .ravel(order="F")
        .tolist()
    )


def pad_img(img: np.ndarray, padded_shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Pad the input image and return the padding matrix used."""
    padding_mat = padding_matrix(img.shape[0:2], padded_shape)
    padded_img = warp_img(img, padding_mat, dst_shape_rc=padded_shape)
    return padded_img, padding_mat


def crop_img(img, crop_info, cushion=50) -> tuple[np.ndarray, np.ndarray]:
    """Crop an image and return the translation matrix."""
    row_min, row_max = (
        crop_info["row_min"] + cushion,
        crop_info["row_max"] - cushion,
    )
    col_min, col_max = (
        crop_info["col_min"] + cushion,
        crop_info["col_max"] - cushion,
    )
    translation_mat = np.array([[1, 0, -col_min], [0, 1, -row_min], [0, 0, 1]])
    return img[row_min:row_max, col_min:col_max], translation_mat


def get_corners_of_image(shape_rc: tuple[int, int]) -> np.ndarray:
    """Get corners of image in clockwise order (TL, TR, BR, BL)."""
    max_x = shape_rc[1]
    max_y = shape_rc[0]
    bottom_left = [0, 0]
    bottom_right = [max_x, 0]
    top_left = [0, max_y]
    top_right = [max_x, max_y]

    corners = np.array([bottom_left, bottom_right, top_right, top_left])
    corners_rc = corners[:, ::-1]
    return corners_rc


def get_warp_scaling_factors(
    src_shape_rc: tuple[int, int] | None = None,
    dst_shape_rc: tuple[int, int] | None = None,
    transformation_src_shape_rc: tuple[int, int] | None = None,
    transformation_dst_shape_rc: tuple[int, int] | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Get scaling factors needed to warp points.

    This function returns the scalings needed to go from src_shape_rc to transformation_src_shape_rc and from transformation_dst_shape_rc to dst_shape_rc, respectively. If a returned value is None, it means there is no need to scale.
    """
    # convert shapes to arrays
    if src_shape_rc is not None:
        src_shape_rc = np.array(src_shape_rc)
    if transformation_src_shape_rc is not None:
        transformation_src_shape_rc = np.array(transformation_src_shape_rc)
    if dst_shape_rc is not None:
        dst_shape_rc = np.array(dst_shape_rc)
    if transformation_dst_shape_rc is not None:
        transformation_dst_shape_rc = np.array(transformation_dst_shape_rc)

    # Get input scalings
    src_sxy = None
    if transformation_src_shape_rc is not None and src_shape_rc is not None:
        if not np.all(transformation_src_shape_rc == src_shape_rc):
            src_sxy = (transformation_src_shape_rc / src_shape_rc)[::-1]

    # Get output scalings
    dst_sxy = None
    if transformation_dst_shape_rc is not None and dst_shape_rc is not None:
        if not np.all(dst_shape_rc == transformation_dst_shape_rc):
            dst_sxy = (dst_shape_rc / transformation_dst_shape_rc)[::-1]

    return src_sxy, dst_sxy


def get_scaled_similarity_transform(
    mat: np.ndarray,
    src_shape_rc: tuple[int, int],
    dst_shape_rc: tuple[int, int],
    transformation_src_shape_rc: tuple[int, int],
    transformation_dst_shape_rc: tuple[int, int],
) -> np.ndarray:
    """Scale a similarity transformation so it can be applied on input/output at new scales.

    After finding the registration between downsampled images, this function can be used to calculate the corresponding registration between the original images.

    Args:
        mat (np.ndarray): 3x3 similarity transformation matrix without reflection.
        src_shape_rc (tuple[int, int]): shape of image on which the scaled transform will be applied. For example, this could be a larger/smaller version of the image that was used for feature detection.
        dst_shape_rc (tuple[int, int]): shape of image (with shape `src_shape_rc`) after warping.
        transformation_src_shape_rc (tuple[int, int]): shape of image that was used to find the transformation. For example, this could be the original image in which features were detected.
        transformation_dst_shape_rc (tuple[int, int]): shape of image (with shape `transformation_src_shape_rc`) after warping.

    Returns:
        np.ndarray: the scaled transform.
    """
    img_corners_xy = get_corners_of_image(src_shape_rc)[::-1]
    warped_corners = warp_xy(
        img_corners_xy,
        mat=mat,
        transformation_src_shape_rc=transformation_src_shape_rc,
        transformation_dst_shape_rc=transformation_dst_shape_rc,
        src_shape_rc=src_shape_rc,
        dst_shape_rc=dst_shape_rc,
    )
    # Similarity transform here does not handle reflection
    tform = sktr.SimilarityTransform()
    tform.estimate(img_corners_xy, warped_corners)
    scaled_mat = tform.params
    return scaled_mat


def warp_xy(
    xy_arr: np.ndarray,
    mat: np.ndarray,
    src_shape_rc: tuple[int, int] | None = None,
    dst_shape_rc: tuple[int, int] | None = None,
    transformation_src_shape_rc: tuple[int, int] | None = None,
    transformation_dst_shape_rc: tuple[int, int] | None = None,
) -> np.ndarray:
    """Warp xy points given a transformation matrix.

    Args:
        xy_arr (np.ndarray): (N, 2) array of xy coordinates for N points.
        mat (np.ndarray): 3x3 affine transformation matrix.
        src_shape_rc (tuple[int, int] | None, optional): shape of image from which the points originated. For example, this could be a larger/smaller version of the image that was used for feature detection. Defaults to None.
        dst_shape_rc (tuple[int, int] | None, optional): shape of image (with shape `src_shape_rc`) after warping. Defaults to None.
        transformation_src_shape_rc (tuple[int, int] | None, optional): shape of image that was used to find the transformation. For example, this could be the original image in which features were detected. Defaults to None.
        transformation_dst_shape_rc (tuple[int, int] | None, optional): shape of image (with shape `transformation_src_shape_rc`) after warping. Defaults to None.

    Returns:
        np.ndarray: (N, 2) array of warped points.
    """
    src_sxy, dst_sxy = get_warp_scaling_factors(
        src_shape_rc=src_shape_rc,
        dst_shape_rc=dst_shape_rc,
        transformation_src_shape_rc=transformation_src_shape_rc,
        transformation_dst_shape_rc=transformation_dst_shape_rc,
    )
    if src_sxy is not None:
        in_src_xy = xy_arr * src_sxy
    else:
        in_src_xy = xy_arr

    rigid_xy = transform_pts_2d(in_src_xy, mat).astype(float)

    if dst_sxy is not None:
        return rigid_xy * dst_sxy
    else:
        return rigid_xy


def warp_img(
    img: np.ndarray,
    mat: np.ndarray,
    dst_shape_rc: tuple[int, int] | None = None,
) -> np.ndarray:
    """Warp an input image.

    Args:
        img (np.ndarray): the input image in uint8.
        mat (np.ndarray): the 3x3 transformation matrix
        dst_shape_rc (tuple[int, int] | None, optional): shape of the warped image. If not provided, this will be set big enough to avoid cropping. Defaults to None.

    Returns:
        np.ndarray: the warped image in uint8.
    """
    if dst_shape_rc is None:
        src_corners_rc = get_corners_of_image(img.shape[:2])
        warped_src_corners_xy = warp_xy(src_corners_rc[:, ::-1], mat)
        dst_shape_rc = np.ceil(np.max(warped_src_corners_xy[:, ::-1], axis=0)).astype(int)
    # Use the inverse transform for warpping an image
    inv_mat = np.linalg.inv(mat)
    warped_img = sktr.warp(img, inv_mat, output_shape=dst_shape_rc, preserve_range=True).astype(
        np.uint8
    )
    return warped_img
