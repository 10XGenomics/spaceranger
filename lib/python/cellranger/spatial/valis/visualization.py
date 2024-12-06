#
# Copyright (c) 2024 10X Genomics, Inc. All rights reserved.
#
"""Various functions used to visualize registration results."""
import numpy as np
from skimage import color, draw

from cellranger.spatial.transform import pad_img, warp_xy


def draw_features(kp_xy: np.ndarray, image: np.ndarray, max_n_ft: int = -1) -> np.ndarray:
    """Draw keypoints on a image."""
    np.random.seed(0)
    if image.ndim == 2:
        feature_img = color.gray2rgb(image)
    else:
        feature_img = image.copy()
    rad = int(np.mean(feature_img.shape) / 100)
    n_kp = len(kp_xy)
    if n_kp < max_n_ft or max_n_ft < 0:
        max_n_ft = n_kp
    for col, row in kp_xy[0:max_n_ft].astype(int):
        circ_r, circ_c = draw.circle_perimeter(row, col, rad, shape=image.shape[0:2])
        feature_img[circ_r, circ_c] = np.random.randint(0, 256, 3)
    return feature_img


def draw_matches(  # pylint: disable=too-many-locals
    src_img: np.ndarray,
    matched_kp1_xy: np.ndarray,
    dst_img: np.ndarray,
    matched_kp2_xy: np.ndarray,
    unfiltered_kp1_xy: np.ndarray | None = None,
    unfiltered_kp2_xy: np.ndarray | None = None,
) -> np.ndarray:
    """Draw feature matches between two images.

    Args:
        src_img (np.ndarray): image associated with `kp1_xy`.
        matched_kp1_xy (np.ndarray): xy coordinates of matched feature points found in `src_img`.
        dst_img (np.ndarray): image associated with `kp2_xy`.
        matched_kp2_xy (np.ndarray): xy coordinates of matched feature points found in `dst_img`.
        unfiltered_kp1_xy (np.ndarray | None, optional): xy coordinates of all feature points found in `src_img`. Defaults to None.
        unfiltered_kp2_xy (np.ndarray | None, optional): xy coordinates of all feature points found in `dst_img`. Defaults to None.

    Returns:
        np.ndarray: Image show corresponding features of `src_img` and `dst_img`.
    """
    np.random.seed(0)
    all_dims = np.array([src_img.shape, dst_img.shape])
    out_shape = np.max(all_dims, axis=0)[0:2]

    # Draw all feature points first if given
    if unfiltered_kp1_xy is not None:
        src_img = draw_features(unfiltered_kp1_xy, src_img)
    if unfiltered_kp2_xy is not None:
        dst_img = draw_features(unfiltered_kp2_xy, dst_img)

    padded_src, src_padding_mat = pad_img(src_img, out_shape)
    padded_dst, dst_padding_mat = pad_img(dst_img, out_shape)

    # Horizontal layout
    feature_img = np.hstack([padded_src, padded_dst])
    dst_xy_shift = np.array([out_shape[1], 0])

    if feature_img.ndim == 2:
        feature_img = color.gray2rgb(feature_img).astype(np.uint8)

    dst_padding_mat[0:2, 2] += dst_xy_shift
    dst_xy_in_feature_img = warp_xy(matched_kp2_xy, mat=dst_padding_mat)
    src_xy_in_feature_img = warp_xy(matched_kp1_xy, mat=src_padding_mat)

    n_pt = np.min([matched_kp1_xy.shape[0], matched_kp2_xy.shape[0]])
    for i in range(n_pt):
        xy1 = src_xy_in_feature_img[i]
        xy2 = dst_xy_in_feature_img[i]
        pt_color = np.random.randint(0, 256, 3)

        rad = 3
        circ_rc_1 = draw.ellipse(*xy1[::-1], rad, rad, shape=feature_img.shape)
        circ_rc_2 = draw.ellipse(*xy2[::-1], rad, rad, shape=feature_img.shape)
        line_rc = np.array(
            draw.line_aa(*np.round(xy1[::-1]).astype(int), *np.round(xy2[::-1]).astype(int))
        )
        line_rc[0] = np.clip(line_rc[0], 0, feature_img.shape[0]).astype(int)
        line_rc[1] = np.clip(line_rc[1], 0, feature_img.shape[1]).astype(int)

        feature_img[line_rc[0].astype(int), line_rc[1].astype(int)] = (
            pt_color * line_rc[2][..., np.newaxis]
        )
        feature_img[circ_rc_1] = pt_color
        feature_img[circ_rc_2] = pt_color

    return feature_img
