# Copyright (c) 2021 10x Genomics, Inc. All rights reserved.

"""Functions to process images used in CytAssist."""

from __future__ import annotations

import shutil

import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes
from scipy.signal import fftconvolve
from scipy.spatial import ConvexHull
from skimage.filters import (  # pylint: disable=no-name-in-module
    gaussian,
    threshold_minimum,
    threshold_otsu,
)
from skimage.morphology import (
    binary_closing,
    binary_dilation,
    binary_erosion,
    binary_opening,
    disk,
    remove_small_holes,
)


def bresenham(
    x0: int,
    y0: int,
    x1: int,
    y1: int,
) -> list[tuple[int, int]]:
    # pylint: disable=invalid-name
    """Go look up bresenham's line algorithm.

    This is that algorithm.

    It gets a set of points in discrete space that make up a line from
    point A to point B.
    Yield integer coordinates on the line from (x0, y0) to (x1, y1).
    Input coordinates should be integers.
    The result will contain both the start and the end point.

    Args:
        x0 (int): start x coordinate
        y0 (int): start y coordinate
        x1 (int): end x coordinate
        y1 (int): end y coordinate

    Returns:
        line (List[Tuple[int, int]]): a list containing the x and y coordinates
            of all points in the line
    """
    dx = x1 - x0
    dy = y1 - y0

    xsign = 1 if dx > 0 else -1
    ysign = 1 if dy > 0 else -1

    dx = int(abs(dx))
    dy = int(abs(dy))

    if dx > dy:
        xx, xy, yx, yy = xsign, 0, 0, ysign
    else:
        dx, dy = dy, dx
        xx, xy, yx, yy = 0, ysign, xsign, 0

    D = 2 * dy - dx
    y = 0

    line = []
    for x in range(dx + 1):
        line.append((x0 + x * xx + y * yx, y0 + x * xy + y * yy))
        if D >= 0:
            y += 1
            D -= 2 * dx
        D += 2 * dy
    return line


def sort_corners(corners: list[np.ndarray]) -> list[np.ndarray]:
    """This function sorts a set of xy corner coordinates representing.

    the corners of a box into top left, top right, bottom right, bottom left.

    Args:
        List[np.ndarray]: A list of unsorted corners as np arrays, as (x, y).

    Returns:
        List[np.ndarray]: The output is a list of 1x2 (x, y) corners
        in the order of top left, top right, bottom right, bottom left.
    """
    top_corners = sorted(corners, key=lambda x: x[1])[:2]
    top_left = sorted(top_corners, key=lambda x: x[0])[0]
    top_right = sorted(top_corners, key=lambda x: x[0])[1]
    bottom_corners = sorted(corners, key=lambda x: x[1])[2:]
    bottom_left = sorted(bottom_corners, key=lambda x: x[0])[0]
    bottom_right = sorted(bottom_corners, key=lambda x: x[0])[1]
    return top_left, top_right, bottom_right, bottom_left


def fit_rectangle(points: np.ndarray) -> list[np.ndarray]:
    """Find the smallest bounding rectangle for a set of points.

    Returns a set of points representing the corners of the bounding box.
    We basically are finding the convex hull of the points, then making a list
    of all the angles between neighboring points on the convex hull, then we
    transform the point set by all of those angles and get the min/max x and y
    values (bounding box). We choose the bounding box with the minimum area.
    Finally corners are sorted.

    Args:
        points: an nx2 matrix of unorganized x, y coordinates that we will try
        to wrap in the smallest possible rectangle.

    Returns:
        List[np.ndarray]: a list of 1x2 matrices (x, y) of sorted coordinates,
            in order of top left, top right, bottom right, bottom left.
    """
    # get the convex hull for the points
    hull_points = points[ConvexHull(points).vertices]

    # calculate edge angles
    edges = np.zeros((len(hull_points) - 1, 2))
    edges = hull_points[1:] - hull_points[:-1]

    angles = np.zeros(len(edges))
    angles = np.arctan2(edges[:, 1], edges[:, 0])

    angles = np.abs(np.mod(angles, np.pi / 2))
    angles = np.unique(angles)

    # find rotation matrices
    rotations = np.vstack(
        [
            np.cos(angles),
            np.cos(angles - np.pi / 2),
            np.cos(angles + np.pi / 2),
            np.cos(angles),
        ]
    ).T
    rotations = rotations.reshape((-1, 2, 2))

    # apply rotations to the hull
    rot_points = np.dot(rotations, hull_points.T)

    # find the bounding points
    min_x = np.nanmin(rot_points[:, 0], axis=1)
    max_x = np.nanmax(rot_points[:, 0], axis=1)
    min_y = np.nanmin(rot_points[:, 1], axis=1)
    max_y = np.nanmax(rot_points[:, 1], axis=1)

    # find the box with the best area
    areas = (max_x - min_x) * (max_y - min_y)
    best_idx = np.argmin(areas)

    x_1 = max_x[best_idx]
    x_2 = min_x[best_idx]
    y_1 = max_y[best_idx]
    y_2 = min_y[best_idx]
    r = rotations[best_idx]

    # sort the corners (top left, top right, bottom right, bottom left)
    return sort_corners(
        [
            np.dot([x_1, y_2], r),
            np.dot([x_2, y_2], r),
            np.dot([x_2, y_1], r),
            np.dot([x_1, y_1], r),
        ],
    )


def crop_spacer_dynamic_window(
    image: np.ndarray,
    perim_thickness: int = 300,
    max_threshold: int = 128,
) -> tuple[int, int, int, int]:
    """Given a Cytassist image where the perimeter of the image is occupied.

    by the Cytassist slide spacer, crop the interior image region within the spacer.

    Args:
        image (np.ndarray): the input image, uncropped.
        perim_thickness (int): This algorithm attemps to crop the spacer portion of
        the image using otsu. This value indicates the number of pixels from the
        edge of all sides (the thickness of the perimeter of the image) to use as
        a subset of the data for calculating a threshold value allowing the segmentation
        of the region interior to the spacer.
        max_threshold (int): the max possible value of the image thresholding step
        used to isolate the spacer from the foreground of the image.

    Returns:
        bounding box of cropped region, (rowmin, colmin, rowmax, colmax)
    """
    perim_data = np.concatenate(
        (
            image[:perim_thickness, :].ravel(),
            image[-perim_thickness:, :].ravel(),
            image[perim_thickness:-perim_thickness, :perim_thickness].ravel(),
            image[perim_thickness:-perim_thickness, -perim_thickness:].ravel(),
        ),
        axis=0,
    )
    mask = image[:, :] > min(threshold_otsu(perim_data), max_threshold)
    mask = binary_opening(mask, disk(5))
    mask = binary_fill_holes(mask)
    mask[:1, :] = 0
    mask[-1:, :] = 0
    mask[:, :1] = 0
    mask[:, -1:] = 0

    perim_y, perim_x = np.where(mask ^ binary_erosion(mask, disk(1)))
    perim_points = np.concatenate(
        (
            perim_x.reshape((-1, 1)),
            perim_y.reshape((-1, 1)),
        ),
        axis=1,
    )
    corners = fit_rectangle(perim_points)
    inner_corners = []
    center = np.mean([c[0] for c in corners]), np.mean([c[1] for c in corners])
    for corner in corners:
        line = bresenham(center[0], center[1], corner[0], corner[1])
        for point in line:
            if not mask[int(point[1]), int(point[0])]:
                inner_corners.append((int(point[0]), int(point[1])))
                break
            if point == line[-1]:
                inner_corners.append((int(point[0]), int(point[1])))

    qc_box_img = np.zeros_like(np.tile(image.reshape(np.hstack([image.shape, 1])), (1, 1, 3)))
    bbox = [
        np.max([inner_corners[0][1], inner_corners[1][1]]),
        np.min([inner_corners[2][1], inner_corners[3][1]]),
        np.max([inner_corners[0][0], inner_corners[3][0]]),
        np.min([inner_corners[1][0], inner_corners[2][0]]),
    ]
    for i in range(4):
        if i == 0:
            line = bresenham(
                bbox[2],
                bbox[0],
                bbox[3],
                bbox[0],
            )
        elif i == 1:
            line = bresenham(
                bbox[3],
                bbox[0],
                bbox[3],
                bbox[1],
            )
        elif i == 2:
            line = bresenham(
                bbox[3],
                bbox[1],
                bbox[2],
                bbox[1],
            )
        elif i == 3:
            line = bresenham(
                bbox[2],
                bbox[1],
                bbox[2],
                bbox[0],
            )
        line = np.asarray(line).reshape((-1, 2)).astype("int")
        line[line[:, 1] < 0, 1] = 0
        line[line[:, 1] > image.shape[0], 1] = image.shape[0]
        line[line[:, 0] < 0, 0] = 0
        line[line[:, 0] > image.shape[1], 0] = image.shape[1]
        qc_box_img[line[:, 1], line[:, 0], 2] = True
    qc_box_img[:, :, 2] = binary_dilation(qc_box_img[:, :, 2], disk(3))
    qc_box_img[qc_box_img == 0] = np.tile(image.reshape(np.hstack([image.shape, 1])), (1, 1, 3))[
        qc_box_img == 0
    ]
    return bbox, qc_box_img


def crop_spacer_interior(
    img_orig: np.ndarray,
    crop_height_px: int,
    crop_width_px: int,
) -> tuple[int, int, int, int]:
    """Crop the interior spacer region of Cytassist image.

    Args:
        img_orig: np.ndarray, original image
        crop_height_px: int, the height of the bounding box in pixel.
        crop_width_px: int, the width of the bounding box in pixel.

    Return:
        Tuple[int, int, int, int]: the lower, upper limit of row and
            column to crop the image
    """
    scale = 0.1  # we will scale things down to 10% for time savings
    square_img = np.ones((crop_height_px, crop_width_px)).astype("float")
    square_img = cv2.resize(square_img, None, fx=scale, fy=scale)

    img = cv2.resize(img_orig, None, fx=scale, fy=scale)
    #  threshold the input image to get a good idea of the inner spacer region
    thresh = threshold_minimum(img)
    img_thresh = (img > thresh).astype("uint8")

    #  since illumination gradients can cause the edges of the spacer area to not be straight,
    #  do a morphological closing with a vertical and horizontal line kernel, and pad the image so that
    #  areas on the side of the region can never be filled in
    stick_kernel_length = 300
    pad_1 = stick_kernel_length + 1
    img_thresh = np.pad(img_thresh, ((pad_1, pad_1), (pad_1, pad_1)))
    img_thresh = binary_closing(img_thresh, np.ones((stick_kernel_length, 1)))
    img_thresh = binary_closing(img_thresh, np.ones((1, stick_kernel_length)))

    #  fill holes in the thresholed region, padding to avoid filling open space on the sides of the image
    max_hole_area = 50000
    pad_2 = np.ceil((max_hole_area - 2 * pad_1) / 6 + 1).astype("int")
    img_thresh = np.pad(img_thresh, ((pad_2, pad_2), (pad_2, pad_2)))
    img_thresh = remove_small_holes(img_thresh, area_threshold=max_hole_area).astype("float")
    img_thresh = img_thresh[pad_1 + pad_2 : -(pad_1 + pad_2), pad_1 + pad_2 : -(pad_1 + pad_2)]

    #  in order to get the correlation to lock more towards the fiducial area, add them to the thresholded mask,
    #  then do a gaussian blur
    img_thresh = gaussian(img_thresh, 5)  # + fiducial_img, 5)

    #  do a 2d fft based cross correlation to get
    xcorr = fftconvolve(img_thresh, square_img, mode="same")
    maxloc = np.where(xcorr == xcorr.max())

    #  max location might be a non singular region of equivalent values in the middle since the data is plateau-like,
    #  so take the mean value, then get crop window from that
    row = maxloc[0][0]
    col = maxloc[1][0]
    rowmin = int(row / scale - crop_height_px / 2)
    rowmax = int(row / scale + crop_height_px / 2)
    colmin = int(col / scale - crop_width_px / 2)
    colmax = int(col / scale + crop_width_px / 2)

    #  adjust the image boundaries in case they run beyond the edges
    if colmin < 0:
        colmax -= colmin
        colmin -= colmin
    if colmax >= img_orig.shape[1]:
        colmin -= colmax - img_orig.shape[1] - 1
        colmax -= colmax - img_orig.shape[1] - 1
    if rowmin < 0:
        rowmax -= rowmin
        rowmin -= rowmin
    if rowmax >= img_orig.shape[0]:
        rowmin -= rowmax - img_orig.shape[0] - 1
        rowmax -= rowmax - img_orig.shape[0] - 1

    return (rowmin, rowmax, colmin, colmax)


def convert_to_eosin_img(
    rgb_img: np.ndarray,
    dist_threshold: float = 50,
    damp_sigma: float = 50,
) -> np.ndarray:
    """Emphasize the eosin pink color from RGB image.

    Not used until cytassist image has the correct white balance.

    The purpose of the algorithm is to remove the non-eosin color as much
    as possible, which enables markers on the back of the glass slide and
    improve the tissue segmentation for even weak eosin color.

    The algorithm uses an empirical color vector for the targeted eosin color.
    If the color of the pixel is different with the eosin color more than a
    threshold value, the saturation of the pixel is reduced with a weight
    factor. The value of such pixels are also increased.

    Args:
        rgb_img (np.ndarray): (height, width, 3) array of RGB image
        dist_threshold (float, optional): threshold of color difference. Defaults to 45.
        damp_sigma (float, optional): how fast to make the color gray-ish.
            Larger make it slower. Defaults to 50.

    Returns:
        np.ndarray: the converted RGB image with mostly the eosin colored region.
    """
    # the eosin_color_vec is an empirical value observed from experimental data
    eosin_color_vec = np.array([40, 37])
    hsv_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
    angle = (179 - hsv_img[:, :, 0]) * np.pi * 2 / 179
    reformated = np.stack(
        [hsv_img[:, :, 1] * np.cos(angle), hsv_img[:, :, 1] * np.sin(angle)], axis=2
    )
    color_distance = np.linalg.norm(reformated - eosin_color_vec, axis=2)
    color_mask = color_distance > dist_threshold
    weights = np.clip(dist_threshold - color_distance, a_min=None, a_max=0)
    hsv_img[:, :, 1] = hsv_img[:, :, 1] * np.exp(weights / damp_sigma)
    hsv_img[:, :, 2][color_mask] = np.percentile(hsv_img[:, :, 2], 99.0) * 0.9
    coverted_rgb = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)
    return coverted_rgb


def prepare_cytassist_img(
    cytassist_image_paths: list[str],
    fid_detect_img_path: str,
    tissue_detect_grayscale_img_path: str,
    tissue_detect_saturation_img_path: str | None,
    out_cyt_img_path: str,
    qc_crop_img_path: str,
    min_crop_size: int | None = 2000,
    is_visium_hd: bool = False,
) -> dict:
    """Prepare cytassist image for further processing.

    During development, CytAssist image(s) can be in the form of a RGB image,
    or a pair of images with green and red illumination. This function generates
    the image used for fiducial detection, tissue segmentation, and tissue registraion.
    The function also includes a pseudo cropping, i.e., the image is not really
    cropped, only the coordinates of the cropping window are saved.

    This function also tries to remove the non eosin color for tissue segmentation
    and tissue registration. The purpose is to enable customer to draw markers of
    a different color on the back of the slide. And also to improve the robustness of
    the tissue segmentation/registration algorithm.

    Args:
        cytassist_image_paths (List[str]): list of paths of the cytassist image.
        fid_detect_img_path (str): path to save the image for fiducial detection.
        tissue_detect_grayscale_img_path (str): path to save the grayscale image for tissue detection.
        tissue_detect_saturation_img_path (str):path to save the saturation image for tissue detection.
        out_cyt_img_path (str): path to save the cytassist image in the output.
        qc_crop_img_path (str): path to save the crop qc image
        min_crop_size (int): minimum cropping size on either dimension. If the calculated
            cropping size on either dimension is smaller than it, do not crop on that dimension.
        is_visium_hd (bool): if this is a visium HD sample. If so, we do not crop the image.
            defaults to False.

    Returns:
        Dict: [description]
    """
    if len(cytassist_image_paths) == 2:
        # The first image is red illumination; second is green illumination.
        shutil.copy(cytassist_image_paths[0], fid_detect_img_path)
        shutil.copy(cytassist_image_paths[1], tissue_detect_grayscale_img_path)
        shutil.copy(cytassist_image_paths[1], out_cyt_img_path)
        fid_img = cv2.imread(cytassist_image_paths[0], cv2.IMREAD_GRAYSCALE)
    elif len(cytassist_image_paths) == 1:
        shutil.copy(cytassist_image_paths[0], out_cyt_img_path)
        bgr_cyt_img = cv2.imread(cytassist_image_paths[0], cv2.IMREAD_COLOR)
        rgb_cyt_img = cv2.cvtColor(bgr_cyt_img, cv2.COLOR_BGR2RGB)
        if rgb_cyt_img.ndim != 3:
            raise ValueError(
                f"expect the cytassist image to be RGB but get dimension of {rgb_cyt_img.ndim}"
            )
        cv2.imwrite(fid_detect_img_path, rgb_cyt_img[:, :, 0])
        cv2.imwrite(tissue_detect_grayscale_img_path, rgb_cyt_img[:, :, 1])
        if tissue_detect_saturation_img_path is not None:
            hsv_cyt_img = cv2.cvtColor(bgr_cyt_img, cv2.COLOR_BGR2HSV)
            cv2.imwrite(tissue_detect_saturation_img_path, hsv_cyt_img[:, :, 1])
        fid_img = rgb_cyt_img[:, :, 0]
    else:
        raise ValueError(
            f"Only support 1 or 2 CytAssist images but {len(cytassist_image_paths)} are passed."
        )
    # Use fiducial image for cropping since the tissue has less contrast
    (rowmin, rowmax, colmin, colmax), qc_box_img = crop_spacer_dynamic_window(
        fid_img, max_threshold=64
    )
    # TODO: generate a warning message if cropping check fails and surface the message.
    # Disable cropping if visium HD
    if is_visium_hd or rowmax - rowmin < min_crop_size:
        rowmin, rowmax = 0, fid_img.shape[0]
    if is_visium_hd or colmax - colmin < min_crop_size:
        colmin, colmax = 0, fid_img.shape[1]
    cv2.imwrite(qc_crop_img_path, qc_box_img)
    crop_info_dict = {"row_min": rowmin, "row_max": rowmax, "col_min": colmin, "col_max": colmax}
    return crop_info_dict
