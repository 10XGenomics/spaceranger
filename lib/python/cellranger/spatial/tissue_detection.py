# Copyright (c) 2021 10X Genomics, Inc. All rights reserved.
"""Algorithm and utility functions for tissue detection."""

from __future__ import annotations

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy
import skimage
import skimage.feature
import skimage.filters
from matplotlib import colors
from matplotlib.patches import Polygon

import cellranger.constants as cr_constants
import cellranger.spatial.image_util as image_util

matplotlib.use("agg")

BACKGROUND = 0
DETECTED_BY_ONE_METHOD = 1
DETECTED_BY_TWO = 2
DETECTED_BY_ALL = 3

PIXEL_THRESHOLD = 1.43  # sqrt 2
SMALL_OBJECT_GC_FG_THRESHOLD = 1000
SMALL_OBJECT_GC_FG_ENTROPY_THRESHOLD = 2000
CANNY_IMAGE_THRESHOLD = 100

cvals = [cv2.GC_BGD, cv2.GC_FGD, cv2.GC_PR_BGD, cv2.GC_PR_FGD]
cmap_colors = ["blue", "red", "deepskyblue", "orange"]

norm = plt.Normalize(min(cvals), max(cvals))
tuples = list(zip(map(norm, cvals), cmap_colors))
cmap_grabcut_markers = colors.LinearSegmentedColormap.from_list("", tuples)

cmap_otsu_sum_colors = ["blue", "peachpuff", "orange", "red"]
otsu_sum_cvals = [0, 1, 2, 3]
norm = plt.Normalize(min(otsu_sum_cvals), max(otsu_sum_cvals))
otsu_sum_tuples = list(zip(map(norm, otsu_sum_cvals), cmap_otsu_sum_colors))
cmap_otsu_sum_markers = colors.LinearSegmentedColormap.from_list("", otsu_sum_tuples)


def float_image_to_ubyte(img) -> np.ndarray:
    """Convert float image to ubyte."""
    return skimage.img_as_ubyte(
        np.interp(
            img,
            (img.min(), img.max()),
            (0.0, 1.0),
        )
    )


def get_grabcut_initialisation(
    gray: np.ndarray, longest_side: int, axes: np.ndarray | None
) -> np.ndarray:
    """Generate grabcut initialisation.

    Args:
        gray (np.ndarray): Grayscale tissue image which is cropped to bounded box.
        longest_side (int): Longest side of the image pre-bounded-box-ing.
        axes (np.ndarray): axes of image being constructed. Note that this gets modified in the fn.

    Returns:
        np.ndarray: grabcut initialisation figure.
    """
    mask_size = np.prod(gray.shape)
    # set sizes of objects and holes (assumes approx square input image)
    small_holes_size = int(longest_side / 2.0)
    small_objects_size = int(longest_side / 2.0)
    large_holes_size = int(longest_side * 50.0)
    # Calculate grayscale intensity Otsu threshold, often good proxy for where tissue is
    # (not as good when background and tissue has similar gray scale histograms)
    otsu_thresh = gray <= skimage.filters.threshold_otsu(gray)
    # Remove holes and tissue debris
    otsu_thresh = skimage.morphology.remove_small_objects(otsu_thresh, small_objects_size)
    otsu_thresh = skimage.morphology.remove_small_holes(otsu_thresh, small_holes_size)

    # Get the gradient (local max - local min) of the gray scale image,
    # high gradient usually indicates tissue (not as good for when larger
    # areas are out of focus or tissue is really smooth potentially
    # affected by low resolution images)
    gradient = skimage.filters.rank.gradient(gray, skimage.morphology.disk(5))

    # Binarize the gradient into two classes 1=FG and 0=BG using Otsu threshold
    inverted_grad = skimage.util.invert(gradient, signed_float=False)
    otsu_of_gradient = inverted_grad <= skimage.filters.threshold_otsu(inverted_grad)
    otsu_of_gradient = skimage.morphology.remove_small_objects(otsu_of_gradient, small_objects_size)
    otsu_of_gradient = skimage.morphology.remove_small_holes(otsu_of_gradient, small_holes_size)

    # Detect canny edges on the grayscale image (many edges usually indicate tissue)
    canny_edges = skimage.feature.canny(gray)
    closed_canny = skimage.morphology.closing(canny_edges)
    closed_canny = scipy.ndimage.distance_transform_edt(1 - closed_canny) <= longest_side * 0.01

    # Sum upp the two estimates of tissue placement
    # (gradient based and Outsu on grayscale intensity)
    otsu_sum = np.add(
        np.add(otsu_of_gradient.astype("uint8"), otsu_thresh.astype("uint8")),
        closed_canny.astype("uint8"),
    )

    # Start making markers for the grabcut
    markers_gc = np.zeros(gray.shape).astype("uint8")
    # to keep track of not yet classed vs obvious background
    classed = np.zeros(otsu_sum.shape).astype("uint8")

    ##### below is order dependent based on priority, pixels may be assign GC_BGD early and then
    ##### the same pixel may get GC_FGD later

    # If classed as background by both methods add a margin of 1% image longest side (in pixels) and
    # set to an obvious background pixels
    background = np.zeros(otsu_sum.shape).astype("uint8")
    background[otsu_sum == BACKGROUND] = 1
    background = scipy.ndimage.distance_transform_edt(background) >= longest_side * 0.01
    markers_gc[background == 1] = cv2.GC_BGD
    classed[background == 1] += 1

    # Take the two estimates (otsu_sum) fill all holes and set everything detected by at least one
    # method to be probable Background (instead of obvious background)
    # This is done so no holes will be classed as obvious background
    no_holes = np.zeros(otsu_sum.shape).astype("bool")
    no_holes[otsu_sum >= DETECTED_BY_ONE_METHOD] = True
    # remove_small_holes treats 0/1 mask different than false/true - use boolean
    no_holes = skimage.morphology.remove_small_holes(no_holes, large_holes_size)
    markers_gc[no_holes >= 1] = cv2.GC_PR_BGD
    classed[no_holes >= 1] += 1

    # If detected by at least one method set to be a possible foreground pixel
    markers_gc[otsu_sum >= DETECTED_BY_ONE_METHOD] = cv2.GC_PR_FGD
    classed[otsu_sum >= DETECTED_BY_ONE_METHOD] += 1

    # If detected by two methods add a margin of 5% (inward) image longest side (in pixels)
    # basically make the estimate smaller by some amount around the boundaries
    # set as an obvious foreground (object) pixel
    foreground = np.zeros(otsu_sum.shape).astype("uint8")
    foreground[otsu_sum == DETECTED_BY_TWO] = 1
    foreground = scipy.ndimage.distance_transform_edt(foreground) >= longest_side * 0.05
    markers_gc[foreground == 1] = cv2.GC_FGD
    classed[foreground == 1] += 1

    # If detected by all methods add a margin of 2.5% image longest side (in pixels)
    # same as above, but with smaller margin of safety due to greater certainty
    # set as an obvious foreground (object) pixel
    foreground = np.zeros(otsu_sum.shape).astype("uint8")
    foreground[otsu_sum == DETECTED_BY_ALL] = 1
    foreground = scipy.ndimage.distance_transform_edt(foreground) >= longest_side * 0.025
    markers_gc[foreground == 1] = cv2.GC_FGD
    classed[foreground == 1] += 1

    # Within tissue estimates (no_holes) but zero in outsu sum should be probable background
    # Essentially encourage interior holes when no method has indicated foreground.
    otsu_background = np.zeros(otsu_sum.shape).astype("uint8")
    otsu_background[otsu_sum == BACKGROUND] = 1
    probable_foreground_hole = np.add(otsu_background.astype("uint8"), no_holes.astype("uint8"))
    temp = np.ones(otsu_sum.shape).astype("uint8")
    temp[probable_foreground_hole == 2] = 0
    temp = scipy.ndimage.distance_transform_edt(temp) >= longest_side * 0.015
    temp = skimage.util.invert(temp, signed_float=False)
    probable_foreground_hole = temp.astype("uint8")
    markers_gc[temp == 1] = cv2.GC_PR_BGD

    # Set any unclassed pixels to be possible background
    markers_gc[classed == 0] = cv2.GC_PR_BGD

    # if statement for switching creation of plots for debugging and evaluation on and off
    if axes is not None:
        axes[0][1].imshow(canny_edges, cmap=plt.cm.gray, interpolation="nearest")
        axes[0][1].set_title("Canny Edges")

        axes[0][2].imshow(otsu_thresh, cmap=plt.cm.gray, interpolation="nearest")
        axes[0][2].set_title(f"Otsu threshold\n({np.sum(otsu_thresh)} / {mask_size})")

        axes[1][0].imshow(closed_canny, cmap=plt.cm.gray, interpolation="nearest")
        axes[1][0].set_title(f"Closed Canny\n({np.sum(closed_canny)} / {mask_size})")

        axes[1][1].imshow(gradient, cmap=plt.cm.gray, interpolation="nearest")
        axes[1][1].set_title("Local Gradient")

        axes[1][2].imshow(otsu_of_gradient, cmap=plt.cm.gray, interpolation="nearest")
        axes[1][2].set_title(f"Otsu of Gradient\n({np.sum(otsu_of_gradient)} / {mask_size})")

        axes[2][0].imshow(otsu_sum, cmap=plt.cm.tab10, interpolation="nearest")
        axes[2][0].set_title("Sum of Otsus")

        annotations = dict(zip(*np.unique(markers_gc.flatten(), return_counts=True)))
        axes[2][1].imshow(markers_gc, cmap=cmap_grabcut_markers, interpolation="none")
        axes[2][1].set_title(
            "Grabcut Markers: \n"
            f"GC_BGD : {annotations.get(cv2.GC_BGD, 0)}, GC_PR_BGD : {annotations.get(cv2.GC_PR_BGD, 0)}\n"
            f"GC_FGD : {annotations.get(cv2.GC_FGD, 0)}, GC_PR_FGD : {annotations.get(cv2.GC_PR_FGD, 0)}"
        )

    return markers_gc


def get_grabcut_initialisation_v2(
    gray: np.ndarray,
    saturation_image: np.ndarray | None,
    bounding_box: np.ndarray | None,
    axes: np.ndarray | None,
) -> np.ndarray:
    """Generate grabcut initialisation.

    Args:
        gray (np.ndarray): Grayscale tissue image which is cropped to bounded box.
        saturation_image (np.ndarray): Saturation image.
        bounding_box (np.ndarray | None): Bounding box to use
        axes (np.ndarray): axes of image being constructed. Note that this gets modified in the fn.

    Returns:
        np.ndarray: grabcut initialisation figure.
    """
    mask_size = np.prod(gray.shape)
    # Get the gradient (local max - local min) of the gray scale image,
    # high gradient usually indicates tissue (not as good for when larger
    # areas are out of focus or tissue is really smooth potentially
    # affected by low resolution images)
    entropy = skimage.filters.rank.entropy(gray, skimage.morphology.disk(5))

    # Binarize the gradient into two classes 1=FG and 0=BG using Otsu treshold
    inverted_entropy = skimage.util.invert(entropy, signed_float=False)
    otsu_of_entropy = inverted_entropy <= skimage.filters.threshold_otsu(inverted_entropy)
    otsu_of_entropy = skimage.morphology.remove_small_holes(
        otsu_of_entropy, area_threshold=SMALL_OBJECT_GC_FG_ENTROPY_THRESHOLD
    )

    # Detect canny edges on the grayscale image (many edges usually indicate tissue)
    canny_edges = skimage.feature.canny(gray)
    closed_canny = skimage.morphology.closing(canny_edges)
    closed_canny = scipy.ndimage.distance_transform_edt(1 - closed_canny) <= PIXEL_THRESHOLD
    closed_canny = skimage.morphology.remove_small_objects(
        closed_canny, min_size=CANNY_IMAGE_THRESHOLD
    )
    closed_canny = skimage.morphology.remove_small_holes(
        closed_canny, area_threshold=10 * CANNY_IMAGE_THRESHOLD
    )

    sum_of_canny_entropy = np.add(
        closed_canny.astype("uint8"),
        otsu_of_entropy.astype("uint8"),
    )

    # Calculate grayscale intensity Otsu treshold, often good proxy for where tissue is
    # (not as good when background and tissue has similar gray scale histograms)
    transformed_gray = float_image_to_ubyte(gray.astype(int) ** 6)
    otsu_thresh_transformed = transformed_gray <= skimage.filters.threshold_otsu(transformed_gray)
    otsu_thresh_original = gray <= skimage.filters.threshold_otsu(gray)
    if saturation_image is not None:
        saturation_transformed = float_image_to_ubyte(saturation_image.astype(int) ** (1 / 6))
        otsu_saturation_transformed = saturation_transformed >= skimage.filters.threshold_otsu(
            saturation_transformed
        )
        # make sure we did not call everything as foreground in saturation
        if not np.all(otsu_thresh_original | otsu_saturation_transformed):
            otsu_thresh_original = otsu_thresh_original | otsu_saturation_transformed
    otsu_thresh_original = skimage.morphology.remove_small_holes(otsu_thresh_original, 2000)

    # Check if the otsu thresholding on sat/intensity is stumped by non destained region
    # if it is, then use otsu on intensity**6. dont use it normally as a lot of garbage gets
    # chucked as foreground if we use the convex transform.

    # This is the number of mask detected by transformed otsu and by canny or entropy in the
    # slide region in the fiducial frames but not detected by otsu on untransformed intensity.
    if bounding_box is not None:
        residuals = box(
            (otsu_thresh_transformed & ~otsu_thresh_original)
            & (sum_of_canny_entropy > 0).astype("uint8"),
            bounding_box,
        )
        num_pixels_in_ca_gained_by_trns_otsu = residuals.sum()
        num_pixels_in_ca_detected_by_all = box(
            otsu_thresh_original & (sum_of_canny_entropy > 0).astype("uint8"), bounding_box
        ).sum()
        if num_pixels_in_ca_detected_by_all > num_pixels_in_ca_gained_by_trns_otsu:
            otsu_thresh = otsu_thresh_original
        elif not np.all(otsu_thresh_transformed):
            # Dont use the convex tranform if it declared everything as foreground
            otsu_thresh = otsu_thresh_transformed
        else:
            otsu_thresh = otsu_thresh_original
    else:
        otsu_thresh = otsu_thresh_original

    otsu_sum = np.add(
        np.add(otsu_of_entropy.astype("uint8"), otsu_thresh.astype("uint8")),
        closed_canny.astype("uint8"),
    )

    # Start making markers for the grabcut
    markers_gc = np.zeros(gray.shape).astype("uint8")
    # to keep track of not yet classed vs obvious background
    classed = np.zeros(otsu_sum.shape).astype("uint8")

    ##### below is order dependent based on priority, pixels may be assign GC_BGD early and then
    ##### the same pixel may get GC_FGD later

    # If classed as background by both methods add a margin of 2 pixels and
    # set to an obvious background pixels
    background = np.zeros(otsu_sum.shape).astype("uint8")
    background[otsu_sum == BACKGROUND] = 1
    background = scipy.ndimage.distance_transform_edt(background) >= PIXEL_THRESHOLD
    markers_gc[background == 1] = cv2.GC_BGD
    classed[background == 1] += 1

    # If detected by at least one method set to be a possible foreground pixel
    markers_gc[otsu_sum >= DETECTED_BY_ONE_METHOD] = cv2.GC_PR_FGD
    classed[otsu_sum >= DETECTED_BY_ONE_METHOD] += 1

    # If detected by two methods add a margin of 2 pixels (inward) image longest side (in pixels)
    # basically make the estimate smaller by some amount around the boundaries
    # set as an obvious foreground (object) pixel
    foreground = np.zeros(otsu_sum.shape).astype(bool)
    foreground[otsu_sum >= DETECTED_BY_TWO] = True
    foreground = skimage.morphology.remove_small_objects(foreground, SMALL_OBJECT_GC_FG_THRESHOLD)
    foreground = scipy.ndimage.distance_transform_edt(foreground) >= PIXEL_THRESHOLD
    markers_gc[foreground == 1] = cv2.GC_FGD
    classed[foreground == 1] += 1

    # Set any unclassed pixels to be possible background
    markers_gc[classed == 0] = cv2.GC_PR_BGD

    # if statement for switching creation of plots for debugging and evaluation on and off
    if axes is not None:
        axes[0][1].imshow(canny_edges, cmap=plt.cm.gray, interpolation="nearest")
        axes[0][1].set_title("Canny Edges")

        axes[0][2].imshow(otsu_thresh, cmap=plt.cm.gray, interpolation="nearest")
        axes[0][2].set_title(f"Otsu threshold\n({np.sum(otsu_thresh)} / {mask_size})")

        axes[1][0].imshow(closed_canny, cmap=plt.cm.gray, interpolation="nearest")
        axes[1][0].set_title(f"Closed Canny\n({np.sum(closed_canny)} / {mask_size})")

        axes[1][1].imshow(entropy, cmap=plt.cm.gray, interpolation="nearest")
        axes[1][1].set_title("Entropy on grayscale")

        axes[1][2].imshow(otsu_of_entropy, cmap=plt.cm.gray, interpolation="nearest")
        axes[1][2].set_title(f"Otsu of Entropy\n({np.sum(otsu_of_entropy)} / {mask_size})")

        axes[2][0].imshow(otsu_sum, cmap=cmap_otsu_sum_markers, interpolation="nearest")
        axes[2][0].set_title("Sum of Otsus")

        annotations = dict(zip(*np.unique(markers_gc.flatten(), return_counts=True)))
        axes[2][1].imshow(markers_gc, cmap=cmap_grabcut_markers, interpolation="none")
        axes[2][1].set_title(
            "Grabcut Markers: \n"
            f"GC_BGD : {annotations.get(cv2.GC_BGD, 0)}, GC_PR_BGD : {annotations.get(cv2.GC_PR_BGD, 0)}\n"
            f"GC_FGD : {annotations.get(cv2.GC_FGD, 0)}, GC_PR_FGD : {annotations.get(cv2.GC_PR_FGD, 0)}"
        )

    return markers_gc


def get_mask(
    original: np.ndarray,
    bounding_box: np.ndarray = None,
    use_full_fov: bool = False,
    plot: bool = False,
):
    """Segment the image and return binary mask of tissue.

    The get_mask function takes an image (array) in grayscale and uses the opencv grabcut
    algorithm to detect tissue section(s) (foreground) on the glass slide surface (background).
    Markers for initialization of the grabcut algorithm are based on otsu thresholding of the
    grayscale and the gradient (local max-min) of the image. Returns a binary mask (where 1=tissue,
    0=background) as well as a qc image with the mask overlayed on the input image and colored with
    cr_constants.TISSUE_COLOR.

    Args:
        original (np.ndarray): image for tissue segmentation.
        bounding_box (np.ndarray): only segment tissue within the bounding box. Defaults to None.
        use_full_fov (bool): whether to use the whole FoV to do tissue detection.
        plot (bool): whether to return the intermediate qc figure. Defaults to False.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: the binary tissue mask, qc figure,
            and intermediate qc figure
    """
    if len(original.shape) != 2:
        raise RuntimeError(f"non-2D image (color?) passed to get_mask nD={len(original.shape)}")

    cv2.setRNGSeed(0)
    np.random.seed(0)

    if bounding_box is None or use_full_fov:
        height, width = original.shape[:2]
        # order matters below - should trace a bounding box by adjacent edges
        bounding_box_to_use = np.array(
            [(0, 0), (width - 1, 0), (width - 1, height - 1), (0, height - 1)]
        )
    else:
        bounding_box_to_use = bounding_box

    gray = box(original, bounding_box_to_use)
    mask_size = np.prod(gray.shape)

    # if statement for switching creation of plots for debugging and evaluation on and off
    if plot:
        init_qc, axes = plt.subplots(nrows=3, ncols=3, figsize=[15, 15.5])
        axes[0][0].imshow(
            unbox(original, bounding_box_to_use, gray, border_thickness=5),
            cmap="gray",
            interpolation="nearest",
        )
        axes[0][0].set_title("Input Image")
        if bounding_box_to_use is not None:
            bbox = Polygon(bounding_box_to_use, closed=True, alpha=0.25, facecolor="red")
            axes[0][0].add_patch(bbox)
    else:
        init_qc, axes = None, None

    longest_side = max(original.shape[:2])
    markers_gc = get_grabcut_initialisation(gray, longest_side, axes)

    # make the image compatible with the grabcut (color)
    img = cv2.cvtColor(gray.astype("uint8"), cv2.COLOR_GRAY2RGB)

    # run the opencv grabcut
    bgmodel = np.zeros((1, 65), dtype="float64")
    fgmodel = np.zeros((1, 65), dtype="float64")

    mask, bgmodel, fgmodel = cv2.grabCut(
        img, markers_gc, None, bgmodel, fgmodel, 6, cv2.GC_INIT_WITH_MASK
    )
    mask = np.where((mask == 2) | (mask == 0), 0, 1)

    # generate qc image as 24-bit RGB instead of crazy float64 RGB
    qc_img = img
    #    qc = image_util.cv_composite_labelmap(qc, mask, {0:(255,0,0), 1:(0,0,255)}, alpha=0.25)
    tissue_color_rgb = tuple(int(255 * x) for x in colors.to_rgb(cr_constants.TISSUE_COLOR))

    # If we use the full FoV for tissue segmentations, crop out the image and the mask
    if use_full_fov and bounding_box is not None:
        mask = box(mask.astype("uint8"), bounding_box)
        qc_img = box(qc_img, bounding_box)
        bounding_box_to_use = bounding_box
        gc_markers = markers_gc
    else:
        gc_markers = unbox(
            np.zeros_like(original), bounding_box_to_use, markers_gc, interp=cv2.INTER_NEAREST
        )

    qc_img = image_util.cv_composite_labelmap(
        qc_img, mask, {1: tissue_color_rgb}, alpha=cr_constants.TISSUE_NOSPOTS_ALPHA
    )  # try skipping background coloring

    # if enabled add the qc to the debugging/evaluation plot and save/show
    if plot:
        axes[2][2].imshow(qc_img)
        axes[2][2].set_title(f"GrabCut\n({np.sum(mask)} / {mask_size})")

        plt.subplots_adjust(left=None, bottom=0, right=None, top=0.95, wspace=0.1, hspace=0.1)

        # convert canvas to image
        # init_qc.canvas.draw()
        # init_qc = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
        # init_qc = init_qc.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        # img is rgb, convert to opencv's default bgr
        # init_qc = cv2.cvtColor(init_qc, cv2.COLOR_RGB2BGR)
        # plt.close()
    else:
        init_qc = None

    # reembed the mask in the original image shape
    mask = unbox(
        np.zeros(original.shape[:2], dtype=np.uint8),
        bounding_box_to_use,
        mask,
        border_thickness=0,
        interp=cv2.INTER_NEAREST,
    )
    # reembed the qc image in the original
    qc_img = unbox(
        original, bounding_box_to_use, qc_img, border_thickness=2, interp=cv2.INTER_NEAREST
    )
    return (mask, qc_img, init_qc, gc_markers)


def get_mask_v2(
    original: np.ndarray,
    saturation_channel: np.ndarray | None = None,
    bounding_box: np.ndarray | None = None,
    use_full_fov: bool = False,
    plot: bool = False,
):
    """Segment the image and return binary mask of tissue.

    The get_mask function takes an image (array) in grayscale and uses the opencv grabcut
    algorithm to detect tissue section(s) (foreground) on the glass slide surface (background).
    Markers for initialization of the grabcut algorithm are based on otsu thresholding of the
    grayscale and the gradient (local max-min) of the image. Returns a binary mask (where 1=tissue,
    0=background) as well as a qc image with the mask overlayed on the input image and colored with
    cr_constants.TISSUE_COLOR.

    Args:
        original (np.ndarray): image for tissue segmentation.
        saturation_channel (np.ndarray): saturation image.
        bounding_box (np.ndarray): only segment tissue within the bounding box. Defaults to None.
        use_full_fov (bool): whether to use the whole FoV to do tissue detection.
        plot (bool): whether to return the intermediate qc figure. Defaults to False.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: the binary tissue mask, qc figure,
            and intermediate qc figure
    """
    if len(original.shape) != 2:
        raise RuntimeError(f"non-2D image (color?) passed to get_mask nD={len(original.shape)}")

    cv2.setRNGSeed(0)
    np.random.seed(0)

    height, width = original.shape[:2]
    if bounding_box is None or use_full_fov:
        # order matters below - should trace a bounding box by adjacent edges
        bounding_box_to_use = np.array(
            [(0, 0), (width - 1, 0), (width - 1, height - 1), (0, height - 1)]
        )
    else:
        bounding_box_to_use = bounding_box

    gray = box(original, bounding_box_to_use)
    if saturation_channel is not None:
        saturation_channel = box(saturation_channel, bounding_box_to_use)
    mask_size = np.prod(gray.shape)

    # if statement for switching creation of plots for debugging and evaluation on and off
    if plot:
        init_qc, axes = plt.subplots(nrows=3, ncols=3, figsize=[15, 15.5])
        axes[0][0].imshow(
            unbox(original, bounding_box_to_use, gray, border_thickness=5),
            cmap="gray",
            interpolation="nearest",
        )
        axes[0][0].set_title("Input Image")
        if bounding_box_to_use is not None:
            bbox = Polygon(bounding_box_to_use, closed=True, alpha=0.25, facecolor="red")
            axes[0][0].add_patch(bbox)
    else:
        init_qc, axes = None, None

    # longest_side = max(original.shape[:2])
    markers_gc = get_grabcut_initialisation_v2(gray, saturation_channel, bounding_box, axes)

    # make the image compatible with the grabcut (color)
    img = cv2.cvtColor(gray.astype("uint8"), cv2.COLOR_GRAY2RGB)

    # run the opencv grabcut
    bgmodel = np.zeros((1, 65), dtype="float64")
    fgmodel = np.zeros((1, 65), dtype="float64")

    mask, bgmodel, fgmodel = cv2.grabCut(
        img, markers_gc, None, bgmodel, fgmodel, 8, cv2.GC_INIT_WITH_MASK, gamma=1.0
    )
    mask = np.where((mask == 2) | (mask == 0), 0, 1)

    # generate qc image as 24-bit RGB instead of crazy float64 RGB
    qc_img = img
    #    qc = image_util.cv_composite_labelmap(qc, mask, {0:(255,0,0), 1:(0,0,255)}, alpha=0.25)
    tissue_color_rgb = tuple(int(255 * x) for x in colors.to_rgb(cr_constants.TISSUE_COLOR))

    # If we use the full FoV for tissue segmentations, crop out the image and the mask
    if use_full_fov and bounding_box is not None:
        mask = box(mask.astype("uint8"), bounding_box)
        qc_img = box(qc_img, bounding_box)
        bounding_box_to_use = bounding_box
        gc_markers = markers_gc
    else:
        gc_markers = unbox(
            np.zeros_like(original), bounding_box_to_use, markers_gc, interp=cv2.INTER_NEAREST
        )

    qc_img = image_util.cv_composite_labelmap(
        qc_img, mask, {1: tissue_color_rgb}, alpha=cr_constants.TISSUE_NOSPOTS_ALPHA
    )  # try skipping background coloring

    # if enabled add the qc to the debugging/evaluation plot and save/show
    if plot:
        axes[2][2].imshow(qc_img)
        axes[2][2].set_title(f"GrabCut\n({np.sum(mask)} / {mask_size})")

        plt.subplots_adjust(left=None, bottom=0, right=None, top=0.95, wspace=0.1, hspace=0.1)

        # convert canvas to image
        # init_qc.canvas.draw()
        # init_qc = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
        # init_qc = init_qc.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        # img is rgb, convert to opencv's default bgr
        # init_qc = cv2.cvtColor(init_qc, cv2.COLOR_RGB2BGR)
        # plt.close()
    else:
        init_qc = None

    # reembed the mask in the original image shape
    mask = unbox(
        np.zeros(original.shape[:2], dtype=np.uint8),
        bounding_box_to_use,
        mask,
        border_thickness=0,
        interp=cv2.INTER_NEAREST,
    )
    # reembed the qc image in the original
    qc_img = unbox(
        original, bounding_box_to_use, qc_img, border_thickness=2, interp=cv2.INTER_NEAREST
    )
    return (mask, qc_img, init_qc, gc_markers)


def get_bounding_box(pts_xy: np.ndarray, padding: float) -> list[tuple[float, float]]:
    """Generate bounding box that includes all the points.

    Generates a rectangular bounding box based on a set of points WITHOUT the assumption that
    the box is axis-aligned.  The box is padded on each side by the amount padding specified in
    original image units. Returns the (x,y) coordinates of four corners.
    Makes heavy use of OpenCV convenience functions.

    Args:
        pts_xy (np.ndarray): (n, 2) shape. (x, y) coordinates of each point
        padding (float): The padding to be used (generally point diameter)

    Returns:
        List[Tuple[float, float]]: box coordinates [(x1,y1),...,(x4,y4)]
    """
    # compute the "rotated rectangle" surrounding the (padded) array - no assumption of axis alignment
    rrect = cv2.minAreaRect(pts_xy.astype(int))
    rrect = _pad_rotated_rectangle(rrect, padding)

    # compute the corners of the rotated rectangle
    #
    # Given points 0, 1, 2, 3, these define four sides 0:1, 1:2, 2:3, and 3:0.
    # OpenCV code guarantees that 0:1 is opposite 2:3, 1:2 is opposite 3:0, and
    # 0:1 is adjacent to 1:2.  This implies that the lengths defined by 0:1 and 1:2
    # give the two side lengths of the rectangle.  Also, drawing the sides "in order"
    # will trace out a continuous contour.
    sbox = np.round(cv2.boxPoints(rrect))
    return sbox


def get_manual_qc_image(image, bounding_box=None):
    """Prepare a QC image to indicate where the user specified tissue vs background,.

    instead of doing automatic tissue detection.
    """
    if bounding_box:
        y_pos, x_pos, height, width = bounding_box
    else:
        y_pos = 0
        x_pos = 0
        height, width = image.shape[:2]

    original = image
    image = original[y_pos : y_pos + height, x_pos : x_pos + width]

    # just show the border
    qc_img = unbox(original, [y_pos, x_pos, height, width], image, border_thickness=2)
    return qc_img


def _get_bbox_transform(bounding_box):
    len1 = np.linalg.norm(bounding_box[1] - bounding_box[0])
    len2 = np.linalg.norm(bounding_box[2] - bounding_box[1])

    cols, rows = int(len1 + 0.5), int(len2 + 0.5)
    srcpts = np.float32(bounding_box[:3])
    dstpts = np.float32(np.array([[0, 0], [cols - 1, 0], [cols - 1, rows - 1]]))

    trans = cv2.getAffineTransform(src=srcpts, dst=dstpts)

    return trans, cols, rows


def box(original, bounding_box, interp=cv2.INTER_LINEAR):
    """Given a bounding box comprising coordinates of vertices (NOT x,y,w,h), extract a possibly rotated.

    rectangle from the original image and return the subimage
    """
    trans, cols, rows = _get_bbox_transform(bounding_box)
    return cv2.warpAffine(
        original, trans, (cols, rows), borderMode=cv2.BORDER_REPLICATE, flags=interp
    )


def _unbox1(original, bounding_box, boxed_image, interp=cv2.INTER_LINEAR):
    sentinel = np.iinfo(boxed_image.dtype).max
    img_max = np.max(boxed_image)
    if img_max == sentinel:
        cv2.normalize(boxed_image, boxed_image, sentinel - 1, 0, cv2.NORM_MINMAX)

    trans, _, _ = _get_bbox_transform(bounding_box)
    output = cv2.warpAffine(
        src=boxed_image,
        M=trans,
        flags=cv2.WARP_INVERSE_MAP | cv2.BORDER_CONSTANT | interp,
        dsize=(original.shape[1], original.shape[0]),
        borderValue=sentinel,
    )
    output[output == sentinel] = original[output == sentinel]
    return output


def unbox(original, bounding_box, boxed_image, border_thickness=None, interp=cv2.INTER_LINEAR):
    """Function for re-embedding a cropped bounding boxed image back in the original image (or shape).

    If border thickness is > 0, then we return a color version of what otherwise MUST be grayscale inputs.
    """
    if len(boxed_image.shape) == 3 and len(original.shape) == 2:
        original = cv2.cvtColor(original, cv2.COLOR_GRAY2RGB)
        output = np.zeros(original.shape, dtype="uint8")
        for i in range(3):
            output[:, :, i] = _unbox1(original[:, :, i], bounding_box, boxed_image[:, :, i], interp)
    else:
        output = _unbox1(original, bounding_box, boxed_image, interp)

    # add a blue border
    if border_thickness:
        if border_thickness < 0:
            raise ValueError("negative border thickness passed to unbox")

        if len(output.shape) == 2:
            output = cv2.cvtColor(output, cv2.COLOR_GRAY2RGB)

        cv2.drawContours(
            output, [np.intp(bounding_box)], 0, color=cr_constants.TISSUE_BBOX_COLOR, thickness=2
        )

    return output


def _pad_rotated_rectangle(rrect, padding):
    """Given an opencv rotated rectangle, add "padding" pixels to all sides and return a reconstructed.

    RotatedRect
    """
    (centerx, centery), (width, height), angle = rrect
    width += 2 * padding
    height += 2 * padding

    return ((centerx, centery), (width, height), angle)
