#!/usr/bin/env python3
#
# Copyright (c) 2022 10X Genomics, Inc. All rights reserved.
#

"""Stage to generate images used by following imaging stages."""

from __future__ import annotations

import os
import shutil
from typing import NamedTuple

import cv2
import martian
import numpy as np

import cellranger.feature.utils as feature_utils
import cellranger.spatial.cytassist_img_preprocess as cyt_img_prep
import cellranger.spatial.data_utils as data_utils
import cellranger.spatial.image_util as image_util
import cellranger.spatial.tiffer as tiffer
from cellranger.spatial.dark_image_contrast import enhance_immunoflourescence_contrast
from cellranger.spatial.data_utils import HIRES_MAX_DIM_DEFAULT, HIRES_MAX_DIM_DICT, LORES_MAX_DIM
from cellranger.spatial.loupe_util import get_remove_image_pages
from cellranger.spatial.pipeline_mode import PipelineMode, Product

__MRO__ = """
stage STANDARDIZE_IMAGES(
    in  PipelineMode pipeline_mode,
    in  file[]       tissue_image_paths,
    in  file[]       cytassist_image_paths,
    in  int          dark_images,
    in  int          dapi_channel_index,
    in  path         loupe_alignment_file,
    in  bool         read_metadata_from_tissue_image,
    in  float        tissue_image_pixel_size_in,
    out json         scalefactors_json,
    out json         crop_info_json,
    out png          fiducials_detection_image,
    out png          tissue_detection_grayscale_image,
    out png          tissue_detection_saturation_image,
    out png          qc_cytassist_crop_image,
    out tiff         registration_target_image,
    out tiff         cytassist_image,
    out bool         skip_tissue_registration,
    out float        tissue_image_pixel_size,
    out png          tissue_hires_image,
    out png          tissue_lowres_image,
    out file[]       cloupe_display_image_paths,
    src py           "stages/spatial/standardize_images",
) split (
) using (
    mem_gb   = 3,
    vmem_gb  = 64,
    volatile = strict,
)
"""


REGIST_TARGET_IMAGE_SCALE_KEY = "regist_target_img_scalef"


class OutputImagePath(NamedTuple):
    """Path of intermediate images to save."""

    fiducials_detection_image: str
    tissue_detection_grayscale_image: str
    tissue_detection_saturation_image: str | None
    registration_target_image: str
    cytassist_image: str
    tissue_hires_image: str
    tissue_lowres_image: str
    qc_cytassist_crop_image: str


def main(_args, _outs):
    martian.throw("main is not supposed to run.")


def split(args):
    # estimate downsampling bytes
    num_tissue_gb = 1  # consider cytassist images and cropping algorithm
    if args.tissue_image_paths is not None and len(args.tissue_image_paths) > 0:
        num_tissue_gb += tiffer.call_tiffer_mem_estimate_gb(args.tissue_image_paths[0], 4000)
    if args.cytassist_image_paths:
        for path in args.cytassist_image_paths:
            num_tissue_gb += tiffer.call_tiffer_mem_estimate_gb(path, 4000)

    # add 3GB for overhead
    num_gbs = num_tissue_gb + 4
    return {
        "chunks": [],
        "join": {
            "__mem_gb": round(num_gbs, 3),
            "__vmem_gb": max(2 * num_gbs + 6, 64),
        },  # martian default is +3
    }


def join(args, outs, _chunk_defs, _chunk_outs):
    print(f"outs={outs}")
    if args.cytassist_image_paths is None or len(args.cytassist_image_paths) != 1:
        outs.tissue_detection_saturation_image = None
    output_path = OutputImagePath(
        fiducials_detection_image=outs.fiducials_detection_image,
        tissue_detection_grayscale_image=outs.tissue_detection_grayscale_image,
        tissue_detection_saturation_image=outs.tissue_detection_saturation_image,
        registration_target_image=outs.registration_target_image,
        cytassist_image=outs.cytassist_image,
        tissue_hires_image=outs.tissue_hires_image,
        tissue_lowres_image=outs.tissue_lowres_image,
        qc_cytassist_crop_image=outs.qc_cytassist_crop_image,
    )
    pipeline_mode = PipelineMode(**args.pipeline_mode)
    try:
        pipeline_mode.validate()
    except ValueError:
        martian.throw(f"Invalid pipeline mode of {pipeline_mode}")
    scalefactors_dict, crop_info_dict = standardize_images(
        pipeline_mode,
        args.tissue_image_paths,
        args.dark_images,
        args.cytassist_image_paths,
        args.dapi_channel_index,
        args.loupe_alignment_file,
        output_path,
        read_metadata_from_tissue_image=args.read_metadata_from_tissue_image,
    )

    scalef = scalefactors_dict.get(REGIST_TARGET_IMAGE_SCALE_KEY)
    outs.tissue_image_pixel_size = None
    if args.tissue_image_pixel_size_in is not None and scalef is not None:
        outs.tissue_image_pixel_size = args.tissue_image_pixel_size_in / scalef
    print(f"Reg image scale factor: {scalef} tissue pixel size out: {outs.tissue_image_pixel_size}")

    feature_utils.write_json_from_dict(scalefactors_dict, outs.scalefactors_json)
    feature_utils.write_json_from_dict(crop_info_dict, outs.crop_info_json)
    if args.tissue_image_paths:
        outs.cloupe_display_image_paths = args.tissue_image_paths
    else:
        outs.cloupe_display_image_paths = args.cytassist_image_paths
    if pipeline_mode.product == Product.CYT and args.tissue_image_paths:
        outs.skip_tissue_registration = False
    else:
        outs.skip_tissue_registration = True


def combine_multiple_channels(ds_images, target_path):
    """Take multiple channels as individual grayscale images and combine into a.

    single color image
    """
    # BGR: green, red, blue, yellow, orange, purple
    #      yellow-green, cyan, magenta
    colors = [
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 1.0),
        (0.0, 0.5, 1.0),
        (1.0, 0.0, 0.5),
        (0.0, 1.0, 0.5),
        (1.0, 1.0, 0.0),
        (1.0, 0.0, 1.0),
    ]

    rgb_float = None  # type: np.ndarray
    rgb_norm = [0.0, 0.0, 0.0]
    for index, path in enumerate(ds_images):
        # This makes sure that the image read in 8-bit irrespective of
        # if the input image is 8-bit or 16-bit
        input_image = image_util.cv_read_image_standard(path)
        row, col = input_image.shape
        if rgb_float is None:
            rgb_float = np.zeros((row, col, 3), np.float32)
        else:
            height, width, _ = rgb_float.shape
            if row != height or col != width:
                martian.throw(
                    f"image channels are not the same shape: {height},{width} vs {path} is {row},{col}"
                )

        color = colors[index % len(colors)]
        rgb_norm[0] += color[0]
        rgb_norm[1] += color[1]
        rgb_norm[2] += color[2]
        rgb_float[:, :, 0] += color[0] * input_image[:, :]
        rgb_float[:, :, 1] += color[1] * input_image[:, :]
        rgb_float[:, :, 2] += color[2] * input_image[:, :]

    assert rgb_float is not None
    if rgb_norm[0] > 0.0:
        rgb_float[:, :, 0] /= rgb_norm[0]
    if rgb_norm[1] > 0.0:
        rgb_float[:, :, 1] /= rgb_norm[1]
    if rgb_norm[2] > 0.0:
        rgb_float[:, :, 2] /= rgb_norm[2]

    # This uses the fact that rgb_float is an image with values in each channel in the range [0, 255]
    # That is enforced by the fact that we read images through image_util.cv_read_image_standard and
    # the code above that makes sure that every channel has same range as input range
    rgb_contrast_enhanced = enhance_immunoflourescence_contrast(rgb_float.astype(np.uint8))

    cv2.imwrite(target_path, rgb_contrast_enhanced)


def _check_sizes(sizes, tmp_json):
    """Checks json response from tiffer against width and height pair in sizes."""
    width = int(tmp_json["width"])
    height = int(tmp_json["height"])
    if sizes is None:
        return (width, height)
    elif sizes[0] != width or sizes[1] != height:
        martian.throw(
            f"input images/pages have inconsistent sizes ({sizes[0]},{sizes[1]}) vs ({width},{height})"
        )

    return sizes


def multi_page_downsample(input_files, skip_pages, target_size, target_path):
    """Downsample multi-channel images."""
    npages = tiffer.call_tiffer_get_num_pages(input_files[0])
    ds_images = []
    ds_scalef = None
    target_basename, target_ext = os.path.splitext(target_path)

    if len(input_files) == 1 and npages > 1:
        # downsample and combine multiple single-channel images
        sizes = None
        for page in range(npages):
            if page not in skip_pages:
                tmp_json = tiffer.call_tiffer_resample(
                    input_files[0],
                    target_size,
                    target_basename + "_" + str(page) + target_ext,
                    page=page,
                )
                ds_images.append(tmp_json["outpath"])
                ds_scalef = tmp_json["scale"]
                _check_sizes(sizes, tmp_json)
    else:
        sizes = None
        for index, image in enumerate(input_files):
            tmp_json = tiffer.call_tiffer_resample(
                image, target_size, target_basename + "_" + str(index) + target_ext
            )
            ds_images.append(tmp_json["outpath"])
            ds_scalef = tmp_json["scale"]
            _check_sizes(sizes, tmp_json)

    if len(ds_images) > 1:
        combine_multiple_channels(ds_images, target_path)
    else:
        shutil.copy(ds_images[0], target_path)

    return ds_scalef


def convert_color_image_to_inverse_grayscale(inpath, outpath):
    inimage = image_util.cv_read_image_standard(inpath)
    outimage = ~inimage
    cv2.imwrite(outpath, outimage)


def dark_image_downsample_all(dark_image_paths, loupe_alignment_file, hires_info, lowres_info):
    """Main handler for IHC/IF images: optimally combining and downsampling."""
    skip_pages = get_remove_image_pages(loupe_alignment_file)

    hr_scalef = multi_page_downsample(dark_image_paths, skip_pages, hires_info[0], hires_info[1])
    lr_scalef = multi_page_downsample(dark_image_paths, skip_pages, lowres_info[0], lowres_info[1])

    return hr_scalef, lr_scalef


def prepare_visualization_images(
    tissue_image_paths: list[str],
    hires_img_path: str,
    lowres_img_path: str,
    dark_images: int,
    hires_max_dim: int,
    lowres_max_dim: int,
    loupe_alignment_file: str,
) -> tuple[float, float]:
    """Save the high and low res image for visualization.

    Args:
        tissue_image_paths (List[str]): list of tissue image paths
        hires_img_path (str): path to save the hires image
        lowres_img_path (str): path to save the low res image
        dark_images (int): whether the image is dark image
        hires_max_dim (int): maximum dimension of the hi res image
        lowres_max_dim (int): maximum dimension of the low res image
        loupe_alignment_file (str): path to the loupe alignment file

    Returns:
        Tuple[float, float]: [description]
    """
    if dark_images != data_utils.DARK_IMAGES_CHANNELS:
        # brightfield or pre-colorized image path
        hires_img_dict = tiffer.call_tiffer_resample(
            tissue_image_paths[0], hires_max_dim, hires_img_path
        )
        lowres_img_dict = tiffer.call_tiffer_resample(
            tissue_image_paths[0], lowres_max_dim, lowres_img_path
        )
        hr_scalef = hires_img_dict["scale"]
        lr_scalef = lowres_img_dict["scale"]
    else:
        # dark image (e.g. IHC/IF) path
        hi_res_info = (hires_max_dim, hires_img_path)
        low_res_info = (lowres_max_dim, lowres_img_path)
        hr_scalef, lr_scalef = dark_image_downsample_all(
            tissue_image_paths,
            loupe_alignment_file,
            hi_res_info,
            low_res_info,
        )
    return hr_scalef, lr_scalef


def standardize_images(
    pipeline_mode: PipelineMode,
    tissue_image_paths: list[str],
    dark_images: int,
    cytassist_image_paths: list[str],
    dapi_channel_index: int,
    loupe_alignment_file: str,
    output_path: OutputImagePath,
    read_metadata_from_tissue_image: bool | None = False,
) -> tuple[dict, dict]:
    """Read tissue images, and return their downsampled copies.

    This function is responsible to create:

    Visualization images:
    - Two images for display with maximum size of `HIRES_MAX_DIM` and `LORES_MAX_DIM`

    Process images:
    - One image for tissue detection
    - One image for fiducial detection and registration
    - (Optional) One image as the targeted image of tissue registration with maximum size
        of `regist_target_im_max_dim`

    Scale factors corresponding to these images are returned.

    The process images are the same as the visualization images for non-CytAssist product.
    """
    # this check can also be moved to determine_pipeline_mode
    if tissue_image_paths is not None:
        if dark_images != data_utils.DARK_IMAGES_CHANNELS and len(tissue_image_paths) > 1:
            martian.throw("Pipeline must be called with only one image in brightfield mode.")

    vis_hires_dim = HIRES_MAX_DIM_DICT.get(pipeline_mode, HIRES_MAX_DIM_DEFAULT)
    # Dictionary to store scale information and crop information output
    scalefactors = {}
    crop_info_dict = {}
    if pipeline_mode.product in (Product.VISIUM, Product.VISIUM_HD_NOCYT_PD):
        hr_scalef, lr_scalef = prepare_visualization_images(
            tissue_image_paths,
            output_path.tissue_hires_image,
            output_path.tissue_lowres_image,
            dark_images,
            vis_hires_dim,
            LORES_MAX_DIM,
            loupe_alignment_file,
        )
        # Non-cytassist assume hi-res visualization image is the same as the process image
        if dark_images != data_utils.DARK_IMAGES_CHANNELS:
            shutil.copy(output_path.tissue_hires_image, output_path.fiducials_detection_image)
        else:
            # invert the dark image for fiducial detection/registration
            convert_color_image_to_inverse_grayscale(
                output_path.tissue_hires_image, output_path.fiducials_detection_image
            )
        # TODO (dongyao): check if the final size is large enough if tiffer do not generate error
        shutil.copy(output_path.tissue_hires_image, output_path.tissue_detection_grayscale_image)
        process_img_scalef = hr_scalef

    elif pipeline_mode.product == Product.CYT:
        crop_info_dict = cyt_img_prep.prepare_cytassist_img(
            cytassist_image_paths=cytassist_image_paths,
            fid_detect_img_path=output_path.fiducials_detection_image,
            tissue_detect_grayscale_img_path=output_path.tissue_detection_grayscale_image,
            tissue_detect_saturation_img_path=output_path.tissue_detection_saturation_image,
            out_cyt_img_path=output_path.cytassist_image,
            qc_crop_img_path=output_path.qc_cytassist_crop_image,
            is_visium_hd=pipeline_mode.is_visium_hd_with_fiducials(),
        )
        process_img_scalef = 1.0
        if tissue_image_paths:
            hr_scalef, lr_scalef = prepare_visualization_images(
                tissue_image_paths,
                output_path.tissue_hires_image,
                output_path.tissue_lowres_image,
                dark_images,
                vis_hires_dim,
                LORES_MAX_DIM,
                loupe_alignment_file,
            )
            skip_pages = get_remove_image_pages(loupe_alignment_file)
            # the dapi_channel_index is 1-based
            selected_pages = [dapi_channel_index - 1] if dapi_channel_index else []
            regist_target_scalef = image_util.prepare_registration_target_image(
                tissue_image_paths,
                dark_images,
                image_util.REGIST_TARGET_IMAGE_MAX_DIM,
                output_path.registration_target_image,
                skip_pages,
                selected_pages=selected_pages,
                read_metadata_from_tissue_image=read_metadata_from_tissue_image,
            )
            scalefactors[REGIST_TARGET_IMAGE_SCALE_KEY] = float(regist_target_scalef)
        else:
            # Use Cytassist image for visualization if tissue image doesn't exist
            hr_scalef, lr_scalef = prepare_visualization_images(
                (
                    cytassist_image_paths
                    if len(cytassist_image_paths) == 1
                    else [cytassist_image_paths[1]]
                ),
                output_path.tissue_hires_image,
                output_path.tissue_lowres_image,
                data_utils.DARK_IMAGES_NONE,
                vis_hires_dim,
                LORES_MAX_DIM,
                loupe_alignment_file,
            )
            scalefactors[REGIST_TARGET_IMAGE_SCALE_KEY] = 1.0
    else:
        raise ValueError(f"Unsupported product mode: {pipeline_mode.product}")

    scalefactors["process_img_scalef"] = float(process_img_scalef)
    scalefactors["tissue_hires_scalef"] = float(hr_scalef)
    scalefactors["tissue_lowres_scalef"] = float(lr_scalef)
    return scalefactors, crop_info_dict
