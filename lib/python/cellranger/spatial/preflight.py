# Copyright (c) 2019 10X Genomics, Inc. All rights reserved.
"""Preflight support routines for spatial analyses."""
from __future__ import annotations

import os
from pathlib import Path
from typing import NamedTuple

import h5py
import martian

import cellranger.spatial.slide as slide
from cellranger.chemistry import CUSTOM_CHEMISTRY_NAME
from cellranger.preflight import PreflightException
from cellranger.spatial.cytassist_constants import (
    CYTA_HD_IMAGE_WIDTH,
    CYTA_IMAGE_ALLOWED_HEIGHTS,
    CYTA_IMAGE_ALLOWED_WIDTHS,
    CYTA_IMAGE_DIM,
)
from cellranger.spatial.data_utils import (
    CYTASSIST_TIFF_CAPTURE_AREA_TO_PIPELINE_CAPTURE_AREA,
    DARK_IMAGES_CHANNELS,
    DARK_IMAGES_COLORIZED,
    DARK_IMAGES_NONE,
    DISALLOWED_DARK_IMAGES_EXTENSION,
    SLIDE_ID_EXCEPTIONS,
    parse_slide_sample_area_id,
)
from cellranger.spatial.image_util import REGIST_TARGET_IMAGE_MAX_DIM
from cellranger.spatial.loupe_util import LoupeParser, get_remove_image_pages
from cellranger.spatial.tiffer import call_tiffer_checksum, call_tiffer_info
from cellranger.spatial.valis.registration import (
    FM_CANONICAL_PIXEL_SIZE,
    MAXIMUM_TISSUE_IMAGE_SCALING,
)

VISIUM_HD_CHEMISTRIES = ["SPATIAL-HD-v1"]


def is_hd_upload(chemistry: str, custom_chemistry_def: dict | None) -> bool:
    """Checks if an upload is a HD run."""
    if chemistry in VISIUM_HD_CHEMISTRIES:
        return True
    elif chemistry == CUSTOM_CHEMISTRY_NAME and custom_chemistry_def is not None:
        return any(
            barcode.get("whitelist", {}).get("slide") is not None
            for barcode in custom_chemistry_def["barcode"]
        )
    else:
        return False


def check_spatial_image_paths(
    tissue_image_paths: list[str],
    cytassist_image_paths: list[str],
    dark_images,
):
    """Check for the existence of GPR files if given to the pipeline."""
    if tissue_image_paths is None or len(tissue_image_paths) == 0:
        if not cytassist_image_paths:
            raise PreflightException("No images specified!")
    else:
        if len(tissue_image_paths) > 1 and dark_images != DARK_IMAGES_CHANNELS:
            raise PreflightException("Can only specify multiple images in dark-image mode")

        for path in tissue_image_paths:
            if not os.path.isfile(path):
                raise PreflightException(f"The image file {path} does not exist")

    if cytassist_image_paths:
        for path in cytassist_image_paths:
            if not os.path.isfile(path):
                raise PreflightException(f"The CytAssist image file {path} does not exist")


def check_pattern_arguments(
    v1_pattern_fix,
):
    """Check special support arguments."""
    if v1_pattern_fix:
        v1_filtered_fbm = v1_pattern_fix.get("v1_filtered_fbm")
        v1_pattern_type = v1_pattern_fix.get("v1_pattern_type")
        if not os.path.isfile(v1_filtered_fbm):
            raise PreflightException(f"The v1_filtered_fbm input {v1_filtered_fbm} does not exist")
        if v1_pattern_type not in [1, 2]:
            raise PreflightException(
                "The v1_pattern_fix value must be either 1 or 2 - Please consult support@10xgenomics.com"
            )
        try:
            with h5py.File(v1_filtered_fbm) as f:
                ffbm_attrs = set(
                    [
                        "chemistry_description",
                        "filetype",
                        "library_ids",
                        "original_gem_groups",
                        "software_version",
                        "version",
                    ]
                )
                if not ffbm_attrs <= set(f.attrs):
                    raise PreflightException(f"v1_filtered_fbm is invalid: {v1_filtered_fbm}")
        except OSError as ex:
            raise PreflightException(f"v1_filtered_fbm is invalid: {v1_filtered_fbm}") from ex


def check_spatial_arguments(
    loupe_alignment_file,
    gpr_file,
    hd_layout_file,
    slide_serial_capture_area,
    is_hd_run: bool,
    hd_log_umi_image,
    is_pd: bool,
):
    """Check for the existence of GPR files if given to the pipeline."""
    if loupe_alignment_file is not None:
        if not os.path.isfile(loupe_alignment_file):
            raise PreflightException(
                f"The loupe alignment file {loupe_alignment_file} does not exist"
            )
        else:
            try:
                loupe_data = LoupeParser(loupe_alignment_file)
                if not (
                    is_hd_run or loupe_data.oligos_preselected() or loupe_data.contain_cyta_info()
                ):
                    raise PreflightException(
                        "No spots were selected as 'in-tissue' in the Loupe alignment file."
                    )
            except OSError as ex:  # file is unreadable
                raise PreflightException(
                    f"Cannot read Loupe alignment file: {loupe_alignment_file}"
                ) from ex
            except (
                ValueError,
                KeyError,
                AssertionError,
            ) as ex:  # file is not a json file or is missing required keys
                raise PreflightException(
                    f"Loupe alignment file is invalid: {loupe_alignment_file}"
                ) from ex

    if gpr_file is not None:
        if not os.path.exists(gpr_file):
            raise PreflightException(f"The slide file {gpr_file} does not exist")

    if is_hd_run:
        if is_pd and (
            not slide_serial_capture_area and not hd_layout_file and not hd_log_umi_image
        ):
            raise PreflightException(
                "HD run with no slide_and_well / slide_serial_capture_area / hd_layout_file specified."
            )

    if hd_layout_file is not None:
        if not os.path.exists(hd_layout_file):
            raise PreflightException(f"The slide layout file {hd_layout_file} does not exist")

    if gpr_file is not None and hd_layout_file is not None:
        raise PreflightException("Cannot provide a gpr and an hd layout file at the same time")

    _check_slide_capture_area_ids(
        slide_serial_capture_area, loupe_alignment_file, gpr_file, hd_layout_file, is_hd_run, is_pd
    )


def _check_slide_capture_area_ids(
    slide_serial_capture_area, loupe_alignment_file, gpr_file, hd_layout_file, is_hd_run, is_pd
):
    """Check that slide_serial_capture_area from command line arguments matches loupe_alignment_file.

    and/or gpr_file if provided and/or hd_layout_file provided.
    """
    cmdline_slide_id, cmdline_area_id = None, None
    alignment_file_slide_id, alignment_file_area_id = None, None
    gpr_slide_id = None

    if slide_serial_capture_area:
        cmdline_slide_id, cmdline_area_id = parse_slide_sample_area_id(slide_serial_capture_area)

    if is_pd and cmdline_slide_id in SLIDE_ID_EXCEPTIONS:
        cmdline_slide_id = None
        cmdline_area_id = None

    if loupe_alignment_file is not None:
        loupe_data = LoupeParser(json_path=loupe_alignment_file)
        if loupe_data.has_hd_slide() != is_hd_run:
            if is_hd_run:
                raise PreflightException(
                    "Loupe alignment file is not Visium HD compatible, "
                    "but this is a Visium HD run."
                )
            else:
                raise PreflightException(
                    "Loupe alignment file seems to be from a Visium HD slide, "
                    "but this is a Visium SD run."
                )
        cytassist_only_non_hd_file = all(
            (not is_hd_run, loupe_data.contain_cyta_info(), not loupe_data.contain_fiducial_info())
        )
        if loupe_data.has_serial_number():
            alignment_file_slide_id = loupe_data.get_serial_number()
        else:
            raise PreflightException(
                f"Loupe alignment file {loupe_alignment_file} may be truncated or malformed as it's missing a slide serial number field"
            )

        if loupe_data.has_area_id():
            alignment_file_area_id = loupe_data.get_area_id()
        else:
            raise PreflightException(
                f"Loupe alignment file {loupe_alignment_file} may be truncated or malformed as it's missing a slide area field"
            )
        if not cytassist_only_non_hd_file and (
            cmdline_slide_id != alignment_file_slide_id or cmdline_area_id != alignment_file_area_id
        ):
            raise PreflightException(
                "You provided {}, but the loupe alignment file {}".format(
                    (
                        "--unknown-slide"
                        if not cmdline_slide_id and not cmdline_area_id
                        else f"--slide={cmdline_slide_id} --area={cmdline_area_id}"
                    ),
                    (
                        f"says {alignment_file_slide_id}-{alignment_file_area_id}"
                        if alignment_file_slide_id is not None or alignment_file_area_id is not None
                        else "was generated without a serial number"
                    ),
                )
            )

    if gpr_file is not None:
        # calling gpr reader
        gpr_data = slide.call_gprreader_info_gpr(gpr_file)
        gpr_slide_id = gpr_data["barcode"]
        if not gpr_slide_id:  # make sure empty strings are interpreted as None
            gpr_slide_id = None

        if cmdline_slide_id != gpr_slide_id:
            raise PreflightException(
                "You provided {}, but the slide file {}".format(
                    (
                        "--unknown-slide"
                        if not cmdline_slide_id and not cmdline_area_id
                        else f"--slide={cmdline_slide_id}"
                    ),
                    f"says {gpr_slide_id}" if gpr_slide_id is not None else "is empty or malformed",
                )
            )

        capture_areas = slide.get_capture_areas_from_gpr_info_json(gpr_data)
        if cmdline_area_id not in capture_areas:
            raise PreflightException(
                "You provided a capture area {} for slide ID {}, but this slide design"
                " only has capture areas {}.".format(
                    cmdline_area_id, cmdline_slide_id, ",".join(capture_areas)
                )
            )

    if hd_layout_file is not None:
        # call the hd layout reader
        hd_layout_data = slide.call_hd_layout_reader_info(hd_layout_file)
        hd_layout_slide_id = hd_layout_data["slide_uid"]
        if cmdline_slide_id != hd_layout_slide_id:
            user_input = (
                "--unknown-slide"
                if not cmdline_slide_id and not cmdline_area_id
                else f"--slide={cmdline_slide_id}"
            )
            use_slide_id = (
                f"says {hd_layout_slide_id}"
                if hd_layout_slide_id is not None
                else "is empty or malformed"
            )
            raise PreflightException(
                f"You provided {user_input}, but the slide file {use_slide_id}"
            )

        capture_areas = slide.get_capture_areas_from_hd_layout_info_json(hd_layout_data)
        if cmdline_area_id not in capture_areas:
            raise PreflightException(
                "You provided a capture area {} for slide ID {}, but this slide design"
                " only has capture areas {}.".format(
                    cmdline_area_id, cmdline_slide_id, ",".join(capture_areas)
                )
            )

    # TODO: special case this for SD only and have HD specific checks
    elif cmdline_slide_id is not None:
        whitelist_name = slide.get_whitelist_for_slide_id(cmdline_slide_id)
        capture_areas = slide.get_capture_areas_for_whitelist(whitelist_name)
        if cmdline_area_id not in capture_areas:
            raise PreflightException(
                "You provided a capture area {} for slide ID {}, but this slide design"
                " only has capture areas {}.".format(
                    cmdline_area_id, cmdline_slide_id, ",".join(capture_areas)
                )
            )


def check_images_exist(
    tissue_image_paths_str: list[str], dark_images: int, loupe_alignment_file: str | None = None
):
    """Check that input images and loupe alignment file exist and are the right extension.

    Args:
        tissue_image_paths_str (List[str]): List of image path given by the rust cmdline
        dark_images (int): Value that determines what type of dark image(s) is/are provided.
        loupe_alignment_file (Optional[str], optional): Loupe file if manual alignment has been performed. Defaults to None.

    Raises:
        PreflightException: Any preflight specific case linked to images
    """
    if loupe_alignment_file is not None and not os.path.exists(loupe_alignment_file):
        raise PreflightException(
            f"Loupe alignment file specified, but doesn't exist: {loupe_alignment_file}"
        )

    bad_image_paths = []
    tissue_image_paths = [Path(path) for path in tissue_image_paths_str]
    for path in tissue_image_paths:
        if not path.exists():
            bad_image_paths.append(path)
    if len(bad_image_paths) > 0:
        if (
            dark_images != DARK_IMAGES_CHANNELS
            and len(bad_image_paths) == 1
            and "," in bad_image_paths[0]
        ):
            raise PreflightException(
                f"Failed to load a single image named {bad_image_paths[0]}\nNote that multiple images cannot be used with --image and --colorizedimage."
            )
        raise PreflightException(
            "Some image paths could not be read:\n{}".format("\n".join(bad_image_paths))
        )
    if dark_images != DARK_IMAGES_NONE:
        for path in tissue_image_paths:
            if path.suffix in DISALLOWED_DARK_IMAGES_EXTENSION:
                raise PreflightException(
                    f"The image input {path} is a PNG image, which is not compatible with the --darkimage option."
                )


def check_images_consistent(tissue_image_paths):
    """Check that we don't mix image formats."""
    canonical_extension = {".tiff": ".tif", ".tif": ".tif", ".jpeg": ".jpg", ".jpg": ".jpg"}
    formats = set()
    for path in tissue_image_paths:
        _, ext = os.path.splitext(path.lower())
        ext = canonical_extension.get(ext, ext)
        formats.add(ext)
    if len(formats) > 1:
        raise PreflightException(
            "Input images cannot mix image types: {}".format(", ".join(sorted(formats)))
        )


def _check_image_checksums(
    tissue_image_paths, cytassist_image_paths: list[str], loupe_alignment_file
):
    """Check that any loupe alignment file matches at least one image file.

    Compares checksums.  Also, ensure that no image files are repeated.
    """
    fiducial_image_checksum = None
    regist_target_checksum = None
    if loupe_alignment_file is not None:
        try:
            loupe_data = LoupeParser(json_path=loupe_alignment_file)
        except OSError as ex:
            raise PreflightException(
                f"Unable to open loupe alignment file: {loupe_alignment_file}\nError {ex.errno}: {ex.strerror}"
            ) from ex
        except ValueError as ex:
            raise PreflightException(
                f"Loupe alignment file {loupe_alignment_file} was not a correctly formatted file"
            ) from ex

        fiducial_image_checksum = loupe_data.fiducial_image_checksum()

        regist_target_checksum = loupe_data.regist_target_checksum()

    tissue_checksums = list(map(call_tiffer_checksum, tissue_image_paths))
    if cytassist_image_paths:
        cytassist_checksums = list(map(call_tiffer_checksum, cytassist_image_paths))
        # at least one of cytassist image is used for fiducial registration and tissue segmentation
        if fiducial_image_checksum and not fiducial_image_checksum in cytassist_checksums:
            martian.log_info(
                f"Loupe manual fiducial alignment checksum: {fiducial_image_checksum}\ncytassist checksums seen {cytassist_checksums}"
            )
            raise PreflightException(
                "Loupe manual fiducial alignment file seems to be created with different CytAssist image than was passed to the pipeline"
            )
        if (
            regist_target_checksum
            and not (regist_target_checksum in tissue_checksums)
            and any(tissue_checksums)
        ):
            martian.log_info(
                f"Loupe manual tissue alignment checksum: {regist_target_checksum}\ntissue checksums seen {tissue_checksums}"
            )
            raise PreflightException(
                "Loupe manual tissue alignment file seems to be created with different microscope image than was passed to the pipeline"
            )
        # if manual loupe file contains checksumHires, no fiducial, and no microscope image
        if (
            regist_target_checksum
            and not loupe_data.contain_fiducial_info()
            and not any(tissue_image_paths)
        ):
            raise PreflightException(
                "The Loupe alignment file passed to --loupe-alignment was produced by registering a CytAssist image and a microscope image, \
                but no microscope image was passed to --image, --darkimage, or --colorizedimage. The microscope image must be supplied in this case"
            )
    else:
        if regist_target_checksum:
            raise PreflightException(
                "Loupe alignment json specifies an Image Registration but no CytAssist image was provided to Space Ranger using --cytaimage"
            )
        # if manual loupe file contains a checksum, see if it matches _any_ file
        if fiducial_image_checksum and not fiducial_image_checksum in tissue_checksums:
            martian.log_info(
                f"loupe file checksum: {fiducial_image_checksum}\ntissue checksums seen {tissue_checksums}"
            )
            raise PreflightException(
                "Loupe manual alignment file seems to be created with different tissue image(s) than was passed to the pipeline"
            )

    # sort the checksums to see if there are any duplicates
    checksum_paths = sorted(zip(tissue_checksums, tissue_image_paths))
    for idx, (checksum, path) in enumerate(checksum_paths):
        if idx > 0 and checksum == checksum_paths[idx - 1][0]:
            raise PreflightException(
                f"Input images seem to be the same: {path} and {checksum_paths[idx - 1][1]}"
            )


class AllowedImageTypes(NamedTuple):
    """Map DARK_IMAGES_* keys to allowable image attributes such as bit depth."""

    name: str
    multi: bool
    tiff_color: list[tuple[str, int]]

    @classmethod
    def from_dark_images(cls, dark_images):
        """Factory method returns the allowable attributes based on the dark_images setting."""
        lookup_dict = {
            DARK_IMAGES_NONE: cls(
                name="--image",
                multi=False,
                tiff_color=[("gray", 8), ("gray", 16), ("RGB", 8), ("RGBA", 8)],
            ),
            DARK_IMAGES_CHANNELS: cls(
                name="--darkimage", multi=True, tiff_color=[("gray", 8), ("gray", 16)]
            ),
            DARK_IMAGES_COLORIZED: cls(
                name="--colorizedimage", multi=False, tiff_color=[("RGBA", 8), ("RGB", 8)]
            ),
        }
        try:
            allowed = lookup_dict[dark_images]
        except KeyError as error:
            raise RuntimeError("unknown value for dark_images in preflight") from error

        return allowed


def check_images_valid(  # pylint: disable=too-many-locals
    tissue_image_paths,
    dark_images,
    chemistry: str,
    cytassist_image_paths: list[str],
    dapi_index: int,
    image_scale: float | None,
    loupe_alignment_file=None,
):
    """Check that tiffer can open an image and its largest dimension is > 2000px.

    This requires more than 1 GB of memory and is run on the run target only.
    """
    _check_image_checksums(tissue_image_paths, cytassist_image_paths, loupe_alignment_file)

    skip_pages = get_remove_image_pages(loupe_alignment_file)

    if dapi_index is not None:
        if dark_images != DARK_IMAGES_CHANNELS or not cytassist_image_paths:
            raise PreflightException(
                "--dapi-index can only be specified with --darkimage and when used along with --cytaimage"
            )
        elif dapi_index < 1:
            raise PreflightException(
                f"--dapi-index was specified as {dapi_index}, but cannot be less than 1. The --dapi-index must be a value from 1 to N where N is the number of channels in the fluorescence --darkimage"
            )
        elif len(tissue_image_paths) > 1 and dapi_index > len(tissue_image_paths):
            raise PreflightException(
                f"--dapi-index was specified as {dapi_index}, but cannot be larger than the total number of --darkimage image files which was {len(tissue_image_paths)}."
            )
        else:
            imageinfo = call_tiffer_info(tissue_image_paths[0])
            npages = len(imageinfo["pages"])
            if dapi_index > npages:
                raise PreflightException(
                    f"--dapi-index was specified as {dapi_index}, but cannot be larger than the number of pages in the --darkimage which was {npages}."
                )

    whsizes = None
    whprotofile = None
    for path in tissue_image_paths:
        allowed = AllowedImageTypes.from_dark_images(dark_images)

        # call tiffer info to get image metadata
        imageinfo = call_tiffer_info(path)
        martian.log_info(f"called tiffer info on {path}:\n\t{imageinfo}")

        npages = len(imageinfo["pages"])
        nfiles = len(tissue_image_paths)

        # fail for any mixture of multi-page and multi-file
        if nfiles > 1 and npages > 1:
            raise PreflightException(f"Cannot pass multiple images and multipage images: {path}")

        # fail if all pages are skipped
        if len(skip_pages) >= npages:
            raise PreflightException(
                f"The loupe alignment file {loupe_alignment_file} would skip all of the pages in the input image"
            )

        # fail for multi-page or multi-file if not allowed
        if not allowed.multi and nfiles > 1:
            raise PreflightException(f"Cannot use multiple images with {allowed.name}")
        elif not allowed.multi and npages > 1:
            raise PreflightException(f"Cannot use multi-page TIFF images with {allowed.name}")

        # interrogate pages for color and size consistency if TIFF
        # interrogate multiple files for size consistency
        colormd = None
        for page in imageinfo["pages"]:
            if colormd is None:
                colormd = (page["colorMode"], page["depth"])
                if imageinfo["format"] == "tiff" and not colormd in allowed.tiff_color:
                    raise PreflightException(
                        f"Image color mode/depth {colormd} not allowed for {allowed.name}"
                    )
            elif colormd != (page["colorMode"], page["depth"]):
                raise PreflightException(
                    f"cannot mix color modes and bit depths across pages in image {path}"
                )

            if whsizes is None:
                whsizes = (page["width"], page["height"])
                whprotofile = path
            elif (page["width"], page["height"]) != whsizes:
                if nfiles > 1:
                    raise PreflightException(
                        "can't mix image sizes between files:\n{} had {} and \n{} has {}".format(
                            whprotofile, whsizes, path, (page["width"], page["height"])
                        )
                    )
                else:
                    raise PreflightException(
                        "can't mix image sizes between pages in a TIFF image:\n{} has both {} and {}".format(
                            path, whsizes, (page["width"], page["height"])
                        )
                    )

            limit, slide_name = (4000, "XL") if chemistry == "SPATIAL3Pv5" else (2000, "standard")
            if max(page["width"], page["height"]) < limit:
                raise PreflightException(
                    f"Image must have at least one dimension >= {limit} for {slide_name} slides"
                )

        if image_scale is not None:
            max_dim = max(max(page["width"], page["height"]) for page in imageinfo["pages"])
            image_physical_size = max_dim * image_scale
            max_image_physical_size = (
                FM_CANONICAL_PIXEL_SIZE * MAXIMUM_TISSUE_IMAGE_SCALING * REGIST_TARGET_IMAGE_MAX_DIM
            )
            if image_physical_size > max_image_physical_size:
                raise PreflightException(
                    f"The maximum dimension of tissue image must be at most {max_image_physical_size/1000:.02f} mm. "
                    f"Slide passed in had max image dimension {image_physical_size/1000:.02f} mm"
                )


def check_cytassist_metadata(metadata, cyta_img_path):
    """Check if cytassist metadata is valid."""
    wrong_num_pages = len(metadata["pages"]) > 1
    wrong_img_format = metadata["format"] != "tiff"
    wrong_bit_depth = metadata["pages"][0]["depth"] != 8
    wrong_img_mode = metadata["pages"][0]["colorMode"] != "RGBA"

    if wrong_num_pages or wrong_img_format or wrong_bit_depth or wrong_img_mode:
        raise PreflightException(
            f"Image at {cyta_img_path} does not seem to be an unaltered image from the CytAssist instrument."
        )


def check_cytassist_img_size(width, height, is_hd, is_pd):
    """Check if cytassist image size is valid."""
    wrong_height = height not in CYTA_IMAGE_ALLOWED_HEIGHTS
    if is_pd:
        wrong_width = width not in CYTA_IMAGE_ALLOWED_WIDTHS
    elif is_hd:
        wrong_width = width != CYTA_HD_IMAGE_WIDTH
    else:
        wrong_width = width != CYTA_IMAGE_DIM

    return wrong_width or wrong_height


def check_cytassist_img_valid(
    cytassist_image_paths: list[str],
    slide_serial_capture_area: str | None,
    chemistry: str,
    custom_chemistry_def: dict | None,
    is_pd: bool,
    override_id: bool | None,
):
    """Check if cytassist image is valid."""
    if not cytassist_image_paths:
        return
    is_hd = is_hd_upload(chemistry, custom_chemistry_def)
    num_cyta_images = len(cytassist_image_paths)
    if num_cyta_images != 1:
        raise PreflightException(
            f"--cytaimage can only be used for 1 CytAssist image but {num_cyta_images} CytAssist images were input."
        )
    cyta_img_path = cytassist_image_paths[0]
    metadata = call_tiffer_info(cyta_img_path)
    check_cytassist_metadata(metadata, cyta_img_path)
    meta_slide_id = metadata.get("slideSerial")
    meta_capture_area = metadata.get("captureArea")
    if meta_capture_area:
        meta_capture_area = CYTASSIST_TIFF_CAPTURE_AREA_TO_PIPELINE_CAPTURE_AREA.get(
            meta_capture_area
        )
    width = metadata["pages"][0]["width"]
    height = metadata["pages"][0]["height"]

    # known slide case
    if slide_serial_capture_area:
        if not override_id:
            cmdline_slide_id, cmdline_capture_area = parse_slide_sample_area_id(
                slide_serial_capture_area
            )

            if meta_slide_id and meta_slide_id not in SLIDE_ID_EXCEPTIONS:
                if meta_slide_id != cmdline_slide_id:
                    raise PreflightException(
                        "Slide serial number is inconsistent between --slide-id provided and CytAssist image metadata."
                        f"Provided --slide {cmdline_slide_id} does not match image metadata: {meta_slide_id}. "
                        "If the slide serial number in the cytassist metadata is incorrect, "
                        "please use `--override-id` to override it with the one specified using `--slide`"
                        "and `--area`."
                    )
            if meta_capture_area:
                if meta_capture_area != cmdline_capture_area:
                    raise PreflightException(
                        "Capture area is inconsistent between --area provided and CytAssist image metadata."
                        f"Provided --area {cmdline_capture_area} does not match image metadata: {meta_capture_area}"
                        "If the capture area in the cytassist metadata is incorrect, "
                        "please use `--override-id` to override it with the one specified using `--area`"
                        "and `--slide`."
                    )

    wrong_size = check_cytassist_img_size(width, height, is_hd, is_pd)
    if wrong_size:
        runtype = " HD" if is_hd else ""
        raise PreflightException(
            f"The CytAssist image at {cyta_img_path} is not the correct size for Visium{runtype}."
        )
