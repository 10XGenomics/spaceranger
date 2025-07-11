#
# Copyright (c) 2025 10X Genomics, Inc. All rights reserved.
#
"""Outputs hight level imaging metrics."""

__MRO__ = """
stage COLLECT_IMAGING_METRICS(
    in  file[] tissue_image_paths,
    in  jpg    detected_tissue_mask,
    in  bool   segmentation_from_user,
    in  json[] summaries,
    out json   imaging_metrics,
    out json   tissue_image_shape,
    src py     "stages/spatial/collect_imaging_metrics",
) using (
    mem_gb   = 8,
    vmem_gb  = 64,
    volatile = strict,
)
"""

import json
import os
from dataclasses import asdict, dataclass

import cv2
import martian
from PIL import Image

import cellranger.spatial.image_util as image_utils
import cellranger.spatial.tiffer as tiffer

Image.MAX_IMAGE_PIXELS = 500_000_000


@dataclass
class TissueImageStats:
    """Statistics of tissue image."""

    height: int | None = None
    width: int | None = None
    depth: int | None = None
    pages: int | None = None


def get_metadata_pillow(image_path: str, key: str):
    """Extract metadata from an image file using Pillow.

    Args:
        image_path (str): The file path to the image.
        key (str): The key to extract from the image metadata.

    Returns:
        dict: A dictionary containing the image metadata if successful.
        None: If an error occurs while extracting metadata.
    """
    try:
        with Image.open(image_path) as img:
            if key not in img.info:
                return None
            else:
                value = img.info.get(key)
                return value[0] if value else None
    except Exception as e:  # pylint: disable=broad-except
        martian.log_info(f"Failed to get metadata from image using Pillow: {e}")
        return None


def get_microscope_info(image_path: str):
    """Extracts microscope information from the metadata of an image file.

    Args:
        image_path (str): A list containing the path to the image file.

    Returns:
        dict or None: A dictionary containing the microscope manufacturer and model if found,
                      otherwise None.
    """
    try:
        xml_tree = image_utils.extract_metadata_xml_tree(image_path)
    except ValueError as e:
        martian.log_info(f"Failed to extract metadata XML tree: {e}")
        return None
    if xml_tree is not None:
        microscope = xml_tree.findall(".//Microscope")
        if not microscope:
            return None
        microscope_info = {
            "tissue_image_manufacturer": microscope[0].attrib.get("Manufacturer"),
            "tissue_image_model": microscope[0].attrib.get("Model"),
        }
        return microscope_info
    return None


def get_objective_info(image_path: str):
    """Extracts objective information from the metadata of an image file.

    Args:
        image_path (str): A list containing the path to the image file.

    Returns:
        dict or None: A dictionary containing the objective magnification information
                      with the key 'tissue_image_objective_magnification', or None if
                      the information is not found.
    """
    try:
        xml_tree = image_utils.extract_metadata_xml_tree(image_path)
    except ValueError as e:
        martian.log_info(f"Failed to extract metadata XML tree: {e}")
        return None
    if xml_tree is not None:
        objective = xml_tree.findall(".//Objective")
        if not objective:
            return None
        magnification = objective[0].attrib.get("NominalMagnification")
        if magnification is None:
            return None
        objective_info = {
            "tissue_image_objective_magnification": float(magnification),
        }
        return objective_info
    return None


def count_objects_and_sizes(mask_path, min_size):
    """Count objects and their sizes in a binary mask image.

    This function reads a binary mask image from the specified path, applies connected components analysis to identify
    distinct objects, and returns a dictionary containing information about objects that meet the minimum size requirement.

    Args:
        mask_path (str): Path to the binary mask image file.
        min_size (int): Minimum size (in pixels) for an object to be included in the output.

    Returns:
        dict: A dictionary where each key is an object index and each value is a list containing:
            - area (float): The area of the object in pixels.
            - centroid x (float): The x-coordinate of the object's centroid.
            - centroid y (float): The y-coordinate of the object's centroid.
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # Ensure binary mask: 0 = background, 255 = objects
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # Apply connected components analysis to get labels, stats, and centroids
    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)

    # Create a dictionary for objects that meet the minimum size requirement.
    # Each entry will be: key -> [area, centroid x, centroid y]
    object_info = {}
    object_index = 0
    for i in range(1, num_labels):  # Skip label 0 (background)
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_size:
            centroid = centroids[i]
            object_info[object_index] = [
                round(float(area), 1),
                round(float(centroid[0]), 1),
                round(float(centroid[1]), 1),
            ]
            object_index += 1

    return object_info


def main(args, outs):
    if not args.tissue_image_paths:
        martian.clear(outs)
        return
    image_info = tiffer.try_call_tiffer_info(args.tissue_image_paths[0])
    num_pages = tiffer.call_tiffer_get_num_pages(args.tissue_image_paths[0])
    mask_info = (
        count_objects_and_sizes(args.detected_tissue_mask, min_size=0)
        if args.detected_tissue_mask
        else None
    )

    tissue_image_stats = TissueImageStats(
        height=(
            image_info["pages"][0].get("height")
            if image_info["pages"][0].get("height") is not None
            else None
        ),
        width=(
            image_info["pages"][0].get("width")
            if image_info["pages"][0].get("width") is not None
            else None
        ),
        depth=(
            image_info["pages"][0].get("depth")
            if image_info["pages"][0].get("depth") is not None
            else None
        ),
        pages=num_pages,
    )
    with open(outs.tissue_image_shape, "w") as f:
        json.dump(asdict(tissue_image_stats), f, indent=2)

    image_output_info = {
        "tissue_image_format": image_info["format"],
        "tissue_image_width": (
            float(tissue_image_stats.width) if tissue_image_stats.width is not None else None
        ),
        "tissue_image_height": (
            float(tissue_image_stats.height) if tissue_image_stats.height is not None else None
        ),
        "tissue_image_color_mode": image_info["pages"][0].get("colorMode"),
        "tissue_image_depth": (
            float(tissue_image_stats.depth) if tissue_image_stats.depth is not None else None
        ),
        "tissue_image_num_pages": (
            float(tissue_image_stats.pages) if tissue_image_stats.pages is not None else None
        ),
        "tissue_mask_info": mask_info,
        "segmentation_from_user": args.segmentation_from_user,
    }
    metadata_pillow = get_metadata_pillow(args.tissue_image_paths[0], key="dpi")
    if metadata_pillow is not None:
        dpi_info = {"tissue_image_dpi": float(metadata_pillow)}
        image_output_info.update(dpi_info)
    microscope_info = get_microscope_info(args.tissue_image_paths[0])
    if microscope_info is not None:
        image_output_info.update(microscope_info)
    objective_info = get_objective_info(args.tissue_image_paths[0])
    if objective_info is not None:
        image_output_info.update(objective_info)
    if args.summaries:
        for summary in args.summaries:
            if summary and os.path.exists(summary):
                with open(summary) as f:
                    summary_data = json.load(f)
                    image_output_info.update(summary_data)

    with open(outs.imaging_metrics, "w") as f:
        json.dump(image_output_info, f, indent=2)
