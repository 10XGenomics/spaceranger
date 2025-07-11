#!/usr/bin/env python
#
# Copyright (c) 2022 10X Genomics, Inc. All rights reserved.
#

"""Utility functions for generating and encoding polygon images."""

import martian

# isort: off
# pylint: disable=wrong-import-position
import matplotlib

matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.pyplot as plt

# pylint: enable=wrong-import-position
# isort: on
import numpy as np
from PIL import Image

# pylint: disable=no-name-in-module
from skimage.draw import polygon, polygon_perimeter  # pylint: disable=no-name-in-module

# pylint: enable=no-name-in-module
from cellranger.spatial.image import base64_encode_image
from cellranger.spatial.webp_image_utils import get_crop_bbox


def heatmap_legend(cmap, title, vmin, vmax, fname):
    """Generate a heatmap legend image for the UMI plot."""
    fig, ax = plt.subplots(figsize=(6, 1))
    fig.subplots_adjust(bottom=0.5)
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    colorbar = matplotlib.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, orientation="horizontal")
    colorbar.ax.set_title(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(fname, dpi=100, pad_inches=0)
    plt.close()
    return base64_encode_image(fname)


def make_path_and_encode(
    chart, filename: str, lossless=True, crop_box: np.ndarray | None = None, padding: int = 0
):
    """Make a path for the chart and encode it as a base64 image.

    Args:
        chart (PIL.Image.Image): An image object to be encoded.
        filename (str): path to the image file.
        lossless (bool): Whether to save the image using lossless compression.
        crop_box (np.ndarray): 2D array of coordinates of crop box.
        padding (int): Padding to use. Defaults to 0.

    Returns:
        str: base64 encoded image.
    """
    if crop_box is not None:
        crop_bbox = get_crop_bbox(chart, crop_box=crop_box, padding=padding)
        chart = chart.crop(crop_bbox)
    chart_path = martian.make_path(f"{filename}.webp").decode()
    chart.save(chart_path, format="WEBP", lossless=lossless)
    return base64_encode_image(chart_path)


def scale_geometry(geometry, scale_factor):
    """Scales the coordinates of a given geometry by a specified scale factor.

    Args:
        geometry (dict): A dictionary representing the geometry to be scaled.
                        It must have a "type" key with the value "Polygon" and a "coordinates" key
                        containing a list of rings, where each ring is a list of [x, y] coordinate pairs.
        scale_factor (float): The factor by which to scale the coordinates.

    Returns:
        dict: A new geometry dictionary with the scaled coordinates.
    """
    if geometry["type"] == "Polygon":
        scaled_coordinates = [
            [[x * scale_factor, y * scale_factor] for x, y in ring]
            for ring in geometry["coordinates"]
        ]
        return {"type": geometry["type"], "coordinates": scaled_coordinates}
    else:
        raise ValueError("Unsupported geometry type")


def generate_genome_images(
    shapes_per_genome, tissue_image_height, tissue_image_width, log_transform=False
):  # pylint: disable=too-many-locals
    """Generate genome images by filling and coloring polygons based on UMI values.

    Args:
        shapes_per_genome (dict): A dictionary where keys are genome names and values are lists of tuples.
                                  Each tuple contains a feature dictionary and a UMI value.
        tissue_image_height (int): The height of the tissue image.
        tissue_image_width (int): The width of the tissue image.
        log_transform (bool, optional): Whether to apply log10 transformation to UMI values. Defaults to False.

    Returns:
        dict: A dictionary where keys are genome names and values are encoded image paths.
    """
    encoded_images = {}
    for genome, shapes in shapes_per_genome.items():
        # Create a canvas for UMI values (filled with NaN)
        raster = np.full((tissue_image_height, tissue_image_width), np.nan, dtype=np.float32)
        # Create an edge mask to mark the polygon perimeters
        edge_mask = np.zeros((tissue_image_height, tissue_image_width), dtype=bool)

        for feature, val in shapes:
            if feature["type"] == "Polygon":
                coords = np.array(feature["coordinates"][0])
                # Optionally log10 transform the UMI value
                if log_transform:
                    val = np.log10(val + 1)
                # Fill the polygon interior with the UMI value
                rr_fill, cc_fill = polygon(
                    coords[:, 1],
                    coords[:, 0],
                    shape=raster.shape,
                )
                raster[rr_fill, cc_fill] = val

                # Mark the polygon edge (perimeter) in the edge mask
                rr_edge, cc_edge = polygon_perimeter(coords[:, 1], coords[:, 0], shape=raster.shape)
                edge_mask[rr_edge, cc_edge] = True

        # Normalize the UMI values from the filled interior (ignoring NaNs)
        if not np.isnan(raster).all():
            normed = (raster - np.nanmin(raster)) / (np.nanmax(raster) - np.nanmin(raster))
            normed[np.isnan(raster)] = 0
        else:
            normed = np.zeros_like(raster)

        # Apply the colormap to the non-edge (filled) pixels
        cmap = cm.get_cmap("turbo")
        colored = np.zeros((tissue_image_height, tissue_image_width, 4), dtype=np.uint8)
        valid = ~np.isnan(raster)
        colored[valid] = (cmap(normed[valid]) * 255).astype(np.uint8)

        # Override the edge pixels to be black
        colored[edge_mask] = [0, 0, 0, 255]

        # Ensure that any pixels that remain NaN (i.e. not covered by any polygon) are transparent
        colored[~valid] = [0, 0, 0, 0]

        img = Image.fromarray(colored, mode="RGBA")
        if log_transform:
            encoded_images[genome] = make_path_and_encode(img, f"log10_genome_{genome}")
        else:
            encoded_images[genome] = make_path_and_encode(img, f"genome_{genome}")

    return encoded_images
