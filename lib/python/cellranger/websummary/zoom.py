#!/usr/bin/env python
#
# Copyright (c) 2023 10X Genomics, Inc. All rights reserved.
#
"""Handle zooming behavior for the spatial HD websummary."""


from dataclasses import dataclass

import numpy as np

from cellranger.spatial.transform import transform_pts_2d


class MinMax:
    """Min and max values."""

    min: float
    max: float

    def __init__(self):
        """Initialize min and max to the opposite extremes."""
        self.min = float("inf")
        self.max = float("-inf")

    def observe(self, value: float):
        """Observe a value and update the min and max if necessary."""
        self.min = min(self.min, value)
        self.max = max(self.max, value)

    def dilate(self, delta: float):
        """Dilate the min and max by a delta."""
        self.min = self.min - delta
        self.max = self.max + delta


class AxisAlignedBoundingBox:
    """An axis aligned bounding box."""

    x: MinMax
    y: MinMax

    def __init__(self):
        """Initialize the bounding box."""
        self.x = MinMax()
        self.y = MinMax()

    def observe(self, x: float, y: float):
        """Update the bounding box with a new point."""
        self.x.observe(x)
        self.y.observe(y)

    def dx(self):  # pylint: disable=invalid-name
        return self.x.max - self.x.min

    def dy(self):  # pylint: disable=invalid-name
        return self.y.max - self.y.min

    def square(self):
        """Make the bounding box square."""
        x_range = self.dx()
        y_range = self.dy()
        if x_range > y_range:
            delta = (x_range - y_range) / 2
            self.y.dilate(delta)
        else:
            delta = (y_range - x_range) / 2
            self.x.dilate(delta)

    def dilate(self, delta: float):
        """Dilate the bounding box by a delta in both x and y."""
        self.x.dilate(delta)
        self.y.dilate(delta)


@dataclass
class InitialZoomPan:
    """Initial zoom and pan data."""

    scale: float
    dx: float  # pylint: disable=invalid-name
    dy: float  # pylint: disable=invalid-name

    @staticmethod
    def compute(image, image_transform, display_width, border_width=10.0):
        """Given an image and a transform, compute the initial zoom and pan.

        The initial zoom and pan are such that the image is scaled to fit within the display width
        with a border_width pixel border around the image.
        """
        bbox = AxisAlignedBoundingBox()
        nrows, ncols = image.shape
        pts_xy = np.array([[0, 0], [ncols - 1, 0], [ncols - 1, nrows - 1], [0, nrows - 1]])
        for transformed_pts in transform_pts_2d(pts_xy, image_transform):
            bbox.observe(transformed_pts[0], transformed_pts[1])

        bbox.square()
        bbox.dilate(border_width)
        initial_scale = display_width / bbox.dx()
        initial_zoom_pan = InitialZoomPan(
            scale=initial_scale,
            dx=-bbox.x.min * initial_scale,
            dy=-bbox.y.min * initial_scale,
        )
        return initial_zoom_pan
