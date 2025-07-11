#
# Copyright (c) 2023 10X Genomics, Inc. All rights reserved.
#
"""Bounding Box class used across different use cases."""

from dataclasses import dataclass
from typing import Self

import numpy as np


@dataclass
class BoundingBox:
    """Bounding box in row/col indices.

    The ranges follow python formats with the min being inclusive and the max being non-inclusive.
    """

    minr: int
    minc: int
    maxr: int
    maxc: int

    @property
    def shape(self) -> tuple[int, int]:
        return (self.maxr - self.minr, self.maxc - self.minc)

    @property
    def slice(self) -> slice:
        return np.s_[self.minr : self.maxr, self.minc : self.maxc]

    def non_overlapping(self, other: Self) -> bool:
        return (
            self.minr >= other.maxr
            or self.maxr <= other.minr
            or self.minc >= other.maxc
            or self.maxc <= other.minc
        )

    def intersect(self, other: Self):
        return BoundingBox(
            minr=max(self.minr, other.minr),
            minc=max(self.minc, other.minc),
            maxr=min(self.maxr, other.maxr),
            maxc=min(self.maxc, other.maxc),
        )

    @classmethod
    def new(cls, minr, minc, maxr, maxc) -> Self:
        return cls(
            minr=int(np.floor(minr)),
            minc=int(np.floor(minc)),
            maxr=int(np.ceil(maxr)),
            maxc=int(np.ceil(maxc)),
        )

    @classmethod
    def from_image_dimensions(cls, width, height) -> Self:
        return cls(
            minr=0,
            minc=0,
            maxr=int(height),
            maxc=int(width),
        )

    def get_array(self) -> tuple[int, int, int, int]:
        """Returns the bounding box as (min_row, min_col, max_row, max_col)."""
        return (self.minr, self.minc, self.maxr, self.maxc)

    def smallest_dimension(self) -> int:
        return min(self.maxr - self.minr, self.maxc - self.minc)

    def __post_init__(self):
        if self.maxr < self.minr:
            raise ValueError("Bounding box needs max row to be at least min row")
        if self.maxc < self.minc:
            raise ValueError("Bounding box needs max column to be at least min column")
