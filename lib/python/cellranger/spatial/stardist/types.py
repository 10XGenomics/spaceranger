#
# Copyright (c) 2025 10X Genomics, Inc. All rights reserved.
#
"""Types used to pass around objects in Stardist."""

from collections.abc import Iterable
from dataclasses import dataclass
from itertools import zip_longest
from typing import Self


@dataclass
class AxisMap:
    """Map an axis to values."""

    X: int | None = None
    Y: int | None = None
    C: int | None = None  # pylint: disable=invalid-name

    @classmethod
    def from_axes(cls, axes: Iterable[str] | str) -> Self:
        return cls(**dict((ax, ind) for (ind, ax) in enumerate(axes)))

    @classmethod
    def from_axes_and_values(cls, axes: str, values: Iterable, default_value=None) -> Self:
        return cls(**dict(zip_longest(axes, values, fillvalue=default_value)))

    def get(self, value: str):
        """Get mapping to a value."""
        match value.upper():
            case "X":
                return self.X
            case "Y":
                return self.Y
            case "C":
                return self.C
            case _:
                raise ValueError(f"Only values accepted are 'X', 'Y', 'C'. Given: {value}")

    def get_tuple(self, query_axes: str) -> tuple:
        return tuple(self.get(x) for x in query_axes)

    def get_non_channel_values(self, query_axes: str) -> tuple:
        return tuple(self.get(x) for x in query_axes if x.upper() != "C")
