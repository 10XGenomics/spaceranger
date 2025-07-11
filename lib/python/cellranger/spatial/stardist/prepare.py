#
# Copyright (c) 2025 10X Genomics, Inc. All rights reserved.
#

"""Classes & Functions related to preparing image patches for training & testing."""

import math
from abc import ABCMeta, abstractmethod

import numpy as np
from six import add_metaclass

from cellranger.spatial.stardist.stardist_utils import (
    _raise,
    axes_check_and_normalize,
    consume,
    normalize_mi_ma,
)
from cellranger.spatial.stardist.types import AxisMap


@add_metaclass(ABCMeta)
class Normalizer:
    """Abstract base class for normalization methods."""

    @abstractmethod
    def before(self, x, axes):
        """Normalization of the raw input image (method stub).

        Parameters
        ----------
        x : :class:`numpy.ndarray`
            Raw input image.
        axes : str
            Axes of input image x

        Returns:
        -------
        :class:`numpy.ndarray`
            Normalized input image with suitable values for neural network input.
        """

    @abstractmethod
    def after(self, mean, scale, axes):
        """Possible adjustment of predicted restored image (method stub).

        Parameters
        ----------
        mean : :class:`numpy.ndarray`
            Predicted restored image or per-pixel ``mean`` of Laplace distributions
            for probabilistic model.
        scale: :class:`numpy.ndarray` or None
            Per-pixel ``scale`` of Laplace distributions for probabilistic model (``None`` otherwise.)
        axes : str
            Axes of ``mean`` and ``scale``

        Returns:
        -------
        :class:`numpy.ndarray`
            Adjusted restored image(s).
        """

    def __call__(self, x, axes):
        """Alias for :func:`before` to make this callable."""
        return self.before(x, axes)

    @property
    @abstractmethod
    def do_after(self):
        """Bool : Flag to indicate whether :func:`after` should be called."""


class MinMaxNormalizer(Normalizer):
    """Normalizer based on min and max values."""

    vmin, vmax = 0, 255

    def before(self, x, _axes):
        return normalize_mi_ma(x, self.vmin, self.vmax, dtype=np.float32)

    def after(self, *_args, **_kwargs):
        raise NotImplementedError

    @property
    def do_after(self):
        return False


class NoNormalizer(Normalizer):
    """No normalization.

    Parameters
    ----------
    do_after : bool
        Flag to indicate whether to undo normalization.

    Raises:
    ------
    ValueError
        If :func:`after` is called, but parameter `do_after` was set to ``False`` in the constructor.
    """

    def __init__(self, do_after=False):
        self._do_after = do_after

    def before(self, x, axes):
        return x

    def after(self, mean, scale, axes):
        if not self.do_after:
            raise ValueError()

        return mean, scale

    @property
    def do_after(self):
        return self._do_after


@add_metaclass(ABCMeta)
class Resizer:
    """Abstract base class for resizing methods."""

    @abstractmethod
    def before(self, x, axes, axes_div_by):
        """Resizing of the raw input image (method stub).

        Parameters
        ----------
        x : :class:`numpy.ndarray`
            Raw input image.
        axes : str
            Axes of input image x
        axes_div_by : iterable of int
            Resized image must be evenly divisible by the provided values for each axis.

        Returns:
        -------
        :class:`numpy.ndarray`
            Resized input image.
        """

    @abstractmethod
    def after(self, x, axes):
        """Resizing of the restored image (method stub).

        Parameters
        ----------
        x : :class:`numpy.ndarray`
            Restored image.
        axes : str
            Axes of restored image x

        Returns:
        -------
        :class:`numpy.ndarray`
            Resized restored image.
        """


class NoResizer(Resizer):
    """No resizing.

    Raises:
    ------
    ValueError
        In :func:`before`, if image resizing is necessary.
    """

    def before(self, x, axes, axes_div_by):
        axes = axes_check_and_normalize(axes, x.ndim)
        consume(
            (s % div_n == 0) or _raise(ValueError(f"{s} (axis {a}) is not divisible by {div_n}."))
            for a, div_n, s in zip(axes, axes_div_by, x.shape)
        )
        return x

    def after(self, x, axes):
        return x


class StarDistPadAndCropResizer(Resizer):
    """StarDist pad and crop resizer."""

    # TODO: check correctness
    def __init__(self, grid: AxisMap, mode="reflect", **kwargs):
        self.mode = mode
        self.grid = grid
        self.kwargs = kwargs
        self.pad = None
        self.padded_shape = None

    def before(self, x, axes, axes_div_by):
        assert all(a % g == 0 for g, a in zip((self.grid.get(a) for a in axes), axes_div_by))
        axes = axes_check_and_normalize(axes, x.ndim)

        def _split(v):
            return 0, v  # only pad at the end

        self.pad = {
            a: _split((div_n - s % div_n) % div_n)
            for a, div_n, s in zip(axes, axes_div_by, x.shape)
        }
        x_pad = np.pad(x, tuple(self.pad[a] for a in axes), mode=self.mode, **self.kwargs)
        self.padded_shape = dict(zip(axes, x_pad.shape))
        if "C" in self.padded_shape:
            del self.padded_shape["C"]
        return x_pad

    def after(self, x, axes):
        # axes can include 'C', which may not have been present in before()
        assert (
            self.pad is not None
        ), "Got self.pad=None in `after` method of StarDistPadAndCropResizer"
        assert (
            self.padded_shape is not None
        ), "Got self.padded_shape=None in `after` method of StarDistPadAndCropResizer"
        axes = axes_check_and_normalize(axes, x.ndim)
        assert all(
            s_pad == s * g
            for s, s_pad, g in zip(
                x.shape,
                (self.padded_shape.get(a, _s) for a, _s in zip(axes, x.shape)),
                (self.grid.get(a) for a in axes),
            )
        )
        crop = tuple(
            slice(0, -(math.floor(p[1] / g)) if p[1] >= g else None)
            for p, g in zip(
                (self.pad.get(a, (0, 0)) for a in axes), (self.grid.get(a) for a in axes)
            )
        )
        return x[crop]

    def filter_points(self, ndim, points, axes):
        """Returns indices of points inside crop region."""
        assert points.ndim == 2
        assert (
            self.pad is not None
        ), "Got self.pad=None in `filter_points` method of StarDistPadAndCropResizer"
        assert (
            self.padded_shape is not None
        ), "Got self.padded_shape=None in `filter_points` method of StarDistPadAndCropResizer"
        axes = axes_check_and_normalize(axes, ndim)

        bounds = np.array(
            tuple(
                self.padded_shape[a] - self.pad[a][1] for a in axes if a.lower() in ("z", "y", "x")
            )
        )
        idx = np.where(np.all(points < bounds, 1))
        return idx
