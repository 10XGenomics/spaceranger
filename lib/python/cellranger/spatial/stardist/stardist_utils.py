#
# Copyright (c) 2025 10X Genomics, Inc. All rights reserved.
#
"""Stardist utilities."""

import collections
import json
from pathlib import Path

import numexpr
import numpy as np

STARDIST_DEFAULT_MODEL_PATH = Path(__file__).parent / "stardist_models"
STARDIST_DEFAULT_MODEL_NAME = "latest"
STARDIST_BLOCK_SIZE = 4096
STARDIST_TILE_SIZE = 1024
STARDIST_BLOCK_MIN_OVERLAP = 256
STARDIST_N_TILES_PER_BLOCK = (4, 4, 1)


def _is_power_of_2(i):
    assert i > 0
    e = np.log2(i)
    return e == int(e)


def _normalize_grid(grid, n):
    try:
        grid = tuple(grid)
        if not (len(grid) == n and all(map(np.isscalar, grid)) and all(map(_is_power_of_2, grid))):
            _raise(TypeError())
        return tuple(int(g) for g in grid)
    except (TypeError, AssertionError) as exc:
        raise ValueError(
            f"grid = {grid} must be a list/tuple of length {n} with values that are power of 2"
        ) from exc


def _is_floatarray(x):
    return isinstance(x.dtype.type(0), np.floating)


def load_json(fpath):
    with open(fpath) as f:
        return json.load(f)


def save_json(data, fpath, **kwargs):
    with open(fpath, "w") as f:
        f.write(json.dumps(data, **kwargs))


def _raise(e):
    if isinstance(e, BaseException):
        raise e
    else:
        raise ValueError(e)


def normalize_mi_ma(x, mi, ma, clip=False, eps=1e-20, dtype=np.float32):
    """Normalize min max."""
    if dtype is not None:
        x = x.astype(dtype, copy=False)
        mi = dtype(mi) if np.isscalar(mi) else mi.astype(dtype, copy=False)
        ma = dtype(ma) if np.isscalar(ma) else ma.astype(dtype, copy=False)
        eps = dtype(eps)

    x = numexpr.evaluate("(x - mi) / ( ma - mi + eps )")

    if clip:
        x = np.clip(x, 0, 1)

    return x


# https://docs.python.org/3/library/itertools.html#itertools-recipes
def consume(iterator):
    collections.deque(iterator, maxlen=0)


def axes_check_and_normalize(axes, length=None, disallowed=None, return_allowed=False):
    """S(ample), T(ime), C(hannel), Z, Y, X."""
    allowed = "STCZYX"
    if axes is None:
        _raise(ValueError("axis cannot be None."))
    axes = str(axes).upper()
    consume(
        a in allowed or _raise(ValueError(f"invalid axis {a}, must be one of {list(allowed)}"))
        for a in axes
    )
    if disallowed is not None:
        consume(a not in disallowed or _raise(ValueError(f"disallowed axis {a}.")) for a in axes)
    consume(
        axes.count(a) == 1 or _raise(ValueError(f"axis {a} occurs more than once.")) for a in axes
    )
    if length is not None and len(axes) != length:
        _raise(ValueError(f"axes ({axes}) must be of length {length}."))
    return (axes, allowed) if return_allowed else axes
