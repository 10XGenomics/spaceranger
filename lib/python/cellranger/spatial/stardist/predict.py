#
# Copyright (c) 2025 10X Genomics, Inc. All rights reserved.
#

"""Tile classes and functions required for tiled predictions of large images."""

import warnings

import numpy as np
import pytest

from cellranger.spatial.stardist.stardist_utils import _raise


class Tile:
    """Class that represents one tile for prediction."""

    def __init__(self, n, size, overlap, prev):
        self.n = int(n)
        self.size = int(size)
        self.overlap = int(overlap)
        if self.n < self.size:
            assert prev is None
            # print("Truncating tile size from %d to %d." % (self.size, self.n))
            self.size = self.n
            self.overlap = 0
        assert self.size > 2 * self.overlap
        # assert self.n >= self.size
        if prev is not None:
            assert not prev.at_end, "Previous tile already at end"
        self.prev = prev
        self.read_slice = self._read_slice
        self.write_slice = self._write_slice

    @property
    def at_begin(self):
        return self.prev is None

    @property
    def at_end(self):
        return self.read_slice.stop == self.n

    @property
    def _read_slice(self):
        if self.at_begin:
            start, stop = 0, self.size
        else:
            prev_read_slice = self.prev.read_slice
            start = prev_read_slice.stop - 2 * self.overlap
            stop = start + self.size
            shift = min(0, self.n - stop)
            start, stop = start + shift, stop + shift
            assert start > prev_read_slice.start
        assert start >= 0, "start must be greater than or equal to 0"
        assert stop <= self.n, f"stop must be less than or equal to {self.n}"
        return slice(start, stop)

    @property
    def _write_slice(self):
        if self.at_begin:
            if self.at_end:
                return slice(0, self.n)
            else:
                return slice(0, self.size - 1 * self.overlap)
        elif self.at_end:
            s = self.prev.write_slice.stop
            return slice(s, self.n)
        else:
            s = self.prev.write_slice.stop
            return slice(s, s + self.size - 2 * self.overlap)

    def __repr__(self):
        s = np.array(list(" " * self.n))
        s[self.read_slice] = "-"
        s[self.write_slice] = "x" if (self.at_begin or self.at_end) else "o"
        return "".join(s)


class Tiling:
    """Tiled collection."""

    def __init__(self, n, size, overlap):
        self.n = n
        self.size = size
        self.overlap = overlap
        tiles = [Tile(prev=None, n=self.n, size=self.size, overlap=self.overlap)]
        while not tiles[-1].at_end:
            tiles.append(Tile(prev=tiles[-1], n=self.n, size=self.size, overlap=self.overlap))
        self.tiles = tiles

    def __len__(self):
        return len(self.tiles)

    def __repr__(self):
        return "\n".join(f"{i:3}. {t}" for i, t in enumerate(self.tiles, 1))

    def slice_generator(self, block_size=1):
        """Generate slices."""

        def scale(sl):
            return slice(block_size * sl.start, block_size * sl.stop)

        def crop_slice(read, write):
            stop = write.stop - read.stop
            return slice(write.start - read.start, stop if stop < 0 else None)

        for t in self.tiles:
            read, write = scale(t.read_slice), scale(t.write_slice)
            yield read, write, crop_slice(read, write)

    @staticmethod
    def for_n_tiles(n, n_tiles, overlap):
        """Get candidate tiling."""
        smallest_size = 2 * overlap + 1
        tile_size = smallest_size  # start with smallest posible tile_size
        while len(Tiling(n, tile_size, overlap)) > n_tiles:
            tile_size += 1
        if tile_size == smallest_size:
            return Tiling(n, tile_size, overlap)
        candidates = (
            Tiling(n, tile_size - 1, overlap),
            Tiling(n, tile_size, overlap),
        )
        diffs = [np.abs(len(c) - n_tiles) for c in candidates]
        return candidates[np.argmin(diffs)]


def total_n_tiles(x, n_tiles, block_sizes, n_block_overlaps, guarantee="size"):
    """Get total tiles."""
    assert x.ndim == len(n_tiles) == len(block_sizes) == len(n_block_overlaps)
    assert guarantee in ("size", "n_tiles")
    n_tiles_used = 1
    for n, n_tile, block_size, n_block_overlap in zip(
        x.shape, n_tiles, block_sizes, n_block_overlaps
    ):
        assert n % block_size == 0
        n_blocks = n // block_size
        if guarantee == "size":
            n_tiles_used *= len(Tiling.for_n_tiles(n_blocks, n_tile, n_block_overlap))
        elif guarantee == "n_tiles":
            n_tiles_used *= n_tile
    return n_tiles_used


def tile_iterator_1d(
    x, axis, n_tiles, block_size, n_block_overlap, guarantee="size"
):  # pylint:disable=too-many-locals
    """Tile iterator for one dimension of array x.

    Parameters
    ----------
    x : numpy.ndarray
        Input array
    axis : int
        Axis which sould be tiled, all other axis not tiled
    n_tiles : int
        Targeted number of tiles for axis of x (see guarantee below)
    block_size : int
        Axis of x is assumed to be evenly divisible by block_size
        All tiles are aligned with the block_size
    n_block_overlap : int
        Tiles will overlap at least this many blocks (see guarantee below)
    guarantee : str
        Can be either 'size' or 'n_tiles':
        'size':    The size of all tiles is guaranteed to be the same,
                   but the number of tiles can be different and the
                   amount of overlap can be larger than requested.
        'n_tiles': The size of tiles can be different at the beginning and end,
                   but the number of tiles is guarantee to be the one requested.
                   The mount of overlap is also exactly as requested.
    """
    n = x.shape[axis]

    if n % block_size != 0:
        _raise(ValueError("'x' must be evenly divisible by 'block_size' along 'axis'"))
    n_blocks = n // block_size

    if guarantee not in ("size", "n_tiles"):
        _raise(ValueError("guarantee must be either 'size' or 'n_tiles'"))

    if guarantee == "size":
        tiling = Tiling.for_n_tiles(n_blocks, n_tiles, n_block_overlap)

        def ndim_slices(t):
            sl = [slice(None)] * x.ndim
            sl[axis] = t
            return tuple(sl)

        for read, write, crop in tiling.slice_generator(block_size):
            tile_in = read  # src in input image     / tile
            tile_out = write  # dst in output image    / s_dst
            tile_crop = crop  # crop of src for output / s_src
            yield x[ndim_slices(tile_in)], ndim_slices(tile_crop), ndim_slices(tile_out)

    elif guarantee == "n_tiles":
        n_tiles_valid = int(np.clip(n_tiles, 1, n_blocks))
        if n_tiles != n_tiles_valid:
            warnings.warn(f"invalid value ({n_tiles}) for 'n_tiles', changing to {n_tiles_valid}")
            n_tiles = n_tiles_valid

        s = n_blocks // n_tiles  # tile size
        r = n_blocks % n_tiles  # blocks remainder
        assert n_tiles * s + r == n_blocks

        # list of sizes for each tile
        tile_sizes = s * np.ones(n_tiles, int)
        # distribute remaining blocks to tiles at beginning and end
        if r > 0:
            tile_sizes[: r // 2] += 1
            tile_sizes[-(r - r // 2) :] += 1

        # n_block_overlap = int(np.ceil(92 / block_size))
        # n_block_overlap -= 1
        # print(n_block_overlap)

        # (pre,post) offsets for each tile
        off = [
            (n_block_overlap if i > 0 else 0, n_block_overlap if i < n_tiles - 1 else 0)
            for i in range(n_tiles)
        ]

        # tile_starts = np.concatenate(([0],np.cumsum(tile_sizes[:-1])))
        # print([(_st-_pre,_st+_sz+_post) for (_st,_sz,(_pre,_post)) in zip(tile_starts,tile_sizes,off)])

        def to_slice(t):
            sl = [slice(None)] * x.ndim
            sl[axis] = slice(t[0] * block_size, t[1] * block_size if t[1] != 0 else None)
            return tuple(sl)

        start = 0
        for i in range(n_tiles):
            off_pre, off_post = off[i]

            # tile starts before block 0 -> adjust off_pre
            if start - off_pre < 0:
                off_pre = start
            # tile end after last block -> adjust off_post
            if start + tile_sizes[i] + off_post > n_blocks:
                off_post = n_blocks - start - tile_sizes[i]

            tile_in = (
                start - off_pre,
                start + tile_sizes[i] + off_post,
            )  # src in input image     / tile
            tile_out = (start, start + tile_sizes[i])  # dst in output image    / s_dst
            tile_crop = (off_pre, -off_post)  # crop of src for output / s_src

            yield x[to_slice(tile_in)], to_slice(tile_crop), to_slice(tile_out)
            start += tile_sizes[i]

    else:
        pytest.fail("parameter guarantee to tile_iterator_1d sohuld be 'size' and 'n_tiles'")


def tile_iterator(x, n_tiles, block_sizes, n_block_overlaps, guarantee="size"):
    """Tile iterator for n-d arrays.

    Yields block-aligned tiles (`block_sizes`) that have at least
    a certain amount of overlapping blocks (`n_block_overlaps`)
    with their neighbors. Also yields slices that allow to map each
    tile back to the original array x.

    Notes:
    -----
    - Tiles will not go beyond the array boundary (i.e. no padding).
      This means the shape of x must be evenly divisible by the respective block_size.
    - It is not guaranteed that all tiles have the same size if guarantee is not 'size'.

    Parameters
    ----------
    x : numpy.ndarray
        Input array.
    n_tiles : int or sequence of ints
        Number of tiles for each dimension of x.
    block_sizes : int or sequence of ints
        Block sizes for each dimension of x.
        The shape of x is assumed to be evenly divisible by block_sizes.
        All tiles are aligned with block_sizes.
    n_block_overlaps : int or sequence of ints
        Tiles will at least overlap this many blocks in each dimension.
    guarantee : str
        Can be either 'size' or 'n_tiles':
        'size':    The size of all tiles is guaranteed to be the same,
                   but the number of tiles can be different and the
                   amount of overlap can be larger than requested.
        'n_tiles': The size of tiles can be different at the beginning and end,
                   but the number of tiles is guarantee to be the one requested.
                   The mount of overlap is also exactly as requested.

    Example:
    -------
    Duplicate an array tile-by-tile:

    >>> x = np.array(...)
    >>> y = np.empty_like(x)
    >>>
    >>> for tile,s_src,s_dst in tile_iterator(x, n_tiles, block_sizes, n_block_overlaps):
    >>>     y[s_dst] = tile[s_src]
    >>>
    >>> np.allclose(x,y)
    True

    """
    if np.isscalar(n_tiles):
        n_tiles = (n_tiles,) * x.ndim
    if np.isscalar(block_sizes):
        block_sizes = (block_sizes,) * x.ndim
    if np.isscalar(n_block_overlaps):
        n_block_overlaps = (n_block_overlaps,) * x.ndim

    assert x.ndim == len(n_tiles) == len(block_sizes) == len(n_block_overlaps)

    def _accumulate(tile_in, axis, src, dst):
        for tile, s_src, s_dst in tile_iterator_1d(
            tile_in, axis, n_tiles[axis], block_sizes[axis], n_block_overlaps[axis], guarantee
        ):
            src[axis] = s_src[axis]
            dst[axis] = s_dst[axis]
            if axis + 1 == tile_in.ndim:
                # remove None and negative slicing
                src = [
                    slice(
                        s.start,
                        size if s.stop is None else (s.stop if s.stop >= 0 else size + s.stop),
                    )
                    for s, size in zip(src, tile.shape)
                ]
                yield tile, tuple(src), tuple(dst)
            else:
                # yield from _accumulate(tile, axis+1, src, dst)
                yield from _accumulate(tile, axis + 1, src, dst)

    return _accumulate(x, 0, [None] * x.ndim, [None] * x.ndim)
