#
# Copyright (c) 2025 10X Genomics, Inc. All rights reserved.
#
"""Functions to do prediction on big images."""

import math
from itertools import product

import numpy as np
from skimage.draw import polygon  # pylint:disable=no-name-in-module
from skimage.measure import regionprops  # pylint:disable=no-name-in-module

from cellranger.spatial.stardist.stardist_utils import axes_check_and_normalize

OBJECT_KEYS = set(("prob", "points", "coord", "dist", "class_prob", "class_id"))
COORD_KEYS = set(("points", "coord"))


class Block:
    """One-dimensional block as part of a chain.

    There are no explicit start and end positions. Instead, each block is
    aware of its predecessor and successor and derives such things (recursively)
    based on its neighbors.

    Blocks overlap with one another (at least min_overlap + 2*context) and
    have a read region (the entire block) and a write region (ignoring context).
    Given a query interval, Block.is_responsible will return true for only one
    block of a chain (or raise an exception if the interval is larger than
    min_overlap or even the entire block without context).

    """

    def __init__(self, size, min_overlap, context, pred):
        self.size = int(size)
        self.min_overlap = int(min_overlap)
        self.context = int(context)
        self.pred = pred
        self.succ = None
        assert 0 <= self.min_overlap + 2 * self.context < self.size
        self.stride = self.size - (self.min_overlap + 2 * self.context)
        self._start = 0
        self._frozen = False
        self.extra_context_start = 0
        self.extra_context_end = 0

    @property
    def start(self):
        return self._start if (self.frozen or self.at_begin) else self.pred.succ_start

    @property
    def end(self):
        return self.start + self.size

    @property
    def succ_start(self):
        return self.start + self.stride

    def add_succ(self):
        """Add successor."""
        assert self.succ is None, "self.succ should be None"
        assert not self.frozen, "self.frozen should be False"
        self.succ = Block(self.size, self.min_overlap, self.context, self)
        return self.succ

    def decrease_stride(self, amount):
        """Decrease stride."""
        amount = int(amount)
        assert 0 <= amount, "amount must be non-negative"
        assert amount < self.stride, "amount must be less than self.stride"
        assert not self.frozen, "object must not be frozen"
        self.stride -= amount

    def freeze(self):
        """Call on first block to freeze entire chain (after construction is done)."""
        assert not self.frozen, "Object must not be frozen"
        assert (
            self.at_begin or self.pred.frozen
        ), "Either at_begin must be True or pred must be frozen"
        self._start = self.start
        self._frozen = True
        if not self.at_end:
            self.succ.freeze()

    @property
    def slice_read(self):
        return slice(self.start, self.end)

    @property
    def slice_crop_context(self):
        """Crop context relative to read region."""
        return slice(self.context_start, self.size - self.context_end)

    @property
    def slice_write(self):
        return slice(self.start + self.context_start, self.end - self.context_end)

    def is_responsible(self, bbox):
        """Responsibility for query interval bbox, which is assumed to be smaller than min_overlap.

        If the assumption is met, only one block of a chain will return true.
        If violated, one or more blocks of a chain may raise a NotFullyVisible exception.
        The exception will have an argument that is
            False if bbox is larger than min_overlap, and
            True if bbox is even larger than the entire block without context.

        bbox: (int,int)
            1D bounding box interval with coordinates relative to size without context

        """
        bmin, bmax = bbox

        r_start = (
            0 if self.at_begin else (self.pred.overlap - self.pred.context_end - self.context_start)
        )
        r_end = self.size - self.context_start - self.context_end
        assert 0 <= bmin < bmax <= r_end

        # assert not (bmin == 0 and bmax >= r_start and not self.at_begin), [(r_start,r_end), bbox, self]

        if bmin == 0 and bmax >= r_start:
            if bmax == r_end:
                # object spans the entire block, i.e. is probably larger than size (minus the context)
                raise NotFullyVisible(True)
            if not self.at_begin:
                # object spans the entire overlap region, i.e. is only partially visible here and also by the predecessor block
                raise NotFullyVisible(False)

        # object ends before responsible region start
        if bmax < r_start:
            return False
        # object touches the end of the responsible region (only take if at end)
        if bmax == r_end and not self.at_end:
            return False
        return True

    # ------------------------

    @property
    def frozen(self):
        return self._frozen

    @property
    def at_begin(self):
        return self.pred is None

    @property
    def at_end(self):
        return self.succ is None

    @property
    def overlap(self):
        return self.size - self.stride

    @property
    def context_start(self):
        return 0 if self.at_begin else self.context + self.extra_context_start

    @property
    def context_end(self):
        return 0 if self.at_end else self.context + self.extra_context_end

    def __repr__(self):
        text = f"{self.start:03}:{self.end:03}"
        text += f", write={self.slice_write.start:03}:{self.slice_write.stop:03}"
        text += f", size={self.context_start}+{self.size-self.context_start-self.context_end}+{self.context_end}"
        if not self.at_end:
            text += f", overlap={self.overlap}R/{self.overlap-self.context_end-self.succ.context_start}W"
        return f"{self.__class__.__name__}({text})"

    @property
    def chain(self):
        """Chain."""
        blocks = [self]
        while not blocks[-1].at_end:
            blocks.append(blocks[-1].succ)
        return blocks

    def __iter__(self):
        return iter(self.chain)

    # ------------------------

    @staticmethod
    def cover(size, block_size, min_overlap, context, grid=1, verbose=True):
        """Return chain of grid-aligned blocks to cover the interval [0,size].

        Parameters block_size, min_overlap, and context will be used
        for all blocks of the chain. Only the size of the last block
        may differ.

        Except for the last block, start and end positions of all blocks will
        be multiples of grid. To that end, the provided block parameters may
        be increased to achieve that.

        Note that parameters must be chosen such that the write regions of only
        neighboring blocks are overlapping.

        """
        assert 0 <= min_overlap + 2 * context < block_size <= size
        assert 0 < grid <= block_size
        block_size = _grid_divisible(grid, block_size, name="block_size", verbose=verbose)
        min_overlap = _grid_divisible(grid, min_overlap, name="min_overlap", verbose=verbose)
        context = _grid_divisible(grid, context, name="context", verbose=verbose)

        # allow size not to be divisible by grid
        size_orig = size
        size = _grid_divisible(grid, size, name="size", verbose=False)

        # divide all sizes by grid
        assert all(v % grid == 0 for v in (size, block_size, min_overlap, context))
        size //= grid
        block_size //= grid
        min_overlap //= grid
        context //= grid

        # compute cover in grid-multiples
        t = first = Block(block_size, min_overlap, context, None)
        while t.end < size:
            t = t.add_succ()
        last = t

        # print(); [print(t) for t in first]

        # move blocks around to make it fit
        excess = last.end - size
        t = first
        while excess > 0:
            t.decrease_stride(1)
            excess -= 1
            t = t.succ
            if t == last:
                t = first
        # print(); [print(t) for t in first]

        # add extra context to avoid overlapping write regions of non-neighboring blocks
        t = first
        while not t.at_end:
            if t.succ is not None and t.succ.succ is not None:
                overlap_write = t.slice_write.stop - t.succ.succ.slice_write.start
                if overlap_write > 0:
                    overlap_split1, overlap_split2 = (
                        overlap_write // 2,
                        overlap_write - overlap_write // 2,
                    )
                    t.extra_context_end += overlap_split1
                    t.succ.succ.extra_context_start += overlap_split2
            t = t.succ
        # print(); [print(t) for t in first]

        # make a copy of the cover and multiply sizes by grid
        if grid > 1:
            size *= grid
            block_size *= grid
            min_overlap *= grid
            context *= grid
            _t = first
            t = first = Block(block_size, min_overlap, context, None)
            t.stride = _t.stride * grid
            t.extra_context_start = _t.extra_context_start * grid
            t.extra_context_end = _t.extra_context_end * grid
            while not _t.at_end:
                _t = _t.succ
                t = t.add_succ()
                t.stride = _t.stride * grid
                t.extra_context_start = _t.extra_context_start * grid
                t.extra_context_end = _t.extra_context_end * grid
            last = t

            # change size of last block
            # will be padded internally to the same size
            # as the others by model.predict_instances
            size_delta = size - size_orig
            last.size -= size_delta
            assert 0 <= size_delta < grid

        # for efficiency (to not determine starts recursively from now on)
        first.freeze()

        blocks = first.chain
        # print(); [print(t) for t in first]

        # sanity checks
        assert first.start == 0, "first.start must be 0"
        assert last.end == size_orig, "last.end must be equal to size_orig"
        assert all(t.overlap - 2 * context >= min_overlap for t in blocks if t != last)
        assert all(
            t.slice_write.stop - t.succ.slice_write.start >= min_overlap
            for t in blocks
            if t != last
        )
        assert all(t.start % grid == 0 and t.end % grid == 0 for t in blocks if t != last)
        # print(); [print(t) for t in first]

        # only neighboring blocks should be overlapping
        if len(blocks) >= 3:
            for t in blocks[:-2]:
                assert t.slice_write.stop <= t.succ.succ.slice_write.start

        return blocks


class BlockND:
    """N-dimensional block.

    Each BlockND simply consists of a 1-dimensional Block per axis and also
    has an id (which should be unique). The n-dimensional region represented
    by each BlockND is the intersection of all 1D Blocks per axis.

    Also see `Block`.

    """

    def __init__(self, nd_block_id, blocks, axes):
        self.nd_block_id = nd_block_id
        self.blocks = tuple(blocks)
        self.axes = axes_check_and_normalize(axes, length=len(self.blocks))
        self.axis_to_block = dict(zip(self.axes, self.blocks))

    def blocks_for_axes(self, axes=None):
        axes = self.axes if axes is None else axes_check_and_normalize(axes)
        return tuple(self.axis_to_block[a] for a in axes)

    def slice_read(self, axes=None):
        return tuple(t.slice_read for t in self.blocks_for_axes(axes))

    def slice_crop_context(self, axes=None):
        return tuple(t.slice_crop_context for t in self.blocks_for_axes(axes))

    def slice_write(self, axes=None):
        return tuple(t.slice_write for t in self.blocks_for_axes(axes))

    def read(self, x, axes=None):
        """Read block "read region" from x (numpy.ndarray or similar)."""
        return x[self.slice_read(axes)]

    def crop_context(self, labels, axes=None):
        return labels[self.slice_crop_context(axes)]

    def write(self, x, labels, axes=None):
        """Write (only entries > 0 of) labels to block "write region" of x (numpy.ndarray or similar)."""
        s = self.slice_write(axes)
        mask = labels > 0
        # x[s][mask] = labels[mask] # doesn't work with zarr
        region = x[s]
        region[mask] = labels[mask]
        x[s] = region

    def is_responsible(self, slices, axes=None):
        return all(
            t.is_responsible((s.start, s.stop)) for t, s in zip(self.blocks_for_axes(axes), slices)
        )

    def __repr__(self):
        slices = ",".join(f"{a}={t.start:03}:{t.end:03}" for t, a in zip(self.blocks, self.axes))
        return f"{self.__class__.__name__}({self.nd_block_id}|{slices})"

    def __iter__(self):
        return iter(self.blocks)

    # ------------------------

    def filter_objects(self, labels, polys, axes=None):
        """Filter out objects that block is not responsible for.

        Given label image 'labels' and dictionary 'polys' of polygon/polyhedron objects,
        only retain those objects that this block is responsible for.

        This function will return a pair (labels, polys) of the modified label image and dictionary.
        It will raise a RuntimeError if an object is found in the overlap area
        of neighboring blocks that violates the assumption to be smaller than 'min_overlap'.

        If parameter 'polys' is None, only the filtered label image will be returned.

        Notes:
        -----
        - Important: It is assumed that the object label ids in 'labels' and
          the entries in 'polys' are sorted in the same way.
        - Does not modify 'labels' and 'polys', but returns modified copies.

        Example:
        -------
        >>> labels, polys = model.predict_instances(block.read(img))
        >>> labels = block.crop_context(labels)
        >>> labels, polys, num_nuclei_too_large = block.filter_objects(labels, polys)

        """
        # TODO: option to update labels in-place
        assert np.issubdtype(labels.dtype, np.integer)
        ndim = len(self.blocks_for_axes(axes))
        num_nuclei_too_large = 0
        assert ndim in (2, 3)
        assert (
            labels.ndim == ndim
        ), f"labels.ndim must be equal to ndim, but got {labels.ndim} != {ndim}"
        expected_shape = tuple(s.stop - s.start for s in self.slice_crop_context(axes))
        assert (
            labels.shape == expected_shape
        ), f"labels.shape must be {expected_shape}, but got {labels.shape}"

        labels_filtered = np.zeros_like(labels)
        # problem_ids = []
        for r in regionprops(labels):
            slices = tuple(slice(r.bbox[i], r.bbox[i + labels.ndim]) for i in range(labels.ndim))
            try:
                if self.is_responsible(slices, axes):
                    labels_filtered[slices][r.image] = r.label
            except NotFullyVisible as exc:
                # shape_block_write = tuple(s.stop-s.start for s in self.slice_write(axes))
                shape_object = tuple(s.stop - s.start for s in slices)
                shape_min_overlap = tuple(t.min_overlap for t in self.blocks_for_axes(axes))
                print(
                    f"Found object of shape {shape_object}, which violates the assumption of being smaller than 'min_overlap' "
                    f"{shape_min_overlap}. Increase 'min_overlap' to avoid this problem.\n"
                    f"Error {exc}"
                )
                num_nuclei_too_large += 1

                # if e.args[0]: # object larger than block write region
                #     assert any(o >= b for o,b in zip(shape_object,shape_block_write))
                #     # problem, since this object will probably be saved by another block too
                #     raise RuntimeError(f"Found object of shape {shape_object}, larger than an entire block's write region of shape {shape_block_write}. Increase 'block_size' to avoid this problem.")
                #     # print("found object larger than 'block_size'")
                # else:
                #     assert any(o >= b for o,b in zip(shape_object,shape_min_overlap))
                #     # print("found object larger than 'min_overlap'")

                # # keep object, because will be dealt with later, i.e.
                # # render the poly again into the label image, but this is not
                # # ideal since the assumption is that the object outside that
                # # region is not reliable because it's in the context
                # labels_filtered[slices][r.image] = r.label
                # problem_ids.append(r.label)

        if polys is None:
            # assert len(problem_ids) == 0
            return labels_filtered
        else:
            # it is assumed that ids in 'labels' map to entries in 'polys'
            assert isinstance(polys, dict), "polys must be a dictionary"
            assert any(
                k in polys for k in COORD_KEYS
            ), f"polys must contain at least one of the keys in {COORD_KEYS}"
            filtered_labels = np.unique(labels_filtered)
            filtered_ind = [i - 1 for i in filtered_labels if i > 0]
            polys_out = {k: (v[filtered_ind] if k in OBJECT_KEYS else v) for k, v in polys.items()}
            for k in COORD_KEYS:
                if k in polys_out.keys():
                    polys_out[k] = self.translate_coordinates(polys_out[k], axes=axes)

        return labels_filtered, polys_out, num_nuclei_too_large  # , tuple(problem_ids)

    def translate_coordinates(self, coordinates, axes=None):
        """Translate local block coordinates (of read region) to global ones based on block position."""
        ndim = len(self.blocks_for_axes(axes))
        assert isinstance(coordinates, np.ndarray), "coordinates must be a numpy ndarray"
        assert coordinates.ndim >= 2, "coordinates must have at least 2 dimensions"
        assert (
            coordinates.shape[1] == ndim
        ), f"coordinates.shape[1] must be equal to ndim, but got {coordinates.shape[1]} != {ndim}"
        start = [s.start for s in self.slice_read(axes)]
        shape = tuple(1 if d != 1 else ndim for d in range(coordinates.ndim))
        start = np.array(start).reshape(shape)
        return coordinates + start

    # ------------------------

    @staticmethod
    def cover(shape, axes, block_size, min_overlap, context, grid=1):
        """Return grid-aligned n-dimensional blocks to cover region of the given shape with axes semantics.

        Parameters block_size, min_overlap, and context can be different per
        dimension/axis (if provided as list) or the same (if provided as
        scalar value).

        Also see `Block.cover`.

        """
        shape = tuple(shape)
        n = len(shape)
        axes = axes_check_and_normalize(axes, length=n)
        if np.isscalar(block_size):
            block_size = n * [block_size]
        if np.isscalar(min_overlap):
            min_overlap = n * [min_overlap]
        if np.isscalar(context):
            context = n * [context]
        if np.isscalar(grid):
            grid = n * [grid]
        assert n == len(block_size) == len(min_overlap) == len(context) == len(grid)

        # compute cover for each dimension
        cover_1d = [
            Block.cover(*args) for args in zip(shape, block_size, min_overlap, context, grid)
        ]
        # return cover as Cartesian product of 1-dimensional blocks
        return tuple(BlockND(i, blocks, axes) for i, blocks in enumerate(product(*cover_1d)))


class Polygon:  # pylint:disable=too-few-public-methods
    """Polygon class."""

    def __init__(self, coord, bbox=None, shape_max=None):
        self.bbox = self.coords_bbox(coord, shape_max=shape_max) if bbox is None else bbox
        self.coord = coord - np.array([r[0] for r in self.bbox]).reshape(2, 1)
        self.slice = tuple(slice(*r) for r in self.bbox)
        self.shape = tuple(r[1] - r[0] for r in self.bbox)
        rr, cc = polygon(*self.coord, self.shape)
        self.mask = np.zeros(self.shape, bool)
        self.mask[rr, cc] = True

    @staticmethod
    def coords_bbox(*coords, shape_max=None):
        """Get bounding box."""
        assert all(isinstance(c, np.ndarray) and c.ndim == 2 and c.shape[0] == 2 for c in coords)
        if shape_max is None:
            shape_max = (np.inf, np.inf)
        coord = np.concatenate(coords, axis=1)
        mins = np.maximum(0, np.floor(np.min(coord, axis=1))).astype(int)
        maxs = np.minimum(shape_max, np.ceil(np.max(coord, axis=1))).astype(int)
        return tuple(zip(tuple(mins), tuple(maxs)))


class NotFullyVisible(Exception):
    pass


def _grid_divisible(grid, size, name=None, verbose=True):
    if size % grid == 0:
        return size
    _size = size
    size = math.ceil(size / grid) * grid
    if bool(verbose):
        print(
            f"{verbose if isinstance(verbose,str) else ''}increasing '{'value' if name is None else name}' from {_size} to {size} to be evenly divisible by {grid} (grid)",
            flush=True,
        )
    assert size % grid == 0
    return size
