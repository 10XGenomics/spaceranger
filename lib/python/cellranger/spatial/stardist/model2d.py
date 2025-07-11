#
# Copyright (c) 2025 10X Genomics, Inc. All rights reserved.
#
"""2d Model."""

import datetime
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
from scipy.ndimage import zoom
from tqdm import tqdm

from cellranger.spatial.stardist.big import OBJECT_KEYS, BlockND, _grid_divisible
from cellranger.spatial.stardist.config import Config2D
from cellranger.spatial.stardist.geom2d import dist_to_coord, polygons_to_label
from cellranger.spatial.stardist.matching import relabel_sequential
from cellranger.spatial.stardist.nms import ind_prob_thresh, non_maximum_suppression_sparse
from cellranger.spatial.stardist.predict import tile_iterator, total_n_tiles
from cellranger.spatial.stardist.prepare import NoNormalizer, StarDistPadAndCropResizer
from cellranger.spatial.stardist.stardist_utils import _is_power_of_2, load_json
from cellranger.spatial.stardist.types import AxisMap

CANONICAL_CONFIG_NAME = "config.json"
CANONICAL_THRESHOLDS_NAME = "thresholds.json"
CANONICAL_WEIGHTS_NAME = "model.onnx"
DEFAULT_PROBS_THRESHOLD = 0.5
DEFAULT_NMS_THRESHOLD = 0.4


@dataclass
class Thresholds:
    """Thresholds used in Stardist2D class."""

    prob: float = DEFAULT_PROBS_THRESHOLD
    nms: float = DEFAULT_NMS_THRESHOLD

    def __post_init__(self):
        """Validate that prob and nms are between 0 and 1 if they are not None."""
        if not 0 <= self.prob <= 1:
            raise ValueError(f"prob must be between 0 and 1, but got {self.prob}")

        if not 0 <= self.nms <= 1:
            raise ValueError(f"nms must be between 0 and 1, but got {self.nms}")


class StarDist2D:
    """StarDist2D model.

    Parameters
    ----------
    config : :class:`Config` or None
        If set to ``None``, will be loaded from disk (must exist).
    name : str or None
        Model name. Uses a timestamp if set to ``None`` (default).
    basedir : str
        Directory that contains (or will contain) a folder with the given model name.

    Raises:
    ------
    FileNotFoundError
        If ``config=None`` and config cannot be loaded from disk.
    ValueError
        Illegal arguments, including invalid configuration.

    Attributes:
    ----------
    config : :class:`Config`
        Configuration, as provided during instantiation.
    onnx_model : Onnx session wrapping the keras model.
    name : str
        Model name.
    basedir : :class:`pathlib.Path`
        Path to model folder (which stores configuration, weights, etc.)
    """

    def __init__(
        self,
        config=None,
        name=None,
        basedir: str | bytes | Path = ".",
        onnx_session_opts: ort.SessionOptions | None = None,
    ):
        if not Path(str(basedir)).exists():
            raise ValueError(f"Basedir provided does not exist: {basedir!s}")

        self.config = config
        self.basedir = Path(str(basedir)) if name is None else Path(str(basedir)) / name
        self.name = (
            name if name is not None else datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%f")
        )

        if self.config is None:
            if (config_file := self.basedir / CANONICAL_CONFIG_NAME).exists():
                config_dict = load_json(str(config_file))
                self.config = Config2D(**config_dict)
            else:
                raise FileNotFoundError(
                    "No config file passed in and none found. "
                    f"config file doesn't exist: {config_file.resolve()!s}"
                )
        if onnx_session_opts is None:
            onnx_session_opts = ort.SessionOptions()
            onnx_session_opts.intra_op_num_threads = 1

        self.onnx_model = self.load_weights(onnx_session_opts)

        self._tile_overlap = self._compute_receptive_field()
        self.axes_to_coordinate = AxisMap.from_axes(self.config.axes)
        self.axes_to_grid = AxisMap.from_axes_and_values(
            self.config.axes, tuple(self.config.grid), default_value=1
        )
        self.axes_divs = self.div_axes()
        if (thresholds_path := self.basedir / CANONICAL_THRESHOLDS_NAME).exists():
            self.thresholds = Thresholds(**load_json(str(thresholds_path)))
        else:
            self.thresholds = Thresholds()
        print(f"Using default values: {self.thresholds}.")

    def __repr__(self):
        s = (
            f"{self.__class__.__name__}({self.name}): {self.config.axes} → {self._axes_out}\n"
            f"├─ Directory: {self.basedir.resolve() if self.basedir is not None else None}\n"
            f"└─ {self.config}"
        )
        return s

    def load_weights(self, onnx_session_opts, name=CANONICAL_WEIGHTS_NAME):
        """Load neural network weights from model folder."""
        print(f"Loading network weights from '{self.basedir / name!s}'.")
        return ort.InferenceSession(str(self.basedir / name), onnx_session_opts)

    def raw_predict(self, inputs):
        return self.onnx_model.run(["prob", "dist"], {"input": inputs})

    def _compute_receptive_field(self):
        """Compute the max distance a pixel effects on every image dimension."""
        img_size = tuple(g * 128 for g in self.config.grid)
        print(img_size)
        assert all(_is_power_of_2(s) for s in img_size)
        mid_img_coord = tuple(s // 2 for s in img_size)
        impulse_at_center_img = np.zeros(
            (1,) + img_size + (self.config.n_channel_in,), dtype=np.float32
        )
        impulse_at_center_img[(0,) + mid_img_coord + (slice(None),)] = 1
        all_zero_image = np.zeros_like(impulse_at_center_img)
        impulse_response_img = self.raw_predict(impulse_at_center_img)[0][0, ..., 0]
        zero_response_img = self.raw_predict(all_zero_image)[0][0, ..., 0]
        grid = tuple(
            (
                np.array(impulse_at_center_img.shape[1:-1]) / np.array(impulse_response_img.shape)
            ).astype(int)
        )
        assert grid == self.config.grid
        impulse_response_img = zoom(impulse_response_img, grid, order=0)
        zero_response_img = zoom(zero_response_img, grid, order=0)
        idx_with_effect_of_impulse = np.where(np.abs(impulse_response_img - zero_response_img) > 0)
        if any(len(i) == 0 for i in idx_with_effect_of_impulse):
            raw_overlaps = [(m, m) for m in mid_img_coord]
        else:
            raw_overlaps = [
                (m - np.min(i), np.max(i) - m)
                for (m, i) in zip(mid_img_coord, idx_with_effect_of_impulse)
            ]
        return AxisMap.from_axes_and_values(
            self.config.axes, tuple(max(rf) for rf in raw_overlaps), default_value=0
        )

    def div_axes(self):
        return AxisMap.from_axes_and_values(
            self.config.axes,
            tuple(
                p**self.config.unet_n_depth * g
                for p, g in zip(self.config.unet_pool, self.config.grid)
            ),
            default_value=1,
        )

    @property
    def _axes_out(self):
        return self.config.axes

    def _predict_direct(self, x, **_kwargs):
        ys = self.raw_predict(x[np.newaxis])
        return tuple(y[0] for y in ys)

    def tiling_setup(self, normalized_img, n_tiles):
        """Setup tiling."""
        assert np.prod(n_tiles) > 1
        axes_net = self.config.axes
        axes_net_div_by = self.axes_divs.get_tuple(axes_net)

        axes_net_tile_overlaps = self._tile_overlap.get_tuple(axes_net)
        if n_tiles[self.axes_to_coordinate.C] != 1:
            raise ValueError("entry of n_tiles > 1 only allowed for axes `X` and `Y`.")

        output_shape = AxisMap(
            Y=normalized_img.shape[self.axes_to_coordinate.Y] // self.axes_to_grid.Y,
            X=normalized_img.shape[self.axes_to_coordinate.X] // self.axes_to_grid.X,
            C=1,
        )

        n_block_overlaps = [
            int(np.ceil(overlap / blocksize))
            for overlap, blocksize in zip(axes_net_tile_overlaps, axes_net_div_by)
        ]

        num_tiles_used = total_n_tiles(
            normalized_img,
            n_tiles,
            block_sizes=axes_net_div_by,
            n_block_overlaps=n_block_overlaps,
        )

        tile_generator = tqdm(
            tile_iterator(
                normalized_img,
                n_tiles,
                block_sizes=axes_net_div_by,
                n_block_overlaps=n_block_overlaps,
            ),
            total=num_tiles_used,
        )

        return tile_generator, output_shape

    def predict(  # pylint: disable=too-many-locals
        self,
        img,
        n_tiles,
        prob_thresh=None,
        normalizer=None,
        b=2,
        **predict_kwargs,
    ):
        """Main prediction module of the model.

        Returns:
        -------
        (prob, dist, points)   flat list of probs, dists, and points
        """
        if prob_thresh is None:
            prob_thresh = self.thresholds.prob

        predict_kwargs.setdefault("verbose", 0)

        grid_dict = self.axes_to_grid
        axes_net = self.config.axes
        channel = self.axes_to_coordinate.C

        if self.config.n_channel_in != img.shape[channel]:
            raise ValueError(
                f"Only expect images of {self.config.n_channel_in}; Input image has {img.shape[channel]}."
            )

        if normalizer is None:
            normalizer = NoNormalizer()
        resizer = StarDistPadAndCropResizer(grid=grid_dict)

        normalized_img = normalizer.before(img, axes_net)
        normalized_img = resizer.before(
            normalized_img, axes_net, self.axes_divs.get_tuple(axes_net)
        )

        def _prep(prob, dist):
            prob = np.take(prob, 0, axis=channel)
            dist = np.moveaxis(dist, channel, -1)
            dist = np.maximum(1e-3, dist)
            return prob, dist

        proba, dista, pointsa = [], [], []

        if np.prod(n_tiles) > 1:
            tile_generator, output_shape = self.tiling_setup(normalized_img, n_tiles)

            proba, dista, pointsa = [], [], []

            for tile, s_src, s_dst in tile_generator:

                results_tile = self._predict_direct(tile, **predict_kwargs)

                # account for grid
                s_src = [
                    slice(s.start // grid_dict.get(a), s.stop // grid_dict.get(a))
                    for s, a in zip(s_src, axes_net)
                ]
                s_dst = [
                    slice(s.start // grid_dict.get(a), s.stop // grid_dict.get(a))
                    for s, a in zip(s_dst, axes_net)
                ]
                s_src[channel] = slice(None)
                s_dst[channel] = slice(None)
                s_src, s_dst = tuple(s_src), tuple(s_dst)

                prob_tile, dist_tile = results_tile[:2]
                prob_tile, dist_tile = _prep(prob_tile[s_src], dist_tile[s_src])

                bs = AxisMap.from_axes_and_values(
                    axes_net,
                    tuple(
                        (b if s.start == 0 else -1, b if s.stop == _sh else -1)
                        for s, _sh in zip(s_dst, output_shape.get_tuple(axes_net))
                    ),
                )
                inds = ind_prob_thresh(
                    prob_tile, prob_thresh, b=bs.get_tuple(axes_net.replace("C", ""))
                )
                proba.extend(prob_tile[inds].copy())
                dista.extend(dist_tile[inds].copy())
                _points = np.stack(np.where(inds), axis=1)
                offset = list(s.start for i, s in enumerate(s_dst))
                offset.pop(channel)
                _points = _points + np.array(offset).reshape((1, len(offset)))
                _points = _points * np.array(self.config.grid).reshape((1, len(self.config.grid)))
                pointsa.extend(_points)

        else:
            # predict_direct -> prob, dist, [prob_class if multi_class]
            results = self._predict_direct(normalized_img, **predict_kwargs)
            prob, dist = results[:2]
            prob, dist = _prep(prob, dist)
            inds = ind_prob_thresh(prob, prob_thresh, b=b)
            proba = prob[inds].copy()
            dista = dist[inds].copy()
            _points = np.stack(np.where(inds), axis=1)
            pointsa = _points * np.array(self.config.grid).reshape((1, len(self.config.grid)))

        proba = np.asarray(proba)
        dista = np.asarray(dista).reshape((-1, self.config.n_rays))
        pointsa = np.asarray(pointsa).reshape((-1, self.config.n_dim))

        idx = resizer.filter_points(normalized_img.ndim, pointsa, axes_net)
        proba = proba[idx]
        dista = dista[idx]
        pointsa = pointsa[idx]

        return proba, dista, pointsa

    def _instances_from_prediction(  # pylint:disable=too-many-arguments,too-many-positional-arguments,too-many-locals
        self,
        img_shape,
        prob,
        dist,
        points,
        prob_thresh=None,
        nms_thresh=None,
        return_labels=True,
    ):
        """Get instance labels."""
        n_poly = dist.shape[0]
        if n_poly == 0:
            return None, None

        if prob_thresh is None:
            prob_thresh = self.thresholds.prob
        if nms_thresh is None:
            nms_thresh = self.thresholds.nms

        points, probi, disti, _indsi = non_maximum_suppression_sparse(
            dist,
            prob,
            points,
            nms_thresh=nms_thresh,
        )

        rescale = (1, 1)
        if return_labels:
            labels = polygons_to_label(
                disti, points, prob=probi, shape=img_shape, scale_dist=rescale
            )
        else:
            labels = None

        coord = dist_to_coord(disti, points, scale_dist=rescale)
        res_dict = dict(coord=coord, points=points, prob=probi)

        return labels, res_dict

    def predict_instances(  # pylint:disable=too-many-arguments,too-many-positional-arguments,too-many-locals
        self,
        img,
        axes,
        normalizer=None,
        prob_thresh=None,
        nms_thresh=None,
        n_tiles=None,
        return_labels=True,
        predict_kwargs=None,
    ):
        """Predict instance segmentation from input image.

        Parameters
        ----------
        img : :class:`numpy.ndarray`
            Input image
        axes : str or None
            Axes of the input ``img``.
        normalizer : :class:`csbdeep.data.Normalizer` or None
            (Optional) normalization of input image before prediction.
            Note that the default (``None``) assumes ``img`` to be already normalized.
        prob_thresh : float or None
            Consider only object candidates from pixels with predicted object probability
            above this threshold (also see `optimize_thresholds`).
        nms_thresh : float or None
            Perform non-maximum suppression that considers two objects to be the same
            when their area/surface overlap exceeds this threshold (also see `optimize_thresholds`).
        n_tiles : iterable or None
            Out of memory (OOM) errors can occur if the input image is too large.
            To avoid this problem, the input image is broken up into (overlapping) tiles
            that are processed independently and re-assembled.
            This parameter denotes a tuple of the number of tiles for every image axis (see ``axes``).
            ``None`` denotes that no tiling should be used.
        return_labels: bool
            Whether to create a label image, otherwise return None in its place.
        predict_kwargs: dict
            Keyword arguments for ``predict`` function of Keras model.

        Returns:
        -------
        (:class:`numpy.ndarray`, dict)
            Returns a tuple of the label instances image and also
            a dictionary with the details (coordinates, etc.) of all remaining polygons/polyhedra.

        """
        if n_tiles is None:
            n_tiles = [1] * img.ndim
        if img.ndim != len(n_tiles):
            raise (
                ValueError(
                    "The image has to have same dimensions as the tiles. "
                    f"Image dimensions {img.ndim}, tiles {n_tiles}."
                )
            )
        if axes != self.config.axes:
            raise ValueError("Only YXC image type accepted.")

        _shape_inst = tuple(
            img.shape[i] for i in self.axes_to_coordinate.get_non_channel_values(self.config.axes)
        )

        print("prediction is starting")  # indicate that prediction is starting
        prob, dist, points = self.predict(
            img,
            n_tiles=n_tiles,
            normalizer=normalizer,
            prob_thresh=prob_thresh,
            **(predict_kwargs if predict_kwargs is not None else {}),
        )

        print("Starting NMS.")
        res_instances = self._instances_from_prediction(
            _shape_inst,
            prob,
            dist,
            points=points,
            prob_thresh=prob_thresh,
            nms_thresh=nms_thresh,
            return_labels=return_labels,
        )

        return res_instances

    def predict_instances_big(  # pylint:disable=too-many-arguments,too-many-positional-arguments,too-many-locals
        self,
        img,
        axes,
        block_size,
        min_overlap,
        context=None,
        labels_out=None,
        labels_out_dtype=np.int32,
        **kwargs,
    ):
        """Predict instance segmentation from very large input images.

        Intended to be used when `predict_instances` cannot be used due to memory limitations.
        This function will break the input image into blocks and process them individually
        via `predict_instances` and assemble all the partial results. If used as intended, the result
        should be the same as if `predict_instances` was used directly on the whole image.

        **Important**: The crucial assumption is that all predicted object instances are smaller than
                       the provided `min_overlap`. Also, it must hold that: min_overlap + 2*context < block_size.

        Example:
        -------
        >>> img.shape
        (20000, 20000)
        >>> labels, polys, num_nuclei_too_large = model.predict_instances_big(img, axes='YX', block_size=4096,
                                                        min_overlap=128, context=128, n_tiles=(4,4))

        Parameters
        ----------
        img: :class:`numpy.ndarray` or similar
            Input image
        axes: str
            Axes of the input ``img`` (only 'YXC' accepted.)
        block_size: int or iterable of int
            Process input image in blocks of the provided shape.
            (If a scalar value is given, it is used for all spatial image dimensions.)
        min_overlap: int or iterable of int
            Amount of guaranteed overlap between blocks.
            (If a scalar value is given, it is used for all spatial image dimensions.)
        context: int or iterable of int, or None
            Amount of image context on all sides of a block, which is discarded.
            If None, uses an automatic estimate that should work in many cases.
            (If a scalar value is given, it is used for all spatial image dimensions.)
        labels_out: :class:`numpy.ndarray` or similar, or None, or False
            numpy array or similar (must be of correct shape) to which the label image is written.
            If None, will allocate a numpy array of the correct shape and data type ``labels_out_dtype``.
            If False, will not write the label image (useful if only the dictionary is needed).
        labels_out_dtype: str or dtype
            Data type of returned label image if ``labels_out=None`` (has no effect otherwise).
        kwargs: dict
            Keyword arguments for ``predict_instances``.

        Returns:
        -------
        (:class:`numpy.ndarray` or False, dict)
            Returns the label image and a dictionary with the details (coordinates, etc.) of the polygons/polyhedra.

        """
        n = img.ndim
        if axes != self.config.axes or n != 3:
            raise ValueError("Only YXC axes image type accepted (and thus three channel images).")
        grid = self.axes_divs.get_tuple(axes)
        axes_out = self._axes_out.replace("C", "")
        shape_dict = dict(zip(axes, img.shape))
        shape_out = tuple(shape_dict[a] for a in axes_out)

        if context is None:
            context = self._tile_overlap.get_tuple(axes)

        if np.isscalar(block_size):
            block_size = n * [block_size]
        if np.isscalar(min_overlap):
            min_overlap = n * [min_overlap]
        if np.isscalar(context):
            context = n * [context]
        block_size, min_overlap, context = list(block_size), list(min_overlap), list(context)
        assert n == len(block_size) == len(min_overlap) == len(context)

        block_size[self.axes_to_coordinate.C] = img.shape[self.axes_to_coordinate.C]
        min_overlap[self.axes_to_coordinate.C] = context[self.axes_to_coordinate.C] = 0

        block_size = tuple(
            _grid_divisible(g, v, name="block_size", verbose=False)
            for v, g in zip(block_size, grid)
        )
        min_overlap = tuple(
            _grid_divisible(g, v, name="min_overlap", verbose=False)
            for v, g in zip(min_overlap, grid)
        )
        context = tuple(
            _grid_divisible(g, v, name="context", verbose=False) for v, g in zip(context, grid)
        )

        print(
            f"effective: block_size={block_size}, min_overlap={min_overlap}, context={context}",
            flush=True,
        )

        for a, c, o in zip(axes, context, self._tile_overlap.get_tuple(axes)):
            if c < o:
                print(f"{a}: context of {c} is small, recommended to use at least {o}", flush=True)

        # create block cover
        blocks = BlockND.cover(img.shape, axes, block_size, min_overlap, context, grid)

        if np.isscalar(labels_out) and bool(labels_out) is False:
            labels_out = None
        elif labels_out is None:
            labels_out = np.zeros(shape_out, dtype=labels_out_dtype)
        elif labels_out.shape != shape_out:
            raise ValueError(f"'labels_out' must have shape {shape_out} (axes {axes_out}).")

        polys_all = {}
        label_offset = 1

        kwargs_override = dict(axes=axes, return_labels=True)
        for k, v in kwargs_override.items():
            if k in kwargs:
                print(f"changing '{k}' from {kwargs[k]} to {v}", flush=True)
            kwargs[k] = v

        blocks = tqdm(blocks)
        # actual computation
        num_nuclei_too_large = 0
        for block in blocks:
            labels, polys = self.predict_instances(block.read(img, axes=axes), **kwargs)
            if labels is None:
                continue
            labels = block.crop_context(labels, axes=axes_out)
            labels, polys, block_num_nuclei_too_large = block.filter_objects(
                labels, polys, axes=axes_out
            )
            num_nuclei_too_large += block_num_nuclei_too_large
            # TODO: relabel_sequential is not very memory-efficient (will allocate memory proportional to label_offset)
            # this should not change the order of labels
            labels = relabel_sequential(labels, label_offset)[0]

            if labels_out is not None:
                block.write(labels_out, labels, axes=axes_out)

            for k, v in polys.items():
                polys_all.setdefault(k, []).append(v)

            label_offset += len(polys["prob"])
            del labels

        polys_all = {
            k: (np.concatenate(v) if k in OBJECT_KEYS else v[0]) for k, v in polys_all.items()
        }

        return labels_out, polys_all, num_nuclei_too_large
