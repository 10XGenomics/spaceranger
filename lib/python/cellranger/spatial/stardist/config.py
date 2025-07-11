#
# Copyright (c) 2025 10X Genomics, Inc. All rights reserved.
#

"""Default config for Unet Model."""

import argparse
from dataclasses import dataclass

from cellranger.spatial.stardist.stardist_utils import _normalize_grid


@dataclass
class Config2D(argparse.Namespace):  # pylint:disable=too-many-instance-attributes
    """Configuration for a :class:`StarDist2D` model.

    Parameters
    ----------
    axes : str or None
        Axes of the input images.
    n_rays : int
        Number of radial directions for the star-convex polygon.
        Recommended to use a power of 2 (default: 32).
    n_channel_in : int
        Number of channels of given input image (default: 1).
    grid : (int,int)
        Subsampling factors (must be powers of 2) for each of the axes.
        Model will predict on a subsampled grid for increased efficiency and larger field of view.
    n_classes : None or int
        Number of object classes to use for multi-class prediction (use None to disable)
    backbone : str
        Name of the neural network architecture to be used as backbone.
    kwargs : dict
        Overwrite (or add) configuration attributes (see below).


    Attributes:
    ----------
    unet_n_depth : int
        Number of U-Net resolution levels (down/up-sampling layers).
    unet_kernel_size : (int,int)
        Convolution kernel size for all (U-Net) convolution layers.
    unet_n_filter_base : int
        Number of convolution kernels (feature channels) for first U-Net layer.
        Doubled after each down-sampling layer.
    unet_pool : (int,int)
        Maxpooling size for all (U-Net) convolution layers.
    net_conv_after_unet : int
        Number of filters of the extra convolution layer after U-Net (0 to disable).
    unet_* : *
        Additional parameters for U-net backbone.

    """

    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        axes="YXC",
        n_rays=32,
        n_channel_in=1,
        n_channel_out=1,
        grid=(1, 1),
        n_classes=None,
        backbone="unet",
        **kwargs,
    ):
        """See class docstring."""
        super().__init__(**kwargs)
        self.n_dim = len(axes) - 1
        self.axes = axes
        self.n_channel_in = int(max(1, n_channel_in))
        self.n_channel_out = int(max(1, n_channel_out))

        # directly set by parameters
        self.n_rays = int(n_rays)
        self.grid = _normalize_grid(grid, 2)
        self.backbone = str(backbone).lower()
        self.n_classes = None if n_classes is None else int(n_classes)

        # default config (can be overwritten by kwargs below)
        self.unet_n_depth = 3
        self.unet_kernel_size = 3, 3
        self.unet_n_filter_base = 32
        self.unet_n_conv_per_depth = 2
        self.unet_pool = 2, 2
        self.unet_activation = "relu"
        self.unet_last_activation = "relu"
        self.unet_batch_norm = False
        self.unet_dropout = 0.0
        self.unet_prefix = ""
        self.net_conv_after_unet = 128
        self.net_input_shape = None, None, self.n_channel_in

        self.update_parameters(**kwargs)

    def update_parameters(self, **kwargs):
        """Update class params."""
        attr_new = []
        attr_valid = []
        for k in kwargs:
            try:
                getattr(self, k)
                attr_valid.append(k)
            except AttributeError:
                attr_new.append(k)
        for attr in attr_valid:
            setattr(self, attr, kwargs[attr])

    def __post_init__(self):
        if self.axes != "YXC":
            raise ValueError(f"Only 'YXC' axes configuration allowed. Found {self.axes} in config")
        if self.backbone != "unet":
            raise ValueError(
                f"Only 'unet' backbone configuration allowed. Found {self.backbone} in config"
            )
        if self.net_conv_after_unet <= 0:
            raise ValueError(
                f"Parameter net_conv_after_unet has to be positive. Got {self.net_conv_after_unet}."
            )
        if len(self.unet_pool) != len(self.grid):
            raise ValueError(
                f"unet_pool needs to be of equal length to that of grid. Got unet_pool: {self.unet_pool}; grid: {self.grid}."
            )
        if len(self.axes) - 1 != len(self.grid):
            raise ValueError(
                f"The axes_net needs to be length one more than to that of grid. Got unet_pool: {self.axes}; grid: {self.grid}."
            )
        if self.n_dim != 2:
            raise ValueError(f"Only works with 2-D images. Got n_dim {self.n_dim}")
