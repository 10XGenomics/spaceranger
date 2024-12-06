#!/usr/bin/env python
#
# Copyright (c) 2024 10X Genomics, Inc. All rights reserved.
#
"""Pick saturation plot of the required bin level."""


import martian

import cellranger.cr_io as cr_io
from cellranger.websummary.analysis_tab_aux import CANONICAL_VISIUM_HD_BIN_NAME

__MRO__ = """
stage PICK_SATURATION_PLOT(
    in  map<json> saturation_plots,
    out json      saturation_plots_picked,
    src py        "stages/spatial/pick_saturation_plot",
)
"""


def main(args, outs):
    if not args.saturation_plots or not args.saturation_plots.get(CANONICAL_VISIUM_HD_BIN_NAME):
        outs.saturation_plots_picked = None
        return

    old_canonical_saturation_plot = args.saturation_plots.get(CANONICAL_VISIUM_HD_BIN_NAME)
    cr_io.hardlink_with_fallback(
        old_canonical_saturation_plot, martian.make_path(outs.saturation_plots_picked).decode()
    )
