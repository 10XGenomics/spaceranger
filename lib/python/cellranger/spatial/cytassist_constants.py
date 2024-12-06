#!/usr/bin/env python
#
# Copyright (c) 2023 10X Genomics, Inc. All rights reserved.
#

"""Constants associated with the cytassist data."""

CYTA_IMAGE_DIM = 3000
CYTA_HD_IMAGE_WIDTH = 3200
CYTA_IMAGE_ALLOWED_WIDTHS = [CYTA_IMAGE_DIM, CYTA_HD_IMAGE_WIDTH]
CYTA_IMAGE_ALLOWED_HEIGHTS = [CYTA_IMAGE_DIM]

VALID_CYTASSIST_IMAGE_EXTENSIONS = [
    ".tiff",
    ".tif",
    ".jpeg",
    ".jpg",
    ".qptiff",
    ".btf",
    ".tf2",
    ".tf8",
]
VALID_CYTASSIST_SUPPORT_EXTENSIONS = [".tgz"]
