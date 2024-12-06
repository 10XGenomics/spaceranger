#!/usr/bin/env python3
#
# Copyright (c) 2022 10X Genomics, Inc. All rights reserved.
#

"""Stage to process loupe alignment file.

Process loupe alignment json and generate data needed for the following stages.
The positions of the spots are scaled properly to the processing image. The benefits:

1. Process of Loupe alignment json is grouped into one stage. The following stages
only need to deal with the output of this stage. So any changes of the Loupe alignment
json would only affect this stage and LoupeParser class.

2. The fiducial and oligo position are scaled properly using the information in
scalefactors_json. The following stages don't need to worry about scaling at
all except the final output stage.

3. Since Loupe alignment json could contain either spots information, or tissue
registration information, or both. This stage save them into separate files. The
following stages can directly know whether each information exists or not without
reading the original Loupe file.

4. More checks can be added to this stage to make sure that information in Loupe
alignment file is consistent with the input
"""

from __future__ import annotations

import json

import martian

import cellranger.feature.utils as feature_utils
from cellranger.spatial.loupe_util import LoupeParser
from cellranger.spatial.pipeline_mode import PipelineMode, Product

__MRO__ = """
stage LOUPE_ALIGNMENT_READER(
    in  path         loupe_alignment_file,
    in  PipelineMode pipeline_mode,
    in  json         scalefactors_json,
    out json         loupe_spots_data_json,
    out json         loupe_cyta_data_json,
    out json         hd_slide_layout_json,
    out string[]     image_page_names,
    src py           "stages/spatial/loupe_alignment_reader",
) split (
) using (
    volatile = strict,
)
"""


def split(args):
    return {
        "chunks": [],
        "join": {
            "__mem_gb": max(
                1.0, LoupeParser.estimate_mem_gb_from_json_file(args.loupe_alignment_file)
            ),
        },
    }


def join(args, outs, _chunk_defs, _chunk_outs):
    if args.loupe_alignment_file is None:
        outs.loupe_spots_data_json = None
        outs.loupe_cyta_data_json = None
        outs.image_page_names = None
        outs.hd_slide_layout_json = None
        return
    pipeline_mode = PipelineMode(**args.pipeline_mode)
    try:
        pipeline_mode.validate()
    except ValueError:
        martian.throw(f"Invalid pipeline mode of {pipeline_mode}")
    (
        loupe_data,
        outs.loupe_spots_data_json,
        outs.loupe_cyta_data_json,
        outs.hd_slide_layout_json,
    ) = parse_loupe_json(
        args.loupe_alignment_file,
        pipeline_mode,
        args.scalefactors_json,
        outs.loupe_spots_data_json,
        outs.loupe_cyta_data_json,
        outs.hd_slide_layout_json,
    )
    outs.image_page_names = loupe_data.get_image_page_names()


def parse_loupe_json(
    loupe_alignment_file_path: str,
    pipeline_mode: PipelineMode,
    scalefactors_json: str,
    loupe_spots_data_json: str | None,
    loupe_cyta_data_json: str | None,
    hd_slide_layout_json: str | None,
) -> tuple[LoupeParser, str | None, str | None, str | None]:
    """Parse the loupe manual json file."""
    with open(scalefactors_json) as f:
        scalefactors_dict = json.load(f)
    process_img_scalef = scalefactors_dict.get("process_img_scalef")
    loupe_data = LoupeParser(loupe_alignment_file_path)

    if loupe_data.contain_spots_info():
        spots_data = loupe_data.get_spots_data(scale=process_img_scalef)
        feature_utils.write_json_from_dict(spots_data, loupe_spots_data_json)
    else:
        loupe_spots_data_json = None
    if loupe_data.contain_cyta_info():
        regist_target_scalef = scalefactors_dict.get("regist_target_img_scalef")
        if pipeline_mode.product != Product.CYT:
            martian.throw(
                "Find Cytassist information in Loupe json but not in Cytassist mode. \
                Possibly wrong image arguments."
            )
        # The requirement here is that CytAssist image is NOT going to be scaled!
        cyta_data = loupe_data.get_cyta_data(scale=1.0 / regist_target_scalef)
        feature_utils.write_json_from_dict(cyta_data, loupe_cyta_data_json)
    else:
        loupe_cyta_data_json = None

    hd_layout = loupe_data.hd_slide_layout()
    if hd_layout:
        hd_layout.save_as_json(hd_slide_layout_json)
    else:
        hd_slide_layout_json = None
    return (loupe_data, loupe_spots_data_json, loupe_cyta_data_json, hd_slide_layout_json)
