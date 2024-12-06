# Copyright (c) 2024 10x Genomics, Inc. All rights reserved.
"""Gets cytassist run metadata."""

import json

import martian

from cellranger.spatial.tiffer import call_tiffer_info

___MRO__ = """
stage GET_CYTASSIST_RUN_METADATA(
    in  file[] cytassist_image_paths,
    out json   cytassist_run_metrics,
    src py     "stages/spatial/get_cytassist_run_metadata",
) using (
    volatile = strict,
)
"""


def main(args, outs):
    if not args.cytassist_image_paths or len(args.cytassist_image_paths) > 1:
        martian.clear(outs)
        return

    cyta_img_path = args.cytassist_image_paths[0]
    metadata = call_tiffer_info(cyta_img_path)
    with open(outs.cytassist_run_metrics, "w") as f:
        json.dump(metadata, f, indent=4)
