#
# Copyright (c) 2024 10X Genomics, Inc. All rights reserved.
#

"""Stage to apply redshift fix to cytassist images if needed."""

import os

import martian

import cellranger.cr_io as cr_io
from cellranger.spatial.cytassist_fix import test_for_cytassist_image_fix
from cellranger.spatial.tiffer import try_call_tiffer_compatibilty_fixes

__MRO__ = """
stage FIX_CYTASSIST_IMAGE_COMPATIBILITY(
    in  file[] cytassist_image_paths_in,
    out file[] cytassist_image_paths,
    src py     "stages/spatial/fix_cytassist_image_compatibility",
) using (
    volatile = strict,
)
"""


def main(args, outs):
    if (
        args.cytassist_image_paths_in
        and len(args.cytassist_image_paths_in) == 1
        and test_for_cytassist_image_fix(args.cytassist_image_paths_in[0])
    ):
        martian.log_info("Doing cytassist channel correction!")
        old_path = args.cytassist_image_paths_in[0]
        tmp_dir = martian.make_path("temp_dir_for_tiff").decode("utf8")
        new_path = try_call_tiffer_compatibilty_fixes(old_path, tmp_dir)
        out_cytassist_path = martian.make_path(os.path.basename(new_path)).decode()
        cr_io.hardlink_with_fallback(new_path, out_cytassist_path)
        outs.cytassist_image_paths = [out_cytassist_path]
    elif args.cytassist_image_paths_in:
        martian.log_info(
            "No cytassist channel correction needed, but some cytassist images provided!"
        )
        outs.cytassist_image_paths = []
        for cyta_img_path in args.cytassist_image_paths_in:
            new_path = martian.make_path(os.path.basename(cyta_img_path)).decode()
            cr_io.hardlink_with_fallback(cyta_img_path, new_path)
            outs.cytassist_image_paths.append(new_path)
    else:
        martian.log_info("No cytassist images provided!")
        outs.cytassist_image_paths = None
