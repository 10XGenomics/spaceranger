#!/usr/bin/env python
#
# Copyright (c) 2018 10X Genomics, Inc. All rights reserved.
#


import os
import shutil
import subprocess
import tempfile

import martian
from six import ensure_binary, ensure_str

import cellranger.spatial.tiffer as tiffer
import tenkit.log_subprocess as tk_subproc
from cellranger.spatial.loupe_util import LoupeParser, get_remove_image_pages

__MRO__ = """
stage CLOUPE_TILE_IMAGES(
    in  file[] tissue_image_paths,
    in  int    tile_size,
    in  bool   skip_stage,
    in  bool   no_secondary_analysis,
    in  path   loupe_alignment_file,
    out json   dzi_info,
    out path[] dzi_tiles_paths,
    src py     "stages/cloupe/cloupe_tile_images",
) split (
) using (
    mem_gb  = 3,
    vmem_gb = 12,
)
"""
TILE_DATASET_NAME = "tiles"


def do_not_make_tiles(args):
    """Returns True if there is a reason why this stage should not make tiles."""
    if args.no_secondary_analysis:
        martian.log_info("Skipping tile generation by instruction (--no-secondary-analysis)")
        return True
    if args.skip_stage:
        martian.log_info("Skipping tile generation as stage disabled")
        return True
    if args.tissue_image_paths is None or len(args.tissue_image_paths) == 0:
        martian.log_info("Skipping tile generation due to unspecified image path")
        return True
    for path in args.tissue_image_paths:
        if not os.path.exists(path):
            martian.log_info(f"Skipping tile generation due to incorrect dark image path {path}")
            return True
    return False


def split(args):
    # low mem usage if skipped
    if do_not_make_tiles(args):
        return {"chunks": [], "join": {"__mem_gb": 1}}

    if args.tissue_image_paths is not None and len(args.tissue_image_paths) >= 1:
        mem_gb = (
            tiffer.call_tiffer_mem_estimate_gb(args.tissue_image_paths[0], "tile") + 2
        )  # account for possible overhead
    else:
        martian.exit("we have no images to work with")

    mem_gb = max(mem_gb, LoupeParser.estimate_mem_gb_from_json_file(args.loupe_alignment_file) + 1)

    # address space requirement is a bit more than usual relative to
    # memory usage because of the golang process.
    vmem_gb = mem_gb + max(5, mem_gb)
    return {
        "chunks": [],
        "join": {"__mem_gb": round(mem_gb, 3), "__vmem_gb": round(vmem_gb, 3) + 1},
    }


def join(args, outs, chunk_defs, chunk_outs):
    if do_not_make_tiles(args):
        outs.dzi_info = None
        outs.dzi_tiles_paths = []
        return

    out_prefix = martian.make_path("dzi_tiles_paths")
    if not os.path.exists(out_prefix):
        os.makedirs(out_prefix)
    outs.dzi_tiles_paths = []

    remove_image_pages = get_remove_image_pages(args.loupe_alignment_file)

    for imageidx, image in enumerate(args.tissue_image_paths):
        npages = tiffer.call_tiffer_get_num_pages(image)
        for page in range(npages):
            # skip processing of specified TIFF pages
            if page in remove_image_pages:
                martian.log_info(f"Skipping image tiling for page: {page}")
                continue

            tmp_outs = tempfile.mkdtemp()

            if page > 0 and imageidx > 0:  # double checking
                martian.exit(
                    "can't specify multiple image files and multi-page image files in the same experiment"
                )

            tile_dataset_name_page = f"{TILE_DATASET_NAME}_{page}_{imageidx}"

            call = [
                "cloupetiler",
                image,
                tmp_outs,
                tile_dataset_name_page,
                "--tilesize",
                str(args.tile_size),
                "--page",
                str(page),
            ]

            # this is copied from crconverter; some paths may require
            # encoding as a utf-8 byte string for use in check_output.
            call_bytes = [arg.encode("utf-8") for arg in call]

            # but keep the arg 'call' here because log_info attempts to
            # decode the message if we give it a byte string rather than
            # a unicode string.
            martian.log_info("Running cloupetiler: {}".format(" ".join(call)))
            try:
                results = tk_subproc.check_output(call_bytes, stderr=subprocess.STDOUT)
                martian.log_info(f"cloupetiler output: {results}")
                generated_dzi_path = os.path.join(
                    ensure_binary(tmp_outs), b"%s.dzi" % ensure_binary(tile_dataset_name_page)
                )
                shutil.move(generated_dzi_path, outs.dzi_info)
                dataset_path = b"%s_files" % ensure_binary(tile_dataset_name_page)
                shutil.move(
                    os.path.join(ensure_binary(tmp_outs), dataset_path),
                    os.path.join(ensure_binary(out_prefix), dataset_path),
                )
                outs.dzi_tiles_paths.append(os.path.join(out_prefix, dataset_path))
            except subprocess.CalledProcessError as e:
                outs.dzi_info = None
                outs.dzi_tiles_paths = []
                martian.exit(f"Could not generate image tiles: \n{ensure_str(e.output)}")
