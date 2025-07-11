#!/usr/bin/env python
#
# Copyright (c) 2019 10X Genomics, Inc. All rights reserved.
#
"""Preflight checks intended to be run on the submission host in a cluster environment."""


import martian

import cellranger.preflight as cr_preflight
import cellranger.spatial.preflight as cr_sp_preflight

__MRO__ = """
stage SPACERANGER_PREFLIGHT_LOCAL(
    in  map[]            sample_def,
    in  csv              target_set,
    in  path             reference_path,
    in  csv              feature_reference,
    in  int              recovered_cells,
    in  int              force_cells,
    in  int              r1_length,
    in  int              r2_length,
    in  file[]           tissue_image_paths,
    in  int              dark_images,
    in  path             loupe_alignment_file,
    in  gpr              gpr_file,
    in  vlf              hd_layout_file,
    in  file[]           cytassist_image_paths,
    in  string           slide_serial_capture_area,
    in  string           targeting_method,
    in  string           chemistry,
    in  ChemistryDef     custom_chemistry_def,
    in  V1PatternFixArgs v1_pattern_fix,
    in  bool             is_pd,
    src py               "stages/common/spaceranger_preflight_local",
)
"""


def main(args, _):
    try:
        run_preflight_checks(args)
    except cr_preflight.PreflightException as ex:
        martian.exit(ex.msg)

    cr_preflight.record_package_versions()


def run_preflight_checks(args):
    """Run checks appropriate for a submission host in a cluster environment."""
    # don't parse files or check executables in the local preflight
    full_check = False

    cr_preflight.check_sample_info(args.sample_def, args.reference_path, full_check)
    cr_preflight.check_common_preflights(
        full_check,
        args.reference_path,
        args.r1_length,
        args.r2_length,
        args.recovered_cells,
        args.force_cells,
    )

    is_hd_run = cr_sp_preflight.is_hd_upload(args.chemistry, args.custom_chemistry_def)
    cr_sp_preflight.check_spatial_image_paths(
        args.tissue_image_paths,
        args.cytassist_image_paths,
        args.dark_images,
    )
    cr_sp_preflight.check_spatial_arguments(
        args.loupe_alignment_file,
        args.gpr_file,
        args.hd_layout_file,
        args.slide_serial_capture_area,
        is_hd_run=is_hd_run,
        hd_log_umi_image=None,
        is_pd=args.is_pd,
    )
    cr_sp_preflight.check_pattern_arguments(
        args.v1_pattern_fix,
    )
    cr_preflight.check_feature_preflights(args.sample_def, args.feature_reference)
    cr_preflight.check_targeting_preflights(
        args.target_set,
        args.reference_path,
        args.feature_reference,
        parse_files=full_check,
        expected_targeting_method=args.targeting_method,
        is_spatial=True,
    )
    if args.tissue_image_paths:
        cr_sp_preflight.check_images_exist(
            args.tissue_image_paths, args.dark_images, args.loupe_alignment_file
        )
        cr_sp_preflight.check_images_consistent(args.tissue_image_paths)
