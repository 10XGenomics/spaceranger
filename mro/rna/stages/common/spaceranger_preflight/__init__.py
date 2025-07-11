#!/usr/bin/env python
#
# Copyright (c) 2019 10X Genomics, Inc. All rights reserved.
#

import tarfile
import tempfile

import martian

import cellranger.csv_utils as cr_csv_utils
import cellranger.preflight as cr_preflight
import cellranger.spatial.preflight as cr_sp_preflight
import cellranger.spatial.tiffer as tiffer
from cellranger.feature_ref import FeatureDefException
from cellranger.spatial.cytassist_fix import test_for_cytassist_image_fix
from cellranger.spatial.data_utils import (
    get_all_images_from_tgz_folder,
    get_cytassist_capture_area,
    get_cytassist_images_from_extracted_tgz_folder,
)

__MRO__ = """
stage SPACERANGER_PREFLIGHT(
    in  map[]              sample_def,
    in  csv                target_set,
    in  path               reference_path,
    in  csv                feature_reference,
    in  int                recovered_cells,
    in  int                force_cells,
    in  int                r1_length,
    in  int                r2_length,
    in  file[]             tissue_image_paths,
    in  int                dark_images,
    in  int                dapi_channel_index,
    in  path               loupe_alignment_file,
    in  bool               override_id,
    in  gpr                gpr_file,
    in  vlf                hd_layout_file,
    in  file[]             cytassist_image_paths,
    in  tgz                cytassist_tgz_path,
    in  bool               check_cytassist_sizes,
    in  string             slide_serial_capture_area,
    in  string             targeting_method,
    in  float              image_scale,
    in  string             chemistry,
    in  ChemistryDef       custom_chemistry_def,
    in  V1PatternFixArgs   v1_pattern_fix,
    in  SegmentationInputs segmentation_inputs,
    in  path               hd_log_umi_image,
    in  bool               is_pd,
    src py                 "stages/common/spaceranger_preflight",
) split (
) using (
    mem_gb  = 3,
    vmem_gb = 32,
)
"""


def split(args):
    mem_gb = 8
    # tiffer memory maps the file, so only use tiffer estimate for vmem
    # start w/ default and increase to file-based estimate + 3 GB for overhead.
    if args.tissue_image_paths:
        vmem_gb = max(mem_gb, tiffer.call_tiffer_mem_estimate_gb(args.tissue_image_paths[0], 0)) + 4
    else:
        vmem_gb = mem_gb + 4
    return {"chunks": [], "join": {"__mem_gb": mem_gb, "__vmem_gb": vmem_gb}}


def main(args, outs):
    martian.throw("main is not supposed to run.")


def join(args, outs, _chunk_defs, _chunk_outs):
    print(f"outs={outs}")
    try:
        run_preflight_checks(args)
    except (
        cr_preflight.PreflightException,
        cr_csv_utils.CSVParseException,
        FeatureDefException,
    ) as e:
        martian.exit(e.msg)

    cr_preflight.record_package_versions()


def run_preflight_checks(args):
    full_check = True
    segmentation_inputs = (
        cr_sp_preflight.SegmentationInputs(**dict(args.segmentation_inputs.items()))
        if args.segmentation_inputs
        else None
    )

    cr_preflight.check_os()

    cr_preflight.check_sample_info(
        args.sample_def, args.reference_path, full_check, args.feature_reference
    )
    cr_preflight.check_common_preflights(
        full_check,
        args.reference_path,
        args.r1_length,
        args.r2_length,
        args.recovered_cells,
        args.force_cells,
    )

    # If no cytassist images passed in but a tgz file and slide ID passed in,
    #  extract images from tgz file.

    cytassist_image_paths = args.cytassist_image_paths
    if (
        not args.cytassist_image_paths
        and args.cytassist_tgz_path
        and args.slide_serial_capture_area
    ):
        # Create a temporary folder to extract the TGZ file into
        tmp_dir = tempfile.mkdtemp()

        # Extract TGZ file
        with tarfile.open(args.cytassist_tgz_path) as tar:
            tar.extractall(tmp_dir)

        # Get capture area and extract images
        capture_area = get_cytassist_capture_area(args.slide_serial_capture_area)
        cytassist_image_paths = get_cytassist_images_from_extracted_tgz_folder(
            tmp_dir, capture_area
        )

        if not cytassist_image_paths:
            sample_images = get_all_images_from_tgz_folder(tmp_dir)
            sample_image_string = ",\n".join(sample_images) if sample_images else "None found."
            raise cr_preflight.PreflightException(
                f"No cytassist images found for capture area {capture_area}.\n"
                f"Sample images in the tarball: \n{sample_image_string}\n"
            )

    if cytassist_image_corrected_in_ppln := (
        cytassist_image_paths
        and len(cytassist_image_paths) == 1
        and test_for_cytassist_image_fix(cytassist_image_paths[0])
    ):
        martian.log_info("Checking CytAssist image channel correction.")
        temporary_tiff_dir = martian.make_path("temp_tiff_dir").decode()
        cytassist_image_paths[0] = tiffer.try_call_tiffer_compatibilty_fixes(
            cytassist_image_paths[0], temporary_tiff_dir
        )

    is_hd_run = cr_sp_preflight.is_hd_upload(args.chemistry, args.custom_chemistry_def)
    cr_sp_preflight.check_spatial_image_paths(
        args.tissue_image_paths,
        cytassist_image_paths,
        args.dark_images,
    )
    cr_sp_preflight.check_spatial_arguments(
        args.loupe_alignment_file,
        args.gpr_file,
        args.hd_layout_file,
        args.slide_serial_capture_area,
        is_hd_run=is_hd_run,
        hd_log_umi_image=args.hd_log_umi_image,
        is_pd=args.is_pd,
    )
    cr_sp_preflight.check_pattern_arguments(args.v1_pattern_fix)
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

    cr_sp_preflight.check_images_valid(
        args.tissue_image_paths,
        args.dark_images,
        args.chemistry,
        cytassist_image_paths,
        args.dapi_channel_index,
        image_scale=args.image_scale,
        loupe_alignment_file=args.loupe_alignment_file,
        segmentation_inputs=segmentation_inputs,
        cytassist_image_corrected_in_ppln=cytassist_image_corrected_in_ppln,
    )

    cr_sp_preflight.check_cytassist_img_valid(
        cytassist_image_paths,
        args.slide_serial_capture_area,
        args.chemistry,
        args.custom_chemistry_def,
        args.is_pd,
        args.override_id,
    )
