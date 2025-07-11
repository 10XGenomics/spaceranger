# Copyright (c) 2024 10x Genomics, Inc. All rights reserved.
"""Disable cell segmentation and/or end to end registration."""

from dataclasses import asdict, dataclass

import martian

from cellranger.spatial.segment_nuclei_preflights import ShouldSegmentSufficientStats

__MRO__ = """
stage DISABLE_IMAGING_STAGES(
    in  bool                  is_visium_hd,
    in  bool                  skip_segmentation,
    in  bool                  segmentation_from_user,
    in  int                   dark_images,
    in  file[]                tissue_image_paths,
    in  UmiRegistrationInputs umi_registration_inputs_in,
    out bool                  disable_imaging_stages,
    out bool                  disable_segmentation,
    out UmiRegistrationInputs umi_registration_inputs,
    src py                    "stages/spatial/disable_imaging_stages",
) using (
    mem_gb  = 2,
    vmem_gb = 32,
)
"""


@dataclass
class HdLayoutOffset:
    """HD layout offset."""

    x_offset: float | None = None
    y_offset: float | None = None


@dataclass
class UmiRegistrationInputs:
    """UMI registration inputs."""

    disable: bool
    offset: HdLayoutOffset | None = None


def main(args, outs):
    if not args.umi_registration_inputs_in:
        outs.umi_registration_inputs = asdict(
            UmiRegistrationInputs(disable=bool(args.dark_images) or not args.is_visium_hd)
        )
    else:
        outs.umi_registration_inputs = args.umi_registration_inputs_in

    # dark images are 0 or None for brightfield images
    tissue_segmentable = len(args.tissue_image_paths) == 1 and not args.dark_images
    if not tissue_segmentable:
        martian.log_info(
            f"Disabling segment nuclei because sample has tissue_image_paths of length {len(args.tissue_image_paths)} and dark images {args.dark_images}."
        )
    if tissue_segmentable:
        should_segment = ShouldSegmentSufficientStats.new(args.tissue_image_paths[0])
        tissue_segmentable = tissue_segmentable and should_segment.tissue_segmentable
        martian.log_info(
            f"{should_segment.is_jpeg=}, {should_segment.is_multi_page_tiff=}, {should_segment.is_grayscale=}, {should_segment.tissue_segmentable=}, {tissue_segmentable=}"
        )
    if not tissue_segmentable:
        martian.log_info("Cannot run segment nuclei on sample.")

    outs.disable_imaging_stages = (
        args.skip_segmentation
        or (not args.is_visium_hd)
        or not (tissue_segmentable)
        or args.segmentation_from_user
    )
    outs.disable_segmentation = (
        args.skip_segmentation
        or (not args.is_visium_hd)
        or not (tissue_segmentable or args.segmentation_from_user)
    )
