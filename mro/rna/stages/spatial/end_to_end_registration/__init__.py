# Copyright (c) 2025 10X Genomics, Inc. All rights reserved.

"""Perform end to end registration (UMI vs H&E)."""

import json
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

import martian
import numpy as np
import skimage
from scipy.sparse import coo_matrix

import cellranger.matrix as cr_matrix
from cellranger.fast_utils import SquareBinIndex
from cellranger.spatial.hd_umi_regist import (
    EndToEndRegistrationResults,
    apply_log1p_or_percentile_max,
    register_phase_correlation,
)
from cellranger.spatial.image_util import cv_read_image_standard
from cellranger.spatial.loupe_util import HdLayoutOffset, LoupeParser
from cellranger.spatial.transform import (
    convert_transform_corner_to_center,
    normalize_perspective_transform,
    translation_matrix,
)

if TYPE_CHECKING:
    from cellranger.spatial.slide_design_o3 import (
        VisiumHdSlideWrapper,  # pylint: disable=no-name-in-module, import-error
    )


__MRO__ = """
stage END_TO_END_REGISTRATION(
    in  png            tissue_detection_image,
    in  tiff           registration_target_image,
    in  json           tissue_transform_json,
    in  h5             raw_feature_bc_matrix_h5,
    in  json           registered_spots_data_json,
    in  json           fiducial_transform,
    in  HdLayoutOffset custom_layout_offset,
    out json           summary,
    out json           e2e_hd_layout_data_json,
    out json           e2e_registered_spots_data_json,
    out jpg            qc_phase_corr_map,
    src py             "stages/spatial/end_to_end_registration",
) using (
    volatile = strict,
    mem_gb   = 8,
    vmem_gb  = 16,
)
"""


MAX_ALLOWED_SHIFT_UM = 20

BARCODE = "barcode"
ROW = "row"
COL = "col"
UMI_COUNT = "umi_count"


def _float_or_none(x):
    return float(x) if x is not None else None


def _nan_to_none(x):
    return None if x is None or np.isnan(x) else x


@dataclass
class AllRegistrationStats:
    """All registration stats."""

    gaussian_fit_offset: HdLayoutOffset | None
    pre_refinement_offset: HdLayoutOffset | None
    statistics: EndToEndRegistrationResults


@dataclass
class HdLayoutOffsetStats:  # pylint: disable=too-many-instance-attributes
    """Stats returned by hd layout offset."""

    # final x/y offset; will be 0 if Gaussian fit offset and argmax offset are both above allowed magnitude
    x_offset: float
    y_offset: float
    # offset was provided at the CLI by the user
    is_custom_offset: bool
    # final predicted offset (from either Gaussian fitting or the argmax fallback)
    # was outside the allowed range. this is true if we see NaNs in the cross power
    # spectrum.
    # this is set to false if there was a custom offset
    is_predicted_offset_outside_range: bool
    # was the phase correlation surface all finite. (i.e. it had no NaNs or infinite values)
    phase_correlation_all_finite: bool | None
    # Gaussian fitting failed to localize the correct offset; final offset falls back to argmax if possible
    has_gaussian_fit_failed: bool | None = None
    # Gaussian fitting found a peak outside the allowed range; final offset falls back to argmax if possible
    is_gaussian_fit_outside_range: bool | None = None
    # single highest point in the phase correlation surface is outside the allowed range
    # (so if Gaussian fit fails or is outside range, then final offset will be 0)
    is_pre_refinement_offset_outside_range: bool | None = None
    # fell back to pre-refinement offset since Gaussian fit either failed or produced
    # a value outside the allowed range
    is_pre_refinement_offset_good_but_fit_bad: bool | None = None
    # argmax of the phase correlation surface, i.e. position of the highest single point
    pre_refinement_x_offset: float | None = None
    pre_refinement_y_offset: float | None = None
    # height of the phase correlation peak from Gaussian fit
    gaussian_amplitude: float | None = None
    # location of the phase correlation peak from Gaussian fit
    # determines the final offset unless outside the allowed range
    gaussian_fit_x_offset: float | None = None
    gaussian_fit_y_offset: float | None = None
    # width (in x and y) of the phase correlation peak from Gaussian fit.
    # lower indicates higher precision in the final offset
    gaussian_sigma_x: float | None = None
    gaussian_sigma_y: float | None = None
    # strength of the background in the phase correlation surface
    # (must be interpreted relative to gaussian_amplitude)
    gaussian_fit_background: float | None = None

    def remove_nans(self):
        """Remove nan metrics."""
        self.gaussian_amplitude = _nan_to_none(self.gaussian_amplitude)
        self.gaussian_fit_x_offset = _nan_to_none(self.gaussian_fit_x_offset)
        self.gaussian_fit_y_offset = _nan_to_none(self.gaussian_fit_y_offset)
        self.gaussian_sigma_x = _nan_to_none(self.gaussian_sigma_x)
        self.gaussian_sigma_y = _nan_to_none(self.gaussian_sigma_y)
        self.gaussian_fit_background = _nan_to_none(self.gaussian_fit_background)


def project_image_to_spots(
    registration_target_image, transform_matrix, slide: "VisiumHdSlideWrapper"
):
    """Project the registration target image to the spots."""
    # skimage.transform.warp uses center-based sub-pixel coordinates
    center_based_transform = convert_transform_corner_to_center(transform_matrix)
    sk_transform = skimage.transform.ProjectiveTransform(matrix=center_based_transform)
    resampled_img = skimage.transform.warp(
        registration_target_image,
        sk_transform,
        output_shape=(slide.num_rows(), slide.num_cols()),
        preserve_range=True,
    ).astype(np.uint8)
    return resampled_img


def umi_array_from_matrix(raw_matrix_path, slide: "VisiumHdSlideWrapper"):
    """Compute UMIs per barcode as a 2d numpy array."""
    matrix = cr_matrix.CountMatrix.load_h5_file(raw_matrix_path)
    counts_per_bc = matrix.get_counts_per_bc()

    rows, cols = list(
        zip(*(((x := SquareBinIndex(barcode=bc.decode())).row, x.col) for bc in matrix.bcs))
    )

    # Create the COO matrix directly
    coo = coo_matrix(
        (counts_per_bc, (rows, cols)),
        shape=(slide.num_rows(), slide.num_cols()),
    )
    return coo.toarray()


def register_he_umi(
    tissue_image: np.ndarray,
    umi_counts: np.ndarray,
    under_tissue_mask: np.ndarray,
    apply_log1p: bool,
    qc_phase_corr_map_path: str | None,
    bin_size_um: float,
) -> AllRegistrationStats:
    """Register the HE image with the UMI counts using phase correlation."""
    # Apply log (1 + x) transformation or normalize to percentile max
    umi_counts = apply_log1p_or_percentile_max(umi_counts, under_tissue_mask, apply_log1p)
    registration_results = register_phase_correlation(
        umi_counts,
        tissue_image,
        qc_phase_corr_map_path,
    )
    gaussian_fit_offset = None
    if (
        registration_results.gaussian_shift_x is not None
        and registration_results.gaussian_shift_y is not None
    ):
        gaussian_fit_offset = HdLayoutOffset(
            x_offset=-registration_results.gaussian_shift_x * bin_size_um,
            y_offset=-registration_results.gaussian_shift_y * bin_size_um,
        )

    pre_refinement_offset = None
    if (
        registration_results.argmax_shift_x is not None
        and registration_results.argmax_shift_y is not None
    ):
        pre_refinement_offset = HdLayoutOffset(
            x_offset=-registration_results.argmax_shift_x * bin_size_um,
            y_offset=-registration_results.argmax_shift_y * bin_size_um,
        )
    return AllRegistrationStats(
        gaussian_fit_offset=gaussian_fit_offset,
        pre_refinement_offset=pre_refinement_offset,
        statistics=registration_results,
    )


def split(args):
    # We load the Feature barcode matrix and basically make a copy of that as a COO-matrix
    mem_gb_estimate = (
        cr_matrix.CountMatrix.get_mem_gb_from_matrix_h5(args.raw_feature_bc_matrix_h5, scale=1) * 2
        + 4
    )
    return {
        "chunks": [],
        "join": {
            "__mem_gb": mem_gb_estimate,
            "__vmem_gb": max(mem_gb_estimate + 4, 16),
        },
    }


def join(args, outs, _chunk_args, _chunk_outs):  # pylint: disable = too-many-locals
    spots_data = LoupeParser(args.registered_spots_data_json)
    if spots_data.hd_slide_layout() is None:
        martian.log_warn("No hd_layout found. Skipping E2E registration.")
        martian.clear(outs)
        return
    slide = spots_data.hd_slide
    gaussian_fit_failed = None
    if args.custom_layout_offset:
        layout_offset = HdLayoutOffset(**args.custom_layout_offset)
        layout_offset_norm_um = layout_offset.total_offset()
        is_offset_above_allowed = bool(layout_offset_norm_um > MAX_ALLOWED_SHIFT_UM)
        if is_offset_above_allowed:
            print(
                f"Warning: shift ({layout_offset.x_offset}µm, {layout_offset.y_offset}µm) has a total \
                    displacement of {layout_offset_norm_um}µm which exceeds maximum allowed shift \
                    ({MAX_ALLOWED_SHIFT_UM} µm)"
            )
        hd_layout_offset_stats = HdLayoutOffsetStats(
            x_offset=float(layout_offset.x_offset),
            y_offset=float(layout_offset.y_offset),
            is_custom_offset=True,
            is_predicted_offset_outside_range=False,
            phase_correlation_all_finite=None,
        )
    else:
        with open(args.fiducial_transform) as f:
            fid_transform = np.array(json.load(f))
        with open(args.tissue_transform_json) as f:
            tissue_registration_transform = np.array(json.load(f)["tissue_transform"])

        transform_spot_colrow_to_cytassist_colrow = normalize_perspective_transform(
            fid_transform @ slide.transform_spot_colrow_to_xy()
        )
        transform_spot_colrow_to_microscope_colrow = normalize_perspective_transform(
            tissue_registration_transform @ transform_spot_colrow_to_cytassist_colrow
        )

        umi_counts = umi_array_from_matrix(args.raw_feature_bc_matrix_h5, slide)
        tissue_image = project_image_to_spots(
            cv_read_image_standard(args.registration_target_image).astype(np.uint8),
            normalize_perspective_transform(
                transform_spot_colrow_to_microscope_colrow
                # spot_colrow is center-based and we need a corner-based transform
                @ translation_matrix(-0.5, -0.5),
            ),
            slide,
        )
        all_registration_results = register_he_umi(
            tissue_image=tissue_image,
            umi_counts=umi_counts,
            under_tissue_mask=umi_counts > 0,
            apply_log1p=False,
            qc_phase_corr_map_path=outs.qc_phase_corr_map,
            bin_size_um=slide.spot_pitch(),
        )
        gaussian_fit_failed = not all_registration_results.statistics.gaussian_fit_success
        is_gaussian_fit_outside_range = (
            all_registration_results.gaussian_fit_offset is not None
            and bool(
                all_registration_results.gaussian_fit_offset.total_offset() > MAX_ALLOWED_SHIFT_UM
            )
        )
        is_pre_refinement_offset_outside_range = (
            all_registration_results.pre_refinement_offset is not None
            and bool(
                all_registration_results.pre_refinement_offset.total_offset() > MAX_ALLOWED_SHIFT_UM
            )
        )
        is_predicted_offset_outside_range = False
        if (
            all_registration_results.gaussian_fit_offset is not None
            and not is_gaussian_fit_outside_range
        ):
            layout_offset = all_registration_results.gaussian_fit_offset
        elif (
            all_registration_results.pre_refinement_offset is not None
            and not is_pre_refinement_offset_outside_range
        ):
            layout_offset = all_registration_results.pre_refinement_offset
        else:
            is_predicted_offset_outside_range = True
            layout_offset = HdLayoutOffset(x_offset=0.0, y_offset=0.0)

        # print warnings
        if (
            all_registration_results.gaussian_fit_offset is not None
            and is_gaussian_fit_outside_range
        ):
            print(
                f"Warning: Gaussian fit shift \
                ({all_registration_results.gaussian_fit_offset.x_offset}µm, {all_registration_results.gaussian_fit_offset.y_offset}µm) \
                has a total displacement of {all_registration_results.gaussian_fit_offset.total_offset()}µm \
                which exceeds maximum allowed shift \
                ({MAX_ALLOWED_SHIFT_UM} µm)"
            )
        if (
            all_registration_results.pre_refinement_offset is not None
            and is_pre_refinement_offset_outside_range
        ):
            print(
                f"Warning: Gaussian fit shift \
                ({all_registration_results.pre_refinement_offset.x_offset}µm, {all_registration_results.pre_refinement_offset.y_offset}µm) \
                has a total displacement of {all_registration_results.pre_refinement_offset.total_offset()}µm \
                which exceeds maximum allowed shift \
                ({MAX_ALLOWED_SHIFT_UM} µm)"
            )

        hd_layout_offset_stats = HdLayoutOffsetStats(
            x_offset=float(layout_offset.x_offset),
            y_offset=float(layout_offset.y_offset),
            is_custom_offset=False,
            is_predicted_offset_outside_range=is_predicted_offset_outside_range,
            phase_correlation_all_finite=all_registration_results.statistics.phase_correlation_all_finite,
            has_gaussian_fit_failed=gaussian_fit_failed,
            is_gaussian_fit_outside_range=is_gaussian_fit_outside_range,
            is_pre_refinement_offset_outside_range=is_pre_refinement_offset_outside_range,
            is_pre_refinement_offset_good_but_fit_bad=(
                (
                    is_gaussian_fit_outside_range
                    or gaussian_fit_failed
                    or all_registration_results.gaussian_fit_offset is None
                )
                and all_registration_results.pre_refinement_offset is not None
                and not is_pre_refinement_offset_outside_range
            ),
            pre_refinement_x_offset=(
                float(all_registration_results.pre_refinement_offset.x_offset)
                if all_registration_results.pre_refinement_offset
                else None
            ),
            pre_refinement_y_offset=(
                float(all_registration_results.pre_refinement_offset.y_offset)
                if all_registration_results.pre_refinement_offset
                else None
            ),
            gaussian_fit_x_offset=(
                float(all_registration_results.gaussian_fit_offset.x_offset)
                if all_registration_results.gaussian_fit_offset
                else None
            ),
            gaussian_fit_y_offset=(
                float(all_registration_results.gaussian_fit_offset.y_offset)
                if all_registration_results.gaussian_fit_offset
                else None
            ),
            gaussian_amplitude=_float_or_none(
                all_registration_results.statistics.gaussian_amplitude
            ),
            gaussian_sigma_x=_float_or_none(all_registration_results.statistics.gaussian_sigma_x),
            gaussian_sigma_y=_float_or_none(all_registration_results.statistics.gaussian_sigma_y),
            gaussian_fit_background=_float_or_none(
                all_registration_results.statistics.gaussian_fit_background
            ),
        )
        hd_layout_offset_stats.remove_nans()

    summary_data = {"hd_layout_offset": asdict(hd_layout_offset_stats)}
    with open(outs.summary, "w") as f:
        json.dump(summary_data, f)

    spots_data.update_hd_layout_with_offset(layout_offset)
    spots_data.save_to_json(outs.e2e_registered_spots_data_json)
    spots_data.hd_slide_layout().save_as_json(outs.e2e_hd_layout_data_json)
