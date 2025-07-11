# Copyright (c) 2025 10X Genomics, Inc. All rights reserved.
"""HD end-to-end UMI to H&E registration utils."""

from dataclasses import dataclass

import numpy as np
from scipy.fft import fft2, fftshift, ifft2
from scipy.optimize import curve_fit

from cellranger.spatial.hd_umi_regist_qc_figures import get_phase_correlation_qc_fig

LINEAR_MAX_PERCENTILE = 98
# Half width of window of phase correlation on which a gaussian is fit
# This is twice the maximum allowed shift
WINDOW_HALF_WIDTH = 20


@dataclass
class EndToEndRegistrationResults:
    """Results of End to End registration."""

    phase_correlation_all_finite: bool = False
    gaussian_amplitude: float | None = None
    gaussian_shift_x: float | None = None
    gaussian_shift_y: float | None = None
    gaussian_sigma_x: float | None = None
    gaussian_sigma_y: float | None = None
    gaussian_fit_background: float | None = None
    argmax_shift_x: float | None = None
    argmax_shift_y: float | None = None
    gaussian_fit_success: bool = False


def get_maximum_coordinate_of_surface(surface):
    """Get argmax of a 2d surface."""
    peak_y, peak_x = np.unravel_index(  # pylint: disable=unbalanced-tuple-unpacking
        np.argmax(surface), surface.shape
    )

    return peak_y, peak_x


def apply_log1p_or_percentile_max(
    umis: np.ndarray,
    under_tissue_mask: np.ndarray,
    log1p: bool,
    percentile_max: int = LINEAR_MAX_PERCENTILE,
) -> np.ndarray:
    """Apply either log+1 or percentile max data transformation to eliminate outliers."""
    if log1p:
        umis = np.log1p(umis)
        percentile_max = np.max(umis)
    else:
        umis_under_mask = umis[under_tissue_mask]
        nonzero_umis_under_mask = umis_under_mask[umis_under_mask > 0]
        percentile_max = max(
            int(
                (
                    np.percentile(nonzero_umis_under_mask, LINEAR_MAX_PERCENTILE)
                    if nonzero_umis_under_mask.size
                    else 0
                ),
            ),
            1,
        )

    umis[umis > percentile_max] = percentile_max
    umis = ((umis / percentile_max) * 255).astype(np.uint8)

    return umis


def fit_2d_gaussian(surface):
    """Fit a 2D Gaussian to the surface using scipy's curve_fit."""

    def _gaussian_2d(coords, A, x0, y0, sigma_x, sigma_y, offset):  # pylint: disable=invalid-name
        x, y = coords
        inner = ((x - x0) ** 2) / (2 * sigma_x**2) + ((y - y0) ** 2) / (2 * sigma_y**2)
        return A * np.exp(-inner) + offset

    def _jac_gaussian_2d(
        coords, A, x0, y0, sigma_x, sigma_y, _offset
    ):  # pylint: disable=invalid-name
        """Jacobian of the gaussian wrt its parameters."""
        x, y = coords
        inner = ((x - x0) ** 2) / (2 * sigma_x**2) + ((y - y0) ** 2) / (2 * sigma_y**2)
        exponential = np.exp(-inner)
        first_term = A * exponential
        return np.column_stack(
            (
                exponential,
                ((x - x0) / (sigma_x**2)) * first_term,
                ((y - y0) / (sigma_y**2)) * first_term,
                (((x - x0) ** 2) / (sigma_x**3)) * first_term,
                (((y - y0) ** 2) / (sigma_y**3)) * first_term,
                np.ones_like(x),
            )
        )

    peak_y, peak_x = get_maximum_coordinate_of_surface(surface)
    ny, nx = surface.shape
    x = np.arange(nx)
    y = np.arange(ny)
    x, y = np.meshgrid(x, y)
    xdata = np.vstack((x.ravel(), y.ravel()))
    zdata = surface.ravel()
    initial_guess = (np.max(zdata), peak_x, peak_y, 10, 10, np.min(zdata))
    try:
        popt, _pcov = curve_fit(  # pylint: disable=unbalanced-tuple-unpacking
            _gaussian_2d, xdata, zdata, p0=initial_guess, jac=_jac_gaussian_2d
        )
        fit_success = True
    except RuntimeError as exc:
        print(f"Runtime error while fitting gaussian to the learned transform: {exc}")
        popt = initial_guess
        fit_success = False
    except ValueError as exc:
        print(f"Value error while fitting gaussian to the learned transform: {exc}")
        popt = initial_guess
        fit_success = False
    return popt, fit_success


def get_registration_results(  # pylint: disable=too-many-locals
    phase_corr_map,
) -> EndToEndRegistrationResults:
    """Get registration results from phase correlation map."""
    mid_y, mid_x = np.array(phase_corr_map.shape) // 2
    argmax_y, argmax_x = get_maximum_coordinate_of_surface(phase_corr_map)
    argmax_shift_y, argmax_shift_x = argmax_y - mid_y, argmax_x - mid_x

    # half width to use. minimum of the image size and the default half width
    window_half_width = min(mid_y, mid_x, WINDOW_HALF_WIDTH)
    if not np.isfinite(phase_corr_map).all():
        registration_results = EndToEndRegistrationResults(
            gaussian_fit_success=False,
            phase_correlation_all_finite=False,
        )
    elif (
        not -window_half_width <= argmax_shift_y < window_half_width
        or not -window_half_width <= argmax_shift_x < window_half_width
    ):
        registration_results = EndToEndRegistrationResults(
            gaussian_fit_success=False,
            phase_correlation_all_finite=True,
            argmax_shift_x=argmax_shift_x,
            argmax_shift_y=argmax_shift_y,
        )
    else:
        # Top left corner of window on which to fit a Gaussian
        window_top_left_y, window_top_left_x = mid_y - window_half_width, mid_x - window_half_width
        # Maxima in the phase correlation corresponds to the translation
        popt, gaussian_fit_success = fit_2d_gaussian(
            phase_corr_map[
                window_top_left_y : window_top_left_y + 2 * window_half_width,
                window_top_left_x : window_top_left_x + 2 * window_half_width,
            ]
        )
        A, peak_x, peak_y, sigma_x, sigma_y, offset = popt  # pylint: disable=invalid-name
        shift_y, shift_x = np.round(peak_y + window_top_left_y - mid_y, 2), np.round(
            peak_x + window_top_left_x - mid_x, 2
        )
        registration_results = EndToEndRegistrationResults(
            gaussian_amplitude=A,
            gaussian_shift_x=shift_x,
            gaussian_shift_y=shift_y,
            gaussian_sigma_x=sigma_x,
            gaussian_sigma_y=sigma_y,
            gaussian_fit_background=offset,
            argmax_shift_x=argmax_shift_x,
            argmax_shift_y=argmax_shift_y,
            gaussian_fit_success=gaussian_fit_success,
            phase_correlation_all_finite=True,
        )

    return registration_results


def register_phase_correlation(  # pylint: disable=too-many-locals
    fixed_img: np.ndarray,
    mv_img: np.ndarray,
    phase_corr_map_path: str | None = None,
) -> EndToEndRegistrationResults:
    """Use phase correlation to register two images related by translation only."""
    assert fixed_img.shape == mv_img.shape
    # Compute normalized cross-power spectrum in frequency domain
    fixed_fft2 = fft2(fixed_img)
    mv_fft2 = fft2(mv_img)
    cross_power_spectrum = (fixed_fft2 * mv_fft2.conj()) / np.abs(  # pylint: disable=no-member
        fixed_fft2 * mv_fft2.conj()  # pylint: disable=no-member
    )
    # Phase correlation surface via inverse FFT
    phase_corr_map = fftshift(np.abs(ifft2(cross_power_spectrum)))

    registration_results = get_registration_results(phase_corr_map=phase_corr_map)

    if phase_corr_map_path:
        get_phase_correlation_qc_fig(
            phase_corr_map_path=phase_corr_map_path,
            phase_corr_map=phase_corr_map,
            registration_results=registration_results,
        )
    return registration_results
