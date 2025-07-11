# Copyright (c) 2025 10X Genomics, Inc. All rights reserved.
"""HD end-to-end UMI to H&E registration QC figures."""


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse


def get_phase_correlation_qc_fig(phase_corr_map_path, phase_corr_map, registration_results) -> None:
    """Get phase correlation QC figure."""
    mid_y, mid_x = np.array(phase_corr_map.shape) // 2
    # Plot the phase correlation surface with peak
    fig, ax = plt.subplots(figsize=(12, 9))
    # Only need to focus on the center of the phase correlation map
    steps = 15  # one step is 2 µm
    center_map = phase_corr_map[
        mid_y - steps : mid_y + steps + 1, mid_x - steps : mid_x + steps + 1
    ]
    im = ax.imshow(
        center_map,
        cmap="hot",
        interpolation="nearest",
        vmin=phase_corr_map.min(),
        vmax=phase_corr_map.max(),
    )
    fig.colorbar(im)
    ax.set_xticks(np.arange(0, 2 * steps + 1, 1))
    ax.set_yticks(np.arange(0, 2 * steps + 1, 1))
    ax.set_xticklabels(np.arange(-steps, steps + 1, 1))
    ax.set_yticklabels(np.arange(-steps, steps + 1, 1))
    ax.set_xlabel("X-axis shift (pixels)")
    ax.set_ylabel("Y-axis shift (pixels)")

    # Mark the peak (shift location)
    if (
        registration_results.gaussian_shift_x is not None
        and registration_results.gaussian_shift_y is not None
    ):
        ax.scatter(
            registration_results.gaussian_shift_x + steps,
            registration_results.gaussian_shift_y + steps,
            color="cyan",
            marker="x",
            label=f"(shift_x={registration_results.gaussian_shift_x:.4f}, "
            f"shift_y={registration_results.gaussian_shift_y:.4f})",
        )

    # Mark the argmax shift
    if (
        registration_results.argmax_shift_x is not None
        and registration_results.argmax_shift_y is not None
    ):
        ax.scatter(
            registration_results.argmax_shift_x + steps,
            registration_results.argmax_shift_y + steps,
            color="slategray",
            marker="+",
            label=f"(argmax_shift_x{registration_results.argmax_shift_x:.4f}, "
            f"argmax_shift_y{registration_results.argmax_shift_y:.4f})",
        )

    if (
        registration_results.gaussian_shift_x is not None
        and registration_results.gaussian_shift_y is not None
        and registration_results.gaussian_sigma_x is not None
        and registration_results.gaussian_sigma_y is not None
    ):
        ellipse = Ellipse(
            xy=(
                registration_results.gaussian_shift_x + steps,
                registration_results.gaussian_shift_y + steps,
            ),
            width=2 * registration_results.gaussian_sigma_x,
            height=2 * registration_results.gaussian_sigma_y,
            edgecolor="lime",
            facecolor="none",
            linewidth=2,
            label=f"(sigma_x={registration_results.gaussian_sigma_x:.4f}, "
            f"sigma_y={registration_results.gaussian_sigma_y:.4f}), "
            f"A={registration_results.gaussian_amplitude:.4f}, "
            f"background={registration_results.gaussian_fit_background:.4f}",
        )
        ax.add_patch(ellipse)

    ax.set_title("Phase Correlation Map")
    ax.legend()
    fig.savefig(phase_corr_map_path, dpi=300)
