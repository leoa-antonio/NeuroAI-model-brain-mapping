# visualization.py

"""
Visualization utilities for encoding models.

Right now:
    - Plot a histogram of voxel-wise R² scores.

This module is imported by a driver script or notebook.
It does not fit models or compute metrics.
"""

from typing import Optional

import numpy as np
import matplotlib.pyplot as plt


def plot_r2_histogram(
    r2_voxels: np.ndarray,
    bins: int = 50,
    title: str = "Voxel-wise encoding performance (R²)",
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Plot a histogram of voxel-wise R² scores.

    Parameters
    ----------
    r2_voxels : np.ndarray
        Array of R² scores, shape (N_voxels,).
    bins : int, default=50
        Number of histogram bins.
    title : str, default="Voxel-wise encoding performance (R²)"
        Plot title.
    save_path : str or None, default=None
        If provided, save the figure to this path (e.g. "figs/subj01_r2_hist.png").
    show : bool, default=True
        Whether to display the figure with plt.show().
    """
    r2_voxels = np.asarray(r2_voxels)

    plt.figure()
    plt.hist(r2_voxels, bins=bins)
    plt.axvline(0.0, linestyle="--")
    plt.xlabel("R²")
    plt.ylabel("Number of voxels")
    plt.title(title)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300)

    if show:
        plt.show()
    else:
        plt.close()
