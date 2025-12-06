# visualization.py

"""
Visualization utilities for encoding models.

Provides:
- plot_r2_histogram: histogram of voxel-wise R² values.
"""

from typing import Optional

import numpy as np
import matplotlib.pyplot as plt


def plot_r2_histogram(
    r2: np.ndarray,
    subject: Optional[str] = None,
    bins: int = 50,
) -> None:
    """
    Plot a histogram of voxel-wise R² values.

    Parameters
    ----------
    r2 : array, shape (n_voxels,)
        Voxel-wise R² values.
    subject : str, optional
        Subject identifier for plot title.
    bins : int, default=50
        Number of histogram bins.
    """
    r2 = np.asarray(r2, dtype=np.float32)
    r2_clean = r2[~np.isnan(r2)]

    plt.figure(figsize=(6, 4))
    plt.hist(r2_clean, bins=bins)
    plt.xlabel("Voxel-wise $R^2$")
    plt.ylabel("Count")

    title = "Voxel-wise $R^2$ distribution"
    if subject is not None:
        title += f" ({subject})"
    plt.title(title)

    plt.tight_layout()
    plt.show()
