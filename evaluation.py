# evaluation.py

"""
Evaluation utilities for encoding models.

Core idea:
    - Compare predicted fMRI responses to true responses
    - Compute voxel-wise R² scores on held-out data
    - Print simple summary stats

This file does NOT fit any models; it only evaluates predictions.
"""

from typing import Tuple

import numpy as np
from sklearn.metrics import r2_score


def compute_voxel_r2(
    Y_true: np.ndarray,
    Y_pred: np.ndarray,
) -> np.ndarray:
    """
    Compute voxel-wise R² on held-out data.

    Parameters
    ----------
    Y_true : np.ndarray
        Ground-truth fMRI responses on the test set.
        Shape: (N_test, N_voxels)
    Y_pred : np.ndarray
        Predicted fMRI responses on the test set.
        Shape: (N_test, N_voxels)

    Returns
    -------
    r2_voxels : np.ndarray
        R² per voxel, shape (N_voxels,).

    Notes
    -----
    - R² = 1   → perfect prediction
    - R² = 0   → model no better than predicting the mean
    - R² < 0   → model worse than predicting the mean
    """
    Y_true = np.asarray(Y_true)
    Y_pred = np.asarray(Y_pred)

    if Y_true.shape != Y_pred.shape:
        raise ValueError(
            f"Y_true and Y_pred must have the same shape. "
            f"Got Y_true: {Y_true.shape}, Y_pred: {Y_pred.shape}"
        )

    r2_voxels = r2_score(
        Y_true,
        Y_pred,
        multioutput="raw_values",  # one R² per voxel
    )
    return r2_voxels


def summarize_r2(r2_voxels: np.ndarray) -> Tuple[float, float, float]:
    """
    Print and return simple summary statistics of voxel-wise R².

    Parameters
    ----------
    r2_voxels : np.ndarray
        Voxel-wise R² scores, shape (N_voxels,).

    Returns
    -------
    mean_r2 : float
    median_r2 : float
    frac_positive : float
        Fraction of voxels with R² > 0.
    """
    r2_voxels = np.asarray(r2_voxels)

    mean_r2 = float(np.mean(r2_voxels))
    median_r2 = float(np.median(r2_voxels))
    frac_positive = float(np.mean(r2_voxels > 0.0))

    print("Voxel-wise R² summary:")
    print(f"  mean R²        : {mean_r2:.4f}")
    print(f"  median R²      : {median_r2:.4f}")
    print(f"  % R² > 0       : {frac_positive * 100:.1f}%")

    return mean_r2, median_r2, frac_positive
