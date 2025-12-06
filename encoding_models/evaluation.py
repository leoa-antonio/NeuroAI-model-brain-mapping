# evaluation.py

"""
Evaluation utilities for encoding models.

Provides:
- voxelwise_r2: voxel-wise coefficient of determination (R²)
- summarize_r2: quick text summary of mean/median/%>0
"""

from typing import Dict, Optional

import numpy as np


def voxelwise_r2(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Compute voxel-wise R² between ground truth and predicted fMRI.

    Parameters
    ----------
    y_true : array, shape (n_samples, n_voxels)
        Ground-truth fMRI responses for held-out samples.
    y_pred : array, shape (n_samples, n_voxels)
        Predicted fMRI responses for the same samples.

    Returns
    -------
    r2 : array, shape (n_voxels,)
        R² for each voxel.
    """
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}"
        )

    # Residual sum of squares for each voxel
    ss_res = np.sum((y_true - y_pred) ** 2, axis=0)

    # Total sum of squares for each voxel
    y_mean = np.mean(y_true, axis=0, keepdims=True)
    ss_tot = np.sum((y_true - y_mean) ** 2, axis=0)

    # Avoid division by zero for constant voxels
    with np.errstate(divide="ignore", invalid="ignore"):
        r2 = 1.0 - ss_res / ss_tot
        r2 = np.where(np.isfinite(r2), r2, np.nan)

    return r2


def summarize_r2(r2: np.ndarray, subject: Optional[str] = None) -> Dict[str, float]:
    """
    Print and return summary statistics for voxel-wise R².

    Parameters
    ----------
    r2 : array, shape (n_voxels,)
        Voxel-wise R² values.
    subject : str, optional
        Subject identifier for printing.

    Returns
    -------
    summary : dict
        Dictionary with mean, median, and percent_positive.
    """
    r2 = np.asarray(r2, dtype=np.float32)
    r2_clean = r2[~np.isnan(r2)]

    mean_r2 = float(np.mean(r2_clean))
    median_r2 = float(np.median(r2_clean))
    pct_pos = float(np.mean(r2_clean > 0.0) * 100.0)

    header = f"Voxel-wise R² summary ({subject}):" if subject else "Voxel-wise R² summary:"
    print("\n" + header)
    print(f"mean R²:   {mean_r2:.4f}")
    print(f"median R²: {median_r2:.4f}")
    print(f"% R² > 0:  {pct_pos:.1f}%")

    return None
