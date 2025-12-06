# model_fitting.py

"""
Model fitting utilities for encoding models.

This version is optimized for large voxel counts:
    - uses a single Ridge(alpha) rather than heavy cross-validation
    - trains one multivariate model for all voxels at once
    - returns train/test split + test predictions
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass
class EncodingModelResult:
    model: Ridge
    scaler: Optional[StandardScaler]
    train_idx: np.ndarray
    test_idx: np.ndarray
    y_pred_test: np.ndarray


def fit_ridge_encoding_model(
    X: np.ndarray,
    Y: np.ndarray,
    alpha: float = 100.0,
    test_size: float = 0.2,
    random_state: int = 0,
    standardize: bool = True,
) -> EncodingModelResult:
    """
    Fit a multivariate Ridge regression model X -> Y.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (N_samples, N_features).
    Y : np.ndarray
        fMRI matrix of shape (N_samples, N_voxels).
    alpha : float, default=100.0
        Ridge regularization strength. Larger alpha = stronger shrinkage.
    test_size : float, default=0.2
        Fraction of samples to hold out for testing.
    random_state : int, default=0
        Seed for train/test split reproducibility.
    standardize : bool, default=True
        If True, standardize X using StandardScaler (fit on training set only).

    Returns
    -------
    EncodingModelResult
        Contains the trained model, scaler, train/test indices, and test predictions.
    """
    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.float32)

    print("[model_fitting] X shape:", X.shape)
    print("[model_fitting] Y shape:", Y.shape)

    # --- Train/test split ---
    n_samples = X.shape[0]
    idx_all = np.arange(n_samples)

    train_idx, test_idx = train_test_split(
        idx_all,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )

    X_train, X_test = X[train_idx], X[test_idx]
    Y_train, Y_test = Y[train_idx], Y[test_idx]

    print(f"[model_fitting] Train samples: {X_train.shape[0]}")
    print(f"[model_fitting] Test samples : {X_test.shape[0]}")

    # --- Optional standardization of features ---
    scaler = None
    if standardize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        print("[model_fitting] Standardized features with StandardScaler.")
    else:
        print("[model_fitting] No feature standardization applied.")

    # --- Fit Ridge regression (single alpha) ---
    print(f"[model_fitting] Fitting Ridge(alpha={alpha}) on full voxel matrix...")
    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(X_train, Y_train)
    print("[model_fitting] Finished training.")

    # --- Predict on held-out test set ---
    Y_pred_test = model.predict(X_test)
    print("[model_fitting] Finished test-set prediction.")
    print("[model_fitting] Y_pred_test shape:", Y_pred_test.shape)

    return EncodingModelResult(
        model=model,
        scaler=scaler,
        train_idx=train_idx,
        test_idx=test_idx,
        y_pred_test=Y_pred_test,
    )
