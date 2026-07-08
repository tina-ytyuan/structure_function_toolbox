"""Functional connectivity from resting-state fMRI.

Input: cleaned 4D BOLD (already preprocessed) + a parcellation.
Output: an N x N functional connectivity matrix in the configured atlas.
"""

from __future__ import annotations

import numpy as np


def region_timeseries(bold_4d: np.ndarray, labels_3d: np.ndarray) -> np.ndarray:
    """Average voxel time series within each parcel.

    Parameters
    ----------
    bold_4d : (X, Y, Z, T) array of preprocessed BOLD.
    labels_3d : (X, Y, Z) integer atlas; 0 = background, 1..N = regions.

    Returns
    -------
    (N, T) array of mean time series per region.
    """
    region_ids = np.unique(labels_3d)
    region_ids = region_ids[region_ids != 0]
    ts = np.zeros((region_ids.size, bold_4d.shape[-1]), dtype=float)
    for i, rid in enumerate(region_ids):
        mask = labels_3d == rid
        ts[i] = bold_4d[mask].mean(axis=0)
    return ts


def functional_connectivity(ts: np.ndarray, config) -> np.ndarray:
    """N x N functional connectivity from region time series.

    Uses Pearson correlation (optionally partial), with optional Fisher z.
    """
    if config.fc_measure == "correlation":
        fc = np.corrcoef(ts)
    elif config.fc_measure == "partial_correlation":
        # Partial correlation via precision matrix.
        cov = np.cov(ts)
        precision = np.linalg.pinv(cov)
        d = np.sqrt(np.diag(precision))
        fc = -precision / np.outer(d, d)
        np.fill_diagonal(fc, 1.0)
    else:
        raise ValueError(f"Unknown fc_measure: {config.fc_measure}")

    if config.fc_fisher_z:
        np.fill_diagonal(fc, 0.0)
        fc = np.arctanh(np.clip(fc, -0.999999, 0.999999))
    return fc
