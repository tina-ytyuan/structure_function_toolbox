"""Structure-function coupling metric.

Given a structural matrix (FA-weighted) and a functional matrix (fMRI FC) in
the SAME parcellation, quantify how well they "align". This is the core number
the whole toolbox is built around.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr, spearmanr


def _upper_tri(mat: np.ndarray) -> np.ndarray:
    iu = np.triu_indices_from(mat, k=1)
    return mat[iu]


def _corr(a: np.ndarray, b: np.ndarray, method: str) -> float:
    # Only compare edges where structure is defined & finite in both.
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return np.nan
    a, b = a[m], b[m]
    if method == "spearman":
        return float(spearmanr(a, b).statistic)
    if method == "pearson":
        return float(pearsonr(a, b).statistic)
    raise ValueError(f"Unknown corr method: {method}")


def global_coupling(struct: np.ndarray, func: np.ndarray, config) -> float:
    """Single scalar: correlation between all structural and functional edges."""
    return _corr(_upper_tri(struct), _upper_tri(func), config.coupling_corr)


def regional_coupling(struct: np.ndarray, func: np.ndarray, config) -> np.ndarray:
    """Per-node coupling: (N,) vector.

    For each region i, correlate its structural connectivity profile (row i)
    against its functional connectivity profile (row i), excluding the diagonal.
    """
    n = struct.shape[0]
    out = np.full(n, np.nan)
    for i in range(n):
        idx = np.arange(n) != i
        out[i] = _corr(struct[i, idx], func[i, idx], config.coupling_corr)
    return out


def coupling(struct: np.ndarray, func: np.ndarray, config):
    """Dispatch to the configured coupling metric.

    Returns a float (global) or an (N,) array (regional).
    """
    if struct.shape != func.shape:
        raise ValueError(
            f"structural {struct.shape} and functional {func.shape} matrices "
            "must share the same parcellation/shape"
        )
    if config.coupling_metric == "global_corr":
        return global_coupling(struct, func, config)
    if config.coupling_metric == "regional_corr":
        return regional_coupling(struct, func, config)
    raise ValueError(f"Unknown coupling_metric: {config.coupling_metric}")
