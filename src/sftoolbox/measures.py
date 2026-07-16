"""Voxelwise fMRI measures computed from cleaned BOLD only.

The numerics come from ``fmri_measures.py``, which is vendored verbatim from
Ajay's DCC code (``fMRI_signal_properties.py``) so the toolbox computes ALFF,
fALFF, ReHo, and RSFA exactly the way the team does. This module is only a thin
adapter: it builds a brain mask, calls Ajay's functions, and returns each result
as a ``(map_3d, mask)`` pair in the (X, Y, Z) grid the rest of the toolbox (web
app, references, viz) expects.

Do not put measure math here — change it upstream in ``fmri_measures.py`` (or,
better, re-vendor from Ajay's file). Defaults follow HCP / Ajay: TR = 0.72 s,
band 0.01-0.08 Hz, 27-voxel ReHo neighborhood.
"""

from __future__ import annotations

import numpy as np

from . import fmri_measures as _fm
from . import fmri_measures_arnav as _fma

# HCP / Ajay defaults.
DEFAULT_TR = 0.72
DEFAULT_LOW = 0.01
DEFAULT_HIGH = 0.08

# Arnav-measure defaults.
DEFAULT_INT_MAX_LAG = 20
DEFAULT_MSE_SCALES = (1, 2, 3, 4, 5)
DEFAULT_MSE_M = 2
DEFAULT_MSE_R = 0.15


def brain_mask(bold_4d: np.ndarray) -> np.ndarray:
    """Analysis mask matching Ajay's pipeline: finite voxels with signal.

    A voxel is kept if its time series is finite everywhere and not identically
    zero. (Ajay's ``process_nifti`` uses exactly this when no explicit mask is
    supplied.)
    """
    finite = np.all(np.isfinite(bold_4d), axis=-1)
    return finite & np.any(bold_4d != 0, axis=-1)


def _inflate(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Put a (V,) masked-voxel vector back into a 3D volume; 0 outside mask."""
    out = np.zeros(mask.shape, dtype=np.float64)
    out[mask] = values
    return out


def reho(bold_4d: np.ndarray, mask: np.ndarray | None = None,
         cluster: int = 27, tr: float = DEFAULT_TR,
         low: float = DEFAULT_LOW, high: float = DEFAULT_HIGH,
         detrend: bool = False, filter_band: bool = False
         ) -> tuple[np.ndarray, np.ndarray]:
    """Kendall's-W regional homogeneity (Ajay's ``compute_reho``).

    ``cluster`` is the neighborhood size (7, 19, or 27). Returns (map_3d, mask).
    """
    if mask is None:
        mask = brain_mask(bold_4d)
    w = _fm.compute_reho(bold_4d, mask=mask, neighborhood=cluster, tr=tr,
                         low=low, high=high, detrend=detrend,
                         filter_band=filter_band)
    return w, mask


def alff_falff(bold_4d: np.ndarray, tr: float = DEFAULT_TR,
               mask: np.ndarray | None = None,
               band: tuple[float, float] = (DEFAULT_LOW, DEFAULT_HIGH)
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ALFF and fALFF maps (Ajay's ``compute_alff`` / ``compute_falff``).

    Returns (alff_3d, falff_3d, mask). Both are DPABI-matched: linear detrend,
    zero-pad to next power of two, amplitude = |FFT|*2/T, DPABI bin cutoffs.
    """
    if mask is None:
        mask = brain_mask(bold_4d)
    flat = bold_4d[mask]                       # (V, T)
    low, high = band
    alff = _fm.compute_alff(flat, tr=tr, low=low, high=high, normalize=False)
    falff = _fm.compute_falff(flat, tr=tr, low=low, high=high, normalize=False)
    return _inflate(alff, mask), _inflate(falff, mask), mask


def rsfa(bold_4d: np.ndarray, mask: np.ndarray | None = None,
         tr: float = DEFAULT_TR,
         band: tuple[float, float] = (DEFAULT_LOW, DEFAULT_HIGH),
         bandpass: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """RSFA / BOLD-SD (Ajay's ``compute_bold_sd_rsfa``).

    Temporal SD (ddof=1) after linear detrend and FFT band-pass. Returns
    (map_3d, mask).
    """
    if mask is None:
        mask = brain_mask(bold_4d)
    flat = bold_4d[mask]                       # (V, T)
    low, high = band
    vals = _fm.compute_bold_sd_rsfa(flat, tr=tr, low=low, high=high,
                                    bandpass=bandpass, normalize=False)
    return _inflate(vals, mask), mask


# --- Arnav's additional measures (adapters over fmri_measures_arnav) ----

_SLOW_BANDS = {"slow5": _fma.SLOW5, "slow4": _fma.SLOW4}


def alff_falff_band(bold_4d: np.ndarray, key: str, tr: float = DEFAULT_TR,
                    mask: np.ndarray | None = None
                    ) -> tuple[np.ndarray, np.ndarray]:
    """One slow-band ALFF/fALFF map (Arnav). ``key`` e.g. 'alff_slow5'."""
    if mask is None:
        mask = brain_mask(bold_4d)
    _kind, name = key.split("_", 1)            # ('alff'|'falff', 'slow4'|'slow5')
    band = _SLOW_BANDS[name]
    res = _fma.compute_alff_falff(bold_4d[mask], tr=tr, bands={name: band},
                                  detrend=True, standardize=False)
    return _inflate(res[key], mask), mask


def int_timescale(bold_4d: np.ndarray, mask: np.ndarray | None = None,
                  tr: float = DEFAULT_TR, max_lag: int = DEFAULT_INT_MAX_LAG
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Intrinsic Neural Timescale (Arnav's ``compute_int``)."""
    if mask is None:
        mask = brain_mask(bold_4d)
    vals = _fma.compute_int(bold_4d[mask], tr=tr, max_lag=max_lag)
    return _inflate(vals, mask), mask


def coherence_reho(bold_4d: np.ndarray, mask: np.ndarray | None = None,
                   tr: float = DEFAULT_TR,
                   band: tuple[float, float] = (DEFAULT_LOW, DEFAULT_HIGH)
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Coherence-ReHo (Arnav's ``compute_coherence_reho``).

    Returns (map_3d, mask). The returned mask marks voxels where a coherence
    value was actually computed (finite); invalid voxels are 0 in the map.
    """
    if mask is None:
        mask = brain_mask(bold_4d)
    out3d = _fma.compute_coherence_reho(bold_4d, tr=tr, band=band, mask=mask)
    valid = mask & np.isfinite(out3d)
    return np.nan_to_num(np.asarray(out3d, dtype=np.float64), nan=0.0), valid


def mse_complexity(bold_4d: np.ndarray, mask: np.ndarray | None = None,
                   tr: float = DEFAULT_TR, scales=DEFAULT_MSE_SCALES,
                   m: int = DEFAULT_MSE_M, r_ratio: float = DEFAULT_MSE_R,
                   method: str = "mean") -> tuple[np.ndarray, np.ndarray]:
    """Multiscale-entropy complexity index (Arnav).

    Sample entropy per coarse-grained scale (``compute_mse``), collapsed across
    scales with a NaN-aware reduction (mean by default; +inf -> NaN first),
    matching ``compute_mse_complexity_index.py``. Requires ``antropy``.
    """
    if mask is None:
        mask = brain_mask(bold_4d)
    scales = tuple(int(s) for s in scales)
    per = _fma.compute_mse(bold_4d[mask], scales=scales, m=m, r_ratio=r_ratio,
                           detrend=True, standardize=False)
    stack = np.stack([per[f"mse_scale_{s}"] for s in scales], axis=0)  # (S, V)
    stack = stack.astype(np.float64)
    stack[np.isinf(stack)] = np.nan
    with np.errstate(invalid="ignore"):
        ci = np.nansum(stack, axis=0) if method == "sum" else np.nanmean(stack, axis=0)
    valid = mask.copy()
    valid[mask] = np.isfinite(ci)
    return _inflate(np.nan_to_num(ci, nan=0.0), mask), valid


def compute(measure: str, bold_4d: np.ndarray, params: dict | None = None):
    """Dispatch by measure key; return (map_3d, mask). Shared by app + scripts."""
    p = params or {}
    tr = float(p.get("tr", DEFAULT_TR))
    low = float(p.get("low", DEFAULT_LOW))
    high = float(p.get("high", DEFAULT_HIGH))
    # Ajay's four.
    if measure == "reho":
        return reho(bold_4d, cluster=int(p.get("cluster", 27)),
                    tr=tr, low=low, high=high)
    if measure in ("alff", "falff"):
        alff, falff, mask = alff_falff(bold_4d, tr=tr, band=(low, high))
        return (alff if measure == "alff" else falff), mask
    if measure == "rsfa":
        return rsfa(bold_4d, tr=tr, band=(low, high))
    # Arnav's additions.
    if measure in ("alff_slow5", "alff_slow4", "falff_slow5", "falff_slow4"):
        return alff_falff_band(bold_4d, measure, tr=tr)
    if measure == "int":
        return int_timescale(bold_4d, tr=tr,
                             max_lag=int(p.get("max_lag", DEFAULT_INT_MAX_LAG)))
    if measure == "coherence_reho":
        return coherence_reho(bold_4d, tr=tr, band=(low, high))
    if measure == "mse":
        scales = p.get("mse_scales", DEFAULT_MSE_SCALES)
        return mse_complexity(bold_4d, tr=tr, scales=scales,
                              m=int(p.get("mse_m", DEFAULT_MSE_M)),
                              r_ratio=float(p.get("mse_r", DEFAULT_MSE_R)))
    raise ValueError(f"unknown measure: {measure}")


# Registry so the UI/back end can look measures up by key. Ajay's four first,
# then Arnav's additional measures.
MEASURES = {
    "reho": {"label": "Regional Homogeneity (ReHo)"},
    "alff": {"label": "ALFF"},
    "falff": {"label": "fALFF (fractional ALFF)"},
    "rsfa": {"label": "RSFA (Resting-State Fluctuation Amplitude)"},
    "alff_slow5": {"label": "ALFF (slow-5, 0.01-0.027 Hz)"},
    "alff_slow4": {"label": "ALFF (slow-4, 0.027-0.073 Hz)"},
    "falff_slow5": {"label": "fALFF (slow-5, 0.01-0.027 Hz)"},
    "falff_slow4": {"label": "fALFF (slow-4, 0.027-0.073 Hz)"},
    "int": {"label": "INT (Intrinsic Neural Timescale)"},
    "coherence_reho": {"label": "Coherence-ReHo"},
    "mse": {"label": "MSE (Multiscale Entropy complexity index)"},
}
