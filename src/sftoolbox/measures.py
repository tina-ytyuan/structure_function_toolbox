"""Voxelwise fMRI measures computed from cleaned BOLD only.

Each function takes a 4D BOLD array (X, Y, Z, T) and returns a 3D map plus a
brain mask. No atlas, diffusion, or extra inputs required — just cleaned BOLD
(and TR for frequency-based measures).

Measures
--------
  reho        Regional homogeneity (Kendall's W over a voxel neighborhood).
  alff_falff  Amplitude of low-frequency fluctuation (and fractional ALFF).
  seed_fc     Seed-based functional connectivity (correlation map).

These are intentionally standard, well-known definitions so results are
comparable to other resting-state pipelines.
"""

from __future__ import annotations

import numpy as np


def brain_mask(bold_4d: np.ndarray) -> np.ndarray:
    """Simple data-driven mask: voxels with non-trivial temporal variance."""
    var = bold_4d.var(axis=-1)
    thr = var.mean() * 0.05
    return var > max(thr, 1e-8)


def _rank_time(bold_4d: np.ndarray) -> np.ndarray:
    """Rank each voxel's time series along time (ties ignored; fine for BOLD)."""
    order = np.argsort(bold_4d, axis=-1)
    ranks = np.empty_like(order, dtype=np.float64)
    T = bold_4d.shape[-1]
    idx = np.arange(1, T + 1)
    np.put_along_axis(ranks, order, np.broadcast_to(idx, order.shape), axis=-1)
    return ranks


def _neighbor_offsets(cluster: int):
    if cluster not in (7, 19, 27):
        raise ValueError("cluster must be 7, 19, or 27")
    offs = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                man = abs(dx) + abs(dy) + abs(dz)
                if cluster == 7 and man > 1:
                    continue
                if cluster == 19 and man > 2:
                    continue
                offs.append((dx, dy, dz))
    return offs


def reho(bold_4d: np.ndarray, mask: np.ndarray | None = None,
         cluster: int = 27) -> tuple[np.ndarray, np.ndarray]:
    """Regional homogeneity (Kendall's coefficient of concordance) map.

    For each voxel, the concordance of its own and its neighbors' time-series
    rankings. cluster = 7 (faces), 19 (+edges), or 27 (+corners).
    """
    if mask is None:
        mask = brain_mask(bold_4d)
    X, Y, Z, T = bold_4d.shape
    ranks = _rank_time(bold_4d)                      # (X,Y,Z,T)
    offs = _neighbor_offsets(cluster)
    K = len(offs)

    summed = np.zeros((X, Y, Z, T), dtype=np.float64)
    count = np.zeros((X, Y, Z), dtype=np.float64)
    for dx, dy, dz in offs:
        shifted = np.roll(ranks, shift=(dx, dy, dz), axis=(0, 1, 2))
        summed += shifted
        count += 1
    # Kendall's W per voxel from the summed neighbor ranks over time.
    mean_r = summed.mean(axis=-1, keepdims=True)
    S = ((summed - mean_r) ** 2).sum(axis=-1)
    denom = (K ** 2) * (T ** 3 - T)
    W = np.where(denom > 0, 12.0 * S / denom, 0.0)
    W[~mask] = 0.0
    return W, mask


def alff_falff(bold_4d: np.ndarray, tr: float, mask: np.ndarray | None = None,
               band: tuple[float, float] = (0.01, 0.08)
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ALFF and fractional ALFF maps.

    ALFF = mean spectral amplitude within ``band``.
    fALFF = band amplitude / total amplitude across all positive frequencies.
    Requires TR (seconds) to build the frequency axis.
    """
    if mask is None:
        mask = brain_mask(bold_4d)
    T = bold_4d.shape[-1]
    x = bold_4d - bold_4d.mean(axis=-1, keepdims=True)      # remove DC
    freqs = np.fft.rfftfreq(T, d=tr)
    amp = np.abs(np.fft.rfft(x, axis=-1))
    inband = (freqs >= band[0]) & (freqs <= band[1])
    total = amp[..., freqs > 0].sum(axis=-1)
    band_amp = amp[..., inband].sum(axis=-1)
    alff = np.where(inband.sum() > 0, amp[..., inband].mean(axis=-1), 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        falff = np.where(total > 0, band_amp / total, 0.0)
    alff[~mask] = 0.0
    falff[~mask] = 0.0
    return alff, falff, mask


def rsfa(bold_4d: np.ndarray, mask: np.ndarray | None = None
         ) -> tuple[np.ndarray, np.ndarray]:
    """Resting-State Fluctuation Amplitude: temporal standard deviation of BOLD.

    RSFA (Kannurpatti & Biswal, 2008) is the standard deviation of each voxel's
    resting-state time series — a simple, robust amplitude measure that needs
    only cleaned BOLD (no TR or frequency band).
    """
    if mask is None:
        mask = brain_mask(bold_4d)
    out = bold_4d.std(axis=-1)
    out[~mask] = 0.0
    return out, mask


def seed_fc(bold_4d: np.ndarray, seed_ijk: tuple[int, int, int],
            mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Seed-based functional connectivity: correlation of every voxel with a seed.

    seed_ijk is a voxel coordinate (i, j, k). Returns a correlation map in
    [-1, 1].
    """
    if mask is None:
        mask = brain_mask(bold_4d)
    i, j, k = seed_ijk
    X, Y, Z, T = bold_4d.shape
    if not (0 <= i < X and 0 <= j < Y and 0 <= k < Z):
        raise ValueError(f"seed {seed_ijk} outside volume {(X, Y, Z)}")
    seed = bold_4d[i, j, k, :]
    seed = (seed - seed.mean()) / (seed.std() + 1e-12)
    xz = bold_4d - bold_4d.mean(axis=-1, keepdims=True)
    xz = xz / (bold_4d.std(axis=-1, keepdims=True) + 1e-12)
    fc = (xz * seed).mean(axis=-1)
    fc[~mask] = 0.0
    return np.clip(fc, -1.0, 1.0), mask


def compute(measure: str, bold_4d: np.ndarray, params: dict | None = None):
    """Dispatch by measure key; return (map_3d, mask). Shared by app + scripts."""
    p = params or {}
    if measure == "reho":
        return reho(bold_4d, cluster=int(p.get("cluster", 27)))
    if measure in ("alff", "falff"):
        alff, falff, mask = alff_falff(
            bold_4d, tr=float(p.get("tr", 2.0)),
            band=(float(p.get("low", 0.01)), float(p.get("high", 0.08))))
        return (alff if measure == "alff" else falff), mask
    if measure == "rsfa":
        return rsfa(bold_4d)
    raise ValueError(f"unknown measure: {measure}")


# Registry so the UI/back end can look measures up by key.
MEASURES = {
    "reho": {"label": "Regional Homogeneity (ReHo)"},
    "alff": {"label": "ALFF"},
    "falff": {"label": "fALFF (fractional ALFF)"},
    "rsfa": {"label": "RSFA (Resting-State Fluctuation Amplitude)"},
}
