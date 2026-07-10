"""Per-measure normative references for voxelwise fMRI measures.

A reference for a measure (ReHo, ALFF, fALFF, RSFA) stores, over a cohort:
  * mean_map, sd_map  -- voxelwise mean and SD (for a z-map comparison), and
  * summaries         -- one number per subject (mean value in mask), for a
                         summary percentile comparison.

Voxelwise comparison requires every subject on the same voxel grid / space, so
a reference records its ``shape`` and comparison checks the incoming subject
matches. The summary percentile works regardless of grid.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def build(measure: str, maps: list[np.ndarray], masks: list[np.ndarray],
          out_path: str) -> dict:
    """Build and save a normative reference from per-subject maps + masks."""
    arr = np.stack([np.asarray(m, float) for m in maps])        # (S, X, Y, Z)
    msk = np.stack([np.asarray(m, bool) for m in masks])        # (S, X, Y, Z)
    group_mask = msk.all(axis=0)
    mean_map = arr.mean(axis=0)
    sd_map = arr.std(axis=0, ddof=1) if arr.shape[0] > 1 else np.zeros_like(mean_map)
    summaries = np.array([a[m].mean() if m.any() else np.nan
                          for a, m in zip(arr, msk)])
    ref = {
        "measure": measure,
        "shape": np.array(mean_map.shape),
        "mean_map": mean_map,
        "sd_map": sd_map,
        "group_mask": group_mask,
        "summaries": summaries,
        "n": arr.shape[0],
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **ref)
    return ref


def load(path: str) -> dict:
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def default_path(measure: str) -> str:
    return f"outputs/measure_ref_{measure}.npz"


def compare(subject_map: np.ndarray, subject_mask: np.ndarray, ref: dict):
    """Compare a subject against a reference.

    Returns (z_map, z_mask, summary_value, summary_percentile). The z-map is
    only meaningful where the subject and cohort grids match; if shapes differ,
    z_map/z_mask are None and only the summary percentile is returned.
    """
    summary = float(subject_map[subject_mask].mean())
    summaries = np.asarray(ref["summaries"], float)
    summaries = summaries[np.isfinite(summaries)]
    percentile = float((summaries < summary).mean() * 100) if summaries.size else np.nan

    if tuple(int(s) for s in ref["shape"]) != subject_map.shape:
        return None, None, summary, percentile

    mean_map = np.asarray(ref["mean_map"], float)
    sd_map = np.asarray(ref["sd_map"], float)
    group_mask = np.asarray(ref["group_mask"], bool)
    z_mask = group_mask & subject_mask & (sd_map > 0)
    z = np.zeros_like(subject_map, dtype=float)
    z[z_mask] = (subject_map[z_mask] - mean_map[z_mask]) / sd_map[z_mask]
    return z, z_mask, summary, percentile
