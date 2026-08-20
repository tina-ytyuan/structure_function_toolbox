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


def build(
    measure: str, maps: list[np.ndarray], masks: list[np.ndarray], out_path: str
) -> dict:
    """Build and save a normative reference from per-subject maps + masks."""
    arr = np.stack([np.asarray(m, float) for m in maps])  # (S, X, Y, Z)
    msk = np.stack([np.asarray(m, bool) for m in masks])  # (S, X, Y, Z)
    group_mask = msk.all(axis=0)
    mean_map = arr.mean(axis=0)
    sd_map = arr.std(axis=0, ddof=1) if arr.shape[0] > 1 else np.zeros_like(mean_map)
    summaries = np.array([a[m].mean() if m.any() else np.nan for a, m in zip(arr, msk)])
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


def global_normalize(map_3d: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Scale a map by its own global mean inside the mask (DPABI 'm' convention).

    Measures in raw BOLD units (ALFF, RSFA) carry arbitrary per-subject scaling
    from scanner gain and intensity normalisation, so the across-subject SD ends
    up dominated by *global* amplitude differences rather than regional ones.
    Dividing each subject by its own in-mask mean removes that nuisance scale,
    which is what mALFF / mReHo do (Zang et al. 2007; Zuo et al. 2010).

    Ratio-valued measures (fALFF, ReHo, MSE) are already largely scale-free, but
    normalising them is harmless and keeps every reference on one convention.
    """
    m = np.asarray(map_3d, float)
    valid = np.asarray(mask, bool) & np.isfinite(m) & (m != 0)
    if not valid.any():
        return m
    g = float(m[valid].mean())
    if not np.isfinite(g) or g == 0:
        return m
    out = np.zeros_like(m)
    out[valid] = m[valid] / g
    return out


def is_normalized(ref: dict) -> bool:
    """True if this reference was built from globally normalised subject maps."""
    return bool(ref["normalized"]) if "normalized" in ref else False


def has_sd(ref: dict) -> bool:
    """True if the reference carries a usable across-subject SD map."""
    if "sd_map" not in ref:
        return False
    return bool(np.any(np.asarray(ref["sd_map"], float) > 0))


def can_ttest(ref: dict) -> bool:
    """True if the reference supports a single-subject t-test (needs SD and n>1)."""
    return has_sd(ref) and int(ref.get("n", 0)) > 1


def fdr_correct(p_values: np.ndarray, alpha: float = 0.05):
    """Benjamini-Hochberg FDR correction over a 1-D array of p-values.

    Returns (q_values, threshold) where ``q_values`` are the BH-adjusted
    p-values (monotone, same order as the input) and ``threshold`` is the
    largest raw p that survives at ``alpha`` (0.0 if nothing survives).

    FDR controls the expected *proportion* of false positives among the voxels
    you declare significant — the standard choice for voxelwise maps, and far
    less conservative than Bonferroni over ~10^5 voxels.
    """
    p = np.asarray(p_values, dtype=float).ravel()
    n = p.size
    if n == 0:
        return p.copy(), 0.0
    order = np.argsort(p)
    ranked = p[order]
    # BH critical values, then enforce monotonicity from the top down.
    q_sorted = np.minimum.accumulate((ranked * n / np.arange(1, n + 1))[::-1])[::-1]
    q_sorted = np.clip(q_sorted, 0.0, 1.0)
    q = np.empty_like(q_sorted)
    q[order] = q_sorted
    passing = ranked <= (np.arange(1, n + 1) / n) * alpha
    threshold = float(ranked[passing].max()) if passing.any() else 0.0
    return q, threshold


def ttest_vs_group(subject_map: np.ndarray, subject_mask: np.ndarray, ref: dict):
    """Single-subject vs group t-test, voxelwise (Crawford & Howell, 1998).

    Tests whether one individual's value differs from a control group, using the
    group mean, SD and size. Unlike a z-score this accounts for the group being a
    finite sample:

        t = (x - mean) / (SD * sqrt((n + 1) / n)),   df = n - 1

    Returns (t_map, p_map, mask, df) where ``p`` is the two-tailed p-value. The
    maps are 0 outside the valid mask. Requires an SD map and n > 1 in the
    reference (``can_ttest``); raises ValueError otherwise.
    """
    from scipy import stats

    if not can_ttest(ref):
        raise ValueError(
            "reference has no across-subject SD map or n <= 1; rebuild it from "
            "individual subjects (scripts/build_measure_reference.py) so it "
            "stores mean, SD and n"
        )
    if tuple(int(s) for s in ref["shape"]) != subject_map.shape:
        raise ValueError(
            f"subject grid {subject_map.shape} does not match the group "
            f"{tuple(int(s) for s in ref['shape'])}"
        )

    mean_map = np.asarray(ref["mean_map"], float)
    sd_map = np.asarray(ref["sd_map"], float)
    group_mask = np.asarray(ref["group_mask"], bool)
    n = int(ref["n"])
    df = n - 1

    # Put the subject on the same scale the cohort was built on. Without this a
    # subject whose global amplitude differs by a few percent shows a uniform
    # whole-brain offset that swamps any regional effect.
    if is_normalized(ref):
        subject_map = global_normalize(subject_map, subject_mask & group_mask)

    valid = group_mask & subject_mask & (sd_map > 0)
    t = np.zeros_like(subject_map, dtype=float)
    p = np.ones_like(subject_map, dtype=float)
    denom = sd_map[valid] * np.sqrt((n + 1.0) / n)
    tv = (subject_map[valid] - mean_map[valid]) / denom
    t[valid] = tv
    p[valid] = 2.0 * stats.t.sf(np.abs(tv), df)
    return t, p, valid, df


def difference(subject_map: np.ndarray, subject_mask: np.ndarray, ref: dict):
    """Compare a subject to the group *average* (no SD needed).

    Returns (diff_map, diff_mask, subject_mean, group_mean) where
    ``diff_map = subject - group_mean`` inside the shared mask. If the subject
    and group grids differ, diff_map/diff_mask are None and only the two summary
    means are returned. Use this when the reference has only a mean map.
    """
    mean_map = np.asarray(ref["mean_map"], float)
    group_mask = np.asarray(ref["group_mask"], bool)

    if tuple(int(s) for s in ref["shape"]) != subject_map.shape:
        subj_mean = float(subject_map[subject_mask].mean())
        return None, None, subj_mean, np.nan

    m = group_mask & subject_mask
    diff = np.zeros_like(subject_map, dtype=float)
    diff[m] = subject_map[m] - mean_map[m]
    subj_mean = float(subject_map[m].mean()) if m.any() else np.nan
    grp_mean = float(mean_map[m].mean()) if m.any() else np.nan
    return diff, m, subj_mean, grp_mean
