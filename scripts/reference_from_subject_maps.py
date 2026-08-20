"""Build a normative reference from per-subject measure maps, with normalisation.

Why this exists
---------------
Group mean/SD maps built from *raw* measure values are dominated by global
amplitude differences between subjects: ALFF and RSFA are in arbitrary BOLD
units, so scanner gain and intensity scaling inflate the across-subject SD
(observed CV ~0.39, versus ~0.14 for bounded measures like ReHo). That inflated
SD sits in the denominator of the single-subject t-test and crushes every t,
producing *fewer* significant voxels than chance and a uniform whole-brain
offset instead of regional effects.

The fix is the standard DPABI 'm' convention: divide each subject's map by its
own global in-mask mean before aggregating (mALFF / mReHo; Zang et al. 2007,
Zuo et al. 2010).

This script re-aggregates from measure maps your pipeline has *already*
computed, so it does not recompute any measure. Reading a few hundred 3D maps
takes minutes, not the hours a full recomputation would.

Usage
-----
    python scripts/reference_from_subject_maps.py --measure rsfa \
        --maps '/path/to/per_subject/*/Rest1LR/rsfa.nii.gz' \
        --mask /path/to/final_mask_no_ventricles.nii \
        --out  refs/measure_ref_rsfa.npz

Add --no-normalize to reproduce the old raw-unit behaviour for comparison.
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np

from sftoolbox import io, measure_norm


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--measure", required=True, help="Measure key, e.g. rsfa")
    ap.add_argument("--maps", required=True,
                    help="Glob matching one 3D measure map per subject")
    ap.add_argument("--mask", help="Mask NIfTI defining the analysis voxels")
    ap.add_argument("--no-normalize", action="store_true",
                    help="Skip global normalisation (raw units; not recommended)")
    ap.add_argument("--limit", type=int, default=0, help="Use only the first N maps")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    paths = sorted(glob.glob(args.maps))
    if args.limit:
        paths = paths[: args.limit]
    if not paths:
        raise SystemExit(f"No maps matched: {args.maps}")
    normalize = not args.no_normalize
    print(f"{len(paths)} subject maps | normalise: {normalize}")

    mask = None
    if args.mask:
        mask = np.asarray(io.load_nifti_data(args.mask)) != 0
        print(f"mask voxels: {int(mask.sum()):,}")

    # Welford's online algorithm: one pass, constant memory. Stacking 936
    # volumes would need ~7 GB; this needs three arrays.
    count = mean = m2 = None
    used = 0
    for i, p in enumerate(paths, 1):
        try:
            arr = np.asarray(io.load_nifti_data(p), dtype=float)
        except Exception as e:
            print(f"  skip {Path(p).name}: {e}")
            continue
        if mean is None:
            shape = arr.shape
            count = np.zeros(shape, dtype=np.int32)
            mean = np.zeros(shape, dtype=float)
            m2 = np.zeros(shape, dtype=float)
            if mask is None:
                mask = np.ones(shape, dtype=bool)
            elif mask.shape != shape:
                raise SystemExit(f"mask {mask.shape} != map {shape}")
        elif arr.shape != mean.shape:
            print(f"  skip {Path(p).name}: shape {arr.shape} != {mean.shape}")
            continue

        if normalize:
            arr = measure_norm.global_normalize(arr, mask)

        # A voxel contributes only where this subject actually has a value.
        v = mask & np.isfinite(arr) & (arr != 0)
        count[v] += 1
        delta = np.zeros_like(mean)
        delta[v] = arr[v] - mean[v]
        mean[v] += delta[v] / count[v]
        m2[v] += delta[v] * (arr[v] - mean[v])
        used += 1
        if i % 50 == 0 or i == len(paths):
            print(f"  {i}/{len(paths)} processed", flush=True)

    if used < 2:
        raise SystemExit(f"Need >=2 usable maps; got {used}")

    # Sample SD (ddof=1), only where enough subjects contributed.
    enough = count >= 2
    sd = np.zeros_like(mean)
    sd[enough] = np.sqrt(m2[enough] / (count[enough] - 1))

    group_mask = mask & enough & np.isfinite(mean) & np.isfinite(sd) & (sd > 0)
    mean = np.nan_to_num(mean, nan=0.0)
    sd = np.nan_to_num(sd, nan=0.0)
    n = int(np.median(count[group_mask])) if group_mask.any() else used

    ref = {
        "measure": args.measure,
        "shape": np.array(mean.shape),
        "mean_map": mean,
        "sd_map": sd,
        "group_mask": group_mask,
        "summaries": np.array([], dtype=float),
        "n": n,
        "normalized": bool(normalize),
        "mask_name": Path(args.mask).name if args.mask else "",
        "mask_voxels": int(group_mask.sum()),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **ref)

    inmask = group_mask
    cv = np.median(sd[inmask] / mean[inmask]) if inmask.any() else float("nan")
    print(
        f"\nWrote {args.measure} -> {args.out}\n"
        f"  subjects used : {used} (n stored = {n})\n"
        f"  mask voxels   : {int(group_mask.sum()):,}\n"
        f"  normalised    : {normalize}\n"
        f"  median CV     : {cv:.3f}   "
        f"(raw-unit measures ran ~0.39 before normalising)"
    )


if __name__ == "__main__":
    main()
