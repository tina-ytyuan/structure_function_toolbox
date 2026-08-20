"""Wrap an existing group-mean map into a toolbox measure reference (.npz).

Your group already computed voxelwise group-mean maps per measure (and, for FA,
a group-SD map). This turns one of those NIfTIs into the reference the toolbox's
compare step / web app loads — no recomputation of a cohort needed.

Usage
-----
    python scripts/reference_from_group_map.py --measure reho \
        --mean .../mean_maps/reho/reho_Rest1LR_group_mean.nii.gz \
        --out  /path/to/project/$USER/refs/measure_ref_reho.npz

Add --std <SD NIfTI> if a group standard-deviation map exists (that's what
enables the voxelwise z-map "how far from average"). Without --std, the
reference still supports the subject-vs-average difference view.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from sftoolbox import io


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--measure", required=True, help="Measure key, e.g. reho")
    ap.add_argument("--mean", required=True, help="Group-mean NIfTI")
    ap.add_argument("--std", help="Group-SD NIfTI (optional; enables z-maps)")
    ap.add_argument("--mask", help="Mask NIfTI (default: finite & nonzero mean)")
    ap.add_argument("--n", type=int, default=0,
                    help="Subjects in the group average. Required (>1) for the "
                         "single-subject t-test; also settable from --count.")
    ap.add_argument("--count",
                    help="Per-voxel subject-count NIfTI; its max sets n if --n "
                         "is not given")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    mean_map = np.asarray(io.load_nifti_data(args.mean), dtype=float)
    if mean_map.ndim != 3:
        raise SystemExit(f"Mean map must be 3D; got shape {mean_map.shape}")

    if args.std:
        sd_map = np.asarray(io.load_nifti_data(args.std), dtype=float)
        if sd_map.shape != mean_map.shape:
            raise SystemExit(
                f"SD shape {sd_map.shape} != mean shape {mean_map.shape}")
    else:
        sd_map = np.zeros_like(mean_map)

    if args.mask:
        group_mask = np.asarray(io.load_nifti_data(args.mask)) != 0
        if group_mask.shape != mean_map.shape:
            raise SystemExit(
                f"Mask shape {group_mask.shape} != mean shape {mean_map.shape}")
    else:
        group_mask = np.isfinite(mean_map) & (mean_map != 0)

    # Group maps often carry NaN outside the analysis mask. Restrict the mask to
    # voxels where mean and SD are both finite, then zero the NaNs, so the stored
    # reference never propagates NaN into a comparison.
    group_mask = group_mask & np.isfinite(mean_map) & np.isfinite(sd_map)
    mean_map = np.nan_to_num(mean_map, nan=0.0)
    sd_map = np.nan_to_num(sd_map, nan=0.0)

    n = int(args.n)
    if n <= 0 and args.count:
        count_map = np.asarray(io.load_nifti_data(args.count), dtype=float)
        n = int(np.nanmax(count_map))
        print(f"n taken from --count map: {n}")

    ref = {
        "measure": args.measure,
        "shape": np.array(mean_map.shape),
        "mean_map": mean_map,
        "sd_map": sd_map,
        "group_mask": group_mask,
        "summaries": np.array([], dtype=float),   # no per-subject values here
        "n": n,
        # Provenance: lets the app tell users which mask this reference assumes
        # and warn when an incoming subject was masked differently.
        "mask_name": Path(args.mask).name if args.mask else "",
        "mask_voxels": int(group_mask.sum()),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **ref)
    tt = "yes" if (args.std and n > 1) else "no"
    print(
        f"Wrote {args.measure} reference -> {args.out}\n"
        f"  mean {mean_map.shape}, SD map: {'yes' if args.std else 'no'}, "
        f"n={n}, mask voxels {int(group_mask.sum())}, t-test enabled: {tt}"
    )
    if args.std and n <= 1:
        print("  WARNING: SD present but n<=1, so the t-test stays disabled. "
              "Pass --n (e.g. --n 936) or --count.")


if __name__ == "__main__":
    main()
