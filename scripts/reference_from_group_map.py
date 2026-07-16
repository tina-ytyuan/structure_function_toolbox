"""Wrap an existing group-mean map into a toolbox measure reference (.npz).

Your group already computed voxelwise group-mean maps per measure (and, for FA,
a group-SD map). This turns one of those NIfTIs into the reference the toolbox's
compare step / web app loads — no recomputation of a cohort needed.

Usage
-----
    python scripts/reference_from_group_map.py --measure reho \
        --mean .../mean_maps/reho/reho_Rest1LR_group_mean.nii.gz \
        --out  /hpc/group/396-brainfun26/tina/refs/measure_ref_reho.npz

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
                    help="Subjects in the group average (provenance only)")
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

    ref = {
        "measure": args.measure,
        "shape": np.array(mean_map.shape),
        "mean_map": mean_map,
        "sd_map": sd_map,
        "group_mask": group_mask,
        "summaries": np.array([], dtype=float),   # no per-subject values here
        "n": int(args.n),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **ref)
    print(
        f"Wrote {args.measure} reference -> {args.out}\n"
        f"  mean {mean_map.shape}, SD map: {'yes' if args.std else 'no'}, "
        f"mask voxels {int(group_mask.sum())}"
    )


if __name__ == "__main__":
    main()
