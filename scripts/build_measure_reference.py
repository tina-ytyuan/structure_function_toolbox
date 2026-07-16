"""Build a normative reference for one fMRI measure from a folder of subjects.

Runs the chosen measure on every subject in a folder and saves a reference the
web app auto-loads to show cohort comparisons (summary percentile + z-map).

Usage
-----
    python scripts/build_measure_reference.py \
        --subjects-root ~/sft_test_subjects --measure reho \
        --out outputs/measure_ref_reho.npz

    # ALFF/fALFF need TR + band:
    python scripts/build_measure_reference.py --subjects-root ~/hcp \
        --measure alff --tr 0.72 --low 0.01 --high 0.08

Voxelwise z-maps only work if subjects share a voxel grid (same space). The
summary percentile works regardless.
"""

from __future__ import annotations

import argparse

import numpy as np

from sftoolbox import io, measure_norm, measures


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--subjects-root", required=True)
    ap.add_argument(
        "--pattern", default="*bold*.nii*",
        help="Glob for BOLD files (DCC HCP: '*_finproc.nii.gz')",
    )
    ap.add_argument(
        "--limit", type=int, default=None,
        help="Only process the first N subjects (for quick tests)",
    )
    ap.add_argument(
        "--measure", required=True, choices=sorted(measures.MEASURES),
        help="Any measure key (core four or additional measures)",
    )
    ap.add_argument("--out")
    ap.add_argument(
        "--mask",
        help="3D NIfTI mask applied to every subject (e.g. the group's "
             "final_mask.nii). Voxels != 0 are kept. Defaults to a per-subject "
             "finite/nonzero mask when omitted.",
    )
    ap.add_argument("--cluster", type=int, default=27)
    ap.add_argument("--tr", type=float, default=0.72)
    ap.add_argument("--low", type=float, default=0.01)
    ap.add_argument("--high", type=float, default=0.08)
    ap.add_argument("--max-lag", type=int, default=measures.DEFAULT_INT_MAX_LAG,
                    help="INT only: autocorrelation lags")
    ap.add_argument("--mse-scales", type=int, nargs="+",
                    default=list(measures.DEFAULT_MSE_SCALES),
                    help="MSE only: coarse-graining scales")
    ap.add_argument("--mse-m", type=int, default=measures.DEFAULT_MSE_M,
                    help="MSE only: sample-entropy embedding order")
    ap.add_argument("--mse-r", type=float, default=measures.DEFAULT_MSE_R,
                    help="MSE only: sample-entropy tolerance ratio")
    args = ap.parse_args()

    mask = None
    if args.mask:
        mask = np.asarray(io.load_nifti_data(args.mask)) != 0
        print(f"Using mask {args.mask}: {int(mask.sum())} voxels, shape {mask.shape}")

    params = {
        "cluster": args.cluster,
        "tr": args.tr,
        "low": args.low,
        "high": args.high,
        "max_lag": args.max_lag,
        "mse_scales": tuple(args.mse_scales),
        "mse_m": args.mse_m,
        "mse_r": args.mse_r,
        "mask": mask,
    }
    out = args.out or measure_norm.default_path(args.measure)

    bolds = io.find_subject_bolds(args.subjects_root, pattern=args.pattern)
    if not bolds:
        raise SystemExit(f"No BOLD NIfTIs found under {args.subjects_root}")
    if args.limit is not None:
        bolds = dict(list(bolds.items())[: args.limit])
        print(f"Limiting to first {len(bolds)} subjects")

    maps, masks, ids, skipped = [], [], [], []
    shapes = set()
    for sid, path in bolds.items():
        try:
            bold = io.load_nifti_data(path)
            if bold.ndim != 4:
                raise ValueError(f"not 4D (shape {bold.shape})")
            m, mask = measures.compute(args.measure, bold, params)
            maps.append(m)
            masks.append(mask)
            ids.append(sid)
            shapes.add(m.shape)
            print(f"  ok   {sid}  {m.shape}")
        except Exception as e:
            skipped.append((sid, str(e)))
            print(f"  skip {sid}: {e}")

    if not maps:
        raise SystemExit("No subjects processed; nothing to save.")
    if len(shapes) > 1:
        print(
            f"\nWARNING: subjects have differing grids {shapes}. "
            "Voxelwise z-maps need a common grid; the summary percentile will "
            "still work. Consider resampling subjects to a common space."
        )

    measure_norm.build(args.measure, maps, masks, out)
    print(f"\nWrote {args.measure} reference for {len(ids)} subjects -> {out}")
    if skipped:
        print(f"Skipped {len(skipped)}.")


if __name__ == "__main__":
    main()
