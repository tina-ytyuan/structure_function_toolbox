"""Validate a normative reference by testing held-out subjects against it.

Why this exists
---------------
A reference is well calibrated if a *typical* subject, drawn from the same
population as the cohort, produces an unremarkable result. Two quantities say
whether that holds, and both have known expected values:

  SD of t across voxels   -> 1.0
  observed / chance ratio -> 1.0   (voxels at p<0.05, over 5% of those tested)

Departures diagnose specific problems:

  SD(t) << 1 with |mean t| large
      The t map is nearly constant across voxels: a whole-brain offset moving
      every voxel together, not regional effects. Typical of amplitude measures
      (ALFF, RSFA) compared without global normalisation, where per-subject
      scanner scaling is perfectly correlated across voxels.

  SD(t) >> 1
      The cohort SD is too small; the test will over-declare significance.

  SD(t) ~ 1, ratio ~ 1
      Calibrated. A real patient effect then shows as a genuine excess.

Held-out subjects are required: a subject inside the cohort is pulled toward
the mean it helped define, which biases t toward 0. Subjects listed in the
reference's own ``subject_ids`` are skipped automatically.

Usage
-----
    python scripts/check_reference_calibration.py \
        --ref outputs/measure_ref_rsfa.npz \
        --maps '/hpc/group/.../results_gpu_raw/Rest1LR/*/*_rsfa.nii.gz' \
        --mask /hpc/group/.../final_mask_no_ventricles.nii --n 20
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
    ap.add_argument("--ref", required=True, help="Reference .npz to validate")
    ap.add_argument("--maps", required=True, help="Glob of per-subject measure maps")
    ap.add_argument("--mask", help="Mask NIfTI (should match the reference's)")
    ap.add_argument("--n", type=int, default=20, help="Held-out subjects to test")
    args = ap.parse_args()

    ref = measure_norm.load(args.ref)
    if not measure_norm.can_ttest(ref):
        raise SystemExit("Reference has no usable SD map or n<=1.")
    in_cohort = set(map(str, ref["subject_ids"])) if "subject_ids" in ref else set()

    print(f"reference   : {args.ref}")
    print(f"  measure   : {str(ref['measure'])}")
    print(f"  n         : {int(ref['n'])}   normalised: "
          f"{measure_norm.is_normalized(ref)}")
    print(f"  cohort ids: {len(in_cohort) or 'not recorded'}")

    mask = None
    if args.mask:
        mask = np.asarray(io.load_nifti_data(args.mask)) != 0

    paths = sorted(glob.glob(args.maps))
    held_out = [p for p in paths if Path(p).parent.name not in in_cohort]
    if not in_cohort:
        print("  WARNING: reference lists no subject_ids, so held-out status "
              "cannot be verified. Results may be biased toward 0.")
        held_out = paths
    print(f"maps        : {len(paths)} found, {len(held_out)} held out\n")
    if not held_out:
        raise SystemExit("No held-out subjects available to test.")

    rows = []
    for p in held_out[: args.n]:
        sid = Path(p).parent.name
        arr = np.asarray(io.load_nifti_data(p), float)
        smask = np.isfinite(arr) & (arr != 0)
        if mask is not None:
            smask &= mask
        try:
            t, pv, valid, _df = measure_norm.ttest_vs_group(arr, smask, ref)
        except ValueError as e:
            print(f"  skip {sid}: {e}")
            continue
        tv, pp = t[valid], pv[valid]
        if tv.size == 0:
            continue
        n_sig = int((pp < 0.05).sum())
        rows.append((sid, float(tv.mean()), float(tv.std()),
                     n_sig / (0.05 * tv.size)))

    if not rows:
        raise SystemExit("No subjects could be tested.")

    print(f"{'subject':12s} {'mean t':>9s} {'SD(t)':>8s} {'obs/chance':>11s}")
    print("-" * 44)
    for sid, mt, st, oe in rows:
        print(f"{sid:12s} {mt:+9.3f} {st:8.3f} {oe:11.2f}")

    mts = np.array([r[1] for r in rows])
    sts = np.array([r[2] for r in rows])
    oes = np.array([r[3] for r in rows])
    print("-" * 44)
    print(f"{'median':12s} {np.median(mts):+9.3f} {np.median(sts):8.3f} "
          f"{np.median(oes):11.2f}")
    print(f"{'expected':12s} {0.0:+9.3f} {1.0:8.3f} {1.0:11.2f}")

    sd_med, oe_med = float(np.median(sts)), float(np.median(oes))
    print()
    if sd_med < 0.7:
        print("VERDICT: NOT calibrated - SD(t) well below 1.")
        print("  The t map is dominated by a whole-brain offset rather than")
        print("  regional variation. For amplitude measures rebuild the")
        print("  reference with global normalisation.")
        if abs(float(np.median(mts))) > 0.2:
            print(f"  Median |mean t| = {abs(float(np.median(mts))):.3f} "
                  "confirms a systematic offset.")
    elif sd_med > 1.4:
        print("VERDICT: NOT calibrated - SD(t) well above 1; cohort SD too "
              "small, so significance will be over-declared.")
    elif oe_med > 1.5:
        print(f"VERDICT: NOT calibrated - {oe_med:.2f}x as many p<0.05 voxels as "
              "chance, so the test over-declares significance.")
        print("  If the reference averaged each subject's runs while test")
        print("  subjects supply a single run, it describes a quieter quantity")
        print("  than what it is compared against. Rebuild with")
        print("  --runs separate so the SD reflects single-run variability.")
    elif oe_med < 0.5:
        print("VERDICT: conservative - fewer p<0.05 voxels than chance despite "
              "reasonable SD(t). Inspect the mask and cohort composition.")
    else:
        print("VERDICT: calibrated. Typical subjects look unremarkable, so a "
              "genuine effect should stand out.")


if __name__ == "__main__":
    main()
