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
    ap.add_argument("--subjects",
                    help="Text file of subject IDs (one per line) to restrict "
                         "to. Use this to reproduce a specific cohort rather "
                         "than whatever happens to be on disk.")
    ap.add_argument("--runs", choices=("separate", "average"), default="separate",
                    help="How to treat multiple runs per subject. 'separate' "
                         "(default) aggregates every run, so the SD describes "
                         "how much a SINGLE run varies across the population — "
                         "which is what a user uploads. 'average' averages a "
                         "subject's runs first, describing a quieter quantity "
                         "and over-declaring significance against single runs.")
    ap.add_argument("--limit", type=int, default=0, help="Use only the first N maps")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    paths = sorted(glob.glob(args.maps))
    if not paths:
        raise SystemExit(f"No maps matched: {args.maps}")

    wanted = None
    if args.subjects:
        wanted = {
            ln.strip() for ln in Path(args.subjects).read_text().splitlines()
            if ln.strip() and not ln.startswith("#")
        }
        # Match a subject ID appearing anywhere in the path (…/100206/… or
        # …/100206_finproc_LR_rsfa.nii.gz), so this works across layouts.
        kept, seen = [], set()
        for p in paths:
            hit = next((s for s in wanted if s in p), None)
            if hit is not None:
                kept.append(p)
                seen.add(hit)
        missing = sorted(wanted - seen)
        print(f"cohort list: {len(wanted)} subjects | matched {len(seen)} | "
              f"maps {len(kept)} of {len(paths)} on disk")
        if missing:
            print(f"  WARNING: {len(missing)} listed subjects have no map, e.g. "
                  f"{', '.join(missing[:5])}")
        paths = kept
        if not paths:
            raise SystemExit("No maps matched the subject list")

    if args.limit:
        paths = paths[: args.limit]
    normalize = not args.no_normalize
    print(f"{len(paths)} subject maps | normalise: {normalize}")

    # Record the grid the reference lives on, so a subject on a different
    # MNI grid can be resampled onto it instead of being rejected.
    ref_affine = np.asarray(io.load_nifti(paths[0]).affine, float)

    mask = None
    if args.mask:
        mask = np.asarray(io.load_nifti_data(args.mask)) != 0
        print(f"mask voxels: {int(mask.sum()):,}")

    # Group maps by subject so runs can be handled deliberately. The reference
    # must describe the same quantity the subject supplies: users upload ONE
    # run, so the SD should be single-run variability. Averaging a subject's
    # runs first describes a quieter quantity and makes ordinary subjects look
    # abnormal — measured at SD(t) 1.31 and 1.88x chance on HCP RSFA.
    by_subject: dict[str, list[str]] = {}
    for p in paths:
        by_subject.setdefault(Path(p).parent.name, []).append(p)
    subjects = sorted(by_subject)
    runs_each = [len(v) for v in by_subject.values()]
    multi_run = bool(runs_each) and max(runs_each) > 1

    if args.runs == "average":
        groups = [by_subject[s] for s in subjects]
        if multi_run:
            print(f"grouping {len(paths)} maps into {len(subjects)} subjects "
                  f"({min(runs_each)}-{max(runs_each)} runs each); runs averaged "
                  f"within subject")
    else:
        # One group per map: every run is its own observation, so the SD
        # describes single-run variability — the same quantity a user's single
        # uploaded run represents. n is still reported as the number of people,
        # keeping the t-test's df conservative rather than counting each run as
        # an independent subject.
        groups = [[p] for s in subjects for p in by_subject[s]]
        if multi_run:
            print(f"using {len(paths)} maps from {len(subjects)} subjects "
                  f"({min(runs_each)}-{max(runs_each)} runs each) as separate "
                  f"observations; SD describes single-run variability")

    # Welford's online algorithm: one pass, constant memory. Stacking 936
    # volumes would need ~7 GB; this needs three arrays.
    count = mean = m2 = None
    used = 0

    # Per-slice summaries, tracked alongside. The spread of *individual*
    # subjects' slice averages cannot be recovered later from mean_map and
    # sd_map: the SD of an average depends on how voxels co-vary within the
    # slice, and averaging discards that. Deriving it as sd/sqrt(voxels)
    # assumes voxels are independent, which understates it roughly 40-fold and
    # makes every subject look abnormal. So measure it here, while the
    # individual maps are still in hand.
    s_count = s_mean = s_m2 = None
    for i, group in enumerate(groups, 1):
        # Each group is one observation: a single map, or a subject's runs
        # averaged. Runs are normalised individually first, so a run with
        # different global scaling cannot dominate.
        acc = None
        acc_n = 0
        for p in group:
            try:
                arr = np.asarray(io.load_nifti_data(p), dtype=float)
            except Exception as e:
                print(f"  skip {Path(p).name}: {e}")
                continue
            if mean is None and acc is None:
                shape = arr.shape
                count = np.zeros(shape, dtype=np.int32)
                mean = np.zeros(shape, dtype=float)
                m2 = np.zeros(shape, dtype=float)
                if mask is None:
                    mask = np.ones(shape, dtype=bool)
                elif mask.shape != shape:
                    raise SystemExit(f"mask {mask.shape} != map {shape}")
            elif arr.shape != (mean.shape if mean is not None else arr.shape):
                print(f"  skip {Path(p).name}: shape {arr.shape} != {mean.shape}")
                continue
            if normalize:
                arr = measure_norm.global_normalize(arr, mask)
            arr = np.where(np.isfinite(arr), arr, 0.0)
            acc = arr if acc is None else acc + arr
            acc_n += 1
        if acc is None or acc_n == 0:
            continue
        arr = acc / acc_n

        # This observation's mean within each axial slice, over the analysis
        # mask, then Welford again over those per-slice values.
        nz = mask.shape[2]
        if s_mean is None:
            s_count = np.zeros(nz, dtype=np.int32)
            s_mean = np.zeros(nz, dtype=float)
            s_m2 = np.zeros(nz, dtype=float)
        for z in range(nz):
            mz = mask[:, :, z] & np.isfinite(arr[:, :, z]) & (arr[:, :, z] != 0)
            if mz.sum() < 20:
                continue
            sv = float(arr[:, :, z][mz].mean())
            s_count[z] += 1
            d = sv - s_mean[z]
            s_mean[z] += d / s_count[z]
            s_m2[z] += d * (sv - s_mean[z])

        # A voxel contributes only where this subject actually has a value.
        v = mask & np.isfinite(arr) & (arr != 0)
        count[v] += 1
        delta = np.zeros_like(mean)
        delta[v] = arr[v] - mean[v]
        mean[v] += delta[v] / count[v]
        m2[v] += delta[v] * (arr[v] - mean[v])
        used += 1
        if i % 50 == 0 or i == len(groups):
            print(f"  {i}/{len(groups)} observations processed", flush=True)

    if used < 2:
        raise SystemExit(f"Need >=2 usable subjects; got {used}")

    # Slice-level sample SD (ddof=1), across observations.
    slice_sd = np.zeros_like(s_mean)
    ok_s = s_count >= 2
    slice_sd[ok_s] = np.sqrt(s_m2[ok_s] / (s_count[ok_s] - 1))

    # Sample SD (ddof=1), only where enough subjects contributed.
    enough = count >= 2
    sd = np.zeros_like(mean)
    sd[enough] = np.sqrt(m2[enough] / (count[enough] - 1))

    group_mask = mask & enough & np.isfinite(mean) & np.isfinite(sd) & (sd > 0)
    mean = np.nan_to_num(mean, nan=0.0)
    sd = np.nan_to_num(sd, nan=0.0)
    # df for the single-subject t-test comes from the number of independent
    # PEOPLE, not the number of maps. With runs kept separate, count reflects
    # ~4 maps per person, so use the subject total instead — the SD benefits
    # from every run while the df stays honest.
    n_obs = int(np.median(count[group_mask])) if group_mask.any() else used
    n = len(subjects) if args.runs == "separate" else n_obs

    ref = {
        "measure": args.measure,
        "shape": np.array(mean.shape),
        "mean_map": mean,
        "sd_map": sd,
        "group_mask": group_mask,
        "summaries": np.array([], dtype=float),
        "n": n,
        "affine": ref_affine,
        # Per-slice distribution of individual subjects, for the profile plot.
        "slice_mean": s_mean,
        "slice_sd": slice_sd,
        "slice_n": s_count,
        "normalized": bool(normalize),
        "mask_name": Path(args.mask).name if args.mask else "",
        "mask_voxels": int(group_mask.sum()),
        # Record exactly which subjects went in, so the cohort behind a
        # reference is always recoverable from the file itself.
        "subject_ids": np.array(subjects),
        "runs_per_subject": int(np.median(runs_each)) if runs_each else 1,
        "subjects_file": Path(args.subjects).name if args.subjects else "",
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
