"""Build the HCP normative reference.

Runs every HCP subject through the same extraction path and stores the
distribution of coupling values. The saved .npz is the redistributable
"standard dataset" the web app and compare step load.

Usage
-----
Real data (each subject is a subfolder under --subjects-root):
    python scripts/build_reference.py \
        --subjects-root /path/to/HCP \
        --atlas /path/to/atlas_labels.nii.gz \
        --out outputs/hcp_reference.npz

Synthetic (no data — makes a placeholder reference so the app's compare panel
works before real HCP data is available):
    python scripts/build_reference.py --synthetic --out outputs/hcp_reference.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from sftoolbox.config import DEFAULT
from sftoolbox import io, pipeline, reference


def build_real(subjects_root: str, atlas_path: str, out: str, cfg=DEFAULT):
    root = Path(subjects_root)
    subject_dirs = sorted(p for p in root.iterdir() if p.is_dir())
    if not subject_dirs:
        raise SystemExit(f"No subject subfolders found under {root}")

    labels_3d = io.load_nifti_data(atlas_path)
    expected = int(np.unique(np.rint(labels_3d).astype(int))[1:].size)

    values, ids, skipped = [], [], []
    for sd in subject_dirs:
        sid = sd.name
        try:
            subj = io.Subject.from_dir(sid, sd)
            val = pipeline.extract_subject(subj, labels_3d, cfg,
                                           expected_regions=expected)
            values.append(val)
            ids.append(sid)
            print(f"  ok   {sid}")
        except Exception as e:
            skipped.append((sid, str(e)))
            print(f"  skip {sid}: {e}")

    if not values:
        raise SystemExit("No subjects processed successfully; nothing to save.")

    reference.build_reference(values, ids, cfg, out_path=out)
    print(f"\nWrote reference for {len(ids)} subjects -> {out}")
    if skipped:
        print(f"Skipped {len(skipped)} subject(s).")


def build_synthetic(out: str, n_subjects: int = 100, cfg=DEFAULT):
    """Placeholder reference so the compare panel works before real data."""
    rng = np.random.default_rng(0)
    if cfg.coupling_metric == "regional_corr":
        n_regions = 200
        vals = list(rng.normal(0.3, 0.1, size=(n_subjects, n_regions)))
    else:
        vals = list(rng.normal(0.3, 0.1, size=n_subjects))
    ids = [f"SYN{i:03d}" for i in range(n_subjects)]
    reference.build_reference(vals, ids, cfg, out_path=out)
    print(f"Wrote SYNTHETIC reference ({n_subjects} subjects) -> {out}")
    print("Replace with a real HCP reference before drawing any conclusions.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subjects-root", help="folder with one subfolder per subject")
    ap.add_argument("--atlas", help="integer label NIfTI matching the config atlas")
    ap.add_argument("--out", default=DEFAULT.reference_path)
    ap.add_argument("--synthetic", action="store_true",
                    help="build a placeholder reference with no data")
    args = ap.parse_args()

    if args.synthetic:
        build_synthetic(args.out)
    else:
        if not (args.subjects_root and args.atlas):
            ap.error("real mode needs --subjects-root and --atlas (or --synthetic)")
        build_real(args.subjects_root, args.atlas, args.out)


if __name__ == "__main__":
    main()
