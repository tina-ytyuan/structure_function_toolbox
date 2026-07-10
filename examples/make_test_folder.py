"""Download a folder of real resting-state subjects to try batch mode.

Fetches N subjects from the ADHD-200 / 1000 Functional Connectome sample via
nilearn and arranges them as the app expects: one subfolder per subject, each
containing a BOLD NIfTI.

Run:
    python examples/make_test_folder.py                 # 4 subjects
    python examples/make_test_folder.py --n 6 --out ~/sft_subjects

Then start the app, choose "Folder of subjects", and pick the printed folder.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4, help="number of subjects (max 40)")
    ap.add_argument("--out", default="~/sft_test_subjects",
                    help="destination folder")
    args = ap.parse_args()

    from nilearn import datasets  # project dependency

    dest = Path(args.out).expanduser()
    dest.mkdir(parents=True, exist_ok=True)

    print(f"Fetching {args.n} ADHD resting-state subjects (cached in ~/nilearn_data)…")
    data = datasets.fetch_adhd(n_subjects=args.n)

    for i, func in enumerate(data.func, start=1):
        sid = f"sub-{i:02d}"
        sd = dest / sid
        sd.mkdir(exist_ok=True)
        target = sd / "rest_bold.nii.gz"
        if not target.exists():
            shutil.copy(func, target)
        print(f"  {sid}  ->  {target}")

    print(f"\nDone. Subjects folder:\n  {dest}\n")
    print("In the app: Input mode = 'Folder of subjects' -> Choose folder -> "
          "select the folder above.")


if __name__ == "__main__":
    main()
