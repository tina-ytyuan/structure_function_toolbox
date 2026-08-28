"""Make publication copies of references with cohort membership removed.

What a reference actually contains
----------------------------------
Almost everything in a reference is aggregate: ``mean_map`` and ``sd_map`` are
averages over the whole cohort, ``group_mask`` is geometry, and the rest is
metadata. None of it lets anyone recover an individual.

Two fields are different:

  subject_ids  the list of cohort members
  summaries    one value per subject, when populated

``subject_ids`` is a membership list rather than a measurement, and the IDs are
public under HCP Open Access on their own. The care is needed because of what
the list implies in combination with the selection criterion: HCP gates family
structure behind Restricted Access, so if a cohort was chosen to be mutually
unrelated, publishing who is in it also says something about who is not.

Nothing is lost by removing them. A reference is *used* by comparing a subject
against the mean and SD; membership only matters for rebuilding a byte-identical
copy, and anyone rebuilding supplies their own cohort list anyway.

Usage
-----
    python scripts/strip_reference_ids.py \
        --refs 'outputs/measure_ref_*.npz' --out-dir outputs/publish

Then verify the maps are untouched:

    python scripts/strip_reference_ids.py --verify outputs/publish
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np

# Fields that describe individuals or cohort membership rather than the group.
SENSITIVE = ("subject_ids", "summaries", "subjects_file")


def strip_one(src: Path, dst: Path) -> dict:
    data = np.load(src, allow_pickle=True)
    kept, dropped = {}, []
    for k in data.files:
        if k in SENSITIVE:
            v = data[k]
            n = int(v.size) if hasattr(v, "size") else 0
            dropped.append(f"{k}({n})" if n else k)
            continue
        kept[k] = data[k]
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez(dst, **kept)
    return {"dropped": dropped, "kept": sorted(kept)}


def verify(pub_dir: Path, orig_glob: str) -> int:
    """Confirm the published copies keep identical maps and no membership."""
    problems = 0
    for pub in sorted(pub_dir.glob("measure_ref_*.npz")):
        orig = Path(orig_glob).parent / pub.name
        p = np.load(pub, allow_pickle=True)
        leaked = [f for f in SENSITIVE if f in p.files]
        if leaked:
            print(f"  {pub.name}: FAIL - still contains {leaked}")
            problems += 1
            continue
        if not orig.exists():
            print(f"  {pub.name}: stripped ok (no original to compare)")
            continue
        o = np.load(orig, allow_pickle=True)
        same = all(
            np.array_equal(np.asarray(o[k]), np.asarray(p[k]))
            for k in ("mean_map", "sd_map", "group_mask", "n")
            if k in p.files and k in o.files
        )
        print(f"  {pub.name}: {'maps identical' if same else 'MAPS DIFFER'}"
              f", membership removed")
        problems += 0 if same else 1
    return problems


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--refs", default="outputs/measure_ref_*.npz",
                    help="Glob of references to copy")
    ap.add_argument("--out-dir", default="outputs/publish")
    ap.add_argument("--verify", metavar="DIR",
                    help="Check an already-stripped directory and exit")
    args = ap.parse_args()

    if args.verify:
        print(f"verifying {args.verify}")
        bad = verify(Path(args.verify), args.refs)
        raise SystemExit(1 if bad else 0)

    srcs = sorted(glob.glob(args.refs))
    if not srcs:
        raise SystemExit(f"No references matched: {args.refs}")
    out_dir = Path(args.out_dir)
    print(f"{len(srcs)} references -> {out_dir}\n")

    for s in srcs:
        src = Path(s)
        info = strip_one(src, out_dir / src.name)
        dropped = ", ".join(info["dropped"]) or "nothing to drop"
        print(f"  {src.name}: removed {dropped}")

    print(f"\nWrote {len(srcs)} publication copies to {out_dir}")
    print("Originals are unchanged; keep them for reproducibility.")
    print(f"\nVerify with:\n  python {Path(__file__).name} --verify {out_dir}")


if __name__ == "__main__":
    main()
