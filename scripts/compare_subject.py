"""
Compare one user-supplied subject against the HCP normative reference.
"""

from __future__ import annotations

import argparse

from sftoolbox import compare, io, pipeline, reference
from sftoolbox.config import DEFAULT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("subject_id")
    ap.add_argument("subject_dir")
    ap.add_argument("--reference", default=DEFAULT.reference_path)
    args = ap.parse_args()

    cfg = DEFAULT
    ref = reference.load_reference(args.reference)
    labels_3d = None  # TODO: load atlas labels (must match cfg.atlas)

    subj = io.Subject.from_dir(args.subject_id, args.subject_dir)
    value = pipeline.extract_subject(subj, labels_3d, cfg)
    result = compare.compare_subject(value, ref)
    print(result)


if __name__ == "__main__":
    main()
