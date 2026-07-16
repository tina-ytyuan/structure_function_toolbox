"""Command-line entry points (thin wrappers; logic lives in the modules)."""

from __future__ import annotations

import argparse

from .config import DEFAULT
from .reference import load_reference


def build_reference_cli(argv=None):
    p = argparse.ArgumentParser(description="Build HCP normative reference.")
    p.add_argument("--subjects-dir", required=True)
    p.add_argument("--out", default=DEFAULT.reference_path)
    p.parse_args(argv)  # validates args / provides --help until this is wired
    raise SystemExit(
        "Not wired yet: iterate HCP subjects via pipeline.extract_subject, "
        "then reference.build_reference. See scripts/build_reference.py."
    )


def compare_cli(argv=None):
    p = argparse.ArgumentParser(description="Compare a subject to the reference.")
    p.add_argument("--subject-dir", required=True)
    p.add_argument("--reference", default=DEFAULT.reference_path)
    args = p.parse_args(argv)
    load_reference(args.reference)  # fail early if the reference is unreadable
    raise SystemExit(
        "Not wired yet: run pipeline.extract_subject on the subject, then "
        "compare.compare_subject(value, ref). See scripts/compare_subject.py."
    )
