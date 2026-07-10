"""Download a real resting-state BOLD file to test the app, via nilearn.

Run:
    python examples/fetch_test_data.py           # ADHD resting-state (default)
    python examples/fetch_test_data.py --dev      # development_fmri (movie)

Prints the path to a 4D NIfTI you can upload in the web app (the "Cleaned BOLD"
field). nilearn caches downloads under ~/nilearn_data.
"""

from __future__ import annotations

import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", action="store_true",
                    help="use fetch_development_fmri instead of fetch_adhd")
    args = ap.parse_args()

    from nilearn import datasets  # nilearn is a project dependency

    if args.dev:
        data = datasets.fetch_development_fmri(n_subjects=1)
        bold_path = data.func[0]
        kind = "development_fmri (movie-watching)"
    else:
        data = datasets.fetch_adhd(n_subjects=1)
        bold_path = data.func[0]
        kind = "ADHD resting-state"

    print(f"\nDataset: {kind}")
    print(f"BOLD 4D NIfTI: {bold_path}")
    print("\nUpload that file in the app's 'Cleaned BOLD' field, pick a measure, "
          "and Generate.")


if __name__ == "__main__":
    main()
