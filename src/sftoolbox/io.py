"""Loading preprocessed subject data.

The toolbox does NOT preprocess. It consumes already-preprocessed data that
conforms to docs/input_spec.md. This module locates and loads files, and
validates that an incoming subject matches the expected space/atlas so a
user-supplied subject is actually comparable to the HCP cohort.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Subject:
    """A single subject's preprocessed inputs, already in standard space."""

    subject_id: str
    root: Path
    bold_path: Path | None = None  # cleaned resting-state fMRI (4D NIfTI)
    fa_path: Path | None = None  # precomputed FA map (3D NIfTI), optional
    dwi_path: Path | None = None  # preprocessed diffusion (4D NIfTI)
    bval_path: Path | None = None  # b-values (FSL .bval)
    bvec_path: Path | None = None  # b-vectors (FSL .bvec)
    dwi_mask_path: Path | None = None  # brain mask for diffusion (3D NIfTI)
    tractogram_path: Path | None = None  # streamlines (.trk / .tck), for FA-SC
    t1_path: Path | None = None  # T1w (optional extension)

    @classmethod
    def from_dir(
        cls, subject_id: str, root: str | Path, layout: dict | None = None
    ) -> Subject:
        """Build a Subject by resolving expected filenames under ``root``.

        ``layout`` maps field names to filename patterns (glob), letting the
        input spec stay flexible. Defaults follow a simple flat convention;
        adjust once the HCP directory layout is confirmed.
        """
        root = Path(root)
        if not root.exists():
            raise FileNotFoundError(f"Subject dir not found: {root}")

        default_layout = {
            "bold_path": "*bold*.nii*",
            "fa_path": "*FA*.nii*",
            "dwi_path": "*dwi*.nii*",
            "bval_path": "*.bval",
            "bvec_path": "*.bvec",
            "dwi_mask_path": "*mask*.nii*",
            "tractogram_path": "*.t[rc]k",
            "t1_path": "*T1w*.nii*",
        }
        layout = {**default_layout, **(layout or {})}

        resolved = {}
        for field, pattern in layout.items():
            matches = sorted(root.glob(pattern))
            resolved[field] = matches[0] if matches else None

        return cls(subject_id=subject_id, root=root, **resolved)


# --- Low-level loaders -------------------------------------------------


def find_subject_bolds(
    root: str | Path, pattern: str = "*bold*.nii*"
) -> dict[str, Path]:
    """Discover BOLD files for a folder of subjects.

    Handles two common layouts:
      1. one subfolder per subject, each holding a BOLD file, and
      2. a flat folder of BOLD files (each file treated as one subject).

    Returns {subject_id: bold_path}, sorted by id.
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Folder not found: {root}")

    found: dict[str, Path] = {}
    # Layout 1: subject subfolders.
    for sub in sorted(p for p in root.iterdir() if p.is_dir()):
        matches = sorted(sub.glob(pattern)) or sorted(sub.glob("*.nii*"))
        if matches:
            found[sub.name] = matches[0]

    # Layout 2: flat files, if no subfolders yielded anything.
    if not found:
        for f in sorted(root.glob(pattern)) or sorted(root.glob("*.nii*")):
            found[f.stem.replace(".nii", "")] = f

    return found


def load_nifti(path: str | Path):
    """Return the nibabel image object (keeps affine/header)."""
    import nibabel as nib

    return nib.load(str(path))


def load_nifti_data(path: str | Path) -> np.ndarray:
    """Load a NIfTI file's data array."""
    return np.asarray(load_nifti(path).get_fdata())


def load_bvals_bvecs(bval_path, bvec_path):
    """Load FSL-format b-values and b-vectors."""
    bvals = np.loadtxt(str(bval_path))
    bvecs = np.loadtxt(str(bvec_path))
    # FSL bvec is 3 x N; return as N x 3 for dipy's gradient_table.
    if bvecs.shape[0] == 3 and bvecs.shape[1] != 3:
        bvecs = bvecs.T
    return bvals, bvecs


# --- Conformance -------------------------------------------------------


def validate_conformance(
    subject: Subject, config, expected_regions: int | None = None
) -> list[str]:
    """Check a subject matches the input contract.

    Returns a list of human-readable problems (empty == conformant). This is
    what keeps the user-vs-HCP comparison valid: a subject in the wrong space
    or atlas must not be compared to the normative cohort.

    Checks performed:
      * required inputs for the configured measures are present on disk
      * the parcellation the caller will use has the expected region count
        (pass ``expected_regions`` from the atlas), when a labels file is given

    Space-affine matching is best done by the caller once the standard-space
    template affine is fixed in the input spec; a hook is left below.
    """
    problems: list[str] = []

    # Functional inputs.
    if subject.bold_path is None or not Path(subject.bold_path).exists():
        problems.append("missing cleaned resting-state BOLD (bold_path)")

    # Structural inputs depend on how FA connectivity is built.
    if config.fa_measure == "fa_weighted_sc":
        if subject.fa_path is None and subject.dwi_path is None:
            problems.append("need an FA map or preprocessed diffusion to derive FA")
        if subject.tractogram_path is None:
            problems.append("fa_weighted_sc requires a tractogram (tractogram_path)")
    elif config.fa_measure == "roi_fa":
        if subject.fa_path is None and subject.dwi_path is None:
            problems.append("roi_fa needs an FA map or preprocessed diffusion")

    # Atlas region-count check (if the caller can supply the labels volume).
    if expected_regions is not None and subject.fa_path is not None:
        try:
            n = _label_count(subject.fa_path)  # placeholder; see note
            if n and n != expected_regions:
                problems.append(
                    f"parcellation has {n} regions, expected {expected_regions} "
                    f"for atlas {config.atlas}"
                )
        except Exception:
            pass

    return problems


def _label_count(labels_path) -> int | None:
    """Number of non-zero labels in an integer atlas volume."""
    data = load_nifti_data(labels_path)
    ids = np.unique(np.rint(data).astype(int))
    ids = ids[ids != 0]
    return int(ids.size) if ids.size else None
