"""Loading preprocessed subject data.

The toolbox does NOT preprocess. It consumes already-preprocessed data that
conforms to docs/input_spec.md. This module only locates and loads files;
it validates that an incoming subject matches the expected space/atlas so a
user-supplied subject is comparable to the HCP cohort.
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
    # Paths to preprocessed inputs (resolved lazily by the extractors).
    bold_path: Path | None = None       # cleaned resting-state fMRI (4D NIfTI)
    dwi_fa_path: Path | None = None      # FA map (3D NIfTI) — see note below
    dwi_dir: Path | None = None          # diffusion dir, if FA computed here
    t1_path: Path | None = None          # T1w (optional extension)
    parcellation_path: Path | None = None  # atlas labels in subject/standard space

    @classmethod
    def from_dir(cls, subject_id: str, root: str | Path) -> "Subject":
        root = Path(root)
        if not root.exists():
            raise FileNotFoundError(f"Subject dir not found: {root}")
        return cls(subject_id=subject_id, root=root)


def validate_conformance(subject: Subject, config) -> list[str]:
    """Check that a subject matches the input contract.

    Returns a list of human-readable problems (empty == conformant). This is
    what protects the user-vs-HCP comparison from being meaningless: a subject
    in the wrong space or atlas cannot be compared to the normative cohort.

    TODO: fill in real checks once the input_spec is frozen, e.g.
      - NIfTI affine/space matches config.space
      - parcellation matches config.atlas (n regions, label set)
      - BOLD TR / n volumes within expected range
      - warn on scanner/site metadata differing from HCP (batch effects)
    """
    problems: list[str] = []
    # Placeholder until spec is frozen.
    return problems


def load_nifti_data(path: str | Path) -> np.ndarray:
    """Load a NIfTI file's data array. Thin wrapper for testability."""
    import nibabel as nib  # imported lazily so config/logic import stays light

    return np.asarray(nib.load(str(path)).get_fdata())
