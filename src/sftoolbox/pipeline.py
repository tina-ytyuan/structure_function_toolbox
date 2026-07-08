"""End-to-end per-subject extraction, tying the modules together.

extract_subject() is the single path every subject (HCP or user) runs through.
Steps that depend on the frozen input spec (loading NIfTIs, tractography) call
into io/fmri/fa; the coupling step is fully implemented.
"""

from __future__ import annotations

from . import coupling as _coupling
from . import fa as _fa
from . import fmri as _fmri
from . import io as _io


def extract_functional(subject: "_io.Subject", labels_3d, config):
    """Load BOLD, parcellate, return functional connectivity matrix."""
    bold = _io.load_nifti_data(subject.bold_path)
    ts = _fmri.region_timeseries(bold, labels_3d)
    return _fmri.functional_connectivity(ts, config)


def extract_structural(subject: "_io.Subject", config):
    """Return FA-weighted structural connectivity matrix."""
    return _fa.fa_structural_connectivity(subject, config)


def extract_subject(subject: "_io.Subject", labels_3d, config):
    """Run the full path for one subject and return its coupling value(s).

    Raises loudly if the subject does not conform to the input contract.
    """
    problems = _io.validate_conformance(subject, config)
    if problems:
        raise ValueError(
            f"Subject {subject.subject_id} does not conform to input_spec:\n  - "
            + "\n  - ".join(problems)
        )
    func = extract_functional(subject, labels_3d, config)
    struct = extract_structural(subject, config)
    return _coupling.coupling(struct, func, config)
