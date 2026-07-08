"""Compare a new subject against the HCP normative reference.

Given a user subject's coupling value(s) and a stored reference, report where
the subject falls in the HCP distribution: z-score and percentile, globally
and/or per region.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def compare_subject(subject_coupling, reference: dict) -> dict:
    """Locate a subject within the normative distribution.

    Parameters
    ----------
    subject_coupling : float or (N,) array — must match the reference's metric.
    reference : dict from reference.build_reference / load_reference.

    Returns
    -------
    dict with z (z-scores), percentile, and the raw subject value.
    """
    x = np.asarray(subject_coupling, dtype=float)
    mean = np.asarray(reference["mean"], dtype=float)
    std = np.asarray(reference["std"], dtype=float)

    if x.shape != mean.shape:
        raise ValueError(
            f"subject coupling shape {x.shape} does not match reference "
            f"{mean.shape} — likely a different atlas or coupling metric"
        )

    with np.errstate(divide="ignore", invalid="ignore"):
        z = (x - mean) / std
    percentile = norm.cdf(z) * 100.0  # normal approximation

    return {
        "value": x,
        "z": z,
        "percentile": percentile,
        "atlas": reference.get("atlas"),
        "coupling_metric": reference.get("coupling_metric"),
    }


def empirical_percentile(subject_coupling, reference: dict) -> np.ndarray:
    """Percentile from the empirical HCP distribution (no normality assumption)."""
    x = np.asarray(subject_coupling, dtype=float)
    vals = np.asarray(reference["values"], dtype=float)  # (S,) or (S, N)
    return (vals < x).mean(axis=0) * 100.0
