"""Build the HCP normative reference distribution.

Run the identical measure-extraction path over many HCP subjects and store the
distribution of coupling values. This saved reference IS the redistributable
"standard dataset": a user drops in their own subject and compares against it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def build_reference(coupling_values: list, subject_ids: list[str], config,
                    out_path: str | None = None) -> dict:
    """Aggregate per-subject coupling into a normative distribution.

    Parameters
    ----------
    coupling_values : list where each item is one subject's coupling output
        (a float for global_corr, or an (N,) array for regional_corr).
    subject_ids : matching subject identifiers.
    config : the Config used (stored for provenance / conformance checks).

    Stores mean, std, and the raw stacked values so percentiles can be computed
    at compare time.
    """
    stacked = np.array(coupling_values, dtype=float)  # (S,) or (S, N)
    ref = {
        "values": stacked,
        "mean": np.nanmean(stacked, axis=0),
        "std": np.nanstd(stacked, axis=0, ddof=1),
        "subject_ids": np.array(subject_ids),
        "atlas": config.atlas,
        "coupling_metric": config.coupling_metric,
        "space": config.space,
    }
    out_path = out_path or config.reference_path
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **ref)
    return ref


def load_reference(path: str) -> dict:
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}
