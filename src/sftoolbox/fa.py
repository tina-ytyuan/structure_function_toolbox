"""FA-based structural connectivity from diffusion data.

This is the toolbox's niche: use fractional anisotropy (FA) as the structural
signal, rather than raw streamline counts. We ship code so a user can compute
FA maps from their own preprocessed diffusion data, then build an FA-weighted
structural connectivity matrix in the same parcellation as the fMRI FC matrix.

Pieces:
  1. compute_fa_map            -- FA volume from preprocessed diffusion (dipy DTI).
  2. roi_fa                    -- mean FA per parcel (node feature).
  3. fa_structural_connectivity-- mean FA along streamlines per region pair.

dipy is imported lazily inside the functions that need it, so importing this
module (and the rest of the package) does not require dipy to be installed.
Install it to run the FA paths:  pip install dipy
"""

from __future__ import annotations

import numpy as np


def compute_fa_map(dwi_data: np.ndarray, bvals, bvecs, mask=None) -> np.ndarray:
    """Fit the diffusion tensor and return an FA map.

    Runs on PREPROCESSED diffusion data (eddy/motion corrected, brain
    extracted). Uses dipy's weighted-least-squares tensor fit.

    Parameters
    ----------
    dwi_data : (X, Y, Z, G) diffusion-weighted volumes.
    bvals : (G,) b-values.       bvecs : (G, 3) unit gradient directions.
    mask : (X, Y, Z) bool brain mask (optional but recommended for speed).

    Returns
    -------
    (X, Y, Z) FA map in [0, 1], zeros outside the mask.
    """
    from dipy.core.gradients import gradient_table
    from dipy.reconst.dti import TensorModel

    gtab = gradient_table(bvals, bvecs)
    tenmodel = TensorModel(gtab, fit_method="WLS")
    tenfit = tenmodel.fit(dwi_data, mask=mask)
    fa = np.asarray(tenfit.fa)
    fa[~np.isfinite(fa)] = 0.0  # tensor fit can yield NaNs in CSF/air
    return np.clip(fa, 0.0, 1.0)


def apply_threshold(fa_map: np.ndarray, fa_threshold: float = 0.0) -> np.ndarray:
    """Zero out voxels with FA below ``fa_threshold``.

    FA below ~0.2 is typically not coherent white matter, so a threshold keeps
    grey-matter / CSF / partial-volume voxels from diluting FA measures. Returns
    a copy; the input is not modified. ``fa_threshold=0`` is a no-op.
    """
    if fa_threshold <= 0.0:
        return np.asarray(fa_map, dtype=float)
    out = np.asarray(fa_map, dtype=float).copy()
    out[out < fa_threshold] = 0.0
    return out


def roi_fa(
    fa_map: np.ndarray, labels_3d: np.ndarray, fa_threshold: float = 0.0
) -> np.ndarray:
    """Mean FA within each parcel -> (N,) node-feature vector.

    Useful directly (regional white-matter FA) and as a building block.
    Pure numpy; no dipy needed.

    If ``fa_threshold`` > 0, only voxels at or above it contribute to a region's
    mean (sub-threshold voxels are considered non-white-matter). A region with
    no surviving voxels yields NaN.
    """
    labels = np.rint(labels_3d).astype(int)
    region_ids = np.unique(labels)
    region_ids = region_ids[region_ids != 0]
    out = np.zeros(region_ids.size, dtype=float)
    for i, rid in enumerate(region_ids):
        mask = labels == rid
        if fa_threshold > 0.0:
            mask = mask & (fa_map >= fa_threshold)
        out[i] = fa_map[mask].mean() if mask.any() else np.nan
    return out


def fa_weighted_connectivity(
    streamlines,
    affine,
    labels_3d,
    fa_map,
    n_regions: int | None = None,
    fa_threshold: float = 0.0,
) -> np.ndarray:
    """N x N structural matrix: mean FA sampled along streamlines per region pair.

    For each streamline, its two endpoints fall in parcels (i, j). We sample FA
    at every point along the streamline, average per streamline, then average
    across all streamlines connecting i and j. Edges with no streamlines are 0.

    Parameters
    ----------
    streamlines : sequence of (P_k, 3) arrays in the same world space as ``fa_map``.
    affine : (4, 4) voxel->world affine of the label/FA volumes.
    labels_3d : (X, Y, Z) integer atlas (0 = background, 1..N = regions).
    fa_map : (X, Y, Z) FA volume aligned to ``labels_3d``.
    n_regions : number of regions N (inferred from labels if None).
    fa_threshold : if > 0, FA samples below this along a streamline are ignored
        (treated as non-white-matter) rather than pulling the edge weight down.
    """
    from dipy.tracking.streamline import values_from_volume
    from dipy.tracking.utils import connectivity_matrix

    # Sub-threshold voxels -> NaN so per-streamline nanmean skips them.
    if fa_threshold > 0.0:
        fa_map = np.where(np.asarray(fa_map) >= fa_threshold, fa_map, np.nan)

    labels = np.rint(labels_3d).astype(np.int_)
    if n_regions is None:
        ids = np.unique(labels)
        n_regions = int(ids[ids != 0].max())

    # Map each region-pair to the streamlines connecting them.
    _, mapping = connectivity_matrix(
        streamlines,
        affine,
        labels,
        return_mapping=True,
        mapping_as_streamlines=True,
        symmetric=True,
    )

    fa_sc = np.zeros((n_regions, n_regions), dtype=float)
    for (i, j), strls in mapping.items():
        if i == 0 or j == 0 or not len(strls):
            continue  # skip background and empty pairs
        # Mean FA along each streamline, then mean across streamlines.
        per_strl = [np.nanmean(v) for v in values_from_volume(fa_map, strls, affine)]
        w = float(np.nanmean(per_strl)) if per_strl else 0.0
        a, b = i - 1, j - 1  # labels are 1-based -> 0-based indices
        fa_sc[a, b] = w
        fa_sc[b, a] = w
    return fa_sc


def fa_structural_connectivity(subject, labels_3d, config) -> np.ndarray:
    """High-level: build the FA structural matrix for a subject.

    Loads the subject's FA map (or computes it from diffusion) and tractogram,
    then delegates to fa_weighted_connectivity.
    """
    if config.fa_measure == "roi_fa":
        raise ValueError(
            "fa_measure='roi_fa' produces node features, not an edge matrix; "
            "call roi_fa() instead."
        )
    if config.fa_measure != "fa_weighted_sc":
        raise ValueError(f"Unknown fa_measure: {config.fa_measure}")

    from . import io as _io

    # FA map: prefer a precomputed one; else fit from preprocessed diffusion.
    if subject.fa_path is not None:
        fa_map = _io.load_nifti_data(subject.fa_path)
        affine = _io.load_nifti(subject.fa_path).affine
    elif subject.dwi_path is not None:
        dwi_img = _io.load_nifti(subject.dwi_path)
        bvals, bvecs = _io.load_bvals_bvecs(subject.bval_path, subject.bvec_path)
        mask = (
            _io.load_nifti_data(subject.dwi_mask_path).astype(bool)
            if subject.dwi_mask_path
            else None
        )
        fa_map = compute_fa_map(np.asarray(dwi_img.get_fdata()), bvals, bvecs, mask)
        affine = dwi_img.affine
    else:
        raise ValueError("subject has neither fa_path nor dwi_path")

    if subject.tractogram_path is None:
        raise ValueError("fa_weighted_sc requires subject.tractogram_path")

    from dipy.io.streamline import load_tractogram

    sft = load_tractogram(
        str(subject.tractogram_path), reference="same", bbox_valid_check=False
    )
    sft.to_rasmm()
    fa_threshold = float(getattr(config, "fa_threshold", 0.0) or 0.0)
    return fa_weighted_connectivity(
        sft.streamlines, affine, labels_3d, fa_map, fa_threshold=fa_threshold
    )
