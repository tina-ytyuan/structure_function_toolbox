"""FA-based structural connectivity from diffusion data.

This is the toolbox's niche: use fractional anisotropy (FA) as the structural
signal, rather than raw streamline counts. We ship code so a user can compute
FA maps from their own preprocessed diffusion data, then build an FA-weighted
structural connectivity matrix in the same parcellation as the fMRI FC matrix.

Two pieces here:
  1. compute_fa_map      -- FA volume from preprocessed diffusion (tensor fit).
  2. fa_structural_connectivity -- FA -> N x N structural matrix.

Both depend on the diffusion pipeline / tractography choice, so the heavy parts
are stubbed with explicit interfaces until we lock the atlas + method.
"""

from __future__ import annotations

import numpy as np


def compute_fa_map(dwi_data, bval, bvec, mask=None) -> np.ndarray:
    """Fit the diffusion tensor and return an FA map.

    Intended to run on the user's PREPROCESSED diffusion data (eddy/motion
    corrected, brain-extracted). Implementation will use dipy's tensor model:

        from dipy.core.gradients import gradient_table
        from dipy.reconst.dti import TensorModel
        gtab = gradient_table(bval, bvec)
        fa = TensorModel(gtab).fit(dwi_data, mask=mask).fa

    Kept as a stub so `dipy` isn't a hard dependency until this path is wired.
    """
    raise NotImplementedError(
        "compute_fa_map: wire up dipy TensorModel once diffusion input spec is frozen"
    )


def roi_fa(fa_map: np.ndarray, labels_3d: np.ndarray) -> np.ndarray:
    """Mean FA within each parcel -> (N,) node feature vector.

    Useful directly (regional white-matter FA) and as a building block.
    """
    region_ids = np.unique(labels_3d)
    region_ids = region_ids[region_ids != 0]
    out = np.zeros(region_ids.size, dtype=float)
    for i, rid in enumerate(region_ids):
        out[i] = fa_map[labels_3d == rid].mean()
    return out


def fa_structural_connectivity(subject, config) -> np.ndarray:
    """N x N FA-weighted structural connectivity matrix.

    For config.fa_measure == "fa_weighted_sc": run/parse tractography and set
    edge (i, j) to the mean FA sampled along streamlines connecting regions
    i and j. Requires a tractogram + the atlas; method TBD with mentor.

    For config.fa_measure == "roi_fa": there is no edge matrix; callers should
    use roi_fa() node features instead. We raise here to fail loudly.
    """
    if config.fa_measure == "roi_fa":
        raise ValueError(
            "fa_measure='roi_fa' produces node features, not an edge matrix; "
            "call roi_fa() instead."
        )
    if config.fa_measure == "fa_weighted_sc":
        raise NotImplementedError(
            "fa_structural_connectivity: implement FA-weighted tractography "
            "aggregation once atlas + tractography method are chosen"
        )
    raise ValueError(f"Unknown fa_measure: {config.fa_measure}")
