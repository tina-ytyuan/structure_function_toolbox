"""Central configuration.

These choices define what the toolbox computes. Change them in ONE place so
HCP subjects and user subjects are always processed identically. The two most
consequential decisions (raise with mentor before scaling up):

  * ATLAS          -> fixes matrix dimensions and region correspondence
  * COUPLING_METRIC -> defines what "structure and function aligning" means
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # --- Parcellation ---------------------------------------------------
    # Name of the atlas used to define network nodes. Both the FA structural
    # matrix and the fMRI functional matrix are built in this parcellation so
    # they share dimensions and node correspondence.
    # TODO(mentor): confirm atlas. Candidates: Schaefer-200, Desikan-Killiany,
    #   or an HCP-MMP1 variant. White-matter extension may need a WM atlas too.
    atlas: str = "schaefer_200"
    space: str = "MNI152NLin6Asym"  # all inputs must be in this standard space

    # --- Functional connectivity ---------------------------------------
    fc_measure: str = "correlation"  # correlation | partial_correlation
    fc_fisher_z: bool = True  # Fisher r->z transform edges

    # --- Structural (FA) connectivity ----------------------------------
    # How FA becomes an edge weight between two regions.
    # "fa_weighted_sc": mean FA sampled along streamlines connecting regions.
    # "roi_fa": FA averaged within each region (node feature, not an edge).
    fa_measure: str = "fa_weighted_sc"
    # Minimum FA a voxel must have to count. Voxels below this are treated as
    # non-white-matter and excluded from regional means / streamline sampling.
    # 0.20 is the conventional white-matter cutoff; 0.0 disables thresholding.
    fa_threshold: float = 0.20

    # --- Coupling ------------------------------------------------------
    # How structural and functional matrices are compared per subject.
    # "global_corr": one Spearman corr between the two matrices' edges.
    # "regional_corr": per-node corr of its structural vs functional profile.
    coupling_metric: str = "regional_corr"
    coupling_corr: str = "spearman"  # spearman | pearson

    # --- Optional extra structural signals (niche extensions) ----------
    use_t1_intensity: bool = False  # add T1 intensity as a structural feature
    use_wm_fmri: bool = False  # functional signal in white matter voxels

    # --- Paths (overridable) -------------------------------------------
    reference_path: str = "outputs/hcp_reference.npz"


DEFAULT = Config()
