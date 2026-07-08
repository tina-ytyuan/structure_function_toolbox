# Input specification (the contract)

The toolbox does **no preprocessing**. Every subject — HCP or user-supplied —
must arrive already preprocessed and conforming to this spec, or the
comparison against the HCP normative cohort is not valid. This file is the
single most important thing to freeze early; everything else depends on it.

> Status: DRAFT. Items marked **[decide]** need confirmation with Dr. Michael.

## Standard space
- All volumes in the same standard space: **[decide]** (default `MNI152NLin6Asym`).
- A subject in a different space is rejected by `io.validate_conformance`.

## Parcellation / atlas
- One atlas defines the network nodes for both structural and functional
  matrices, so they share dimensions and node correspondence. **[decide]**
  (candidates: Schaefer-200, Desikan-Killiany, HCP-MMP1).
- Provided as an integer label volume aligned to the standard space.

## Resting-state fMRI (functional)
- Cleaned 4D BOLD (motion/nuisance regressed, in standard space).
  HCP: use the ICA-FIX cleaned rfMRI. **[decide]** exact cleaning level users
  must match.
- Record TR and number of volumes; flag subjects far from the HCP range.

## Diffusion (structural / FA)
- Preprocessed diffusion (eddy/motion corrected, brain-extracted) **or** a
  precomputed FA map in standard space.
- We ship `fa.compute_fa_map` so users can generate FA from their preprocessed
  diffusion data with the same tensor-fit settings we use.
- Structural edges = **[decide]** FA-weighted streamline connectivity
  (`fa_weighted_sc`) vs. regional mean FA (`roi_fa`).

## Optional extensions (niche)
- T1 intensity as an additional structural feature (`config.use_t1_intensity`).
- White-matter fMRI signal vs. FA (`config.use_wm_fmri`) — the underexplored
  angle from the project brief.

## Batch / site caveat
A single outside subject acquired on a different scanner can look "abnormal"
purely from acquisition differences, not biology. Report this limitation with
any single-subject comparison; consider harmonization if it becomes a problem.
