"""Structure-Function Coupling Toolbox.

Pipeline overview
-----------------
1. Extract measures (identical for HCP and user subjects):
     - functional connectivity  (resting-state fMRI)  -> fmri.py
     - FA-weighted structural connectivity (diffusion) -> fa.py
     - structure-function coupling metric              -> coupling.py
2. Build a normative reference distribution from HCP  -> reference.py
3. Compare a new subject against that distribution     -> compare.py

Every subject (HCP or user-supplied) goes through the SAME extraction path,
so results are only comparable when inputs conform to docs/input_spec.md.
"""

__version__ = "0.0.1"
