# structure_function_toolbox

Compare **structural** (FA-based) and **functional** (resting-state fMRI)
connectivity in the same brain, and benchmark any subject against an **HCP
normative cohort**. Our niche: using **fractional anisotropy (FA)** as the
structural signal (with T1 intensity and white-matter fMRI as planned
extensions), rather than raw streamline counts.

The toolbox does **not** preprocess. It consumes already-preprocessed data
that conforms to [`docs/input_spec.md`](docs/input_spec.md), computes measures,
and compares.

## How it works

Every subject — HCP or a user's own — runs through the *same* extraction path:

1. **Functional connectivity** from cleaned rs-fMRI — `sftoolbox/fmri.py`
2. **FA-weighted structural connectivity** from diffusion — `sftoolbox/fa.py`
3. **Structure-function coupling** metric — `sftoolbox/coupling.py`

Then:

4. **Normative reference**: run all HCP subjects, store the coupling
   distribution (`sftoolbox/reference.py`). This saved reference is the
   redistributable "standard dataset".
5. **Compare**: a new user subject → same path → z-score / percentile vs. the
   HCP distribution (`sftoolbox/compare.py`).

## Status

Implemented and tested: coupling metrics, normative reference, comparison.
Stubbed pending decisions: FA/tractography and NIfTI loading (depend on the
chosen atlas + diffusion pipeline — see `[decide]` items in the input spec).

## Two decisions to lock first

- **Atlas / parcellation** — fixes matrix size and node correspondence.
- **Coupling metric** — `global_corr` vs. `regional_corr`; defines what
  "structure and function aligning" means.

Both live in `sftoolbox/config.py`.

## Quick start

```bash
pip install -e .            # or: pip install -r requirements.txt
pytest                      # runs the coupling + comparison tests
```

## Layout

```
src/sftoolbox/   package (config, io, fmri, fa, coupling, reference, compare, pipeline, cli)
scripts/         build_reference.py, compare_subject.py
docs/            input_spec.md  <- the input contract
tests/           synthetic-data tests for the settled logic
```
