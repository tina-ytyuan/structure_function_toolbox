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

Implemented and verified: coupling metrics, normative reference, comparison,
NIfTI/atlas loading + conformance checks, and the full functional (fMRI) path.
FA computation (`fa.compute_fa_map`, dipy tensor fit) and FA-weighted structural
connectivity (`fa.fa_weighted_connectivity`, mean FA along streamlines per
region pair) are implemented against dipy's API; run them locally where dipy is
installed. Still to confirm: the `[decide]` items in the input spec (atlas,
exact cleaning level, structural edge definition).

## Local web app

A small local server with an upload-and-compare interface (opens in your
browser; nothing leaves your machine):

```bash
pip install -e .                 # includes flask, matplotlib, dipy
python -m sftoolbox.webapp       # opens http://127.0.0.1:5000
```

Upload a subject's atlas + BOLD (and optionally an FA map + tractogram) to
generate connectivity, coupling, and the cohort comparison. Click **Run demo**
to see the whole flow on synthetic data first.

The compare panel uses a saved HCP reference at `outputs/hcp_reference.npz`
(auto-loaded if present). Build it once:

```bash
# real HCP data (one subfolder per subject):
python scripts/build_reference.py --subjects-root /path/to/HCP \
       --atlas /path/to/atlas_labels.nii.gz --out outputs/hcp_reference.npz
# or a placeholder before you have data:
python scripts/build_reference.py --synthetic --out outputs/hcp_reference.npz
```

## Cohort comparison (per measure)

Build a normative reference for a measure from a folder of subjects; the app
auto-loads `outputs/measure_ref_<measure>.npz` and then shows, for any subject,
its summary percentile and a voxelwise z-map (subject vs cohort mean/SD):

```bash
python scripts/build_measure_reference.py \
    --subjects-root ~/sft_test_subjects --measure rsfa
# ALFF/fALFF also take --tr --low --high; ReHo takes --cluster
```

Voxelwise z-maps require subjects on a common voxel grid (same space); the
summary percentile works regardless.

## Test data

No HCP download needed to try the app — `nilearn` fetches real resting-state
BOLD for you:

```bash
python examples/fetch_test_data.py     # ADHD resting-state, 1 subject
# prints a path to a 4D NIfTI; upload it in the app's "Cleaned BOLD" field
```

## Try it end-to-end

```bash
python check.py                        # all logic, no data, no network
python examples/run_subject.py --synthetic   # functional path on synthetic NIfTIs
# real subject (needs dipy + a tractogram + an atlas label volume):
python examples/run_subject.py --subject-dir /path/to/HCP/100307 \
       --id 100307 --atlas /path/to/atlas_labels.nii.gz
```

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
