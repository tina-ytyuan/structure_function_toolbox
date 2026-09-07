# Getting started

A complete walkthrough, from an empty machine to your first result. No prior
experience with this toolbox is assumed.

## What this toolbox does

You give it one subject's cleaned resting-state fMRI scan. It computes measures
of brain activity at every voxel, then tells you how that subject compares with
936 healthy adults from the Human Connectome Project.

The comparison is the part most tools cannot do on their own. Normally, judging
whether one person's brain measure is unusual means collecting a control group
yourself. Here the control group is already built and ships with the toolbox.

## What you need before starting

**A computer running macOS, Linux, or Windows.** Everything runs locally. No
data is uploaded anywhere.

**About 1 GB of free disk space.** The reference files are 162 MB, and the
software and its dependencies take up a few hundred more.

**Your fMRI data, already preprocessed.** This is the one requirement the
toolbox cannot help with. See [Preparing your data](#preparing-your-data) below.

## Step 1: install Python

Check whether you already have it. Open a terminal and run:

```bash
python3 --version
```

If you see `Python 3.9` or higher, you are set. If the command is not found, or
the version is older, install Python from [python.org/downloads](https://www.python.org/downloads/).
On macOS you can also use Homebrew: `brew install python`.

## Step 2: download the toolbox

```bash
git clone https://github.com/tina-ytyuan/structure_function_toolbox.git
cd structure_function_toolbox
```

If you do not have `git`, download the ZIP from the repository page using the
green **Code** button, unzip it, and `cd` into the folder.

## Step 3: create an isolated environment

This keeps the toolbox's dependencies separate from anything else on your
machine, so nothing you already have can break.

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows the second line is `.venv\Scripts\activate` instead.

Your prompt should now start with `(.venv)`. You need to run that `activate`
line each time you open a new terminal to work with the toolbox.

## Step 4: install

```bash
pip install -e .
```

This pulls in numpy, scipy, nibabel, nilearn, matplotlib, flask, and a few
others. It takes a couple of minutes.

## Step 5: download the reference files

The reference files hold the HCP cohort statistics. They are too large for the
repository, so they are published separately as a release.

```bash
python scripts/fetch_references.py
```

This downloads eleven files, one per measure, into the `outputs/` folder and
checks each against a published checksum. Expect it to take a few minutes on a
normal connection.

To fetch only what you need:

```bash
python scripts/fetch_references.py rsfa alff reho
```

If you would rather download by hand, go to the
[releases page](https://github.com/tina-ytyuan/structure_function_toolbox/releases),
download every `measure_ref_*.npz` file, and put them in the `outputs/` folder.

Confirm they arrived:

```bash
ls outputs/measure_ref_*.npz
```

You should see eleven files.

## Step 6: start the toolbox

```bash
python -m sftoolbox.webapp
```

Then open **http://127.0.0.1:5000** in your web browser.

To stop it later, press `Ctrl+C` in the terminal.

## Step 7: try the demo first

Before using your own data, click any **Run demo** button. This computes a
measure on synthetic data and shows you the layout of a results page. Nothing
is downloaded or uploaded; it confirms the installation works.

## Step 8: analyse your own subject

On the main page:

1. **Cleaned BOLD.** Choose your subject's preprocessed 4D NIfTI file.
2. **Mask (optional but recommended).** If you have the cohort mask, select it
   here. You can download it from any results page afterwards.
3. **Measures.** Tick the measures you want. Start with one; several at once
   takes longer.
4. **Parameters.** TR defaults to 0.72 seconds, the HCP value. Change it to
   match your scan.
5. Click **Run**.

A large scan takes a minute or two per measure. The page shows a progress
message while it works.

## Preparing your data

The toolbox does not preprocess. It expects a 4D NIfTI file that has already
been through a standard pipeline, meaning:

- motion and nuisance signals regressed out,
- registered to MNI standard space,
- one file per resting-state run.

If your data is raw scanner output, run it through
[fMRIPrep](https://fmriprep.org/) or DPARSF first. That is a substantial
process in its own right and is outside what this toolbox covers.

Your data does **not** need to be at a specific resolution. If it is in MNI
space at a different voxel size, the toolbox resamples it onto the reference
grid automatically and tells you it has done so.

## Reading your results

**Measure map.** Slices through the brain showing the measure's value at each
voxel.

**Value distribution.** A histogram of those values across the brain.

**Comparison to group.** This is the normative part. Key numbers:

| What you see | What it means |
|---|---|
| `mean t` | Average difference from the cohort. Near 0 is typical. |
| `voxels tested` | How many brain locations were compared. |
| `p<0.05 uncorrected` | Voxels that look different before correcting for the number of tests. |
| `FDR q<0.05` | Voxels that survive multiple comparison correction. **Report this one.** |
| `SD of t (exp. 1.0)` | A calibration check. Should be near 1.0. |
| `observed / chance (exp. 1.0)` | Another calibration check. Should be near 1.0. |

Those last two are diagnostics. If either is far from 1.0, the toolbox shows a
warning, because a result that finds far more or far fewer differences than
chance usually points to a problem rather than a discovery.

**Axial profile.** The subject plotted against the cohort by height in the
brain. The grey band is the normal range. Staying inside it is typical.

**Downloads.** Buttons at the top of each result save the measure map, the
statistical maps, and the cohort mask as NIfTI files you can open in any
neuroimaging viewer.

## If something goes wrong

**`command not found: python3`**
Python is not installed, or not on your PATH. Revisit Step 1.

**`No module named sftoolbox`**
The environment is not activated, or the install did not complete. Run
`source .venv/bin/activate`, then `pip install -e .` again.

**No comparison section appears in the results**
The reference files are missing. Run `python scripts/fetch_references.py` and
check that `outputs/` contains eleven `.npz` files.

**`CERTIFICATE_VERIFY_FAILED` while fetching references**
Your Python cannot verify HTTPS certificates. This is common on macOS with
Python installed from python.org, which ships its own certificate store rather
than using the system one. Fix it with `pip install certifi`, or run the
installer's own command once:

```bash
open "/Applications/Python 3.12/Install Certificates.command"
```

Adjust the version number to match your install.

**`HTTP 404` while fetching references**
The release could not be reached. Check the releases page in a browser. If the
repository is private, the files are not publicly downloadable.

**`BOLD must be 4D (X,Y,Z,T)`**
You selected a 3D file, such as a mask or an already-computed measure map. The
input must be a time series.

**The page hangs on a large file**
Some measures, especially multiscale entropy, are slow. Start with one measure
on one subject to gauge timing.

## Using it without the browser

Everything is scriptable. For example, building your own reference from a
folder of subjects:

```bash
python scripts/reference_from_subject_maps.py --measure rsfa \
    --maps '/path/to/maps/*/*_rsfa.nii.gz' \
    --mask /path/to/mask.nii \
    --out outputs/measure_ref_rsfa.npz
```

And checking that a reference is well calibrated:

```bash
python scripts/check_reference_calibration.py \
    --ref outputs/measure_ref_rsfa.npz \
    --maps '/path/to/maps/*/*_rsfa.nii.gz' \
    --mask /path/to/mask.nii --n 20
```

Every script accepts `--help`.

## What the references contain

Group averages only. There are no individual subject values and no list of who
was in the cohort, so no one is identifiable from them. Details are in
[REFERENCES_DEPOSIT.md](REFERENCES_DEPOSIT.md).

If you publish work using them, please include the HCP acknowledgement given on
the release page.
