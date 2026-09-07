# Release notes to paste into GitHub

Copy the block below into the release description on GitHub. It is written for
someone who has never seen this project before.

---

## Normative references for eleven resting-state fMRI measures

These files let you compare a single subject's brain measures against 936
healthy adults from the Human Connectome Project, without collecting a control
group of your own.

### What is in here

Eleven files, one per measure, about 15 MB each. Each holds the average and the
spread of one measure at every voxel across the cohort, plus the analysis mask.

Measures included: ALFF, fALFF, ALFF and fALFF in the slow-4 and slow-5 bands,
RSFA, ReHo, coherence based ReHo, intrinsic neural timescale, and multiscale
entropy.

`SHA256SUMS.txt` lets you verify your downloads.

### How to use them

Install the toolbox, then run one command:

```
python scripts/fetch_references.py
```

That downloads every file into the right place and checks it. Full setup
instructions, starting from an empty machine, are in
[docs/GETTING_STARTED.md](../blob/main/docs/GETTING_STARTED.md).

To install by hand instead, download every `measure_ref_*.npz` file below and
put them in the toolbox's `outputs/` folder.

### How they were built

936 HCP Young Adult subjects, all four resting-state runs, giving 3,744 maps.
Each map was scaled by its own average before combining, following the DPABI
convention, so that arbitrary differences in scanner signal level do not
dominate. Each run counts as a separate observation, because a user supplies a
single run and the spread should describe single runs. The stored cohort size
is 936, the number of people, so repeated runs from one person are not treated
as separate individuals.

Grid: MNI 2 mm, 91 x 109 x 91. Mask: ventricles excluded, 226,304 voxels.

### Checking they work

Each reference was tested against held out subjects. A good reference makes a
typical person look unremarkable, which shows up as two numbers landing near
1.0: the spread of t values across voxels, and the ratio of apparently
significant voxels to what chance alone would produce.

Ten of the eleven measures fall between 0.80 and 0.99 on that ratio. ReHo is
more conservative at 0.32, meaning it under reports differences rather than
over reporting them. This is expected: ReHo is a bounded, skewed quantity while
the underlying test assumes a symmetric distribution. Treat ReHo findings as
cautious.

### Limitations

The cohort is healthy adults roughly aged 22 to 35, scanned on one platform
with one protocol. Comparing subjects from other populations, age ranges, or
scanners may be biased. Check calibration on your own held out data before
relying on it.

### Privacy

Group averages only. No individual subject values and no list of cohort
members, so no one is identifiable from these files.

### Acknowledgement

Data were provided in part by the Human Connectome Project, WU-Minn Consortium
(Principal Investigators: David Van Essen and Kamil Ugurbil; 1U54MH091657)
funded by the 16 NIH Institutes and Centers that support the NIH Blueprint for
Neuroscience Research; and by the McDonnell Center for Systems Neuroscience at
Washington University.

Use of these files remains subject to the HCP Open Access Data Use Terms.
