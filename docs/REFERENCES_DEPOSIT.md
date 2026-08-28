# Normative reference deposit: draft metadata

Draft text for depositing the reference files (Zenodo, OSF, or a GitHub
release). Confirm authorship and the HCP acknowledgement wording before
submitting.

---

## Title

Normative voxelwise references for eleven resting-state fMRI measures, derived
from the Human Connectome Project Young Adult cohort (n = 936)

## Authors

*To be decided.*

## Description

Eleven normative reference files supporting single-subject comparison of
resting-state fMRI measures. Each file records, at every voxel, the mean and
standard deviation of one measure across a cohort of 936 Human Connectome
Project Young Adult subjects, together with the analysis mask and the cohort
size.

Only aggregate quantities are included. The files carry no per-subject values
and no list of which subjects contributed, so no individual is identifiable and
no inference can be drawn about who was or was not in the cohort.

These references let a single individual's measure map be compared against a
normative distribution without the user assembling a control cohort of their
own, the usual barrier to normative comparison in existing tools. They are the
data component of the Structure-Function Toolbox, which computes the measures
and performs the comparison.

**Measures.** ALFF; fALFF; ALFF and fALFF restricted to the slow-4
(0.027-0.073 Hz) and slow-5 (0.01-0.027 Hz) bands; RSFA; ReHo (Kendall's *W*);
coherence-based ReHo; intrinsic neural timescale; and multiscale entropy
complexity index over scales 1-5.

**Construction.** Per-subject measure maps were computed from minimally preprocessed HCP resting-state runs (TR = 0.72 s) and resampled to MNI 2 mm space (91 × 109 × 91). Each map was normalized by its own mean within the analysis mask, following the DPABI m convention before aggregation. All four resting-state runs (Rest1LR, Rest1RL, Rest2LR, and Rest2RL) were treated as separate observations, resulting in 3,744 maps. Therefore, the standard deviation reflects single-run variability, consistent with the single run provided by a user. The stored cohort size is 936, corresponding to the number of independent individuals. Repeated runs were not treated as independent subjects when determining the degrees of freedom for the single-subject t-test. Standard deviations were calculated using the N-1 denominator. The analysis mask excludes the ventricles and contains 226,304 voxels.

**Intended use.** With the accompanying toolbox, comparison uses the Crawford &
Howell (1998) single-case t-test,

t = (x − mean) / (SD × sqrt((n + 1) / n)),   df = n − 1
which treats the cohort as a finite sample rather than a known population.
Voxelwise p-values should be corrected for multiple comparisons;
Benjamini-Hochberg FDR is implemented in the toolbox.

**Calibration.** Validated against held-out subjects. A well-calibrated
reference gives a subject an SD of *t* across voxels near 1.0 and a
ratio of p<0.05 voxels to chance near 1.0. Ten of the eleven measures fall
between 0.80 and 0.99. ReHo is conservative (0.32), consistent
with Kendall's *W* being bounded and left-skewed while the t-test assumes a
normal control distribution; it under-declares significance rather than
over-declaring it. Per-measure figures are in the toolbox README.

**Limitations.** 

## Keywords

resting-state fMRI, normative modelling, single-subject inference, Human
Connectome Project, ALFF, ReHo, multiscale entropy, intrinsic neural timescale,
structure-function coupling

## Files

`measure_ref_<measure>.npz`, eleven files, ~162 MB total. NumPy `.npz`
containing:

| field | contents |
|---|---|
| `mean_map` | voxelwise cohort mean, 91 × 109 × 91 |
| `sd_map` | voxelwise cohort SD, N−1 denominator |
| `group_mask` | analysis mask, 226,304 voxels |
| `shape` | voxel grid dimensions |
| `n` | 936, the number of contributing individuals |
| `normalized` | True; maps were globally normalised before aggregation |
| `mask_name` | name of the source mask |
| `mask_voxels` | 226,304 |
| `runs_per_subject` | 4 |

Load with `numpy.load(path, allow_pickle=True)` or
`sftoolbox.measure_norm.load`.

These files contain aggregate statistics only; rebuilding an equivalent
reference requires your own HCP cohort.

## Licence

*To be confirmed.* CC BY 4.0 is the usual choice for derived neuroimaging data
and is compatible with the toolbox's MIT licence.

## Acknowledgement

Data were provided in part by the Human Connectome Project, WU-Minn Consortium
(Principal Investigators: David Van Essen and Kamil Ugurbil; 1U54MH091657)
funded by the 16 NIH Institutes and Centers that support the NIH Blueprint for
Neuroscience Research; and by the McDonnell Center for Systems Neuroscience at
Washington University.

Only aggregate, group-level products are distributed here. No per-subject
values and no cohort membership list are included, so no individual is
identifiable from these files. Users remain bound by the HCP Open Access Data
Use Terms.

## References

Crawford, J. R., & Howell, D. C. (1998). Comparing an individual's test score
against norms derived from small samples. *The Clinical Neuropsychologist*,
12(4), 482–486.

Zang, Y.-F., et al. (2007). Altered baseline brain activity in children with
ADHD revealed by resting-state functional MRI. *Brain and Development*, 29(2),
83–91.

Zuo, X.-N., et al. (2010). The oscillating brain: complex and reliable.
*NeuroImage*, 49(2), 1432–1445.

Van Essen, D. C., et al. (2013). The WU-Minn Human Connectome Project: an
overview. *NeuroImage*, 80, 62–79.
