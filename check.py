"""
Standalone sanity check — no pytest required.

Run with the venv Python (has numpy/scipy):

    python check.py

Exercises the fully-implemented logic (coupling + normative compare) on
synthetic data and prints PASS/FAIL for each check.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np

from sftoolbox import compare, coupling, fa, fmri, io, reference
from sftoolbox.config import Config

_failures = []


def check(name, cond):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}")
    if not cond:
        _failures.append(name)


def skip(name, why):
    print(f"[SKIP] {name} ({why})")


def sym(mat):
    mat = (mat + mat.T) / 2
    np.fill_diagonal(mat, 0.0)
    return mat


def main():
    rng = np.random.default_rng(0)

    # 1. Global coupling: identical matrices -> ~perfect correlation.
    struct = sym(rng.random((20, 20)))
    cfg_g = Config(coupling_metric="global_corr", coupling_corr="spearman")
    check(
        "global coupling of identical matrices ~ 1",
        coupling.coupling(struct, struct.copy(), cfg_g) > 0.99,
    )

    # 2. Regional coupling returns one value per node.
    func = sym(rng.random((15, 15)))
    struct15 = sym(rng.random((15, 15)))
    cfg_r = Config(coupling_metric="regional_corr")
    check(
        "regional coupling shape == (15,)",
        coupling.coupling(struct15, func, cfg_r).shape == (15,),
    )

    # 3. Shape mismatch raises.
    try:
        coupling.coupling(np.zeros((10, 10)), np.zeros((8, 8)), Config())
        check("shape mismatch raises ValueError", False)
    except ValueError:
        check("shape mismatch raises ValueError", True)

    # 4. Normative reference + compare roundtrip.
    with tempfile.TemporaryDirectory() as d:
        cfg = Config(
            coupling_metric="global_corr", reference_path=str(Path(d) / "ref.npz")
        )
        vals = list(rng.normal(0.5, 0.1, size=100))
        ids = [f"HCP{i:03d}" for i in range(100)]
        ref = reference.build_reference(vals, ids, cfg)

        at_mean = compare.compare_subject(ref["mean"], ref)
        check(
            "subject at cohort mean -> percentile ~ 50",
            abs(float(at_mean["percentile"]) - 50.0) < 5.0,
        )

        high = compare.compare_subject(0.8, ref)
        check(
            "clearly high subject -> percentile > 95", float(high["percentile"]) > 95.0
        )

        emp = compare.empirical_percentile(
            0.5,
            {
                "values": np.linspace(0, 1, 101),
            },
        )
        check("empirical percentile of median ~ 50", abs(float(emp) - 50.0) < 2.0)

    # 7. roi_fa: mean FA per parcel (pure numpy).
    fa_map = np.zeros((4, 4, 4))
    labels = np.zeros((4, 4, 4), dtype=int)
    labels[0] = 1  # region 1
    labels[1] = 2  # region 2
    fa_map[0] = 0.3
    fa_map[1] = 0.7
    out = fa.roi_fa(fa_map, labels)
    check(
        "roi_fa returns per-region means",
        out.shape == (2,) and abs(out[0] - 0.3) < 1e-9 and abs(out[1] - 0.7) < 1e-9,
    )

    # 7b. roi_fa with an FA threshold: sub-threshold voxels are excluded.
    #     Region 1 (all FA=0.3) drops below thr=0.5 -> NaN; region 2 (0.7) stays.
    thr_out = fa.roi_fa(fa_map, labels, fa_threshold=0.5)
    check(
        "roi_fa fa_threshold excludes sub-threshold voxels",
        np.isnan(thr_out[0]) and abs(thr_out[1] - 0.7) < 1e-9,
    )

    # 7c. apply_threshold zeros sub-threshold voxels and is a no-op at 0.
    at = fa.apply_threshold(fa_map, 0.5)
    check(
        "apply_threshold zeros below cutoff, keeps above",
        (at[0] == 0).all()
        and abs(at[1].mean() - 0.7) < 1e-9
        and (fa.apply_threshold(fa_map, 0.0) == fa_map).all(),
    )

    # 8. Functional pipeline end-to-end on synthetic NIfTIs (needs nibabel).
    try:
        import nibabel as nib
    except ImportError:
        skip("functional pipeline end-to-end", "nibabel not installed")
    else:
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            dim, nvol, nreg = 8, 100, 5
            lab = np.zeros((dim, dim, dim), dtype=np.int16).reshape(-1)
            lab[:] = (np.arange(lab.size) % nreg) + 1
            lab = lab.reshape(dim, dim, dim)
            sig = rng.standard_normal((nreg, nvol))
            bold = np.zeros((dim, dim, dim, nvol))
            for r in range(1, nreg + 1):
                bold[lab == r] = sig[r - 1]
            nib.save(nib.Nifti1Image(bold, np.eye(4)), str(tmp / "b.nii.gz"))
            nib.save(nib.Nifti1Image(lab, np.eye(4)), str(tmp / "l.nii.gz"))

            labels3d = io.load_nifti_data(tmp / "l.nii.gz")
            bold4d = io.load_nifti_data(tmp / "b.nii.gz")
            ts = fmri.region_timeseries(bold4d, labels3d)
            fc = fmri.functional_connectivity(ts, Config())
            check(
                "functional pipeline: FC is (nreg, nreg)",
                ts.shape == (nreg, nvol) and fc.shape == (nreg, nreg),
            )

    # 9. FA/dipy paths (needs dipy).
    try:
        import dipy  # noqa: F401

        check("dipy available for FA computation", True)
    except ImportError:
        skip("compute_fa_map / fa_weighted_connectivity", "dipy not installed")

    print()
    if _failures:
        print(f"{len(_failures)} check(s) FAILED: {_failures}")
        sys.exit(1)
    print("All checks passed.")


if __name__ == "__main__":
    main()
