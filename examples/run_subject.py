"""Run one subject end-to-end: FC + FA structural connectivity + coupling.

Two modes:

  # Real subject (files resolved from the directory by io.Subject.from_dir):
  python examples/run_subject.py --subject-dir /path/to/HCP/100307 --id 100307 \
         --atlas /path/to/atlas_labels.nii.gz

  # Synthetic self-test (no data, no dipy needed) — proves the functional
  # path and the wiring run end-to-end:
  python examples/run_subject.py --synthetic
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np

from sftoolbox import coupling, fmri, io, pipeline, viz
from sftoolbox.config import DEFAULT, Config


def _make_synthetic(tmp: Path, n_regions=10, n_vols=120, dim=12):
    """Write a tiny synthetic BOLD + integer atlas as NIfTIs; return paths."""
    import nibabel as nib

    rng = np.random.default_rng(0)
    # Atlas: assign each voxel to one of n_regions blocks (0 kept as background rim).
    labels = np.zeros((dim, dim, dim), dtype=np.int16)
    flat = labels.reshape(-1)
    inner = np.arange(flat.size)
    flat[inner] = (inner % n_regions) + 1
    labels = flat.reshape(dim, dim, dim)

    # BOLD: each region driven by its own signal so FC is well-defined.
    region_sig = rng.standard_normal((n_regions, n_vols))
    bold = np.zeros((dim, dim, dim, n_vols))
    for rid in range(1, n_regions + 1):
        m = labels == rid
        bold[m] = region_sig[rid - 1] + 0.1 * rng.standard_normal((m.sum(), n_vols))

    affine = np.eye(4)
    bold_p = tmp / "sub_bold.nii.gz"
    lab_p = tmp / "atlas_labels.nii.gz"
    nib.save(nib.Nifti1Image(bold, affine), str(bold_p))
    nib.save(nib.Nifti1Image(labels, affine), str(lab_p))
    return bold_p, lab_p, n_regions


def run_synthetic():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        bold_p, lab_p, n = _make_synthetic(tmp)
        cfg = Config(coupling_metric="global_corr")

        labels = io.load_nifti_data(lab_p)
        bold = io.load_nifti_data(bold_p)

        ts = fmri.region_timeseries(bold, labels)
        fc = fmri.functional_connectivity(ts, cfg)
        print(f"region timeseries: {ts.shape}  (expected ({n}, 120))")
        print(f"functional connectivity: {fc.shape}")

        # Stand in a synthetic structural matrix to exercise coupling wiring
        # (real FA-SC needs a tractogram + dipy; see run on real data).
        rng = np.random.default_rng(1)
        struct = rng.random((n, n))
        struct = (struct + struct.T) / 2
        np.fill_diagonal(struct, 0.0)
        val = coupling.coupling(struct, fc, cfg)
        print(f"global coupling (synthetic struct vs real-ish FC): {val:.3f}")

        # Save the local "UI": figures for FC/FA and subject-vs-cohort.
        out = Path("outputs")
        viz.plot_fc_vs_fa(fc, struct, out_path=out / "synthetic_fc_vs_fa.png")
        cohort = rng.normal(0.1, 0.05, size=200)
        ref = {"values": cohort}
        viz.plot_subject_vs_reference(
            val, ref, out_path=out / "synthetic_subject_vs_hcp.png"
        )
        print(
            f"Figures written to {out}/  (synthetic_fc_vs_fa.png, "
            "synthetic_subject_vs_hcp.png)"
        )
        print("Synthetic end-to-end run OK.")


def run_real(args):
    cfg = DEFAULT
    subj = io.Subject.from_dir(args.id, args.subject_dir)
    labels = io.load_nifti_data(args.atlas)
    n_regions = int(np.unique(np.rint(labels).astype(int))[1:].size)
    value = pipeline.extract_subject(subj, labels, cfg, expected_regions=n_regions)
    print(f"[{args.id}] coupling ({cfg.coupling_metric}): {value}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--subject-dir")
    ap.add_argument("--id", default="subject")
    ap.add_argument("--atlas", help="integer label NIfTI matching cfg.atlas")
    args = ap.parse_args()

    if args.synthetic:
        run_synthetic()
    else:
        if not (args.subject_dir and args.atlas):
            ap.error("real mode needs --subject-dir and --atlas (or use --synthetic)")
        run_real(args)


if __name__ == "__main__":
    main()
