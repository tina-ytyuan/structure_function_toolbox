"""Check that a toolbox-computed measure matches a teammate's saved output map.

Given a subject's BOLD and a reference measure map produced by the team's own
pipeline (e.g. one of the per-subject outputs on the DCC), this recomputes the
measure with the toolbox and compares the two voxelwise: max/mean absolute
difference and Pearson correlation inside the mask. It optionally writes a
side-by-side PNG (toolbox | reference | difference) so you can confirm the plots
look identical.

Usage
-----
    python scripts/compare_to_reference_map.py --measure reho \
        --bold  /path/to/project/data/Rest1LR/100206_finproc.nii.gz \
        --reference /path/to/teammate/100206_reho.nii.gz \
        --mask  /path/to/project/data/final_mask.nii \
        --tr 0.72 --figure /tmp/reho_100206_check.png

A max abs difference near 0 (and correlation ~1.0) means the toolbox reproduces
the team's result. Larger differences flag a mismatch to investigate.
"""

from __future__ import annotations

import argparse

import numpy as np

from sftoolbox import io, measures


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--measure", required=True, choices=sorted(measures.MEASURES))
    ap.add_argument("--bold", required=True, help="Subject 4D BOLD NIfTI")
    ap.add_argument("--reference", required=True,
                    help="Teammate's saved measure map (3D NIfTI) for this subject")
    ap.add_argument("--mask", help="3D NIfTI mask; else the toolbox default")
    ap.add_argument("--tr", type=float, default=0.72)
    ap.add_argument("--low", type=float, default=0.01)
    ap.add_argument("--high", type=float, default=0.08)
    ap.add_argument("--cluster", type=int, default=27)
    ap.add_argument("--figure", help="Optional output PNG (toolbox | ref | diff)")
    args = ap.parse_args()

    bold = io.load_nifti_data(args.bold)
    if bold.ndim != 4:
        raise SystemExit(f"BOLD must be 4D; got {bold.shape}")
    ref = np.asarray(io.load_nifti_data(args.reference), dtype=float)

    mask = None
    if args.mask:
        mask = np.asarray(io.load_nifti_data(args.mask)) != 0

    params = {"tr": args.tr, "low": args.low, "high": args.high,
              "cluster": args.cluster, "mask": mask}
    ours, our_mask = measures.compute(args.measure, bold, params)

    if ours.shape != ref.shape:
        raise SystemExit(
            f"Shape mismatch: toolbox {ours.shape} vs reference {ref.shape}")

    # Compare inside the mask where both are finite.
    m = our_mask & np.isfinite(ours) & np.isfinite(ref)
    a, b = ours[m], ref[m]
    diff = a - b
    max_abs = float(np.max(np.abs(diff))) if a.size else float("nan")
    mean_abs = float(np.mean(np.abs(diff))) if a.size else float("nan")
    denom = a.std() * b.std()
    corr = float(np.corrcoef(a, b)[0, 1]) if a.size and denom > 0 else float("nan")

    print(f"measure        : {args.measure}")
    print(f"compared voxels: {int(m.sum()):,}")
    print(f"max |diff|     : {max_abs:.3e}")
    print(f"mean |diff|    : {mean_abs:.3e}")
    print(f"correlation    : {corr:.6f}")
    verdict = "MATCH" if (max_abs < 1e-4 or corr > 0.9999) else "DIFFERS — investigate"
    print(f"verdict        : {verdict}")

    if args.figure:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        z = int(np.argmax(our_mask.reshape(-1, our_mask.shape[2]).sum(0)))
        fig, axes = plt.subplots(1, 3, figsize=(11, 4))
        for ax, img, title, cmap in (
            (axes[0], ours, "toolbox", "magma"),
            (axes[1], ref, "reference", "magma"),
            (axes[2], ours - ref, "difference", "RdBu_r"),
        ):
            sl = np.rot90(img[:, :, z])
            vlim = np.nanpercentile(np.abs(sl), 99) or 1.0
            kw = dict(vmin=-vlim, vmax=vlim) if title == "difference" else {}
            im = ax.imshow(sl, cmap=cmap, **kw)
            ax.set_title(title)
            ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.suptitle(f"{args.measure}  (axial z={z})   max|diff|={max_abs:.2e}")
        fig.tight_layout()
        fig.savefig(args.figure, dpi=140, bbox_inches="tight")
        print(f"figure         : {args.figure}")


if __name__ == "__main__":
    main()
