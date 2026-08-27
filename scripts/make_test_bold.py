"""Generate a synthetic 4D BOLD volume for testing the toolbox end to end.

Real HCP BOLD runs are ~9 GB and must stay on the cluster, which makes local
testing of the full pipeline awkward. This writes a synthetic run on the same
grid as the normative references (91x109x91, MNI 2 mm), so measures compute and
the single-subject comparison actually engages.

The signal is built to exercise the measures rather than be neurally realistic:

  * spatially smooth mixing weights, so neighbouring voxels share signal and
    ReHo / coherence-ReHo are not degenerate,
  * low-frequency components in the 0.01-0.08 Hz band that ALFF and fALFF
    respond to,
  * a per-run global scale factor, mimicking the arbitrary BOLD units that
    made global normalisation necessary in the first place.

The voxel geometry comes from a reference's own ``group_mask``, so the output
lines up with whatever cohort you built.

Usage
-----
    python scripts/make_test_bold.py --ref outputs/measure_ref_rsfa.npz \
        --out outputs/test_subject_bold.nii.gz --timepoints 120
"""

from __future__ import annotations

import argparse

import numpy as np

# Standard MNI152 2 mm affine for the 91x109x91 grid the references use.
MNI2MM_AFFINE = np.array([
    [-2.0, 0.0, 0.0, 90.0],
    [0.0, 2.0, 0.0, -126.0],
    [0.0, 0.0, 2.0, -72.0],
    [0.0, 0.0, 0.0, 1.0],
])


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--ref", help="Reference .npz whose group_mask defines the brain")
    ap.add_argument("--out", required=True)
    ap.add_argument("--timepoints", type=int, default=120)
    ap.add_argument("--tr", type=float, default=0.72)
    ap.add_argument("--components", type=int, default=24,
                    help="Number of shared low-frequency sources")
    ap.add_argument("--global-scale", type=float, default=1.0,
                    help="Multiplies the whole run, mimicking scanner gain. Try "
                         "1.3 to confirm normalisation removes it.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import nibabel as nib
    from scipy import ndimage

    rng = np.random.default_rng(args.seed)
    t_n, tr = args.timepoints, args.tr

    if args.ref:
        from sftoolbox import measure_norm
        ref = measure_norm.load(args.ref)
        mask = np.asarray(ref["group_mask"], bool)
        print(f"mask from {args.ref}: {int(mask.sum()):,} voxels {mask.shape}")
    else:
        # Fall back to a centred ellipsoid on the reference grid.
        shape = (91, 109, 91)
        zz, yy, xx = np.meshgrid(*[np.linspace(-1, 1, s) for s in shape],
                                 indexing="ij")
        mask = (xx**2 / 0.75 + yy**2 / 0.95 + zz**2 / 0.8) < 1.0
        print(f"synthetic ellipsoid mask: {int(mask.sum()):,} voxels")

    shape = mask.shape
    t = np.arange(t_n) * tr

    # Shared sources in the resting-state band. Every voxel is a smoothly
    # varying mixture of these, which is what gives neighbouring voxels the
    # correlation that ReHo measures.
    freqs = rng.uniform(0.01, 0.08, args.components)
    phases = rng.uniform(0, 2 * np.pi, args.components)
    sources = np.sin(2 * np.pi * freqs[:, None] * t[None, :] + phases[:, None])
    sources += 0.3 * rng.normal(size=sources.shape)

    print(f"building {shape} x {t_n} volumes "
          f"({args.components} sources, TR={tr}s)")
    bold = np.zeros(shape + (t_n,), dtype=np.float32)
    idx = np.flatnonzero(mask.ravel())
    for k in range(args.components):
        # Smooth random field -> spatially coherent weights for this source.
        w = ndimage.gaussian_filter(rng.normal(size=shape), sigma=4.0)
        w /= np.abs(w).max() or 1.0
        wv = w.ravel()[idx].astype(np.float32)
        bold.reshape(-1, t_n)[idx] += np.outer(wv, sources[k]).astype(np.float32)

    flat = bold.reshape(-1, t_n)
    # Thermal noise, then a BOLD-like baseline. Raw BOLD units are arbitrary,
    # which is exactly why references need global normalisation.
    flat[idx] += 0.35 * rng.normal(size=(idx.size, t_n)).astype(np.float32)
    flat[idx] *= 120.0
    flat[idx] += 10000.0
    flat[idx] *= args.global_scale

    img = nib.Nifti1Image(bold, MNI2MM_AFFINE)
    img.header.set_zooms((2.0, 2.0, 2.0, tr))
    nib.save(img, args.out)

    ts = flat[idx[len(idx) // 2]]
    print(f"\nwrote {args.out}")
    print(f"  shape        : {bold.shape}")
    print(f"  in-mask voxels: {idx.size:,}")
    print(f"  example voxel : mean {ts.mean():.1f}, SD {ts.std():.2f}")
    print(f"  global scale  : {args.global_scale}")


if __name__ == "__main__":
    main()
