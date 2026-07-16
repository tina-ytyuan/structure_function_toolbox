"""Local visualization (matplotlib).

The toolbox's current "UI": static figures you can open or drop into a report.
Everything here returns a matplotlib Figure and can optionally save to disk, so
the same functions feed a future browser view later.

matplotlib is imported lazily so the rest of the package doesn't require it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _range(mat, symmetric):
    """Safe color range that tolerates all-NaN / empty matrices."""
    finite = mat[np.isfinite(mat)]
    if finite.size == 0:
        return (-1.0, 1.0) if symmetric else (0.0, 1.0)
    if symmetric:
        vmax = float(np.max(np.abs(finite))) or 1.0
        return -vmax, vmax
    return float(finite.min()), float(finite.max()) or 1.0


def _save(fig, out_path):
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig


def plot_matrix(
    mat: np.ndarray,
    title: str = "",
    out_path=None,
    cmap: str = "RdBu_r",
    symmetric: bool = True,
):
    """Heatmap of a connectivity matrix (FC or FA structural)."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 4.2))
    vmin, vmax = _range(mat, symmetric)
    im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("region")
    ax.set_ylabel("region")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return _save(fig, out_path)


def plot_fc_vs_fa(fc: np.ndarray, fa_sc: np.ndarray, out_path=None):
    """Side-by-side FC and FA structural heatmaps for one subject."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, mat, title, cmap, sym in (
        (axes[0], fc, "Functional connectivity", "RdBu_r", True),
        (axes[1], fa_sc, "FA structural connectivity", "viridis", False),
    ):
        vmin, vmax = _range(mat, sym)
        im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(title)
        ax.set_xlabel("region")
        ax.set_ylabel("region")
        if not np.isfinite(mat).any():
            ax.text(
                0.5,
                0.5,
                "not computed",
                transform=ax.transAxes,
                ha="center",
                va="center",
                color="0.5",
                fontsize=11,
            )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return _save(fig, out_path)


def plot_brain_map(
    map_3d: np.ndarray,
    mask: np.ndarray | None = None,
    title: str = "",
    cmap: str = "magma",
    diverging: bool = False,
    out_path=None,
):
    """Representative axial slice + a histogram of in-mask values.

    Used for voxelwise measures (ReHo, ALFF, seed FC). Picks the axial slice
    with the most in-mask voxels so the preview is informative.
    """
    import matplotlib.pyplot as plt

    if mask is None:
        mask = np.isfinite(map_3d) & (map_3d != 0)
    # Choose the most-covered axial slice.
    per_slice = mask.reshape(-1, mask.shape[2]).sum(axis=0)
    z = int(np.argmax(per_slice)) if per_slice.any() else map_3d.shape[2] // 2

    sl = map_3d[:, :, z]
    vals = map_3d[mask]
    if diverging:
        vmax = float(np.max(np.abs(vals))) if vals.size else 1.0
        vmin = -vmax
    else:
        vmin = float(np.min(vals)) if vals.size else 0.0
        vmax = float(np.max(vals)) if vals.size else 1.0

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(9.5, 4))
    im = ax0.imshow(np.rot90(sl), cmap=cmap, vmin=vmin, vmax=vmax)
    ax0.set_title(f"{title}  (axial z={z})" if title else f"axial z={z}")
    ax0.axis("off")
    fig.colorbar(im, ax=ax0, fraction=0.046, pad=0.04)

    ax1.hist(vals, bins=40, color="#a9c8e0", edgecolor="#7fa9cc")
    ax1.set_title("value distribution (in mask)")
    ax1.set_xlabel("value")
    ax1.set_ylabel("voxels")
    fig.tight_layout()
    return _save(fig, out_path)


def plot_brain_montage(
    map_3d: np.ndarray,
    mask: np.ndarray | None = None,
    rows: int = 3,
    cols: int = 8,
    title: str = "",
    cmap: str = "magma",
    diverging: bool = False,
    out_path=None,
):
    """Montage of evenly-spaced axial slices + a value histogram.

    Shows ``rows`` x ``cols`` slices spanning the masked z-range (default 3x8 =
    24 slices), a shared colorbar, and a histogram of in-mask values below.
    """
    import matplotlib.pyplot as plt

    if mask is None:
        mask = np.isfinite(map_3d) & (map_3d != 0)
    n = rows * cols

    zsum = mask.reshape(-1, mask.shape[2]).sum(axis=0)
    zvalid = np.where(zsum > 0)[0]
    if zvalid.size:
        zslices = np.linspace(zvalid.min(), zvalid.max(), n).astype(int)
    else:
        zslices = np.linspace(0, map_3d.shape[2] - 1, n).astype(int)

    vals = map_3d[mask]
    if diverging:
        vmax = float(np.max(np.abs(vals))) if vals.size else 1.0
        vmin = -vmax
    else:
        vmin = float(np.min(vals)) if vals.size else 0.0
        vmax = float(np.max(vals)) if vals.size else 1.0

    fig = plt.figure(figsize=(cols * 1.35, rows * 1.35 + 3.4))
    gs = fig.add_gridspec(
        rows + 1, cols, height_ratios=[1] * rows + [1.6], hspace=0.5, wspace=0.05
    )
    slice_axes = []
    im = None
    for idx, z in enumerate(zslices):
        r, c = divmod(idx, cols)
        ax = fig.add_subplot(gs[r, c])
        im = ax.imshow(np.rot90(map_3d[:, :, z]), cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(f"z={z}", fontsize=7, pad=1)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        slice_axes.append(ax)

    if im is not None:
        fig.colorbar(im, ax=slice_axes, fraction=0.015, pad=0.01)

    axh = fig.add_subplot(gs[rows, :])
    axh.hist(vals, bins=40, color="#a9c8e0", edgecolor="#7fa9cc")
    axh.set_title("value distribution (in mask)", fontsize=9)
    axh.set_xlabel("value")
    axh.set_ylabel("voxels")

    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold", y=0.99)
    return _save(fig, out_path)


_MEASURE_CMAP = {
    "reho": "YlOrRd",
    "alff": "cold_hot",
    "falff": "cold_hot",
    "rsfa": "viridis",
    # Arnav's additional measures.
    "alff_slow5": "cold_hot",
    "alff_slow4": "cold_hot",
    "falff_slow5": "cold_hot",
    "falff_slow4": "cold_hot",
    "int": "magma",
    "coherence_reho": "YlOrRd",
    "mse": "viridis",
}


def plot_orientations(
    map_3d: np.ndarray,
    affine,
    title: str = "",
    cmap: str = "cold_hot",
    symmetric: bool = False,
    n_cuts: int = 7,
    out_path=None,
):
    """Radiological-style slice panels: axial, coronal, sagittal rows.

    Uses nilearn to render each orientation as a labelled row of anatomical
    slices (mm coordinates, L/R markers), like standard neuroimaging figures.
    Requires the map's affine so cuts are in world (mm) space.
    """
    import matplotlib.pyplot as plt
    import nibabel as nib
    from nilearn import plotting

    img = nib.Nifti1Image(np.asarray(map_3d, dtype=float), affine)
    vals = map_3d[np.isfinite(map_3d) & (map_3d != 0)]
    if symmetric:
        vmax = float(np.percentile(np.abs(vals), 99)) if vals.size else 1.0
        vmin = -vmax
    else:
        vmax = float(np.percentile(vals, 99)) if vals.size else 1.0
        vmin = 0.0

    fig = plt.figure(figsize=(13, 9))
    rows = [("z", "axial"), ("y", "coronal"), ("x", "sagittal")]
    for i, (mode, label) in enumerate(rows):
        ax = fig.add_subplot(3, 1, i + 1)
        plotting.plot_stat_map(
            img,
            bg_img=None,
            display_mode=mode,
            cut_coords=n_cuts,
            colorbar=True,
            cmap=cmap,
            axes=ax,
            black_bg=False,
            annotate=True,
            threshold=None,
            symmetric_cbar=symmetric,
            vmin=vmin,
            vmax=vmax,
        )
        # Orientation label in a black box, top-left of the row.
        ax.annotate(
            label,
            xy=(0.005, 0.9),
            xycoords="axes fraction",
            fontsize=12,
            color="white",
            weight="bold",
            bbox=dict(boxstyle="square,pad=0.3", fc="black", ec="none"),
        )
    if title:
        fig.suptitle(title, fontsize=13, y=0.99)
    fig.subplots_adjust(hspace=0.25, top=0.95, bottom=0.02)
    return _save(fig, out_path)


def plot_value_hist(map_3d: np.ndarray, mask: np.ndarray | None = None, out_path=None):
    """Small histogram of in-mask values, shown alongside the slice panels."""
    import matplotlib.pyplot as plt

    if mask is None:
        mask = np.isfinite(map_3d) & (map_3d != 0)
    vals = map_3d[mask]
    fig, ax = plt.subplots(figsize=(7, 2.6))
    ax.hist(vals, bins=40, color="#a9c8e0", edgecolor="#7fa9cc")
    ax.set_title("value distribution (in mask)", fontsize=10)
    ax.set_xlabel("value")
    ax.set_ylabel("voxels")
    fig.tight_layout()
    return _save(fig, out_path)


def plot_subject_vs_reference(
    subject_value, reference: dict, out_path=None, region: int | None = None
):
    """Show where a subject's coupling falls in the HCP distribution.

    For a global (scalar) coupling metric, plots the HCP histogram with the
    subject's value marked. For a regional metric, pass ``region`` to inspect
    one node; otherwise the mean across regions is used.
    """
    import matplotlib.pyplot as plt

    vals = np.asarray(reference["values"], dtype=float)
    x = np.asarray(subject_value, dtype=float)

    if vals.ndim == 2:  # regional: (S, N)
        if region is not None:
            cohort = vals[:, region]
            sval = float(np.ravel(x)[region])
            label = f"region {region}"
        else:
            cohort = np.nanmean(vals, axis=1)
            sval = float(np.nanmean(x))
            label = "mean across regions"
    else:
        cohort = vals
        sval = float(x)
        label = "global"

    pct = float((cohort < sval).mean() * 100)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(cohort, bins=25, color="#d6e6f2", edgecolor="#a9c8e0")
    ax.axvline(
        sval,
        color="#3b82c4",
        lw=2.2,
        label=f"subject ({label}) = {sval:.3f}\n{pct:.0f}th percentile",
    )
    ax.set_xlabel("structure-function coupling")
    ax.set_ylabel("HCP subjects")
    ax.set_title("Subject vs. HCP normative distribution")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    return _save(fig, out_path)
