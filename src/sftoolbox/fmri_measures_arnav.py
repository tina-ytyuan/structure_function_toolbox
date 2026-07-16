"""Vendored fMRI signal measures from Arnav's DCC code.

These are additional measures beyond Ajay's four, taken verbatim from Arnav's
``fMRI_signal_calculations.py`` on the DCC:

  * slow-5 / slow-4 ALFF & fALFF  (band split, DPABI/REST amplitude convention)
  * INT  -- Intrinsic Neural Timescale (Watanabe et al. 2019)
  * coherence-ReHo  -- REST rest_Cohe_ReHo.m spectral local homogeneity
  * MSE  -- Multiscale Entropy (sample entropy over coarse-grained scales)

Source: /hpc/group/396-brainfun26/arnav/fMRI_signal_calculations.py
As with Ajay's file, do NOT refactor the numerics: the point is bit-for-bit
agreement with Arnav's pipeline. The toolbox-facing wrappers that adapt these to
the (map_3d, mask) interface live in measures.py.

``antropy`` is required only for MSE and is imported lazily, so importing this
module (and the package) does not require it. Install it to run MSE:
    pip install antropy
"""

from __future__ import annotations

import numpy as np
from scipy import signal

# Arnav's defaults.
TR = 0.72
SLOW5 = (0.01, 0.027)
SLOW4 = (0.027, 0.073)


# --- utilities ---------------------------------------------------------

def divide(num, den, eps=1e-12):
    return num / np.clip(den, eps, None)


def next_pow2(n):
    return 1 << (n - 1).bit_length()


def matlab_hanning(n):
    """MATLAB's hanning(N): w(k) = 0.5*(1-cos(2*pi*k/(N+1))), k=1..N.

    Nonzero endpoints, unlike numpy/scipy's default Hann window. REST's
    rest_Cohe_ReHo.m uses hanning(N), so this matches it exactly.
    """
    k = np.arange(1, n + 1, dtype=np.float64)
    return (0.5 * (1 - np.cos(2 * np.pi * k / (n + 1)))).astype(np.float32)


def freq_band_bin_range(lo, hi, n_fft, tr):
    """0-based inclusive [start, end] FFT-bin range for a frequency band.

    Matches the discrete bin selection used by DPABI/REST's y_alff_falff.m /
    alff.m (idx_LowCutoff = ceil(...), idx_HighCutoff = fix(...)).
    """
    nyquist_bin = n_fft // 2
    sample_freq = 1.0 / tr

    if lo >= sample_freq / 2:
        idx_lo = nyquist_bin
    else:
        idx_lo = int(np.ceil(lo * n_fft * tr))

    if hi >= sample_freq / 2 or hi == 0:
        idx_hi = nyquist_bin
    else:
        idx_hi = int(np.floor(hi * n_fft * tr))

    return idx_lo, idx_hi


def _coarse_grain(flat, scale):  # input array (n_vox, n_tp)
    n_vox, n_tp = flat.shape
    trim = (n_tp // scale) * scale
    if trim < scale:
        raise ValueError(f"Time series too short for scale={scale}")
    return flat[:, :trim].reshape(n_vox, trim // scale, scale).mean(axis=2)


# --- 1. ALFF / fALFF (slow-4 / slow-5) --------------------------------

def compute_alff_falff(flat, tr=TR, bands=None, detrend=True, standardize=False):
    if bands is None:
        bands = {"slow5": SLOW5, "slow4": SLOW4}

    x = np.asarray(flat, dtype=np.float32)

    if detrend:
        x = signal.detrend(x, axis=1, type="linear")
    else:
        x = x - x.mean(axis=1, keepdims=True)

    if standardize:
        x = divide(x, x.std(axis=1, keepdims=True))

    n_tp = x.shape[1]
    n_fft = next_pow2(n_tp)
    freqs = np.fft.rfftfreq(n_fft, d=tr)
    fft_vals = np.fft.rfft(x, n=n_fft, axis=1).astype(np.complex64)
    # DPABI/REST's y_alff_falff.m applies 2*abs(fft(x))/N uniformly to every
    # bin, including DC and Nyquist, rather than halving them.
    amps = np.abs(fft_vals) * 2.0 / n_tp
    del fft_vals

    nonzero = freqs > 0
    total_amp = amps[:, nonzero].sum(axis=1)

    results = {}
    for name, (lo, hi) in bands.items():
        idx_lo, idx_hi = freq_band_bin_range(lo, hi, n_fft, tr)
        band_amps = amps[:, idx_lo:idx_hi + 1]

        results[f"alff_{name}"] = band_amps.mean(axis=1).astype(np.float32)

        band_amp = band_amps.sum(axis=1)
        results[f"falff_{name}"] = divide(band_amp, total_amp).astype(np.float32)

    return results


# --- 2. Intrinsic Neural Timescale (INT) ------------------------------

def compute_int(flat, tr=TR, max_lag=20, chunk_size=10000):
    x = np.asarray(flat, dtype=np.float32)
    n_vox, n_tp = x.shape

    # Watanabe et al. 2019: mean-center only, no linear detrend.
    x = x - x.mean(axis=1, keepdims=True)

    max_lag = min(max_lag, n_tp - 1)

    n_fft = 2 * next_pow2(n_tp)
    int_map = np.zeros(n_vox, dtype=np.float32)

    for start in range(0, n_vox, chunk_size):
        end = min(start + chunk_size, n_vox)
        chunk = x[start:end]
        F = np.fft.rfft(chunk, n=n_fft, axis=1).astype(np.complex64)
        acf = np.fft.irfft((F * np.conj(F)).astype(np.complex64), n=n_fft, axis=1)[:, :max_lag + 1]
        acf = divide(acf, acf[:, [0]])

        for i in range(end - start):
            ac = acf[i, 1:]
            pos_len = 0
            # '- 1': TW_AutoCorrFactor01.m's `while ACF(j) > 0 && j <= nLags`
            # (j starting at 2) never reaches j = nLags+1, so it never sums the
            # last lag even when positive. Matched here for bit-parity.
            while pos_len < len(ac) - 1 and ac[pos_len] > 0:
                pos_len += 1
            if pos_len > 0:
                int_map[start + i] = np.float32(tr * ac[:pos_len].sum())

    return int_map


# --- 3. Coherence-ReHo ------------------------------------------------

def compute_coherence_reho(
    brain_4d,
    tr=TR,
    band=(0.01, 0.08),
    mask=None,
    nperseg=None,
    noverlap=None,
    detrend="constant",
    cluster_chunk=8192,
):
    """Coherence-ReHo, following REST's rest_Cohe_ReHo.m.

    Band-integrated cross/auto spectrum coherence, averaged over all pairs
    within each voxel's 27-voxel cluster.
    """
    brain_4d = np.asarray(brain_4d, dtype=np.float32)
    X, Y, Z, T = brain_4d.shape

    if mask is None:
        mask = np.ones((X, Y, Z), dtype=bool)
    else:
        mask = np.asarray(mask).astype(bool)

    lo, hi = band

    if nperseg is None:
        # REST "Auto" segment sizing: long enough to resolve full cycles of the
        # band's low cutoff, with 50% overlap.
        min_len_seg = 1.0 / lo / tr
        k = int(np.floor(T / min_len_seg * 2)) - 1
        k = max(k, 1)
        nperseg = max(int(np.floor(T / (k + 1) * 2)), 2)
    if noverlap is None:
        noverlap = int(np.ceil(0.5 * nperseg))

    out = np.full((X, Y, Z), np.nan, dtype=np.float32)

    coords = np.argwhere(mask)
    n_vox = len(coords)
    if n_vox == 0:
        return out

    idx_vol = np.full((X, Y, Z), -1, dtype=np.int32)
    idx_vol[mask] = np.arange(n_vox, dtype=np.int32)

    ts_all = brain_4d[mask].astype(np.float32)
    valid_vox = np.isfinite(ts_all).all(axis=1) & (np.std(ts_all, axis=1) > 1e-12)

    step = nperseg - noverlap
    n_seg = (T - nperseg) // step + 1
    if n_seg < 1:
        return out

    seg_starts = np.arange(n_seg) * step
    seg_idx = seg_starts[:, None] + np.arange(nperseg)[None, :]

    segments = ts_all[:, seg_idx]
    del ts_all

    if detrend == "linear":
        segments = signal.detrend(segments, axis=-1, type="linear")
    elif detrend == "constant":
        segments = segments - segments.mean(axis=-1, keepdims=True)

    win = matlab_hanning(nperseg)
    segments *= win[np.newaxis, np.newaxis, :]

    stft = np.fft.rfft(segments, axis=-1).astype(np.complex64)
    del segments

    freqs = np.fft.rfftfreq(nperseg, d=tr)
    band_sel = (freqs >= lo) & (freqs <= hi)
    if not np.any(band_sel):
        return out
    stft_band = stft[:, :, band_sel]
    del stft

    n_band_freqs = stft_band.shape[-1]
    # band-integration: flatten (segment, freq) so a single matmul sums the
    # cross-spectrum over both segments and band frequencies at once.
    stft_flat = stft_band.reshape(n_vox, n_seg * n_band_freqs)
    del stft_band
    stft_padded = np.vstack(
        [stft_flat, np.zeros((1, stft_flat.shape[1]), dtype=stft_flat.dtype)]
    )
    del stft_flat

    # 27-voxel cluster: center + all face/edge/corner neighbors.
    offsets = np.array([
        (dx, dy, dz)
        for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
    ], dtype=np.int32)
    K = len(offsets)

    member_idx = np.full((n_vox, K), -1, dtype=np.int64)
    for k_off, off in enumerate(offsets):
        nc = coords + off
        in_bounds = (
            (nc[:, 0] >= 0) & (nc[:, 0] < X) &
            (nc[:, 1] >= 0) & (nc[:, 1] < Y) &
            (nc[:, 2] >= 0) & (nc[:, 2] < Z)
        )
        nb_idx = np.full(n_vox, -1, dtype=np.int64)
        nb_idx[in_bounds] = idx_vol[nc[in_bounds, 0], nc[in_bounds, 1], nc[in_bounds, 2]]
        member_idx[:, k_off] = nb_idx

    member_valid = member_idx >= 0
    member_valid[member_valid] &= valid_vox[member_idx[member_valid]]
    member_idx[~member_valid] = n_vox  # point invalid slots at the zero sentinel row

    pair_i, pair_j = np.triu_indices(K, k=1)
    result = np.full(n_vox, np.nan, dtype=np.float32)

    for s in range(0, n_vox, cluster_chunk):
        e = min(s + cluster_chunk, n_vox)
        midx = member_idx[s:e]
        mvalid = member_valid[s:e]

        Xv = stft_padded[midx]  # (c, K, L)
        M = np.matmul(Xv, np.conj(np.transpose(Xv, (0, 2, 1))))  # (c, K, K)
        ap = np.real(np.diagonal(M, axis1=1, axis2=2))  # (c, K), band-integrated auto-power

        num = np.abs(M[:, pair_i, pair_j]) ** 2
        den = ap[:, pair_i] * ap[:, pair_j]
        coh = num / np.clip(den, 1e-12, None)

        pair_valid = mvalid[:, pair_i] & mvalid[:, pair_j]
        n_valid_pairs = pair_valid.sum(axis=1)
        coh = np.where(pair_valid, coh, 0.0)

        has_pairs = n_valid_pairs > 0
        chunk_result = np.full(e - s, np.nan, dtype=np.float32)
        chunk_result[has_pairs] = (
            coh[has_pairs].sum(axis=1) / n_valid_pairs[has_pairs]
        ).astype(np.float32)
        result[s:e] = chunk_result

    out[mask] = result

    return out


# --- 4. Multiscale Entropy (MSE) --------------------------------------

def compute_mse(
    flat,
    scales=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
    m=2,
    r_ratio=0.15,
    detrend=True,
    standardize=False,
    recompute_r_each_scale=False,
):
    # float64 throughout: antropy.sample_entropy upcasts to float64 internally.
    import antropy as ant

    x = np.asarray(flat, dtype=np.float64)
    n_vox, n_tp = x.shape

    if detrend:
        # scipy.signal.detrend(axis=1) returns an F-ordered array; make it
        # C-contiguous for antropy's numba kernel.
        x = np.ascontiguousarray(signal.detrend(x, axis=1, type="linear"))
    else:
        x = x - x.mean(axis=1, keepdims=True)

    orig_sd = x.std(axis=1)

    if standardize:
        x = divide(x, x.std(axis=1, keepdims=True))

    results = {}

    for scale in scales:
        coarse = _coarse_grain(x, scale)
        ent = np.full(n_vox, np.nan, dtype=np.float32)

        for v in range(n_vox):
            ts = coarse[v]

            if not np.isfinite(ts).all():
                continue
            if len(ts) < (m + 2):
                continue
            if np.std(ts) < 1e-12:
                continue

            if recompute_r_each_scale:
                r = float(r_ratio * np.std(ts))
            else:
                r = float(r_ratio * orig_sd[v])

            if r <= 0 or not np.isfinite(r):
                continue

            try:
                ent[v] = np.float32(
                    ant.sample_entropy(np.ascontiguousarray(ts), order=m, tolerance=r)
                )
            except Exception:
                ent[v] = np.nan

        results[f"mse_scale_{scale}"] = ent

    return results
