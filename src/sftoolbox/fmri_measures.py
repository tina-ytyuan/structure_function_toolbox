"""Vendored fMRI signal-property measures (ALFF, fALFF, ReHo, RSFA).

These are the project's canonical measure definitions, taken verbatim from the
team's DCC fMRI signal-property pipeline (DPABI/REST-matched). They are kept here
as the single source of truth so the toolbox computes measures exactly the way
the team does, rather than an independent reimplementation.

Do not "improve" or refactor the numerics here — the point is bit-for-bit
agreement with the reference pipeline (and, through it, DPABI). The toolbox-facing
wrappers that adapt these to the (map_3d, mask) interface live in measures.py.

Conventions (DPABI/REST):
  * linear detrend before ALFF/fALFF/RSFA,
  * zero-pad to the next power of two, amplitude = |FFT| * 2 / T,
  * DPABI integer frequency-bin cutoffs (not float-frequency comparison),
  * Kendall's W ReHo with tie-corrected ranks over a 7/19/27 neighborhood,
  * RSFA = temporal SD (ddof=1) after detrend + FFT band-pass.
"""

from __future__ import annotations

import numpy as np

# --- shared helpers ----------------------------------------------------

def remove_linear_trend(data):
    """Remove the best-fit linear trend from each voxel time series.

    Input:
        data: V x T or X x Y x Z x T (time must be the last axis)
    Returns:
        detrended: same shape as data
    """
    x = np.asarray(data, dtype=np.float64)
    T = x.shape[-1]

    t = np.arange(T, dtype=np.float64)
    t = t - t.mean()

    # slope and intercept for each voxel along time
    slope = np.sum(x * t, axis=-1, keepdims=True) / np.sum(t ** 2)
    intercept = np.mean(x, axis=-1, keepdims=True)

    trend = slope * t + intercept
    return x - trend


def amplitude_spectrum(data, tr):
    """DPABI-style positive-frequency FFT amplitude for each voxel.

    Input:
        data: V x T or X x Y x Z x T (time must be the last axis)
        tr: repetition time in seconds
    Returns:
        freqs: F
        amp: same leading shape as data, but time axis becomes F
    """
    x = np.asarray(data, dtype=np.float64)
    T = x.shape[-1]
    padded_T = 1 << (T - 1).bit_length()

    freqs = np.fft.rfftfreq(padded_T, d=tr)
    fft_vals = np.fft.rfft(x, n=padded_T, axis=-1)
    amp = np.abs(fft_vals) * 2.0 / T

    return freqs, amp


def dpabi_frequency_indices(t_len, tr, low=0.01, high=0.08):
    """Python indices matching DPABI y_alff_falff.m cutoff indices."""
    padded_T = 1 << (t_len - 1).bit_length()
    sample_freq = 1.0 / tr

    if low >= sample_freq / 2.0:
        idx_low = padded_T // 2
    else:
        idx_low = int(np.ceil(low * padded_T * tr + 1.0)) - 1

    if high >= sample_freq / 2.0 or high == 0:
        idx_high = padded_T // 2
    else:
        idx_high = int(np.fix(high * padded_T * tr + 1.0)) - 1

    idx_low = max(0, min(idx_low, padded_T // 2))
    idx_high = max(0, min(idx_high, padded_T // 2))
    return idx_low, idx_high, padded_T


def fft_bandpass(data, tr, low=0.01, high=0.08):
    """FFT bandpass filter.

    Input:
        data: V x T or X x Y x Z x T (time must be the last axis)
        tr: repetition time in seconds
        low/high: cutoff frequencies in Hz
    Returns:
        filtered: same shape as data
    """
    x = np.asarray(data, dtype=np.float64)
    T = x.shape[-1]

    freqs = np.fft.rfftfreq(T, d=tr)
    fft_vals = np.fft.rfft(x, axis=-1)

    keep = (freqs >= low) & (freqs <= high)
    fft_vals[..., ~keep] = 0.0

    return np.fft.irfft(fft_vals, n=T, axis=-1)


def dpabi_ideal_filter(data, tr, low=0.01, high=0.08):
    """DPABI/REST-style ideal FFT filter used by y_IdealFilter.m."""
    x = np.asarray(data, dtype=np.float64)
    T = x.shape[-1]
    padded_T = 1 << (T - 1).bit_length()
    sample_freq = 1.0 / tr

    if low >= sample_freq / 2.0:
        idx_low = padded_T // 2
    else:
        idx_low = int(np.ceil(low * padded_T * tr + 1.0)) - 1

    if high >= sample_freq / 2.0 or high == 0:
        idx_high = padded_T // 2
    else:
        idx_high = int(np.fix(high * padded_T * tr + 1.0)) - 1

    idx_low = max(0, min(idx_low, padded_T - 1))
    idx_high = max(0, min(idx_high, padded_T - 1))

    keep = np.zeros(padded_T, dtype=bool)
    keep[idx_low:idx_high + 1] = True
    neg_idx = np.arange(padded_T - idx_high, padded_T - idx_low + 1)
    neg_idx = neg_idx[(neg_idx >= 0) & (neg_idx < padded_T)]
    keep[neg_idx] = True

    x = x - x.mean(axis=-1, keepdims=True)
    fft_vals = np.fft.fft(x, n=padded_T, axis=-1)
    fft_vals[..., ~keep] = 0.0
    return np.fft.ifft(fft_vals, axis=-1)[..., :T].real


def rank_rows(matrix):
    """Rank each row of a K x T matrix along time (1..T), tie-corrected."""
    x = np.asarray(matrix, dtype=np.float64)
    order = np.argsort(x, axis=1)
    sorted_x = np.take_along_axis(x, order, axis=1)
    sorted_ranks = np.broadcast_to(
        np.arange(1, x.shape[1] + 1, dtype=np.float64),
        x.shape,
    ).copy()

    tie_rows = np.where(np.any(np.diff(sorted_x, axis=1) == 0, axis=1))[0]
    for row in tie_rows:
        values = sorted_x[row]
        starts = np.r_[0, np.flatnonzero(np.diff(values) != 0) + 1]
        ends = np.r_[starts[1:], values.size]
        for start, end in zip(starts, ends):
            if end - start > 1:
                sorted_ranks[row, start:end] = (start + 1 + end) / 2.0

    ranks = np.empty_like(sorted_ranks, dtype=np.float64)
    row_ids = np.arange(x.shape[0])[:, None]
    ranks[row_ids, order] = sorted_ranks

    return ranks


def kendalls_w(time_series):
    """Kendall's coefficient of concordance for a K x T rank matrix."""
    ranks = rank_rows(time_series)

    K, T = ranks.shape
    rank_sums = ranks.sum(axis=0)
    mean_rank_sum = rank_sums.mean()

    S = np.sum((rank_sums - mean_rank_sum) ** 2)
    W = 12.0 * S / (K ** 2 * (T ** 3 - T) + 1e-8)

    return W


def reho_neighbor_offsets(neighborhood=27):
    """Relative voxel offsets for a ReHo neighborhood (7, 19, or 27)."""
    if neighborhood not in (7, 19, 27):
        raise ValueError("neighborhood must be 7, 19, or 27")

    offsets = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                distance_sq = dx * dx + dy * dy + dz * dz

                if neighborhood == 7 and distance_sq <= 1:
                    offsets.append((dx, dy, dz))
                elif neighborhood == 19 and distance_sq <= 2:
                    offsets.append((dx, dy, dz))
                elif neighborhood == 27:
                    offsets.append((dx, dy, dz))
    return offsets


# --- measures ----------------------------------------------------------

def compute_alff(data_2d, tr=0.72, low=0.01, high=0.08, normalize=False):
    """ALFF = average FFT amplitude in the low-frequency band. Input V x T."""
    x = remove_linear_trend(data_2d)
    _, amp = amplitude_spectrum(x, tr)

    idx_low, idx_high, _ = dpabi_frequency_indices(x.shape[-1], tr, low, high)
    alff = amp[:, idx_low:idx_high + 1].mean(axis=1)

    if normalize:
        alff = alff / (alff.mean() + 1e-8)

    return alff


def compute_falff(data_2d, tr=0.72, low=0.01, high=0.08, normalize=False):
    """fALFF = low-frequency amplitude / total amplitude. Input V x T."""
    x = remove_linear_trend(data_2d)
    _, amp = amplitude_spectrum(x, tr)

    idx_low, idx_high, padded_T = dpabi_frequency_indices(x.shape[-1], tr, low, high)
    low_amp = amp[:, idx_low:idx_high + 1].sum(axis=1)
    total_amp = amp[:, 1:padded_T // 2 + 1].sum(axis=1)

    falff = np.divide(
        low_amp,
        total_amp,
        out=np.zeros_like(low_amp, dtype=np.float64),
        where=total_amp != 0,
    )

    if normalize:
        falff = falff / (falff.mean() + 1e-8)

    return falff


def compute_reho(
    data_4d,
    mask=None,
    neighborhood=27,
    tr=0.72,
    low=0.01,
    high=0.08,
    detrend=False,
    filter_band=False,
):
    """Compute ReHo using a 7, 19 or 27 voxel neighborhood.

    Input:
        data_4d: X x Y x Z x T
        mask: X x Y x Z boolean array; if None, all voxels are included
        neighborhood: 7, 19, or 27
        tr: repetition time in seconds
        low/high: optional filter cutoff frequencies in Hz
        detrend: remove linear trend before ReHo (DPABI IsNeedDetrend)
        filter_band: apply DPABI-style ideal filter before ReHo
    Output:
        reho_map: X x Y x Z with ReHo values; zero outside mask
    """
    data = np.asarray(data_4d, dtype=np.float64)
    if data.ndim != 4:
        raise ValueError("data_4d must have shape X x Y x Z x T")

    X, Y, Z, T = data.shape

    if mask is None:
        mask = np.ones((X, Y, Z), dtype=bool)
    else:
        mask = mask.astype(bool)
        if mask.shape != (X, Y, Z):
            raise ValueError("mask must have shape X x Y x Z")

    offsets = reho_neighbor_offsets(neighborhood)
    if detrend or filter_band:
        masked_data = data[mask]
        if detrend:
            masked_data = remove_linear_trend(masked_data)
        if filter_band:
            masked_data = dpabi_ideal_filter(masked_data, tr=tr, low=low, high=high)
        data[:] = 0.0
        data[mask] = masked_data

    reho = np.zeros((X, Y, Z), dtype=np.float64)

    for i in range(1, X - 1):
        for j in range(1, Y - 1):
            for k in range(1, Z - 1):
                if not mask[i, j, k]:
                    continue

                neighbor_series = []
                for dx, dy, dz in offsets:
                    ni = i + dx
                    nj = j + dy
                    nk = k + dz

                    if ni < 0 or ni >= X or nj < 0 or nj >= Y or nk < 0 or nk >= Z:
                        continue
                    if not mask[ni, nj, nk]:
                        continue

                    neighbor_series.append(data[ni, nj, nk, :])

                neighbors = np.asarray(neighbor_series)         # K x T

                if neighbors.shape[0] < 1:
                    continue

                reho[i, j, k] = kendalls_w(neighbors)

    return reho


def compute_bold_sd_rsfa(data, tr=0.72, low=0.01, high=0.08, bandpass=True,
                         normalize=False):
    """BOLD-SD / RSFA = temporal standard deviation at each voxel.

    Input:
        data: V x T or X x Y x Z x T (time must be the last axis)
    Output:
        rsfa: same leading shape as data
    """
    x = remove_linear_trend(data)

    if bandpass:
        x = fft_bandpass(x, tr=tr, low=low, high=high)

    rsfa = x.std(axis=-1, ddof=1)

    if normalize:
        rsfa = rsfa / (rsfa.mean() + 1e-8)

    return rsfa
