"""Tests for the fully-implemented logic: coupling metrics + normative compare.

These use synthetic matrices so they run without any imaging data and prove the
math is correct end-to-end.
"""

import numpy as np

from sftoolbox.config import Config
from sftoolbox import coupling, reference, compare


def _sym(mat):
    mat = (mat + mat.T) / 2
    np.fill_diagonal(mat, 0.0)
    return mat


def test_global_coupling_perfect_alignment():
    rng = np.random.default_rng(0)
    struct = _sym(rng.random((20, 20)))
    func = struct.copy()  # identical -> perfect rank correlation
    cfg = Config(coupling_metric="global_corr", coupling_corr="spearman")
    val = coupling.coupling(struct, func, cfg)
    assert val > 0.99


def test_regional_coupling_shape():
    rng = np.random.default_rng(1)
    struct = _sym(rng.random((15, 15)))
    func = _sym(rng.random((15, 15)))
    cfg = Config(coupling_metric="regional_corr")
    val = coupling.coupling(struct, func, cfg)
    assert val.shape == (15,)


def test_shape_mismatch_raises():
    cfg = Config()
    struct = np.zeros((10, 10))
    func = np.zeros((8, 8))
    try:
        coupling.coupling(struct, func, cfg)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_reference_and_compare_roundtrip(tmp_path):
    cfg = Config(coupling_metric="global_corr", reference_path=str(tmp_path / "ref.npz"))
    # 100 HCP subjects, coupling ~ N(0.5, 0.1)
    rng = np.random.default_rng(2)
    vals = list(rng.normal(0.5, 0.1, size=100))
    ids = [f"HCP{i:03d}" for i in range(100)]
    ref = reference.build_reference(vals, ids, cfg)

    # A subject exactly at the mean -> z ~ 0, percentile ~ 50.
    res = compare.compare_subject(ref["mean"], ref)
    assert abs(float(res["z"])) < 0.05
    assert abs(float(res["percentile"]) - 50.0) < 5.0

    # A clearly high subject -> high percentile.
    res_hi = compare.compare_subject(0.8, ref)
    assert float(res_hi["percentile"]) > 95.0


def test_empirical_percentile(tmp_path):
    cfg = Config(coupling_metric="global_corr")
    vals = list(np.linspace(0, 1, 101))
    ids = [str(i) for i in range(101)]
    ref = reference.build_reference(vals, ids, cfg, out_path=str(tmp_path / "r.npz"))
    pct = compare.empirical_percentile(0.5, ref)
    assert abs(float(pct) - 50.0) < 2.0
