"""Standalone sanity check — no pytest required.

Run with the venv Python (has numpy/scipy):

    python check.py

Exercises the fully-implemented logic (coupling + normative compare) on
synthetic data and prints PASS/FAIL for each check.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np

from sftoolbox.config import Config
from sftoolbox import coupling, reference, compare

_failures = []


def check(name, cond):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}")
    if not cond:
        _failures.append(name)


def sym(mat):
    mat = (mat + mat.T) / 2
    np.fill_diagonal(mat, 0.0)
    return mat


def main():
    rng = np.random.default_rng(0)

    # 1. Global coupling: identical matrices -> ~perfect correlation.
    struct = sym(rng.random((20, 20)))
    cfg_g = Config(coupling_metric="global_corr", coupling_corr="spearman")
    check("global coupling of identical matrices ~ 1",
          coupling.coupling(struct, struct.copy(), cfg_g) > 0.99)

    # 2. Regional coupling returns one value per node.
    func = sym(rng.random((15, 15)))
    struct15 = sym(rng.random((15, 15)))
    cfg_r = Config(coupling_metric="regional_corr")
    check("regional coupling shape == (15,)",
          coupling.coupling(struct15, func, cfg_r).shape == (15,))

    # 3. Shape mismatch raises.
    try:
        coupling.coupling(np.zeros((10, 10)), np.zeros((8, 8)), Config())
        check("shape mismatch raises ValueError", False)
    except ValueError:
        check("shape mismatch raises ValueError", True)

    # 4. Normative reference + compare roundtrip.
    with tempfile.TemporaryDirectory() as d:
        cfg = Config(coupling_metric="global_corr",
                     reference_path=str(Path(d) / "ref.npz"))
        vals = list(rng.normal(0.5, 0.1, size=100))
        ids = [f"HCP{i:03d}" for i in range(100)]
        ref = reference.build_reference(vals, ids, cfg)

        at_mean = compare.compare_subject(ref["mean"], ref)
        check("subject at cohort mean -> percentile ~ 50",
              abs(float(at_mean["percentile"]) - 50.0) < 5.0)

        high = compare.compare_subject(0.8, ref)
        check("clearly high subject -> percentile > 95",
              float(high["percentile"]) > 95.0)

        emp = compare.empirical_percentile(0.5, {
            "values": np.linspace(0, 1, 101),
        })
        check("empirical percentile of median ~ 50",
              abs(float(emp) - 50.0) < 2.0)

    print()
    if _failures:
        print(f"{len(_failures)} check(s) FAILED: {_failures}")
        sys.exit(1)
    print("All checks passed.")


if __name__ == "__main__":
    main()
