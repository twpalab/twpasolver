"""
Regression reference: gains of a few devices against a stored copy.

Run ``python tests/test_regression.py`` to record a new reference after an
intended change of the results, and say why in the commit message.
"""

import logging
from pathlib import Path

import numpy as np
import pytest

from twpasolver import ModeSet, TWPAnalysis
from twpasolver.models import TWPA

DATA = Path(__file__).resolve().parent / "data"
REF = DATA / "regression_gains.npz"
HOLOTYPE = DATA / "imskit_v3_holotype.json"
SIG = np.array([2.0, 6.0, 9.188, 9.196, 10.5, 12.0])
EXT2 = ModeSet.extended_3wm(
    n_pump_harmonics=3, n_frequency_conversion=2, n_signal_harmonics=0
)
CELL_U = {"name": "LCLfBaseCell", "N": 30, "L": 49.9e-12, "C": 24.2e-15, "Lf": 0.87e-9}
CELL_L = {"name": "LCLfBaseCell", "N": 6, "L": 48.6e-12, "C": 8.2e-15, "Lf": 0.28e-9}
PORT_GAINS = ("gain_db", "gain_db_11")  # S12 and S22 are NaN without backward inputs
# Loose enough for float differences across platforms, tight enough for physics changes.
ATOL_DB = 1e-3


def _holotype(scale=(1.0,)):
    base = TWPA.from_file(HOLOTYPE)
    for cell in base.cells:
        cell.delta = 2e-4
    segs = []
    for g in scale:
        seg = base.model_copy(deep=True, update={"N": base.N // len(scale)})
        for cell in seg.cells:
            cell.L, cell.Lf = cell.L * g, cell.Lf * g
        segs.append(seg)
    twpa = (
        segs[0]
        if len(scale) == 1
        else TWPA(cells=segs, Z0_ref=base.Z0_ref, Ip0=base.Ip0)
    )
    return TWPAnalysis(twpa=twpa, freqs=(0.2, 80, 2e-3))


def compute() -> dict[str, np.ndarray]:
    """Every array of the reference, computed with the current code."""
    out = {}

    def gains(tag, r):
        for k in PORT_GAINS:
            out[f"{tag}/{k}"] = np.asarray(getattr(r, k))

    for tag, an in (("holo", _holotype()), ("seg", _holotype((1.0, 1.02, 0.98, 1.0)))):
        for model in ("simple", "bloch"):
            gains(
                f"{tag}_{model}", an.gain(SIG, 16.0, Is0=1e-9, model=model, modes=EXT2)
            )

    s3 = np.array([3.0, 5.0, 6.0])
    small = TWPAnalysis(
        twpa=TWPA(
            cells=[CELL_U, CELL_L, CELL_U], N=20, Istar=6.3e-3, Idc=1e-3, Ip0=1e-3
        ),
        freqs=(0.05, 20, 1e-2),
    )
    for model in ("simple", "bloch"):
        gains(f"small_{model}", small.gain(s3, pump=12.0, Is0=1e-8, model=model))
    out["small_pm/delta"] = small.phase_matching((1, 6, 0.5), (11, 14, 0.5)).delta

    segs = [
        TWPA(cells=[CELL_U, CELL_L, CELL_U], N=20, Istar=6.3e-3, Idc=1e-3, Ip0=1e-3),
        TWPA(cells=[CELL_U, CELL_L, CELL_U], N=10, Istar=9e-3, Idc=1e-3, Ip0=1e-3),
    ]
    two = TWPAnalysis(twpa=TWPA(cells=segs, Ip0=1e-3), freqs=(0.05, 20, 1e-2))
    gains("two_bloch", two.gain(s3, pump=12.0, Is0=1e-8, model="bloch"))
    out["two_pm/delta"] = two.phase_matching((1, 6, 0.5), (11, 14, 0.5)).delta
    return out


@pytest.fixture(scope="module")
def current():
    logging.getLogger("twpasolver").setLevel(logging.ERROR)
    return compute()


def test_reference_keys(current):
    assert sorted(np.load(REF).files) == sorted(current)


@pytest.mark.parametrize("key", sorted(np.load(REF).files) if REF.exists() else [])
def test_matches_reference(current, key):
    np.testing.assert_allclose(current[key], np.load(REF)[key], rtol=0, atol=ATOL_DB)


if __name__ == "__main__":
    np.savez(REF, **compute())
    print(f"recorded {REF}")
