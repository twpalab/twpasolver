"""
Full-band gain of the V3 holotype with and without the fixed edge basis.

Used by ``06_edge_basis.ipynb``. Two runs of the forward-backward Bloch CMEs
(``model="bloch"``) on the same signal grid:

* ``none`` : native Bloch basis everywhere (``edge_basis_threshold=None``);
* ``edge`` : fixed edge basis for every mode without input whose splitting is
             below ``E_WINDOW`` (the default ``FBSolverConfig``).

The signal grid is coarse across the band and fine inside every window where a
signal-dependent mode is within ``E_WINDOW`` of a stopband edge. Results are
cached to ``edge_basis_data/full_band.npz``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from twpasolver import TWPAnalysis
from twpasolver.cmes.bloch import supercell_matrix
from twpasolver.cmes.solver_config import FBSolverConfig
from twpasolver.logger import log
from twpasolver.modes_rwa import ModeArrayFactory

HERE = Path(__file__).resolve().parent
MODEL = HERE / "imskit_v3_holotype.json"
CACHE = HERE / "edge_basis_data" / "full_band.npz"

PORT_Z = 40
FP = 15.8  # pump (GHz)
IS0 = 1e-8  # signal amplitude (A): small-signal gain
IP0 = 1.0e-4
LOSS_TANGENT = 2e-4
FS_RANGE = (1.0, 14.8)  # signal band (GHz)
COARSE_STEP = 0.05
FINE_STEP = 0.004
E_WINDOW = FBSolverConfig().edge_threshold  # splitting below which a mode is flagged
MARGIN = 0.02  # GHz added on both sides of every window
FB = dict(passes=60, anderson_depth=30)

# Mode set of the thesis: minimal, 1st FC, 2nd FC layers.
MODES = {
    "p": (1, 0),
    "s": (0, 1),
    "i": (1, -1),
    "p2": (2, 0),
    "ps": (1, 1),
    "pi": (2, -1),
    "p3": (3, 0),
    "p2s": (2, 1),
    "p2i": (3, -1),
}
SECTOR_OF = {
    "p": 0,
    "s": 0,
    "i": 0,
    "p2": 1,
    "ps": 1,
    "pi": 1,
    "p3": 2,
    "p2s": 2,
    "p2i": 2,
}
SECTOR_NAMES = ("minimal", "1st FC", "2nd FC")
WEAK = ("s", "i", "ps", "pi", "p2s", "p2i")  # modes that depend on the signal

RUNS = {"none": None, "edge": E_WINDOW}  # edge_basis_threshold of each run


def make_analysis() -> TWPAnalysis:
    """Holotype with loss and the 9-mode array of the thesis (``ext2``)."""
    an = TWPAnalysis(twpa=str(MODEL), f_arange=(0.2, 60, 2e-3), port_Z_ext=PORT_Z)
    an.twpa.Ip0 = IP0
    for cell in an.twpa.cells:
        cell.delta = LOSS_TANGENT
    an.update_base_data()
    an.add_mode_array(
        "ext2",
        ModeArrayFactory.create_extended_3wm(
            an.data, n_pump_harmonics=2, n_frequency_conversion=2, n_signal_harmonics=0
        ),
    )
    return an


def mode_splitting(twpa, fs: np.ndarray, fp: float = FP) -> dict[str, np.ndarray]:
    """``|sinh(gamma d)|`` of every mode of ``MODES`` (lossy supercell), per signal frequency."""
    out = {}
    for m, (a, b) in MODES.items():
        f = np.abs(a * fp + b * fs) * 1e9
        T = supercell_matrix(twpa, f)
        h = 0.5 * (T[:, 0, 0] + T[:, 1, 1])
        out[m] = np.abs(np.sqrt(h**2 - 1 + 0j))
    return out


def windows(fs: np.ndarray, split: np.ndarray, e: float = E_WINDOW):
    """Contiguous signal intervals where ``split < e``: list of (start, stop, min split)."""
    low = split < e
    if not low.any():
        return []
    idx = np.flatnonzero(low)
    groups = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
    return [(fs[g[0]], fs[g[-1]], split[g].min()) for g in groups]


def signal_grid(twpa) -> tuple[np.ndarray, dict]:
    """Coarse band grid plus a fine grid around every near-edge window of a weak mode."""
    dense = np.arange(*FS_RANGE, 1e-3)
    split = mode_splitting(twpa, dense)
    wins = {m: windows(dense, split[m]) for m in WEAK}
    pts = [np.arange(FS_RANGE[0], FS_RANGE[1] + 1e-9, COARSE_STEP)]
    for m in WEAK:
        for a, b, _ in wins[m]:
            pts.append(np.arange(a - MARGIN, b + MARGIN + 1e-9, FINE_STEP))
    fs = np.unique(np.round(np.concatenate(pts), 6))
    return fs[(fs >= FS_RANGE[0]) & (fs <= FS_RANGE[1])], wins


def run_sweep(recompute: bool = False, verbose: bool = True) -> dict:
    """Run (or load) both runs on the same grid."""
    if CACHE.exists() and not recompute:
        d = np.load(CACHE, allow_pickle=True)
        return {k: d[k].item() if d[k].dtype == object else d[k] for k in d.files}

    log.setLevel(logging.ERROR)
    an = make_analysis()
    fs, wins = signal_grid(an.twpa)
    out: dict = {"fs": fs, "windows": wins}
    for name, threshold in RUNS.items():
        r = an.gain(
            signal_freqs=fs,
            Is0=IS0,
            pump=FP,
            model="bloch",
            mode_array_config="ext2",
            fb_config=FBSolverConfig(**FB, edge_basis_threshold=threshold),
        )
        b = r["bloch"]
        out[name] = {
            "gain_db": r["gain_db"],
            "converged": b["converged"],
            "residuals": b["residuals"],
            "edge_flag": b.get("edge_flag", np.zeros(b["split"].shape, bool)),
            "split": b["split"],
            "labels": list(r["mode_info"]["mode_map"]),
        }
        if verbose:
            print(f"{name:5s} done: {len(fs)} points")
    CACHE.parent.mkdir(exist_ok=True)
    np.savez(
        CACHE,
        **{
            k: (np.array(v, dtype=object) if isinstance(v, dict) else v)
            for k, v in out.items()
        },
    )
    return out
