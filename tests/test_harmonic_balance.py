"""
``bloch`` gain of the holotype against a converged harmonic-balance simulation.

The reference (``data/hb_holotype_gain.csv``, provenance in its header) solves the
same circuit without CMEs, so this checks the physics, not only stability.
"""

from pathlib import Path

import numpy as np

from twpasolver import ModeSet, TWPAnalysis

DATA = Path(__file__).resolve().parent / "data"
# Error ceilings (dB) at points the CMEs are known to miss: 12.9 GHz is a narrow
# ripple near the stopband, shifted by ~50 MHz.
KNOWN = {12.9: 0.7}


def test_bloch_matches_harmonic_balance():
    ref = np.loadtxt(DATA / "hb_holotype_gain.csv", delimiter=",")
    fs, g_hb = ref[:, 0], ref[:, 1]
    an = TWPAnalysis(twpa=DATA / "imskit_v3_holotype.json", freqs=(0.2, 80, 2e-3))
    modes = ModeSet.extended_3wm(
        n_pump_harmonics=2, n_frequency_conversion=2, n_signal_harmonics=0
    )
    # Default settings; kappa_cutoff 0.1 would miss 5.4/10.4 GHz (edge mode p+i) by 2 dB.
    r = an.gain(fs, pump=15.8, Ip0=1e-4, Is0=1e-9, model="bloch", modes=modes)
    assert r.converged.all()

    err = np.abs(r.gain_db - g_hb)
    # A level error of 0.25 dB or a 30 MHz shift of a steep flank.
    tol = 0.25 + 0.03 * np.abs(np.gradient(g_hb, fs))
    for f, ceiling in KNOWN.items():
        tol[np.isclose(fs, f)] = ceiling
    bad = err > tol
    assert not bad.any(), dict(zip(fs[bad], err[bad].round(3)))
    assert np.median(err) < 0.03
