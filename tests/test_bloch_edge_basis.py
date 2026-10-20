"""Fixed ("edge") Bloch basis for modes close to a stopband edge."""

import numpy as np
import pytest
from scipy.linalg import expm
from test_bloch import HOLOTYPE, _holotype_analysis

from twpasolver.cmes.bloch import (
    edge_basis_tables,
    edge_reference_frequency,
    logm_unimodular_2x2,
    lossless_copy,
    supercell_matrix,
)
from twpasolver.cmes.bloch_fb import _edge_bvp_solve, _expm_2x2


def test_logm_expm_2x2():
    """Closed-form log/exp agree with scipy, including the nilpotent (EP) case."""
    rng = np.random.default_rng(0)
    A = 0.05 * (rng.normal(size=(6, 2, 2)) + 1j * rng.normal(size=(6, 2, 2)))
    A -= np.trace(A, axis1=1, axis2=2)[:, None, None] / 2 * np.eye(2)
    Q = np.array([expm(a) for a in A])
    assert np.allclose(logm_unimodular_2x2(Q), A, atol=1e-14)
    for a in A:
        assert np.allclose(_expm_2x2(a + 0.01), expm(a + 0.01), atol=1e-14)
    N = np.array([[0.0, 0.02], [0.0, 0.0]], dtype=complex)
    assert np.allclose(logm_unimodular_2x2(np.eye(2) + N), N, atol=1e-16)
    assert np.allclose(_expm_2x2(N), np.eye(2) + N, atol=1e-16)


def test_edge_bvp_solve_inside_gap():
    """Riccati sweep: end conditions exact, ODE satisfied, bounded for |K01| L = 50."""
    L, n = 1000.0, 4001
    x = np.linspace(0, L, n)
    h = x[1] - x[0]
    K = np.array([[-0.001 - 0.002j, 0.05], [0.05, -0.001 + 0.002j]])
    Sp = 1e-3 * np.exp(1j * 0.01 * x)
    Sm = 2e-3 * np.exp(-1j * 0.02 * x)
    cp, cm = _edge_bvp_solve(K, h, Sp, Sm, 1.0 + 0j, 0.3 + 0j, 0.1j, -0.2 + 0j)
    assert abs(cp[0] - (1 + 0.3 * cm[0])) < 1e-14
    assert abs(cm[-1] - (0.1j - 0.2 * cp[-1])) < 1e-14
    c = np.vstack([cp, cm])
    d = (c[:, 2:] - c[:, :-2]) / (2 * h)
    r = d - K @ c[:, 1:-1] - np.vstack([Sp, Sm])[:, 1:-1]
    assert np.abs(r).max() < 1e-5
    assert np.abs(c).max() < 2


@pytest.mark.skipif(not HOLOTYPE.exists(), reason="holotype model not available")
def test_edge_basis_tables_exact():
    """K reproduces the cascade of N lossy supercells; lossless K conserves flux."""
    from twpasolver.models import TWPA

    tw = TWPA.from_file(str(HOLOTYPE))
    for cell in tw.cells:
        cell.delta = 2e-4
    ll = lossless_copy(tw)
    f_ref, f_edge, parity = edge_reference_frequency(ll, 41.215e9, 0.10)
    assert 41.20e9 < f_edge < 41.22e9 and f_ref > f_edge and parity == -1
    f = np.array([41.26e9, f_edge, 41.19e9])  # passband, edge, gap
    tab = edge_basis_tables(tw, ll, f, np.full(3, f_ref), np.full(3, 3))
    nc = tab["n_cells"]
    assert np.all(tab["W"] > 0)
    assert np.all(tab["ham_defect"] < 1e-12)
    N = 50
    for q in range(3):
        U = np.array([[tab["Zb_plus"][q], -tab["Zb_minus"][q]], [1, 1]])
        T = supercell_matrix(tw, f[q : q + 1])[0]
        c0 = np.array([1.0, 0.3])
        lhs = (-1) ** (3 * N) * U @ expm(tab["K"][q] * nc * N) @ c0
        rhs = np.linalg.matrix_power(np.linalg.inv(T), N) @ U @ c0
        assert np.max(np.abs(lhs - rhs)) < 1e-10 * np.max(np.abs(rhs))


def _gain(analysis, sig, basis, passes, depth, monkeypatch):
    from twpasolver.cmes import solver_config
    from twpasolver.cmes.solver_config import FBSolverConfig

    monkeypatch.setattr(solver_config, "BLOCH_EDGE_BASIS", basis)
    return analysis.gain(
        signal_freqs=np.asarray(sig),
        Is0=1e-9,
        pump=16.0,
        model="bloch",
        mode_array_config="ext2",
        fb_config=FBSolverConfig(passes=passes, anderson_depth=depth),
    )


@pytest.mark.skipif(not HOLOTYPE.exists(), reason="holotype model not available")
def test_edge_basis_matches_native_basis_in_passband(monkeypatch):
    """p2s in the passband: fixed and native bases give the same gain."""
    analysis = _holotype_analysis()
    native = _gain(analysis, [9.25], None, 80, 15, monkeypatch)["gain_db"]
    fixed = _gain(analysis, [9.25], 0.09, 80, 15, monkeypatch)
    p2s = fixed["mode_info"]["mode_map"]["p2s"]
    assert fixed["bloch"]["edge_flag"][0, p2s]
    assert np.allclose(fixed["gain_db"], native, atol=0.02)


@pytest.mark.skipif(not HOLOTYPE.exists(), reason="holotype model not available")
def test_edge_basis_independent_of_threshold(monkeypatch):
    """At the edge of p2s the gain barely depends on the threshold around 0.1."""
    analysis = _holotype_analysis()
    sig = [9.212]
    g10 = _gain(analysis, sig, 0.10, 80, 15, monkeypatch)
    g12 = _gain(analysis, sig, 0.12, 80, 15, monkeypatch)
    p2s = g10["mode_info"]["mode_map"]["p2s"]
    assert g10["bloch"]["edge_flag"][0, p2s]
    assert np.all(np.isfinite(g10["gain_db"]))
    assert np.allclose(g10["gain_db"], g12["gain_db"], atol=0.02)


@pytest.mark.skipif(not HOLOTYPE.exists(), reason="holotype model not available")
def test_edge_basis_seamless_at_passband_switch(monkeypatch):
    """Just inside the window on the passband side, the basis is the native one."""
    from twpasolver.cmes import solver_config

    analysis = _holotype_analysis()
    sig = [9.2711]  # p2s splitting 0.099
    fixed = _gain(analysis, sig, 0.10, 80, 15, monkeypatch)
    p2s = fixed["mode_info"]["mode_map"]["p2s"]
    assert fixed["bloch"]["edge_flag"][0, p2s]
    monkeypatch.setattr(solver_config, "BLOCH_EDGE_GUARD", None)
    native = _gain(analysis, sig, None, 80, 15, monkeypatch)
    assert np.allclose(fixed["gain_db"], native["gain_db"], atol=0.01)
