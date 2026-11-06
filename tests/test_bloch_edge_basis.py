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
from twpasolver.cmes.bloch_common import _expm_2x2
from twpasolver.cmes.bloch_fb import _edge_bvp_solve_segmented


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


def _edge_bvp_one_segment(K, x, S_plus, S_minus, cp_in, rho_in, cm_out, rho_out):
    """Exact two-point solve of dc/dx = K c + S on a uniform grid (one segment)."""
    one, zero = np.ones(1, complex), np.zeros(1, complex)
    return _edge_bvp_solve_segmented(
        np.asarray(K)[None],
        x,
        np.array([0, len(x)], dtype=np.int64),
        S_plus,
        S_minus,
        cp_in,
        rho_in,
        cm_out,
        rho_out,
        one,
        one,
        zero,
        zero,
        np.zeros(2),
    )


def test_edge_bvp_solve_inside_gap():
    """Riccati sweep: end conditions exact, ODE satisfied, bounded for |K01| L = 50."""
    L, n = 1000.0, 4001
    x = np.linspace(0, L, n)
    h = x[1] - x[0]
    K = np.array([[-0.001 - 0.002j, 0.05], [0.05, -0.001 + 0.002j]])
    Sp = 1e-3 * np.exp(1j * 0.01 * x)
    Sm = 2e-3 * np.exp(-1j * 0.02 * x)
    cp, cm = _edge_bvp_one_segment(K, x, Sp, Sm, 1.0 + 0j, 0.3 + 0j, 0.1j, -0.2 + 0j)
    assert abs(cp[0] - (1 + 0.3 * cm[0])) < 1e-14
    assert abs(cm[-1] - (0.1j - 0.2 * cp[-1])) < 1e-14
    c = np.vstack([cp, cm])
    d = (c[:, 2:] - c[:, :-2]) / (2 * h)
    r = d - K @ c[:, 1:-1] - np.vstack([Sp, Sm])[:, 1:-1]
    assert np.abs(r).max() < 1e-5
    assert np.abs(c).max() < 2


def test_edge_basis_tables_exact():
    """K reproduces the cascade of N lossy supercells; lossless K conserves flux."""
    from twpasolver.models import TWPA

    tw = TWPA.from_file(str(HOLOTYPE))
    for cell in tw.cells:
        cell.delta = 2e-4
    ll = lossless_copy(tw)
    f_ref, f_edge, parity = edge_reference_frequency(ll, 41.199e9, 0.10)
    assert 41.185e9 < f_edge < 41.205e9 and f_ref > f_edge and parity == -1
    f = np.array([41.244e9, f_edge, 41.174e9])  # passband, edge, gap
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


def _gain(analysis, sig, threshold, passes=80, depth=15):
    from twpasolver.cmes.solver_config import FBSolverConfig

    return analysis.gain(
        signal_freqs=np.asarray(sig),
        Is0=1e-9,
        pump=16.0,
        model="bloch",
        mode_array_config="ext2",
        fb_config=FBSolverConfig(
            passes=passes, anderson_depth=depth, edge_basis_threshold=threshold
        ),
    )


def test_edge_basis_needed_at_the_edge():
    """Without the edge basis the iteration fails at the p2s edge; with it, it converges."""
    analysis = _holotype_analysis()
    sig = [9.188, 9.196]
    native = _gain(analysis, sig, None)
    fixed = _gain(analysis, sig, 0.10)
    bad = ~native["bloch"]["converged"] | ~np.isfinite(native["gain_db"])
    assert bad.any()
    assert fixed["bloch"]["converged"].all()
    assert np.all(np.isfinite(fixed["gain_db"]))


def test_edge_basis_matches_native_basis_in_passband():
    """p2s in the passband: fixed and native bases give the same gain."""
    analysis = _holotype_analysis()
    native = _gain(analysis, [9.234], None)["gain_db"]
    fixed = _gain(analysis, [9.234], 0.09)
    p2s = fixed["mode_info"]["mode_map"]["p2s"]
    assert fixed["bloch"]["edge_flag"][0, p2s]
    assert np.allclose(fixed["gain_db"], native, atol=0.02)


def test_edge_basis_independent_of_threshold():
    """At the edge of p2s the gain barely depends on the threshold around 0.1."""
    analysis = _holotype_analysis()
    sig = [9.196]
    g10 = _gain(analysis, sig, 0.10)
    g12 = _gain(analysis, sig, 0.12)
    p2s = g10["mode_info"]["mode_map"]["p2s"]
    assert g10["bloch"]["edge_flag"][0, p2s]
    assert np.all(np.isfinite(g10["gain_db"]))
    assert np.allclose(g10["gain_db"], g12["gain_db"], atol=0.02)


def test_edge_basis_seamless_at_passband_switch():
    """Just inside the window on the passband side, the basis is the native one."""
    analysis = _holotype_analysis()
    sig = [9.2481]  # p2s splitting 0.099
    fixed = _gain(analysis, sig, 0.10)
    p2s = fixed["mode_info"]["mode_map"]["p2s"]
    assert fixed["bloch"]["edge_flag"][0, p2s]
    native = _gain(analysis, sig, None)
    assert np.allclose(fixed["gain_db"], native["gain_db"], atol=0.01)


def test_zone_centre_is_not_an_edge():
    """A low-frequency idler has a small splitting (k -> 0) but no stopband edge."""
    analysis = _holotype_analysis()
    res = analysis.gain(
        signal_freqs=np.array([15.65]),  # idler at 0.35 GHz, splitting ~0.08
        Is0=1e-9,
        pump=16.0,
        model="bloch",
        mode_array_config="ext2",
    )
    i = res["mode_info"]["mode_map"]["i"]
    assert res["bloch"]["split"][0, i] < 0.10
    assert not res["bloch"].get("edge_flag", np.zeros((1, i + 1), bool))[0, i]
    assert np.all(np.isfinite(res["gain_db"]))


# --------------------------------------------------------------------------- #
# Segmented devices
# --------------------------------------------------------------------------- #


def _random_segmented_problem(seed=3):
    rng = np.random.default_rng(seed)
    lens = np.array([200.0, 150.0, 300.0, 120.0, 230.0])
    bounds = np.r_[0, np.cumsum(lens)]
    sub = [41, 31, 61, 25, 47]
    sub_off = np.r_[0, np.cumsum(sub)].astype(np.int64)
    xs = np.concatenate(
        [np.linspace(bounds[s], bounds[s + 1], sub[s]) for s in range(len(lens))]
    )

    def rand_K(gap):
        d = rng.uniform(0.005, 0.02)
        k = rng.uniform(0.01, 0.03) if gap else rng.uniform(0.001, 0.004)
        a = rng.uniform(0, 1e-3)
        return np.array(
            [
                [-a - 1j * d, k * np.exp(1j * rng.uniform(0, 6))],
                [k * np.exp(-1j * rng.uniform(0, 6)), a + 1j * d],
            ]
        )

    n_seg = len(lens)
    K = np.array([rand_K(s % 2 == 0) for s in range(n_seg)])
    cplx = lambda scale: scale * (rng.normal(size=n_seg) + 1j * rng.normal(size=n_seg))
    t, tp = 1 + cplx(0.2), 1 + cplx(0.2)
    rL, rR = cplx(0.3), cplx(0.3)
    t[0] = tp[0] = 1
    rL[0] = rR[0] = 0
    Phi = rng.uniform(0, 6, n_seg + 1)
    return lens, bounds, sub_off, xs, K, t, tp, rL, rR, Phi


def test_segmented_bvp_matches_transfer_matrix():
    """Riccati sweep across junctions: exact against the transfer matrix, junctions exact."""
    lens, bounds, sub_off, xs, K, t, tp, rL, rR, Phi = _random_segmented_problem()
    n_seg = len(lens)
    bc = (1.0 + 0j, 0.2 - 0.1j, 0.05j, -0.3 + 0.1j)  # cp_in, rho_in, cm_out, rho_out
    z = np.zeros(len(xs), complex)
    cp, cm = _edge_bvp_solve_segmented(K, xs, sub_off, z, z, *bc, t, tp, rL, rR, Phi)

    def junction_T(s):  # (c+_L, c-_L) -> (c+_R, c-_R)
        e2 = np.exp(2j * Phi[s])
        cm_row = np.array([-rL[s] * e2 / tp[s], 1 / tp[s]])
        cp_row = np.array([t[s], 0]) + rR[s] / e2 * cm_row
        return np.vstack([cp_row, cm_row])

    M = np.eye(2, dtype=complex)
    starts = []
    for s in range(n_seg):
        if s > 0:
            M = junction_T(s) @ M
        starts.append(M.copy())
        M = expm(K[s] * lens[s]) @ M
    A = np.array([[1, -bc[1]], [M[1, 0] - bc[3] * M[0, 0], M[1, 1] - bc[3] * M[0, 1]]])
    c0 = np.linalg.solve(A, [bc[0], bc[2]])
    for s in range(n_seg):
        for j in range(sub_off[s], sub_off[s + 1]):
            ref = expm(K[s] * (xs[j] - bounds[s])) @ (starts[s] @ c0)
            assert abs(cp[j] - ref[0]) + abs(cm[j] - ref[1]) < 1e-11

    # with sources: junction relations and end conditions hold exactly
    Sp, Sm = 1e-3 * np.exp(0.01j * xs), 2e-3 * np.exp(-0.02j * xs)
    cp, cm = _edge_bvp_solve_segmented(K, xs, sub_off, Sp, Sm, *bc, t, tp, rL, rR, Phi)
    for s in range(1, n_seg):
        left, right = sub_off[s] - 1, sub_off[s]
        e2 = np.exp(2j * Phi[s])
        assert abs(cp[right] - (t[s] * cp[left] + rR[s] / e2 * cm[right])) < 1e-14
        assert abs(cm[left] - (tp[s] * cm[right] + rL[s] * e2 * cp[left])) < 1e-14
    assert abs(cp[0] - (bc[0] + bc[1] * cm[0])) < 1e-14
    assert abs(cm[-1] - (bc[2] + bc[3] * cp[-1])) < 1e-14


def test_segmented_bvp_trivial_junctions_equal_uniform():
    """Identical segments with transparent junctions reproduce a single segment."""
    _, _, _, _, K, *_ = _random_segmented_problem()
    n_seg, h, n_pts = 5, 5.0, 41
    L = h * (n_pts - 1) * n_seg
    bounds = np.linspace(0, L, n_seg + 1)
    sub_off = (np.arange(n_seg + 1) * n_pts).astype(np.int64)
    xs = np.concatenate(
        [np.linspace(bounds[s], bounds[s + 1], n_pts) for s in range(n_seg)]
    )
    src = lambda x: (1e-3 * np.exp(0.01j * x), 2e-3 * np.exp(-0.02j * x))
    bc = (1.0 + 0j, 0.2 - 0.1j, 0.05j, -0.3 + 0.1j)
    ones, zeros = np.ones(n_seg, complex), np.zeros(n_seg, complex)
    cps, _ = _edge_bvp_solve_segmented(
        np.array([K[0]] * n_seg),
        xs,
        sub_off,
        *src(xs),
        *bc,
        ones,
        ones,
        zeros,
        zeros,
        np.zeros(n_seg + 1),
    )
    xu = np.linspace(0, L, (n_pts - 1) * n_seg + 1)
    cpu, _ = _edge_bvp_one_segment(K[0], xu, *src(xu), *bc)
    ref = np.interp(xs, xu, cpu.real) + 1j * np.interp(xs, xu, cpu.imag)
    assert np.allclose(cps, ref, atol=1e-14)


def test_relative_residual_is_infinite_for_nonfinite_iterates():
    """A blown-up iterate never passes for converged."""
    from twpasolver.cmes.bloch_common import _relative_residual

    f = np.zeros((2, 5), dtype=complex)
    assert _relative_residual(f, np.ones(2)) == 0.0
    f[1, 3] = np.nan
    assert _relative_residual(f, np.ones(2)) == np.inf
    f[1, 3] = 0.0
    assert _relative_residual(f, np.array([1.0, np.inf])) == np.inf


def _segmented_holotype(scale=(1.0, 1.0, 1.0, 1.0)):
    """The holotype cut into equal segments; ``scale`` multiplies L and Lf per segment."""
    from twpasolver import TWPAnalysis
    from twpasolver.models import TWPA
    from twpasolver.modes_rwa import ModeArrayFactory

    base = TWPA.from_file(str(HOLOTYPE))
    for cell in base.cells:
        cell.delta = 2e-4
    segs = []
    for g in scale:
        seg = base.model_copy(deep=True, update={"N": base.N // len(scale)})
        for cell in seg.cells:
            cell.L, cell.Lf = cell.L * g, cell.Lf * g
        segs.append(seg)
    dev = TWPA(cells=segs, Z0_ref=base.Z0_ref, Ip0=base.Ip0)
    analysis = TWPAnalysis(twpa=dev, f_arange=(0.2, 80, 2e-3))
    analysis.update_base_data()
    analysis.add_mode_array(
        "ext2",
        ModeArrayFactory.create_extended_3wm(
            analysis.data,
            n_pump_harmonics=3,
            n_frequency_conversion=2,
            n_signal_harmonics=0,
        ),
    )
    return analysis


def test_segmented_copy_matches_uniform():
    """Four identical segments reproduce the uniform edge-basis gain at the p2s edge."""
    sig = [9.188, 9.196, 9.234]
    uniform = _gain(_holotype_analysis(), sig, 0.10)
    seg = _gain(_segmented_holotype(), sig, 0.10)
    p2s = seg["mode_info"]["mode_map"]["p2s"]
    assert seg["bloch"]["edge_flag"][:, :, p2s].all()
    assert seg["bloch"]["converged"].all()
    assert np.allclose(seg["gain_db"], uniform["gain_db"], atol=2e-3)


def test_segmented_mixed_flags():
    """p2s at its edge in some segments only: converges, weak threshold dependence."""
    analysis = _segmented_holotype(scale=(1.0, 1.002, 1.004, 1.006))
    sig = [9.152, 9.192]  # p2s flagged in segments 0-2 / 0-1
    g10 = _gain(analysis, sig, 0.10)
    g12 = _gain(analysis, sig, 0.12)
    native = _gain(analysis, sig, None)
    p2s = g10["mode_info"]["mode_map"]["p2s"]
    flags = g10["bloch"]["edge_flag"][:, :, p2s]
    assert flags.any(axis=1).all() and not flags.all(axis=1).any()
    assert g10["bloch"]["converged"].all() and g12["bloch"]["converged"].all()
    assert not native["bloch"]["converged"].all()
    assert np.allclose(g10["gain_db"], g12["gain_db"], atol=0.03)


def test_per_mode_harmonic_limits_in_filter():
    """Row limit = largest limit among its modes; uniform limits = scalar m_max."""
    from test_bloch import CELL_L, CELL_U

    from twpasolver import TWPAnalysis
    from twpasolver.analysis import prepare_relations_coefficients
    from twpasolver.cmes.bloch_common import _filter_wm_terms_bloch
    from twpasolver.models import TWPA
    from twpasolver.modes_rwa import ModeArrayFactory

    analysis = TWPAnalysis(
        twpa=TWPA(cells=[CELL_U, CELL_L, CELL_U], N=10), f_arange=(0.05, 20, 1e-2)
    )
    analysis.update_base_data()
    ma = ModeArrayFactory.create_extended_3wm(
        analysis.data, n_pump_harmonics=1, n_frequency_conversion=1
    )
    rel3, rel4, _, _ = prepare_relations_coefficients(
        ma.get_rwa_terms(2), ma.get_rwa_terms(3), 1.0, 1.0
    )
    n_modes = len(ma.modes)
    rng = np.random.default_rng(5)
    for _ in range(10):
        kappas = -rng.uniform(0.01, 0.3, n_modes)
        lim = rng.integers(0, 4, n_modes)
        full = _filter_wm_terms_bloch(
            kappas, rel3, rel4, 0.5, 66, np.full(n_modes, lim.max())
        )
        auto = _filter_wm_terms_bloch(kappas, rel3, rel4, 0.5, 66, lim)
        for rows_full, rows_auto, rels in zip(full, auto, (rel3, rel4)):

            def row_lim(row):
                modes = [m % n_modes for m in rels[row[-2]]]
                return lim[modes].max()

            expected = [tuple(r) for r in rows_full if abs(r[-1]) <= row_lim(r)]
            assert [tuple(r) for r in rows_auto] == expected


def test_auto_m_max_config():
    from twpasolver.cmes.solver_config import FBSolverConfig

    assert FBSolverConfig(m_max="auto").m_max_fixed == 1
    assert FBSolverConfig(m_max=3).m_max_fixed == 3
    assert FBSolverConfig(edge_basis_threshold=0.12).edge_threshold == 0.12
    with pytest.raises(ValueError):
        FBSolverConfig(m_max=-1)
    with pytest.raises(ValueError):
        FBSolverConfig(m_max="automatic")


def _gain_m(analysis, sig, m_max):
    from twpasolver.cmes.solver_config import FBSolverConfig

    return analysis.gain(
        signal_freqs=np.asarray(sig),
        Is0=1e-9,
        pump=16.0,
        model="bloch",
        mode_array_config="ext2",
        fb_config=FBSolverConfig(
            passes=80,
            anderson_depth=15,
            edge_threshold=0.10,
            m_max=m_max,
        ),
    )


@pytest.mark.parametrize("segmented", [False, True])
def test_auto_m_max_raises_edge_modes_only(segmented):
    """At the p2s edge only p2s gets its Bragg order; elsewhere auto = 1."""
    analysis = _segmented_holotype() if segmented else _holotype_analysis()
    sig = [6.0, 9.188]
    auto = _gain_m(analysis, sig, "auto")
    lim = auto["bloch"]["m_lim"]
    p2s = auto["mode_info"]["mode_map"]["p2s"]
    s = auto["mode_info"]["mode_map"]["s"]
    lim_p2s = lim[1, ..., p2s]
    assert np.all(lim_p2s >= 2)
    assert np.all(lim[..., s] == 1)
    # where no mode is raised, auto is the fixed run with m_max = 1
    plain = np.all(lim[0] == 1)
    fixed1 = _gain_m(analysis, sig, 1)
    if plain:
        assert np.isclose(auto["gain_db"][0], fixed1["gain_db"][0], atol=1e-9)
    # at the edge, auto agrees with the fixed run at the raised order
    fixed = _gain_m(analysis, sig, int(lim_p2s.max()))
    assert abs(auto["gain_db"][1] - fixed["gain_db"][1]) < 0.05


def test_mismatch_cutoff_capped_at_zone_edge():
    """Kept rows: |dk + G_m| < cutoff * min(k_proc, pi/d), never beyond pi/d; complete."""
    from test_bloch import CELL_L, CELL_U

    from twpasolver import TWPAnalysis
    from twpasolver.analysis import prepare_relations_coefficients
    from twpasolver.cmes.bloch_common import _filter_wm_terms_bloch
    from twpasolver.models import TWPA
    from twpasolver.modes_rwa import ModeArrayFactory

    analysis = TWPAnalysis(
        twpa=TWPA(cells=[CELL_U, CELL_L, CELL_U], N=10), f_arange=(0.05, 20, 1e-2)
    )
    analysis.update_base_data()
    ma = ModeArrayFactory.create_extended_3wm(
        analysis.data, n_pump_harmonics=1, n_frequency_conversion=1
    )
    rels = prepare_relations_coefficients(
        ma.get_rwa_terms(2), ma.get_rwa_terms(3), 1.0, 1.0
    )[:2]
    nm, d = len(ma.modes), 20
    g, k_edge = 2 * np.pi / d, np.pi / d
    rng = np.random.default_rng(2)

    def residuals(kappas, rel_tab, m_lim, cutoff):
        """All (row, variant, m) with their residual, by brute force."""
        out = []
        for r, rel in enumerate(rel_tab):
            modes = [rel[0]] + [m % nm for m in rel[1:]]
            k_proc = np.abs(kappas[modes]).max()
            thr = min(cutoff * min(k_proc, k_edge), k_edge)
            m_row = m_lim[modes].max()
            for v in range(2 ** (len(rel) - 1)):
                dk = -kappas[rel[0]]
                for q, m in enumerate(rel[1:]):
                    sign = -1.0 if m >= nm else 1.0
                    if (v >> q) & 1:
                        sign = -sign
                    dk += sign * kappas[m % nm]
                for mg in range(-m_row, m_row + 1):
                    out.append((abs(dk + mg * g), thr))
        return out

    for cutoff in (0.2, 0.6, 2.0):
        for _ in range(5):
            kappas = -rng.uniform(0.01, 0.6, nm)  # up to k d / pi ~ 4
            m_lim = rng.integers(0, 4, nm)
            kept = _filter_wm_terms_bloch(kappas, *rels, cutoff, d, m_lim)
            for rows, rel_tab in zip(kept, rels):
                expected = sum(
                    res < thr for res, thr in residuals(kappas, rel_tab, m_lim, cutoff)
                )
                assert len(rows) == expected
                assert np.all(np.abs(rows[:, -1]) <= m_lim.max())


def test_cubic_table_interpolation():
    """4-point stencil: exact for cubics, clamped like np.interp, linear on short grids."""
    from twpasolver.cmes.bloch_common import _interp_rows

    rng = np.random.default_rng(4)
    xp = np.sort(rng.uniform(0, 10, 12))
    xp[0], xp[-1] = 0.0, 10.0
    c = rng.normal(size=4) + 1j * rng.normal(size=4)
    poly = lambda x: c[0] + c[1] * x + c[2] * x**2 + c[3] * x**3  # noqa: E731
    arr = np.vstack([poly(xp), 2 * poly(xp)])
    for x in np.linspace(0, 10, 57):
        assert np.allclose(_interp_rows(arr, xp, x), [poly(x), 2 * poly(x)], atol=1e-9)
    assert np.allclose(_interp_rows(arr, xp, -1.0), arr[:, 0])
    assert np.allclose(_interp_rows(arr, xp, 11.0), arr[:, -1])
    short = np.array([0.0, 1.0, 3.0])
    vals = np.array([[1.0 + 0j, 3.0, 4.0]])
    assert np.allclose(
        _interp_rows(vals, short, 2.0), np.interp(2.0, short, vals[0].real)
    )


def test_stopband_guard_weights():
    """1 away from edges, smoothstep to 0 at the edge, 0 in the gap, 1 at the zone centre."""
    from twpasolver.analysis import _stopband_guard_weights

    split = np.array([[0.5, 0.1, 0.05, 0.0, 0.3, 0.02]])
    W = np.array([[50.0, 50.0, 50.0, 50.0, 40j, 50.0]])
    kd = np.array([[0.7, 1.0, 1.0, 1.0, 2.0, 0.1]])
    w = _stopband_guard_weights(split, W, kd, 0.1)
    assert np.allclose(w, [[1.0, 1.0, 0.5, 0.0, 0.0, 1.0]])


@pytest.mark.parametrize("segmented", [False, True])
def test_stopband_modes_switched_off(segmented):
    """simulate_stopband_modes=False: no edge basis, p2s processes off at its edge."""
    from twpasolver.cmes.solver_config import FBSolverConfig

    analysis = _segmented_holotype() if segmented else _holotype_analysis()
    res = analysis.gain(
        signal_freqs=np.array([9.188, 9.234]),
        Is0=1e-9,
        pump=16.0,
        model="bloch",
        mode_array_config="ext2",
        fb_config=FBSolverConfig(
            passes=80, anderson_depth=15, simulate_stopband_modes=False
        ),
    )
    p2s = res["mode_info"]["mode_map"]["p2s"]
    s = res["mode_info"]["mode_map"]["s"]
    w = res["bloch"]["edge_weights"]
    assert "edge_flag" not in res["bloch"]
    assert np.all(w[0, ..., p2s] < 1.0) and np.all(w[..., s] == 1.0)
    assert res["bloch"]["converged"].all()
    assert np.all(np.isfinite(res["gain_db"]))


def test_stopband_edges_match_per_frequency_search():
    """One-pass edge table: nearest edge gives the reference of edge_reference_frequency."""
    from twpasolver.cmes.bloch import stopband_edges
    from twpasolver.models import TWPA

    ll = lossless_copy(TWPA.from_file(str(HOLOTYPE)))
    rng = np.random.default_rng(6)
    # near and inside the stopbands around 27 and 41 GHz, plus passband points
    f = np.concatenate(
        [rng.uniform(26.0e9, 28.5e9, 12), rng.uniform(40.8e9, 41.6e9, 12)]
    )
    e_tab, ref_tab, par_tab = stopband_edges(ll, f, 0.10)
    for fq in f:
        q = np.argmin(np.abs(e_tab - fq))
        f_ref, f_edge, parity = edge_reference_frequency(ll, fq, 0.10)
        assert abs(e_tab[q] - f_edge) < 10.0
        assert abs(ref_tab[q] - f_ref) < 10.0
        assert par_tab[q] == parity


# --------------------------------------------------------------------------- #
# Zone-folding points: small splitting without a gap
# --------------------------------------------------------------------------- #


def test_passband_mask_ignores_touching_bands():
    """|h| = 1 up to rounding (bands touching) is passband; a real gap is not."""
    from twpasolver.cmes.bloch import STOPBAND_TOL, passband_mask

    h = np.array([0.5, 0.99, 1 + 1e-10, 0.99, 0.9, 1.05, 1.2, 1.05, 0.9, -1 - 1e-12])
    expect = np.array([1, 1, 1, 1, 1, 0, 0, 0, 1, 1], dtype=bool)
    assert np.array_equal(passband_mask(h), expect)
    assert not passband_mask(np.array([1 + 2 * STOPBAND_TOL]))[0]


def _unfolded_holotype(q=3, nl_scale=None, delta=None):
    """Holotype with one period unfolded to q copies of single cells (N_tot unchanged).

    ``nl_scale`` (length 34 q) scales the nonlinear inductance of each cell only;
    ``delta`` overrides the loss tangent.
    """
    from twpasolver.models.twpa_cells import LCLfBaseCell

    class _NLCell(LCLfBaseCell):
        nl: float = 1.0

        def bloch_nl_elements(self):
            return [(0, self.L / 2 * self.nl), (1, self.L / 2 * self.nl)]

    analysis = _holotype_analysis()
    tw = analysis.twpa
    cells = []
    for _ in range(q):
        for c in list(tw.cells):
            for _ in range(c.N):
                d = c.model_dump()
                d.pop("name")
                d["N"] = 1
                d["nl"] = 1.0 if nl_scale is None else nl_scale[len(cells)]
                if delta is not None:
                    d["delta"] = delta
                cells.append(_NLCell(**d))
    tw.cells[:] = cells
    tw.N = tw.N // q
    analysis.update_base_data()
    return analysis


def test_folding_point_is_not_an_edge():
    """On an unfolded supercell, k D = n pi without a gap keeps the native basis."""
    from twpasolver.analysis import _edge_basis_entries

    analysis = _unfolded_holotype()
    tw, D = analysis.twpa, 102
    k = lambda f: np.interp(f, analysis.data["freqs"], analysis.data["k"])
    # 22.736 GHz: k D = 5 pi, the 34-cell line has no gap there.
    # 26.25 GHz: passband side of the real gap starting at 26.253 GHz.
    f = np.array([[22.736, 26.25]]) * 1e9
    T = supercell_matrix(lossless_copy(tw), f.ravel())
    h = 0.5 * (T[:, 0, 0] + T[:, 1, 1]).real
    split = np.sqrt(np.abs(1 - h**2))[None, :]
    assert np.all(split < 0.10)
    assert abs(k(22.736) * D / np.pi - 5) < 0.01
    _, flag, ent = _edge_basis_entries(
        tw, split, k(f / 1e9), f, D, np.array([False, False]), 0.10
    )
    assert not flag[0, 0] and flag[0, 1]
    assert list(ent["outside"]) == [0]


def test_folding_point_modes_match_native_basis():
    """Unfolded holotype: modes at folding points give the native-basis gain exactly."""
    analysis = _unfolded_holotype()
    from twpasolver.modes_rwa import ModeArrayFactory

    analysis.add_mode_array(
        "ext2",
        ModeArrayFactory.create_extended_3wm(
            analysis.data,
            n_pump_harmonics=3,
            n_frequency_conversion=2,
            n_signal_harmonics=0,
        ),
    )
    sig = [3.75, 13.75]  # p2s and pi at folding points (split < 0.1, no gap)
    fixed = _gain(analysis, sig, 0.10)
    native = _gain(analysis, sig, None)
    assert not fixed["bloch"].get("edge_flag", np.zeros(1, bool)).any()
    assert np.allclose(fixed["gain_db"], native["gain_db"], atol=1e-9)


def test_contra_qpm_survives_edge_basis():
    """Backward conversion phase matched by a nonlinearity-only grating.

    The converted tone sits on the folding point k_c D = 5 pi by construction;
    the default edge threshold must leave it in the native basis.
    """
    from twpasolver.cmes.solver_config import FBSolverConfig
    from twpasolver.modes_rwa import ModeArrayFactory

    D, fp, Ip, Is = 102, 18.3, 1e-4, 1e-9
    eta = 1 + 0.5 * np.cos(2 * np.pi * 5 * (np.arange(D) + 0.5) / D)
    analysis = _unfolded_holotype(nl_scale=eta, delta=0.0)
    analysis.add_mode_array(
        "c",
        ModeArrayFactory.create_custom(
            analysis.data, ["p", "s", "ps"], [["ps", "p+s"]]
        ),
    )
    out = {}
    for th in (0.10, None):
        r = analysis.gain(
            signal_freqs=np.array([4.5335]),
            pump=fp,
            model="bloch",
            mode_array_config="c",
            initial_amplitudes_fwd=[Ip, Is, 0],
            initial_amplitudes_bwd=[0, 0, 0],
            fb_config=FBSolverConfig(
                passes=100, anderson_depth=20, m_max=6, edge_threshold=th
            ),
        )
        c = r["mode_info"]["mode_map"]["ps"]
        out[th] = np.abs(r["I_tuples_bwd"][0, c, -1] / Is) ** 2 * 4.5335 / (4.5335 + fp)
    assert out[None] > 0.8
    assert abs(out[0.10] - out[None]) < 1e-9
