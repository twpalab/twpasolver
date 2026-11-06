"""Tests for the Bloch-basis CMEs (``bloch`` model) and the ``simple`` model."""

import numpy as np
import pytest

from twpasolver.analysis import GainModel, TWPAnalysis, _drop_segment_axis
from twpasolver.cmes.bloch import bloch_mode_tables
from twpasolver.cmes.bloch_fb import bloch_fb_segmented_cmes_solve
from twpasolver.cmes.solver_config import BlochSolverConfig
from twpasolver.models import TWPA, LCLfBaseCell

CELL_U = {"name": "LCLfBaseCell", "N": 30, "L": 49.9e-12, "C": 24.2e-15, "Lf": 0.87e-9}
CELL_L = {"name": "LCLfBaseCell", "N": 6, "L": 48.6e-12, "C": 8.2e-15, "Lf": 0.28e-9}


def _forward_gamma(twpa, freqs, n_cells):
    T = np.asarray(twpa.single_abcd(freqs)).astype(complex)
    tr = T[:, 0, 0] + T[:, 1, 1]
    gamma = np.log((tr + np.sqrt(tr**2 - 4 + 0j)) / 2) / n_cells
    return np.where(gamma.imag < 0, -gamma, gamma)


def _segment_bloch_tables(analysis, mode_array, cme_inputs):
    """Bloch tables of a uniform device solved as one segment."""
    ci = {**cme_inputs, "gammas": cme_inputs["gammas"][..., None]}
    return analysis._prepare_bloch_segmented(mode_array, ci)


def _uniform_bloch_tables(analysis, mode_array, cme_inputs):
    """Bloch tables of a uniform device, without the segment axis."""
    tables = _segment_bloch_tables(analysis, mode_array, cme_inputs)
    return _drop_segment_axis(tables, cme_inputs["n_freq"])


def _solve_one_segment(x, y0_fwd, y0_bwd, ci, sg, **kwargs):
    """``bloch_fb_segmented_cmes_solve`` on the one-segment tables ``sg``."""
    return bloch_fb_segmented_cmes_solve(
        x,
        sg["seg_bounds"],
        y0_fwd,
        y0_bwd,
        ci["gammas"][:, None],
        ci["relations_3wm"],
        ci["relations_4wm"],
        sg["base_3wm"],
        sg["base_4wm"],
        sg["prefac"],
        sg["phi_plus"],
        sg["phi_minus"],
        sg["L_nl"],
        sg["positions"],
        sg["n_cells"],
        sg["Zb_plus"],
        sg["Zb_minus"],
        **kwargs,
    )


@pytest.mark.parametrize("n_cells, centered", [(1, False), (5, False), (3, True)])
def test_uniform_ladder_limit(n_cells, centered):
    """For a homogeneous ladder the Bloch prefactor is gamma*tanh(gamma/2)/(gamma/2)."""
    L = 49.9e-12
    twpa = TWPA(
        cells=[LCLfBaseCell(N=n_cells, centered=centered, L=L, C=24.2e-15, Lf=0.87e-9)],
        N=100,
    )
    freqs = np.linspace(1e9, 15e9, 8)
    tab = bloch_mode_tables(twpa, freqs, _forward_gamma(twpa, freqs, n_cells))
    gamma = tab["gamma_exact"]

    assert np.allclose(np.abs(tab["phi_plus"]), 1, atol=1e-9)
    assert np.all(tab["eig_residual"] < 1e-9)
    eta = (tab["L_nl"] / L * tab["phi_minus"] * tab["phi_plus"]).sum(-1) / n_cells
    ratio = 2j * 2 * np.pi * freqs * L / tab["W"] * eta / gamma
    assert np.allclose(ratio, np.tanh(gamma / 2) / (gamma / 2), rtol=1e-9)


def _mismatched_analysis(Ip0, Z_ext=70.0, N=100):
    twpa = TWPA(
        cells=[CELL_U, CELL_L, CELL_U],
        N=N,
        Istar=6.3e-3,
        Idc=1e-3,
        Ip0=Ip0,
        Z0_ref=Z_ext,
    )
    analysis = TWPAnalysis(twpa=twpa, f_arange=(0.05, 20, 1e-3), port_Z_ext=Z_ext)
    analysis.update_base_data()
    return analysis


def test_phase_matching_optimal_pump():
    analysis = _mismatched_analysis(Ip0=1e-12)
    pm = analysis.phase_matching(
        process="PA",
        mixing="3WM",
        signal_arange=(2.0, 6.5, 0.25),
        pump_arange=(14.25, 18.0, 0.25),
    )
    best = pm["optimal_pump_freq"]
    assert best in pm["pump_freqs"]
    mean = np.nanmean(np.abs(pm["delta"]), axis=0)
    assert best == pm["pump_freqs"][np.argmin(mean)]


def test_phase_matching_masks_stopband_pumps():
    """Pumps inside a stopband get NaN mismatch and are never the optimum."""
    analysis = _mismatched_analysis(Ip0=1e-12)
    pm = analysis.phase_matching(
        process="PA",
        mixing="3WM",
        signal_arange=(2.0, 6.5, 0.25),
        pump_arange=(12.5, 16.0, 0.05),
    )
    sb = analysis.data["stopbands"]
    pumps = pm["pump_freqs"]
    in_gap = ((pumps[:, None] >= sb[:, 0]) & (pumps[:, None] <= sb[:, 1])).any(axis=1)
    assert in_gap.any() and (~in_gap).any()
    assert np.isnan(pm["delta"][:, in_gap]).all()
    assert not in_gap[pumps == pm["optimal_pump_freq"]].any()


def test_fb_linear_limit_matches_abcd():
    """Without pump, the Bloch FB model reproduces the exact device S-parameters."""
    analysis = _mismatched_analysis(Ip0=1e-12)
    freqs = analysis.data["freqs"]
    idx = np.searchsorted(freqs, [2.0, 6.0, 10.0, 12.0])
    res = analysis.gain(
        freqs[idx],
        pump=16.0,
        Is0=1e-6,
        model=GainModel.BLOCH,
        bloch_config=BlochSolverConfig(passes=40, convergence_threshold=1e-8),
        thin=20,
    )
    s21 = 20 * np.log10(np.abs(analysis.data["S21"][idx]))
    s11 = 20 * np.log10(np.abs(analysis.data["S11"][idx]))
    assert np.allclose(res["gain_db"], s21, atol=1e-3)
    assert np.allclose(res["gain_db_11"], s11, atol=1e-2)


def _connector(z0, length, L_series, C_shunt, R_series):
    """Lossy connector model: line section, series L, shunt C, series R."""
    from twpasolver.models.oneport import Capacitance, Inductance, Resistance
    from twpasolver.models.transmission_lines import LosslessTL
    from twpasolver.models.twoportarrays import TwoPortArray

    return TwoPortArray(
        cells=[
            LosslessTL.from_z_vp(z0, 1.2e8, length),
            Inductance(L=L_series),
            Capacitance(C=C_shunt, twoport_parallel=True),
            Resistance(R=R_series),
        ]
    )


@pytest.mark.parametrize("segmented", [False, True])
def test_fb_linear_limit_with_connectors(segmented):
    """
    Without pump, the Bloch FB model with different connectors at the two ends
    reproduces all four S-parameters of the cascade connector-device-connector.

    Both connectors have port 2 facing the device, so the output one enters the
    cascade reversed (A and D swapped). S12 and S22 come from a signal injected
    at x = L.
    """
    Ze = 50.0
    conn_in = _connector(45.0, 3e-3, 0.4e-9, 60e-15, 1.5)
    conn_out = _connector(58.0, 5e-3, 0.2e-9, 120e-15, 0.5)
    base = TWPA.from_file(str(HOLOTYPE))
    base.Ip0 = 1e-12
    if segmented:
        segs = []
        for g in (1.0, 1.01, 0.99, 1.02):
            seg = base.model_copy(deep=True, update={"N": base.N // 4})
            for cell in seg.cells:
                cell.L, cell.Lf = cell.L * g, cell.Lf * g
            segs.append(seg)
        device = TWPA(cells=segs, Z0_ref=base.Z0_ref, Ip0=base.Ip0)
    else:
        device = base
    analysis = TWPAnalysis(
        twpa=device,
        f_arange=(0.2, 40, 1e-3),
        port_Z_ext=Ze,
        port=conn_in,
        port_out=conn_out,
    )
    analysis.update_base_data()
    idx = np.searchsorted(analysis.data["freqs"], [2.0, 5.0, 8.0, 11.0, 12.5])
    f = analysis.data["freqs"][idx]

    T_in = np.asarray(conn_in.get_abcd(f * 1e9))
    T_out = np.asarray(conn_out.get_abcd(f * 1e9))[:, ::-1, ::-1].copy()
    T_out[:, 0, 1], T_out[:, 1, 0] = T_out[:, 1, 0].copy(), T_out[:, 0, 1].copy()
    T = T_in @ np.asarray(analysis.data["abcd"])[idx] @ T_out
    A, B, C, D = T[:, 0, 0], T[:, 0, 1], T[:, 1, 0], T[:, 1, 1]
    den = A + B / Ze + C * Ze + D
    ref = {
        "21": 2 / den,
        "11": (A + B / Ze - C * Ze - D) / den,
        "12": 2 * (A * D - B * C) / den,
        "22": (-A + B / Ze - C * Ze + D) / den,
    }

    kw = dict(
        pump=16.0,
        model=GainModel.BLOCH,
        bloch_config=BlochSolverConfig(passes=60, convergence_threshold=1e-10),
        thin=20,
    )
    fwd = analysis.gain(f, Is0=1e-6, **kw)
    labels = list(fwd["mode_info"]["mode_map"])
    y_f, y_b = np.zeros(len(labels)), np.zeros(len(labels))
    y_f[labels.index("p")] = base.Ip0
    y_b[labels.index("s")] = 1e-6
    bwd = analysis.gain(f, initial_amplitudes_fwd=y_f, initial_amplitudes_bwd=y_b, **kw)
    sim = {
        "21": fwd["gain_db"],
        "11": fwd["gain_db_11"],
        "12": bwd["gain_db_12"],
        "22": bwd["gain_db_22"],
    }
    for k, tol in (("21", 1e-3), ("12", 1e-3), ("11", 1e-2), ("22", 1e-2)):
        assert np.allclose(sim[k], 20 * np.log10(np.abs(ref[k])), atol=tol), k


@pytest.mark.parametrize("m_max", [0, 5])
def test_fb_energy_balance_lossless(m_max):
    """Lossless FB solution: net power flux sum W(|A+|^2 - |A-|^2) equal at both ends.

    With m_max = 0 the forward and backward pump only exchange phase, and the
    balance holds to a small fraction of the signal power. With m_max > 0 the
    quasi-phase-matched (Bragg-type) terms exchange power between forward and
    backward pump through their relative phase, so the pass-to-pass noise of
    the forward-backward iteration shows up at the level of the total power
    (~1e-7); the equations themselves conserve energy exactly.
    """
    Z_ext, Ip0 = 70.0, 2.5e-4
    analysis = _mismatched_analysis(Ip0=Ip0, Z_ext=Z_ext)
    signal_freqs = np.array([1.75, 2.0, 12.5])
    mode_array = analysis.get_mode_array("basic_3wm")
    ci = analysis._prepare_cme_inputs(mode_array, signal_freqs, 14.3, "s", "p")
    sg = _segment_bloch_tables(analysis, mode_array, ci)
    bl = _drop_segment_axis(sg, ci["n_freq"])
    labels = ci["mode_labels"]
    p, s = labels.index("p"), labels.index("s")
    n = len(signal_freqs)
    y0 = np.zeros((n, len(labels)), dtype=complex)
    y0[:, p], y0[:, s] = Ip0, 1e-6
    x = np.linspace(0, analysis.twpa.N_tot, 500)
    out = _solve_one_segment(
        x,
        y0,
        0 * y0,
        ci,
        sg,
        passes=60,
        convergence_threshold=1e-8,
        Ze=Z_ext,
        m_max=m_max,
        anderson_depth=5,
    )
    W = bl["W"].real
    Zp, Zm = bl["Zb_plus"], bl["Zb_minus"]
    rho_in = (Zm - Z_ext) / (Zp + Z_ext)
    rho_out = (Zp - Z_ext) / (Zm + Z_ext)
    # Exact integration endpoints (interior points are interpolated).
    fwd_0, bwd_L = out[:n, :, 0], out[n:, :, 0]
    fwd_L = -out[:n, :, -1] / (1 + rho_out)
    bwd_0 = out[n:, :, -1] / (1 + rho_in)
    P0 = (W * (np.abs(fwd_0) ** 2 - np.abs(bwd_0) ** 2)).sum(1)
    PL = (W * (np.abs(fwd_L) ** 2 - np.abs(bwd_L) ** 2)).sum(1)
    P_signal = W[:, s] * np.abs(fwd_L[:, s]) ** 2
    if m_max == 0:
        assert np.all(np.abs(PL - P0) < 1e-4 * P_signal)
    else:
        assert np.all(np.abs(PL - P0) < 1e-6 * np.abs(P0))


@pytest.mark.parametrize("model", ["simple", "general_no_reflections"])
def test_backward_inputs_rejected_without_reflections(model):
    """The simple model propagates only +x waves: backward inputs must raise."""
    twpa = TWPA(cells=[CELL_U, CELL_L, CELL_U], N=10, Ip0=2.5e-4)
    analysis = TWPAnalysis(twpa=twpa, f_arange=(0.05, 20, 1e-2))
    analysis.update_base_data()
    with pytest.raises(ValueError, match="initial_amplitudes_bwd"):
        analysis.gain(
            [2.0], pump=14.3, model=model, initial_amplitudes_bwd=[2.5e-4, 0.0, 0.0]
        )


def test_mode_array_without_relations():
    """A ModeArray with only independent modes builds (used to hang)."""
    from twpasolver.modes_rwa import ModeArrayFactory

    twpa = TWPA(cells=[CELL_U, CELL_L, CELL_U], N=10)
    analysis = TWPAnalysis(twpa=twpa, f_arange=(0.05, 20, 1e-2))
    analysis.update_base_data()
    mode_array = ModeArrayFactory.create_custom(analysis.data, ["s", "c"], [])
    assert mode_array.independent_modes == {"s", "c"}


def _process_signatures(rows, n_modes, n_in):
    """Physical process of every (row, solve direction) of a filtered row set.

    A row ``target <- inputs`` in the solve along direction ``d`` stands for the
    Hamiltonian term ``conj(target) * prod(inputs)`` times the Fourier
    component ``m`` of the coupling profile; its partners (each field of the
    term taken as target) describe the same process. The signature is the
    multiset of (mode, conjugated, direction) of the term plus the physical
    ``m`` (``-m`` in the mirrored backward solve), up to overall complex
    conjugation (which flips every conjugation flag and ``m``).
    """
    signatures = []
    for row in rows:
        m_row = int(row[n_in + 2]) if len(row) > n_in + 2 else 0
        for solve_dir in (+1, -1):
            elems = [(int(row[0]), True, solve_dir)]
            for q in range(n_in):
                idx = int(row[1 + q])
                block, mode = divmod(idx, n_modes)
                direction = solve_dir if block < 2 else -solve_dir
                elems.append((mode, block in (1, 3), direction))
            m_phys = m_row * solve_dir
            sig = (tuple(sorted(elems)), m_phys)
            flipped = (tuple(sorted((m, not c, d) for m, c, d in elems)), -m_phys)
            signatures.append(min(sig, flipped))
    return signatures


def test_mismatch_filter_keeps_partner_terms_together():
    """Partner terms of a process are always kept or dropped together."""
    from twpasolver.analysis import prepare_relations_coefficients
    from twpasolver.cmes.bloch_common import _filter_wm_terms_bloch
    from twpasolver.modes_rwa import ModeArrayFactory

    twpa = TWPA(cells=[CELL_U, CELL_L, CELL_U], N=10)
    analysis = TWPAnalysis(twpa=twpa, f_arange=(0.05, 20, 1e-2))
    analysis.update_base_data()
    mode_array = ModeArrayFactory.create_extended_3wm(
        analysis.data, n_pump_harmonics=1, n_frequency_conversion=1
    )
    rel3, rel4, _, _ = prepare_relations_coefficients(
        mode_array.get_rwa_terms(2), mode_array.get_rwa_terms(3), 1.0, 1.0
    )
    n_modes = len(mode_array.modes)
    rng = np.random.default_rng(1)

    def run(kappas, cutoff):
        return _filter_wm_terms_bloch(
            kappas, rel3, rel4, cutoff, 66, np.full(len(kappas), 5)
        )

    for _ in range(20):
        kappas = -rng.uniform(0.01, 0.3, n_modes)
        for cutoff in (0.05, 0.2, 0.5):
            for rows_all, rows_kept, n_in in zip(
                run(kappas, 1e9), run(kappas, cutoff), (2, 3)
            ):
                all_sigs = _process_signatures(rows_all, n_modes, n_in)
                kept_sigs = _process_signatures(rows_kept, n_modes, n_in)
                # every physical row of a kept process must be kept
                for sig in set(kept_sigs):
                    assert all_sigs.count(sig) == kept_sigs.count(sig)


def test_bloch_rhs_conserves_energy_with_reciprocal_vectors():
    """Lossless Bloch FB right-hand sides conserve the net power flux exactly.

    Random forward and backward fields at a random position: the nonlinear part
    of d/dx sum_n W_n (|A+_n|^2 - |A-_n|^2), summed over the kept rows of both
    solves (including G != 0 rows), must vanish.
    """
    from twpasolver.cmes.bloch_common import _bloch_row_coeffs, _filter_wm_terms_bloch
    from twpasolver.cmes.bloch_fb import bloch_fb_segment_rhs

    analysis = _mismatched_analysis(Ip0=2.5e-4, Z_ext=50.0, N=100)
    mode_array = analysis.get_mode_array("basic_3wm")
    ci = analysis._prepare_cme_inputs(mode_array, np.array([12.5]), 14.3, "s", "p")
    bl = _uniform_bloch_tables(analysis, mode_array, ci)
    kappas = np.ascontiguousarray(ci["gammas"][0].imag)
    W = bl["W"][0].real
    L = float(analysis.twpa.N_tot)
    rows3, rows4 = _filter_wm_terms_bloch(
        kappas,
        ci["relations_3wm"],
        ci["relations_4wm"],
        0.2,
        bl["n_cells"],
        np.full(len(kappas), 5),
    )
    assert np.any(rows4[:, -1] != 0)  # G != 0 terms are present
    g_unit = 2 * np.pi / bl["n_cells"]
    args = (
        bl["prefac"][0],
        np.ascontiguousarray(bl["phi_plus"][0]),
        np.ascontiguousarray(bl["phi_minus"][0]),
        bl["L_nl"],
        bl["positions"],
        bl["n_cells"],
    )
    coeffs = {
        (k, fwd): _bloch_row_coeffs(rows, bl[f"base_{k}"], *args, fwd)
        for k, rows in (("3wm", rows3), ("4wm", rows4))
        for fwd in (True, False)
    }
    rng = np.random.default_rng(3)
    grid = np.array([0.0, L])
    for _ in range(5):
        Bf = (rng.normal(size=3) + 1j * rng.normal(size=3)) * 1e-4
        Bb = (rng.normal(size=3) + 1j * rng.normal(size=3)) * 1e-4
        x = rng.uniform(0, L)
        total = 0.0
        for fwd, own, ctr, pos in ((True, Bf, Bb, x), (False, Bb, Bf, L - x)):
            rhs = bloch_fb_segment_rhs(
                pos,
                own,
                np.zeros(3),
                kappas,
                rows3,
                rows4,
                coeffs[("3wm", fwd)],
                coeffs[("4wm", fwd)],
                rows3[:, -1] * g_unit,
                rows4[:, -1] * g_unit,
                0.0,  # segment start x0 and its phase: one segment from x = 0
                np.zeros(3),
                grid,
                np.ascontiguousarray(np.stack([ctr, ctr], -1)),
                0.0,
                np.zeros(3),
                L,
            )
            total += np.sum(W * 2 * np.real(np.conj(own) * rhs))
        # largest single-row contribution sets the scale of the cancellation
        scale = (
            np.sum(W * (np.abs(Bf) ** 2 + np.abs(Bb) ** 2))
            * np.max(np.abs(coeffs[("4wm", True)]))
            * np.max(np.abs(np.concatenate([Bf, Bb]))) ** 2
        )
        assert abs(total) < 1e-9 * scale


def _circuit_reference_3wm(twpa, pump, signal_freqs, Ip0, Ze):
    """
    Cell-by-cell boundary-value reference for 3WM gain with reflections.

    Undepleted pump with its own standing wave; signal and idler are solved
    exactly through every unit cell (no envelope approximation), and both
    end conditions are imposed with one linear solve.
    """
    from twpasolver.cmes.bloch import expand_unit_cells

    eps = twpa.epsilon
    runs = expand_unit_cells(twpa)
    sig = np.atleast_1d(signal_freqs)
    n = len(sig)
    fp, fs = pump * 1e9, sig * 1e9
    fi = fp - fs
    wp, ws, wi = 2 * np.pi * fp, 2 * np.pi * fs, 2 * np.pi * fi

    def shunt(c, f):
        return np.asarray(c.single_abcd(np.atleast_1d(f))).astype(complex)[:, 1, 0]

    Yp = {id(c): shunt(c, fp)[0] for c, _ in runs}
    Ys = {id(c): shunt(c, fs) for c, _ in runs}
    Yi = {id(c): np.conj(shunt(c, fi)) for c, _ in runs}
    Tp = np.asarray(twpa.single_abcd(np.array([fp]))).astype(complex)[0]
    psi0 = np.linalg.matrix_power(Tp, twpa.N) @ np.array([Ze, 1.0])
    psi0 *= Ip0 / ((psi0[0] / Ze + psi0[1]) / 2)
    Vp, Ipp = psi0
    # states: particular (unit incident signal), two homogeneous solutions
    st = np.zeros((3, 4, n), complex)
    st[0, 0], st[0, 1] = Ze, 1.0
    st[1, 0], st[1, 1] = -Ze, 1.0
    st[2, 2], st[2, 3] = -Ze, 1.0
    for _ in range(twpa.N):
        for c, cnt in runs:
            for _ in range(cnt):
                Ipp = Ipp - Yp[id(c)] * Vp
                a = Ipp
                Vp = Vp - 1j * wp * c.L * a
                st[:, 1] -= Ys[id(c)] * st[:, 0]
                st[:, 3] -= Yi[id(c)] * st[:, 2]
                b, ct = st[:, 1].copy(), st[:, 3].copy()
                st[:, 0] -= 1j * ws * c.L * (b + eps / 2 * a * ct)
                st[:, 2] += 1j * wi * c.L * (ct + eps / 2 * np.conj(a) * b)
    res = st[:, 0] - Ze * st[:, 1], st[:, 2] - Ze * st[:, 3]
    gain = np.empty(n)
    for k in range(n):
        M = np.array([[res[0][1, k], res[0][2, k]], [res[1][1, k], res[1][2, k]]])
        u, w = np.linalg.solve(M, -np.array([res[0][0, k], res[1][0, k]]))
        gain[k] = 20 * np.log10(abs(st[0, 1, k] + u * st[1, 1, k] + w * st[2, 1, k]))
    return gain


def test_anderson_converges_near_bragg_coupling():
    """
    Near a stopband the forward signal is Bragg-coupled to the backward idler
    (G != 0 term). With strong coupling the plain forward/backward iteration
    diverges (its pass map has an eigenvalue outside the unit circle) and
    settles on a spurious pump-depleted state; Anderson acceleration converges
    to the circuit result.
    """
    Ze, Ip0, Is0, pump = 50.0, 2.5e-4, 1e-8, 14.3
    twpa = TWPA(
        cells=[CELL_U, CELL_L, CELL_U],
        N=300,
        Istar=6.3e-3,
        Idc=1e-3,
        Ip0=Ip0,
        Z0_ref=Ze,
    )
    analysis = TWPAnalysis(twpa=twpa, f_arange=(0.05, 20, 1e-3), port_Z_ext=Ze)
    analysis.update_base_data()
    sig = np.array([7.46, 7.48, 7.50, 7.52])
    mode_array = analysis.get_mode_array("basic_3wm")
    ci = analysis._prepare_cme_inputs(mode_array, sig, pump, "s", "p")
    sg = _segment_bloch_tables(analysis, mode_array, ci)
    sg["base_4wm"] = 0 * sg["base_4wm"]
    labels = ci["mode_labels"]
    ip, is_ = labels.index("p"), labels.index("s")
    x = np.linspace(0, twpa.N_tot, 400)
    y0f = np.zeros((len(sig), 3), complex)
    y0f[:, ip], y0f[:, is_] = Ip0, Is0
    y0b = np.zeros_like(y0f)

    def solve(depth, passes):
        out = _solve_one_segment(
            x,
            y0f,
            y0b,
            ci,
            sg,
            passes=passes,
            convergence_threshold=1e-6,
            Ze=Ze,
            m_max=5,
            anderson_depth=depth,
        )
        return 20 * np.log10(np.abs(out[: len(sig), is_, -1]) / Is0)

    ref = _circuit_reference_3wm(twpa, pump, sig, Ip0, Ze)
    accelerated = solve(5, 30)
    assert np.allclose(accelerated, ref, atol=0.25)


def test_result_does_not_depend_on_anderson_depth():
    """Acceleration changes only the path to the fixed point, not the result."""
    analysis = _mismatched_analysis(Ip0=1.5e-4)
    freqs = analysis.data["freqs"]
    idx = np.searchsorted(freqs, [2.0, 6.0, 10.0])
    kw = dict(pump=14.3, Is0=1e-8, model=GainModel.BLOCH, thin=20)
    shallow = analysis.gain(
        freqs[idx],
        bloch_config=BlochSolverConfig(
            passes=60, convergence_threshold=1e-8, anderson_depth=1
        ),
        **kw,
    )
    acc = analysis.gain(
        freqs[idx],
        bloch_config=BlochSolverConfig(
            passes=60, convergence_threshold=1e-8, anderson_depth=5
        ),
        **kw,
    )
    assert np.allclose(acc["gain_db"], shallow["gain_db"], atol=1e-3)
    assert np.allclose(acc["gain_db_11"], shallow["gain_db_11"], atol=1e-2)


def test_nan_iterate_stops_unconverged():
    """A NaN pass ends the iteration at once and is never flagged converged."""
    analysis = _mismatched_analysis(Ip0=1.5e-4)
    freqs = analysis.data["freqs"]
    idx = np.searchsorted(freqs, [2.0, 6.0])
    res = analysis.gain(
        freqs[idx],
        pump=14.3,
        Is0=np.nan,
        model=GainModel.BLOCH,
        bloch_config=BlochSolverConfig(passes=30, save_all_passes=True),
        thin=20,
    )
    assert not res["bloch"]["converged"].any()
    assert np.isinf(res["bloch"]["residuals"]).all()


@pytest.mark.parametrize("model", ["simple", "bloch"])
def test_rk_config_step_cap_is_live_and_reported(caplog, model):
    """RKSolverConfig reaches the cached kernels; a capped run is reported."""
    from twpasolver.cmes.solver_config import RKSolverConfig
    from twpasolver.logger import log

    analysis = _mismatched_analysis(Ip0=1.5e-4)
    kw = dict(pump=14.3, Is0=1e-8, model=model, thin=20)
    ok = analysis.gain(np.array([6.0]), **kw)
    assert np.isfinite(ok["gain_db"]).all()
    log.addHandler(caplog.handler)
    try:
        capped = analysis.gain(
            # 2 steps (at most 10 + 100 cells) cannot cross the 6600-cell device
            np.array([6.0]),
            rk_config=RKSolverConfig(max_steps_per_cell=1e-6),
            **kw,
        )
    finally:
        log.removeHandler(caplog.handler)
    assert np.isnan(capped["gain_db"]).all()
    expected = "step cap" if model == "simple" else "did not converge"
    assert any(expected in r.getMessage() for r in caplog.records)


def test_unconverged_points_are_reported(caplog):
    """A run that stops before convergence logs a warning with the indices."""
    analysis = _mismatched_analysis(Ip0=1.5e-4)
    freqs = analysis.data["freqs"]
    idx = np.searchsorted(freqs, [2.0, 6.0])
    from twpasolver.logger import log

    log.addHandler(caplog.handler)
    try:
        analysis.gain(
            freqs[idx],
            pump=14.3,
            Is0=1e-8,
            model=GainModel.BLOCH,
            bloch_config=BlochSolverConfig(passes=1),
            thin=20,
        )
    finally:
        log.removeHandler(caplog.handler)
    assert any("did not converge" in r.getMessage() for r in caplog.records)


# --------------------------------------------------------------------------- #
# Stopband-edge guard
# --------------------------------------------------------------------------- #

# Frozen copy of the docs holotype (V3 KI-TWPA), so that editing the docs model
# does not move the frequencies hard-coded in the tests.
HOLOTYPE = (
    __import__("pathlib").Path(__file__).resolve().parent
    / "data"
    / "imskit_v3_holotype.json"
)


def test_reactive_angle():
    """Reactive angle of W: 0 for real W, 90 deg for imaginary W, sign-independent."""
    from twpasolver.cmes.bloch import reactive_angle_deg

    ang = np.linspace(0, 90, 181)
    W = 50 * np.exp(-1j * np.radians(ang))
    assert np.allclose(reactive_angle_deg(W), ang)
    assert np.allclose(reactive_angle_deg(-W), ang)  # sign of W is irrelevant


def test_lossless_passband_W_is_real_for_asymmetric_cells():
    """W (unlike Zb) is real in a lossless passband for any supercell boundary."""
    twpa = TWPA(cells=[CELL_U, CELL_L], N=50, Istar=6.3e-3, Idc=1e-3, Ip0=1e-4)
    analysis = TWPAnalysis(twpa=twpa, f_arange=(0.05, 20, 1e-3))
    analysis.update_base_data()
    f = np.array([3.0, 6.0, 9.0])
    k = np.interp(f, analysis.data["freqs"], analysis.data["k"])
    tab = bloch_mode_tables(twpa, f * 1e9, 1j * k)
    assert np.all(np.abs(tab["Zb_plus"].imag) > 1.0)  # Zb alone is reactive
    assert np.all(np.abs(np.angle(tab["W"])) < 1e-8)


@pytest.mark.parametrize("perturb_one_row", [False, True])
def test_process_weights_preserve_energy_identity(perturb_one_row):
    """
    Scaling every process by the product of its mode weights keeps the exact
    lossless energy identity of the FB right-hand sides (including G != 0
    rows); scaling a single row without its partners breaks it.
    """
    from twpasolver.cmes.bloch import process_weights
    from twpasolver.cmes.bloch_common import _bloch_row_coeffs, _filter_wm_terms_bloch
    from twpasolver.cmes.bloch_fb import bloch_fb_segment_rhs

    analysis = _mismatched_analysis(Ip0=2.5e-4, Z_ext=50.0, N=100)
    mode_array = analysis.get_mode_array("basic_3wm")
    ci = analysis._prepare_cme_inputs(mode_array, np.array([12.5]), 14.3, "s", "p")
    bl = _uniform_bloch_tables(analysis, mode_array, ci)
    kappas = np.ascontiguousarray(ci["gammas"][0].imag)
    W = bl["W"][0].real
    L = float(analysis.twpa.N_tot)
    rows3, rows4 = _filter_wm_terms_bloch(
        kappas,
        ci["relations_3wm"],
        ci["relations_4wm"],
        0.2,
        bl["n_cells"],
        np.full(len(kappas), 5),
    )
    rng = np.random.default_rng(11)
    mode_w = rng.uniform(0.1, 1.0, size=(1, 3))
    pw = {
        "3wm": process_weights(ci["relations_3wm"], mode_w)[0][rows3[:, 3]],
        "4wm": process_weights(ci["relations_4wm"], mode_w)[0][rows4[:, 4]],
    }
    if perturb_one_row:
        pw["3wm"] = pw["3wm"].copy()
        pw["3wm"][0] *= 0.5
    g_unit = 2 * np.pi / bl["n_cells"]
    args = (
        bl["prefac"][0],
        np.ascontiguousarray(bl["phi_plus"][0]),
        np.ascontiguousarray(bl["phi_minus"][0]),
        bl["L_nl"],
        bl["positions"],
        bl["n_cells"],
    )
    rows = {"3wm": rows3, "4wm": rows4}
    coeffs = {
        (k, fwd): _bloch_row_coeffs(rows[k], bl[f"base_{k}"], *args, fwd) * pw[k]
        for k in ("3wm", "4wm")
        for fwd in (True, False)
    }
    grid = np.array([0.0, L])
    worst = 0.0
    for _ in range(5):
        Bf = (rng.normal(size=3) + 1j * rng.normal(size=3)) * 1e-4
        Bb = (rng.normal(size=3) + 1j * rng.normal(size=3)) * 1e-4
        x = rng.uniform(0, L)
        total = 0.0
        for fwd, own, ctr, pos in ((True, Bf, Bb, x), (False, Bb, Bf, L - x)):
            rhs = bloch_fb_segment_rhs(
                pos,
                own,
                np.zeros(3),
                kappas,
                rows3,
                rows4,
                coeffs[("3wm", fwd)],
                coeffs[("4wm", fwd)],
                rows3[:, -1] * g_unit,
                rows4[:, -1] * g_unit,
                0.0,  # segment start x0 and its phase: one segment from x = 0
                np.zeros(3),
                grid,
                np.ascontiguousarray(np.stack([ctr, ctr], -1)),
                0.0,
                np.zeros(3),
                L,
            )
            total += np.sum(W * 2 * np.real(np.conj(own) * rhs))
        scale = (
            np.sum(W * (np.abs(Bf) ** 2 + np.abs(Bb) ** 2))
            * np.max(np.abs(coeffs[("3wm", True)]))
            * np.max(np.abs(np.concatenate([Bf, Bb])))
        )
        worst = max(worst, abs(total) / scale)
    if perturb_one_row:
        assert worst > 1e-4
    else:
        assert worst < 1e-9


def _holotype_analysis():
    from twpasolver.modes_rwa import ModeArrayFactory

    analysis = TWPAnalysis(twpa=str(HOLOTYPE), f_arange=(0.2, 80, 2e-3))
    for cell in analysis.twpa.cells:
        cell.delta = 2e-4
    analysis.update_base_data()
    for nfc in (1, 2):
        analysis.add_mode_array(
            f"ext{nfc}",
            ModeArrayFactory.create_extended_3wm(
                analysis.data,
                n_pump_harmonics=3,
                n_frequency_conversion=nfc,
                n_signal_harmonics=0,
            ),
        )
    return analysis
