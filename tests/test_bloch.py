"""Tests for the Bloch-basis CMEs (bloch_no_reflections and bloch models)."""

import numpy as np
import pytest

from twpasolver.analysis import GainModel, TWPAnalysis
from twpasolver.cmes.bloch import bloch_mode_tables
from twpasolver.cmes.bloch_fb import bloch_fb_cmes_solve
from twpasolver.cmes.solver_config import FBSolverConfig
from twpasolver.models import TWPA, LCLfBaseCell

CELL_U = {"name": "LCLfBaseCell", "N": 30, "L": 49.9e-12, "C": 24.2e-15, "Lf": 0.87e-9}
CELL_L = {"name": "LCLfBaseCell", "N": 6, "L": 48.6e-12, "C": 8.2e-15, "Lf": 0.28e-9}


def _forward_gamma(twpa, freqs, n_cells):
    T = np.asarray(twpa.single_abcd(freqs)).astype(complex)
    tr = T[:, 0, 0] + T[:, 1, 1]
    gamma = np.log((tr + np.sqrt(tr**2 - 4 + 0j)) / 2) / n_cells
    return np.where(gamma.imag < 0, -gamma, gamma)


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


def test_energy_conservation_lossless():
    """Lossless 3WM: sum_n W_n |A_n|^2 is conserved by the Bloch CMEs."""
    twpa = TWPA(
        cells=[CELL_U, CELL_L, CELL_U], N=200, Istar=6.3e-3, Idc=1e-3, Ip0=2.5e-4
    )
    analysis = TWPAnalysis(twpa=twpa, f_arange=(0.05, 20, 1e-3))
    analysis.update_base_data()
    signal_freqs = np.array([1.75, 2.0, 12.25, 12.5])
    res = analysis.gain(
        signal_freqs,
        pump=14.3,
        Is0=1e-6,
        model=GainModel.BLOCH_NO_REFLECTIONS,
        thin=50,
    )
    W = res["bloch"]["W"]
    assert np.allclose(W.imag, 0, atol=1e-9 * np.abs(W).max())

    power = W.real[:, :, None] * np.abs(res["I_tuples"]) ** 2
    delta = power[:, :, -1] - power[:, :, 0]
    labels = list(res["mode_info"]["mode_map"])
    s, i = labels.index("s"), labels.index("i")
    transferred = delta[:, s] + delta[:, i]
    # Floor: solver tolerance on the (much larger) pump power.
    floor = 1e-8 * power[:, labels.index("p"), 0]
    assert np.all(np.abs(delta.sum(axis=1)) < 1e-3 * np.abs(transferred) + floor)


def test_segmented_not_supported_without_reflections():
    """The no-reflections Bloch model is restricted to non-segmented devices."""
    seg = TWPA(cells=[CELL_U, CELL_L, CELL_U], N=10)
    twpa = TWPA(cells=[seg, seg], Ip0=2.5e-4)
    analysis = TWPAnalysis(twpa=twpa, f_arange=(0.05, 20, 1e-2))
    analysis.update_base_data()
    with pytest.raises(NotImplementedError):
        analysis.gain([2.0], pump=14.3, model="bloch_no_reflections")


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
        fb_config=FBSolverConfig(passes=40, convergence_threshold=1e-8),
        thin=20,
    )
    s21 = 20 * np.log10(np.abs(analysis.data["S21"][idx]))
    s11 = 20 * np.log10(np.abs(analysis.data["S11"][idx]))
    assert np.allclose(res["gain_db"], s21, atol=1e-3)
    assert np.allclose(res["gain_db_11"], s11, atol=1e-2)


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
    bl = analysis._prepare_bloch_coefficients(mode_array, ci)
    labels = ci["mode_labels"]
    p, s = labels.index("p"), labels.index("s")
    n = len(signal_freqs)
    y0 = np.zeros((n, len(labels)), dtype=complex)
    y0[:, p], y0[:, s] = Ip0, 1e-6
    x = np.linspace(0, analysis.twpa.N_tot, 500)
    out = bloch_fb_cmes_solve(
        x,
        y0,
        0 * y0,
        ci["gammas"],
        ci["relations_3wm"],
        ci["relations_4wm"],
        bl["base_3wm"],
        bl["base_4wm"],
        bl["prefac"],
        bl["phi_plus"],
        bl["phi_minus"],
        bl["L_nl"],
        bl["positions"],
        bl["n_cells"],
        bl["Zb_plus"],
        bl["Zb_minus"],
        passes=60,
        convergence_threshold=1e-8,
        Ze=Z_ext,
        m_max=m_max,
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


@pytest.mark.parametrize("model", ["general_no_reflections", "bloch_no_reflections"])
def test_backward_inputs_rejected_without_reflections(model):
    """No-reflection models propagate only +x waves: backward inputs must raise."""
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


@pytest.mark.parametrize("which", ["general", "bloch"])
def test_mismatch_filter_keeps_partner_terms_together(which):
    """Partner terms of a process are always kept or dropped together."""
    from twpasolver.analysis import prepare_relations_coefficients
    from twpasolver.cmes.bloch_common import _filter_wm_terms_bloch
    from twpasolver.cmes.cmes_fb import _filter_wm_terms
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
        if which == "general":
            return _filter_wm_terms(kappas, rel3, rel4, cutoff)
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
    bl = analysis._prepare_bloch_coefficients(mode_array, ci)
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


def test_anderson_converges_where_plain_iteration_diverges():
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
    bl = analysis._prepare_bloch_coefficients(mode_array, ci)
    labels = ci["mode_labels"]
    ip, is_ = labels.index("p"), labels.index("s")
    x = np.linspace(0, twpa.N_tot, 400)
    y0f = np.zeros((len(sig), 3), complex)
    y0f[:, ip], y0f[:, is_] = Ip0, Is0
    y0b = np.zeros_like(y0f)

    def solve(depth, passes):
        out = bloch_fb_cmes_solve(
            x,
            y0f,
            y0b,
            ci["gammas"],
            ci["relations_3wm"],
            ci["relations_4wm"],
            bl["base_3wm"],
            0 * bl["base_4wm"],
            bl["prefac"],
            bl["phi_plus"],
            bl["phi_minus"],
            bl["L_nl"],
            bl["positions"],
            bl["n_cells"],
            bl["Zb_plus"],
            bl["Zb_minus"],
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
    plain = solve(0, 30)
    assert np.max(plain - ref) > 20  # the plain iteration runs away


def test_anderson_matches_plain_iteration_when_it_converges():
    """Acceleration changes only the path to the fixed point, not the result."""
    analysis = _mismatched_analysis(Ip0=1.5e-4)
    freqs = analysis.data["freqs"]
    idx = np.searchsorted(freqs, [2.0, 6.0, 10.0])
    kw = dict(pump=14.3, Is0=1e-8, model=GainModel.BLOCH, thin=20)
    plain = analysis.gain(
        freqs[idx],
        fb_config=FBSolverConfig(
            passes=60, convergence_threshold=1e-8, anderson_depth=0
        ),
        **kw,
    )
    acc = analysis.gain(
        freqs[idx],
        fb_config=FBSolverConfig(
            passes=60, convergence_threshold=1e-8, anderson_depth=5
        ),
        **kw,
    )
    assert np.allclose(acc["gain_db"], plain["gain_db"], atol=1e-3)
    assert np.allclose(acc["gain_db_11"], plain["gain_db_11"], atol=1e-2)


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
            fb_config=FBSolverConfig(passes=1),
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


def test_edge_guard_weights():
    """Weights: 1 for real W, 0 for reactive W, continuous and monotone between."""
    from twpasolver.cmes.bloch import edge_guard_mode_weights, reactive_angle_deg

    ang = np.linspace(0, 90, 181)
    W = 50 * np.exp(-1j * np.radians(ang))
    assert np.allclose(reactive_angle_deg(W), ang)
    assert np.allclose(reactive_angle_deg(-W), ang)  # sign of W is irrelevant
    w = edge_guard_mode_weights(W, 5.0, 20.0)
    assert np.all(w[ang <= 5] == 1.0) and np.all(w[ang >= 20] == 0.0)
    assert np.all(np.diff(w) <= 0)
    assert np.max(np.abs(np.diff(w))) < 0.1  # no jumps on a 0.5 deg grid


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
    bl = analysis._prepare_bloch_coefficients(mode_array, ci)
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


def test_edge_guard_switches_off_edge_conversion():
    """
    imskit holotype pumped at 16 GHz: around fs = 9.19 GHz the second
    conversion product 2 fp + fs sits at the upper edge of the third stopband.
    The guard of the no-reflections Bloch model switches p2s off there, and the
    result matches the run without the second conversion.

    (With the earlier holotype parameters the unguarded run also showed tens of
    dB of spurious ps <-> p2s gain here; with the current ones it does not, so
    only the guard itself is checked.)
    """
    analysis = _holotype_analysis()
    sig = np.array([9.188, 9.194, 9.198])
    kw = dict(signal_freqs=sig, Is0=1e-9, pump=16.0, model="bloch_no_reflections")
    ref = analysis.gain(mode_array_config="ext1", **kw)["gain_db"]
    guarded = analysis.gain(mode_array_config="ext2", **kw)
    p2s = guarded["mode_info"]["mode_map"]["p2s"]
    assert np.all(guarded["bloch"]["edge_weights"][:, p2s] == 0.0)
    assert np.allclose(guarded["gain_db"], ref, atol=0.1)


def test_edge_guard_never_drops_injected_modes(monkeypatch, caplog):
    """An injected mode flagged by the guard is kept and a warning is logged."""
    from twpasolver.cmes import solver_config
    from twpasolver.logger import log

    analysis = _holotype_analysis()
    # Absurdly strict guard: every lossy mode is "reactive".
    monkeypatch.setattr(solver_config, "BLOCH_EDGE_GUARD_DEG", (0.0, 1e-9))
    log.addHandler(caplog.handler)
    try:
        res = analysis.gain(
            signal_freqs=np.array([6.0]),
            Is0=1e-9,
            pump=16.0,
            model="bloch_no_reflections",
        )
    finally:
        log.removeHandler(caplog.handler)
    mm = res["mode_info"]["mode_map"]
    w = res["bloch"]["edge_weights"][0]
    assert w[mm["p"]] == 1.0 and w[mm["s"]] == 1.0
    assert w[mm["i"]] == 0.0
    assert any("injected mode" in r.getMessage() for r in caplog.records)
