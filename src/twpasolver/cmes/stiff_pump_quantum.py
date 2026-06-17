"""Stiff-pump quantum TWPA solver — Bogoliubov basis with κ normalization."""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from twpasolver.cmes.adjoint import *
from twpasolver.cmes.adjoint import (
    apply_transfer_matrices,
    build_transfer_matrices,
    compute_loss_noise_adjoint,
)


def _process_single_freq(args):
    """Worker function for parallel frequency sweep."""
    (
        i,
        kappas_i,
        alphas_i,
        gamma_signal_i,
        Zb_i,
        kappas_array_i,
        alphas_array_i,
        signal_indices,
        pump_indices,
        relations_3wm,
        relations_4wm,
        coeffs_3wm,
        coeffs_4wm,
        kappa_cutoff,
        L,
        pump_fwd,
        pump_bwd_rev,
        x,
        n_modes,
        M,
        idler_row,
    ) = args

    e3, e4, coeff3, coeff4, deltakappa3, deltakappa4 = precompute_B_structure(
        signal_indices,
        pump_indices,
        kappas_array_i,
        alphas_array_i,
        relations_3wm,
        relations_4wm,
        coeffs_3wm,
        coeffs_4wm,
        kappa_cutoff,
        L,
    )
    U_traj, V_traj = build_UV_trajectory(
        pump_fwd,
        pump_bwd_rev,
        alphas_i,
        e3,
        e4,
        coeff3,
        coeff4,
        deltakappa3,
        deltakappa4,
        signal_indices,
        n_modes,
        M,
        x,
    )
    Ts = build_transfer_matrices(x, U_traj, V_traj)
    P, Q = apply_transfer_matrices(Ts, M)
    N_loss, Sigma_bath = compute_loss_noise_adjoint(
        x, U_traj, V_traj, P, Q, alphas_i, kappas_i, M, Ts=Ts
    )
    S0_current = apply_boundary_conditions_smatrix(P, Q, gamma_signal_i, kappas_i, L, M)
    S0_photon = apply_kappa_normalization(S0_current, kappas_i, Zb_i, M)
    obs = quantum_observables(
        S0_photon,
        M,
        signal_row=0,
        idler_row=idler_row,
        N_loss=N_loss,
        Sigma_bath=Sigma_bath,
    )
    return i, obs, Sigma_bath


def solve_pump_trajectory(
    x,
    signal_indices,
    pump_indices,
    y0_fwd,
    y0_bwd,
    gammas,
    reflections,
    relations_3wm,
    relations_4wm,
    coeffs_3wm,
    coeffs_4wm,
    passes=5,
    kappa_cutoff=0.2,
    Z0=50.0,
):
    """
    Run the forward-backward CME solver with signal modes zeroed out.

    Returns
    -------
    pump_fwd, pump_bwd : (n_modes, n_x) complex
    """
    from twpasolver.cmes.cmes_fb import general_cmes_solve_fb_cutoff

    y0_fwd_p, y0_bwd_p = y0_fwd.copy(), y0_bwd.copy()
    for idx in signal_indices:
        y0_fwd_p[idx] = y0_bwd_p[idx] = 0.0

    I_out = general_cmes_solve_fb_cutoff(
        x,
        y0_fwd_p[np.newaxis, :],
        y0_bwd_p[np.newaxis, :],
        gammas[np.newaxis, :],
        reflections[np.newaxis, :],
        relations_3wm,
        relations_4wm,
        coeffs_3wm,
        coeffs_4wm,
        passes=passes,
        kappa_cutoff=kappa_cutoff,
        Z0=Z0,
    )
    return I_out[0], I_out[1]


def precompute_B_structure(
    signal_indices,
    pump_indices,
    kappas,
    alphas,
    relations_3wm,
    relations_4wm,
    coeffs_3wm,
    coeffs_4wm,
    relative_deltak_cutoff,
    L,
):
    """
    Scan all RWA relations and extract phase-matched linear coupling terms.

    Returns
    -------
    entries_3wm, entries_4wm : int arrays
    coeffs3, coeffs4         : complex
    dk3, dk4                 : float
    """
    num_modes = len(kappas)
    signal_set = set(signal_indices)
    pump_set = set(pump_indices)

    signs_3wm = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    signs_4wm = [(s1, s2, s3) for s1 in (1, -1) for s2 in (1, -1) for s3 in (1, -1)]

    e3, coeff3, deltakappa3 = [], [], []
    e4, coeff4, deltakappa4 = [], [], []

    for id3, r in enumerate(relations_3wm):
        r0, r1, r2 = int(r[0]), int(r[1]), int(r[2])
        if r0 not in signal_set:
            continue
        c1 = -1 if r1 >= num_modes else 1
        k1 = r1 - num_modes if c1 == -1 else r1
        c2 = -1 if r2 >= num_modes else 1
        k2 = r2 - num_modes if c2 == -1 else r2
        if sum(1 for idx in [k1, k2] if idx in signal_set) != 1:
            continue
        if sum(1 for idx in [k1, k2] if idx in pump_set) != 1:
            continue
        for s0 in (1, -1):
            for s1, s2 in signs_3wm:
                dk = -s0 * kappas[r0] + s1 * c1 * kappas[k1] + s2 * c2 * kappas[k2]
                if (
                    abs(kappas[r0]) > 0
                    and abs(dk / kappas[r0]) < relative_deltak_cutoff
                ):
                    bnd = np.exp(-1j * (1 - s0) / 2.0 * kappas[r0] * L)
                    for sj, cj, kj in zip([s1, s2], [c1, c2], [k1, k2]):
                        if sj == -1:
                            bnd *= np.exp(+1j * cj * kappas[kj] * L)
                    coeff = coeffs_3wm[id3] * s0 * (alphas[r0] + 1j * kappas[r0]) * bnd
                    e3.append(
                        [
                            r0 + (2 * num_modes if s0 == -1 else 0),
                            r1 + (2 * num_modes if s1 == -1 else 0),
                            r2 + (2 * num_modes if s2 == -1 else 0),
                        ]
                    )
                    coeff3.append(coeff)
                    deltakappa3.append(dk)

    for id4, r in enumerate(relations_4wm):
        r0, r1, r2, r3 = int(r[0]), int(r[1]), int(r[2]), int(r[3])
        if r0 not in signal_set:
            continue
        c1 = -1 if r1 >= num_modes else 1
        k1 = r1 - num_modes if c1 == -1 else r1
        c2 = -1 if r2 >= num_modes else 1
        k2 = r2 - num_modes if c2 == -1 else r2
        c3 = -1 if r3 >= num_modes else 1
        k3 = r3 - num_modes if c3 == -1 else r3
        if sum(1 for idx in [k1, k2, k3] if idx in signal_set) != 1:
            continue
        if sum(1 for idx in [k1, k2, k3] if idx in pump_set) != 2:
            continue
        for s0 in (1, -1):
            for s1, s2, s3 in signs_4wm:
                dk = (
                    -s0 * kappas[r0]
                    + s1 * c1 * kappas[k1]
                    + s2 * c2 * kappas[k2]
                    + s3 * c3 * kappas[k3]
                )
                if abs(dk / kappas[r0]) < relative_deltak_cutoff:
                    bnd = np.exp(-1j * (1 - s0) / 2.0 * kappas[r0] * L)
                    for sj, cj, kj in zip([s1, s2], [c1, c2], [k1, k2]):
                        if sj == -1:
                            bnd *= np.exp(+1j * cj * kappas[kj] * L)
                    coeff = coeffs_4wm[id4] * s0 * (alphas[r0] + 1j * kappas[r0]) * bnd
                    e4.append(
                        [
                            r0 + (2 * num_modes if s0 == -1 else 0),
                            r1 + (2 * num_modes if s1 == -1 else 0),
                            r2 + (2 * num_modes if s2 == -1 else 0),
                            r3 + (2 * num_modes if s3 == -1 else 0),
                        ]
                    )
                    coeff4.append(coeff)
                    deltakappa4.append(dk)

    e3 = (
        np.array(e3, dtype=np.int64).reshape(-1, 3)
        if e3
        else np.empty((0, 3), dtype=np.int64)
    )
    e4 = (
        np.array(e4, dtype=np.int64).reshape(-1, 4)
        if e4
        else np.empty((0, 4), dtype=np.int64)
    )
    return (
        e3,
        e4,
        np.array(coeff3, dtype=np.complex128),
        np.array(coeff4, dtype=np.complex128),
        np.array(deltakappa3, dtype=np.float64),
        np.array(deltakappa4, dtype=np.float64),
    )


def build_UV_trajectory(
    pump_fwd,
    pump_bwd,
    alphas_signal,
    entries_3wm,
    entries_4wm,
    coeffs3,
    coeffs4,
    dk3,
    dk4,
    signal_indices,
    num_modes,
    M,
    x,
):
    """Build U(x) and V(x) trajectories in one vectorised pass."""
    n = num_modes
    n_x = len(x)

    ar = np.empty((4 * n, n_x), dtype=np.complex128)
    ar[:n, :] = pump_fwd
    ar[n : 2 * n, :] = np.conj(pump_fwd)
    ar[2 * n : 3 * n, :] = pump_bwd
    ar[3 * n :, :] = np.conj(pump_bwd)

    sig_row = np.full(n, -1, dtype=np.int64)
    for m, idx in enumerate(signal_indices):
        sig_row[idx] = m

    U_traj = np.zeros((n_x, 2 * M, 2 * M), dtype=np.complex128)
    V_traj = np.zeros((n_x, 2 * M, 2 * M), dtype=np.complex128)

    for m in range(M):
        U_traj[:, m, m] = -alphas_signal[m]
        U_traj[:, M + m, M + m] = alphas_signal[m]

    def _fill(entries, coeffs, dks):
        for idx in range(len(entries)):
            enc = entries[idx]
            coeff = coeffs[idx]
            dk = dks[idx]

            tgt_enc = int(enc[0])
            tgt_bwd = tgt_enc >= 2 * n
            tgt_base = tgt_enc - (2 * n if tgt_bwd else 0)
            tgt_mode = tgt_base - n if tgt_base >= n else tgt_base
            t_row = sig_row[tgt_mode]
            if t_row < 0:
                continue
            t_row_2M = t_row + (M if tgt_bwd else 0)

            src_found = False
            src_is_conj = False
            src_is_bwd = False
            s_col = 0
            pump_product = np.ones(n_x, dtype=np.complex128)

            for j in range(1, len(enc)):
                e = int(enc[j])
                e_bwd = e >= 2 * n
                e_base = e - (2 * n if e_bwd else 0)
                e_conj = e_base >= n
                e_mode = e_base - n if e_conj else e_base

                if sig_row[e_mode] >= 0:
                    src_found = True
                    src_is_bwd = e_bwd
                    src_is_conj = e_conj
                    s_col = sig_row[e_mode] + (M if src_is_bwd else 0)
                else:
                    pump_product = pump_product * ar[e]

            if not src_found:
                continue

            vals = coeff * pump_product * np.exp(1j * dk * x)
            if src_is_conj:
                V_traj[:, t_row_2M, s_col] += vals
            else:
                U_traj[:, t_row_2M, s_col] += vals

    _fill(entries_3wm, coeffs3, dk3)
    _fill(entries_4wm, coeffs4, dk4)
    return U_traj, V_traj


def apply_boundary_conditions(
    P, Q, gamma_signal, kappas_signal, L, M, gamma_signal_L=None
):
    if gamma_signal_L is None:
        gamma_signal_L = gamma_signal

    γ0 = np.asarray(gamma_signal, dtype=np.complex128)
    γL = np.asarray(gamma_signal_L, dtype=np.complex128)

    t0 = np.sqrt(1.0 - γ0 * γ0)
    r0 = γ0
    tL = np.sqrt(1.0 - γL * γL)
    rL = γL
    t0c = np.conj(t0)
    r0c = np.conj(r0)
    tLc = np.conj(tL)
    rLc = np.conj(rL)
    t0out = t0
    tLout = tL
    t0outc = t0c
    tLoutc = tLc

    el = np.exp(1j * kappas_signal * L)
    elc = np.conj(el)

    M_int = np.block([[P, Q], [np.conj(Q), np.conj(P)]])
    A_sys = np.zeros((4 * M, 4 * M), dtype=np.complex128)

    A_sys[0:M, 0:M] = np.eye(M)
    A_sys[0:M, M : 2 * M] = -np.diag(r0 * el)
    A_sys[M : 2 * M, :] = M_int[M : 2 * M, :] - np.diag(rL * el) @ M_int[0:M, :]
    A_sys[2 * M : 3 * M, 2 * M : 3 * M] = np.eye(M)
    A_sys[2 * M : 3 * M, 3 * M : 4 * M] = -np.diag(r0c * elc)
    A_sys[3 * M : 4 * M, :] = (
        M_int[3 * M : 4 * M, :] - np.diag(rLc * elc) @ M_int[2 * M : 3 * M, :]
    )

    rhs = np.zeros((4 * M, 4 * M), dtype=np.complex128)
    rhs[0:M, 0:M] = np.diag(t0)
    rhs[M : 2 * M, M : 2 * M] = np.diag(tL)
    rhs[2 * M : 3 * M, 2 * M : 3 * M] = np.diag(t0c)
    rhs[3 * M : 4 * M, 3 * M : 4 * M] = np.diag(tLc)

    psi0 = np.linalg.solve(A_sys, rhs)
    print(f"cond(A_sys) = {np.linalg.cond(A_sys):.3e}")
    print(f"M_int[M:2M] max = {np.max(np.abs(M_int[M:2*M])):.3e}")
    print(f"M_int[0:M]  max = {np.max(np.abs(M_int[0:M])):.3e}")
    psiL = M_int @ psi0

    S0 = np.zeros((4 * M, 4 * M), dtype=np.complex128)
    S0[0:M, :] = np.diag(el) @ psiL[0:M, :]
    S0[0:M, M : 2 * M] -= np.diag(rL)
    S0[M : 2 * M, :] = np.diag(el) @ psi0[M : 2 * M, :]
    S0[M : 2 * M, 0:M] -= np.diag(r0)
    S0[2 * M : 3 * M, :] = np.diag(elc) @ psiL[2 * M : 3 * M, :]
    S0[2 * M : 3 * M, 3 * M : 4 * M] -= np.diag(rLc)
    S0[3 * M : 4 * M, :] = np.diag(elc) @ psi0[3 * M : 4 * M, :]
    S0[3 * M : 4 * M, 2 * M : 3 * M] -= np.diag(r0c)
    return S0


def apply_boundary_conditions_smatrix(
    P, Q, gamma_signal, kappas_signal, L, M, gamma_signal_L=None
):
    """
    Numerically stable S-matrix from Bogoliubov blocks.

    Parameters
    ----------
    P, Q            : (2M, 2M) complex
    gamma_signal    : (M,) reflection at x=0
    kappas_signal   : (M,) wavevectors
    L               : float
    M               : int
    gamma_signal_L  : (M,) or None

    Returns
    -------
    S0 : (4M, 4M) complex
    """
    if gamma_signal_L is None:
        gamma_signal_L = gamma_signal

    γ0 = np.asarray(gamma_signal, dtype=np.complex128)
    γL = np.asarray(gamma_signal_L, dtype=np.complex128)

    t0 = np.sqrt(1.0 - γ0 * γ0)
    r0 = γ0
    tL = np.sqrt(1.0 - γL * γL)
    rL = γL
    t0c = t0.conj()
    r0c = r0.conj()
    tLc = tL.conj()
    rLc = rL.conj()

    el = np.exp(1j * kappas_signal * L)
    elc = el.conj()

    M_int = np.block([[P, Q], [np.conj(Q), np.conj(P)]])

    D = np.diag(rL * el)
    Dc = np.diag(rLc * elc)
    G = M_int[M : 2 * M, :] - D @ M_int[0:M, :]
    Gc = M_int[3 * M : 4 * M, :] - Dc @ M_int[2 * M : 3 * M, :]

    G_f, G_b, G_fc, G_bc = (
        G[:, 0:M],
        G[:, M : 2 * M],
        G[:, 2 * M : 3 * M],
        G[:, 3 * M : 4 * M],
    )
    Gc_f, Gc_b, Gc_fc, Gc_bc = (
        Gc[:, 0:M],
        Gc[:, M : 2 * M],
        Gc[:, 2 * M : 3 * M],
        Gc[:, 3 * M : 4 * M],
    )

    r0el = np.diag(r0 * el)
    r0celc = np.diag(r0c * elc)

    A_red = np.block(
        [
            [G_f @ r0el + G_b, G_fc @ r0celc + G_bc],
            [Gc_f @ r0el + Gc_b, Gc_fc @ r0celc + Gc_bc],
        ]
    )

    rhs_f = np.zeros((M, 4 * M), dtype=np.complex128)
    rhs_f[:, 0:M] = np.diag(t0)
    rhs_b = np.zeros((M, 4 * M), dtype=np.complex128)
    rhs_b[:, M : 2 * M] = np.diag(tL)
    rhs_fc = np.zeros((M, 4 * M), dtype=np.complex128)
    rhs_fc[:, 2 * M : 3 * M] = np.diag(t0c)
    rhs_bc = np.zeros((M, 4 * M), dtype=np.complex128)
    rhs_bc[:, 3 * M : 4 * M] = np.diag(tLc)

    B_red = np.vstack(
        [
            rhs_b - G_f @ rhs_f - G_fc @ rhs_fc,
            rhs_bc - Gc_f @ rhs_f - Gc_fc @ rhs_fc,
        ]
    )

    x_red = np.linalg.solve(A_red, B_red)
    psi0_b = x_red[0:M, :]
    psi0_bc = x_red[M : 2 * M, :]
    psi0_f = rhs_f + r0el @ psi0_b
    psi0_fc = rhs_fc + r0celc @ psi0_bc
    psi0 = np.vstack([psi0_f, psi0_b, psi0_fc, psi0_bc])
    psiL = M_int @ psi0

    S0 = np.zeros((4 * M, 4 * M), dtype=np.complex128)
    S0[0:M, :] = np.diag(el) @ psiL[0:M, :]
    S0[0:M, M : 2 * M] -= np.diag(rL)
    S0[M : 2 * M, :] = np.diag(el) @ psi0[M : 2 * M, :]
    S0[M : 2 * M, 0:M] -= np.diag(r0)
    S0[2 * M : 3 * M, :] = np.diag(elc) @ psiL[2 * M : 3 * M, :]
    S0[2 * M : 3 * M, 3 * M : 4 * M] -= np.diag(rLc)
    S0[3 * M : 4 * M, :] = np.diag(elc) @ psi0[3 * M : 4 * M, :]
    S0[3 * M : 4 * M, 2 * M : 3 * M] -= np.diag(r0c)
    return S0


def apply_kappa_normalization(S0_current, kappas_signal, Zb_signal, M):
    """Convert current-basis S0 to photon-flux basis (a_m = I_m / sqrt(κ_m))."""
    kZ_2M = np.r_[kappas_signal, kappas_signal]
    kZ_4M = np.r_[kZ_2M, np.conj(kZ_2M)]
    norm = np.sqrt(kZ_4M[np.newaxis, :] / kZ_4M[:, np.newaxis])
    return S0_current * norm


def quantum_observables(
    S0, M, signal_row=0, idler_row=1, sigma_arr=None, N_loss=None, Sigma_bath=None
):
    """
    Extract quantum observables from the photon-basis scattering matrix S0.

    Parameters
    ----------
    S0          : (4M, 4M) complex
    M           : int
    signal_row  : int
    idler_row   : int
    sigma_arr   : (M,) float or None   Bogoliubov signs, defaults to +1
    N_loss      : (4M, 4M) float or None
    Sigma_bath  : (4M, 4M) complex Hermitian or None

    Returns
    -------
    dict with gain_db, squeezing_db, antisqueezing_db,
         quantum_efficiency, quantum_efficiency_lossless,
         commutator, commutator_with_loss, bogoliubov_residual,
         loss_noise_total, S0, sigma_out.
    """
    if sigma_arr is None:
        sigma_arr = np.ones(M)

    sigma_cols = np.r_[sigma_arr, sigma_arr, -sigma_arr, -sigma_arr]

    G = np.abs(S0[signal_row, signal_row]) ** 2
    gain_db = 10 * np.log10(G)

    commutator_discrete = float(np.dot(sigma_cols, np.abs(S0[signal_row, :]) ** 2))
    loss_bath_cont = (
        float(np.dot(sigma_cols, N_loss[signal_row, :])) if N_loss is not None else 0.0
    )
    commutator_with_loss = commutator_discrete + loss_bath_cont

    sigma_out = 0.5 * S0 @ S0.conj().T
    if Sigma_bath is not None:
        sigma_out = sigma_out + Sigma_bath
    sigma_out = 0.5 * (sigma_out + sigma_out.conj().T)

    rows = [signal_row, idler_row, signal_row + 2 * M, idler_row + 2 * M]
    block = sigma_out[np.ix_(rows, rows)]
    variances = np.linalg.eigvalsh(block)
    squeezing_db = 10 * np.log10(2 * variances[0] + 1e-30)
    antisqueezing_db = 10 * np.log10(2 * variances[-1] + 1e-30)

    def balanced_squeezing_db(sigma_out, s, i, M):
        """Minimum variance of X_-(θ_s, θ_i) over both LO phases."""
        ss = np.real(sigma_out[s, s])
        ii = np.real(sigma_out[i, i])
        C_an = sigma_out[s, i + 2 * M]
        C_no = sigma_out[s, i]
        return 10 * np.log10(2 * (0.5 * (ss + ii) - abs(C_an) - abs(C_no)))

    sq_fixed_db = balanced_squeezing_db(sigma_out, signal_row, idler_row, M)

    noise_discrete = 0.5 * np.sum(np.abs(S0[signal_row, :]) ** 2)
    loss_noise_signal = (
        float(N_loss[signal_row, :].sum()) if N_loss is not None else 0.0
    )
    noise_total = noise_discrete + 0.5 * loss_noise_signal

    qe_lossless = (0.5 * G) / noise_discrete if noise_discrete > 0 else 0.0
    qe = (0.5 * G) / noise_total if noise_total > 0 else 0.0
    eta_ideal = G / (2.0 * G - 1.0) if G > 1.0 else G
    qe_normalized = qe / eta_ideal if eta_ideal > 0 else 0.0
    eta_bar = 1.0 - qe_normalized
    added_noise = noise_total / G - 0.5 if G > 0 else np.inf

    return dict(
        gain_db=gain_db,
        squeezing_db=squeezing_db,
        squeezing_db_fixedquad=sq_fixed_db,
        antisqueezing_db=antisqueezing_db,
        quantum_efficiency=qe,
        quantum_efficiency_lossless=qe_lossless,
        quantum_efficiency_normalized=qe_normalized,
        eta_bar=eta_bar,
        added_noise_photons=added_noise,
        commutator=commutator_discrete,
        commutator_with_loss=commutator_with_loss,
        bogoliubov_residual=commutator_with_loss - 1.0,
        loss_noise_total=loss_noise_signal / G,
        S0=S0,
        sigma_out=sigma_out,
        Sn=N_loss,
    )


def stiff_pump_quantum(
    analysis,
    signal_freqs: np.ndarray,
    pump: float | None = None,
    Ip0: float | None = None,
    mode_array_config: str = "basic_3wm",
    signal_mode: str = "s",
    pump_mode: str = "p",
    idler_mode: str = "i",
    thin: int = 100,
    pump_passes: int = 5,
    kappa_cutoff: float = 0.2,
    camp: float = 1e-15,
    n_workers: int | None = None,
):
    """
    Full quantum stiff-pump solver.

    Returns dict with gain_db, squeezing_db, antisqueezing_db,
    quantum_efficiency, quantum_efficiency_lossless, commutator,
    commutator_with_loss, bogoliubov_residual, loss_noise_total,
    S0_array, sigma_out_array, pump_fwd, pump_bwd, signal_freqs.
    """
    from twpasolver.analysis import prepare_relations_coefficients

    if Ip0 is not None:
        analysis.twpa.Ip0 = Ip0
    if pump is None:
        pump = analysis.data["optimal_pump_freq"]

    analysis.update_base_data()
    signal_freqs = np.asarray(signal_freqs)
    n_freq = len(signal_freqs)

    mode_array = analysis.get_mode_array(mode_array_config)
    mode_labels = list(mode_array.modes.keys())
    n_modes = len(mode_labels)
    mode_to_idx = {label: i for i, label in enumerate(mode_labels)}
    signal_idx = mode_to_idx[signal_mode]
    idler_idx = mode_to_idx[idler_mode]
    pump_idx = mode_to_idx[pump_mode]

    N_tot = analysis.twpa.N_tot
    Z0 = analysis.twpa.Z0_ref
    x = np.linspace(0, N_tot, int(N_tot / thin), endpoint=True)
    n_x = len(x)
    L = float(x[-1])

    mode_array.update_frequencies({pump_mode: pump})
    mode_params = mode_array.process_frequency_array(signal_mode, signal_freqs)

    def _param(key):
        return np.array(
            [[mode_params[m][key][i] for m in mode_labels] for i in range(n_freq)]
        )

    freqs_array = _param("freqs")
    kappas_array = _param("k")
    alphas_array = _param("alpha")
    reflections_array = _param("gamma")
    reflections_array = np.full_like(reflections_array, 0.0)
    gammas = -alphas_array + 1j * kappas_array
    Zb_array = (1.0 + reflections_array.imag) / (1.0 - reflections_array.imag) * Z0

    terms_3wm = mode_array.get_rwa_terms(power=2)
    terms_4wm = mode_array.get_rwa_terms(power=3)
    (
        relations_3wm,
        relations_4wm,
        coeffs_3wm,
        coeffs_4wm,
    ) = prepare_relations_coefficients(
        terms_3wm, terms_4wm, analysis.twpa.epsilon, analysis.twpa.xi
    )

    signal_labels = mode_array.get_dependent_modes(signal_mode) | {signal_mode}
    signal_indices = [mode_to_idx[l] for l in mode_labels if l in signal_labels]
    pump_indices = [mode_to_idx[l] for l in mode_labels if l not in signal_labels]
    M = len(signal_indices)

    signal_indices.remove(signal_idx)
    signal_indices = [signal_idx] + signal_indices

    if idler_mode is not None:
        idler_row = signal_indices.index(mode_to_idx[idler_mode])
    elif "i" in signal_labels:
        idler_row = signal_indices.index(mode_to_idx["i"])
    else:
        idler_row = 1 if M > 1 else 0
    print(
        f"Two-mode squeezing block: signal row 0 ({signal_mode}), "
        f"idler row {idler_row} ({mode_labels[signal_indices[idler_row]]})"
    )

    y0_fwd = np.zeros(n_modes, dtype=np.complex128)
    y0_bwd = np.zeros(n_modes, dtype=np.complex128)
    y0_fwd[0] = analysis.twpa.Ip0
    print(pump_indices)

    gammas_pump = gammas[0]
    reflections_pump = reflections_array[0]

    print("Computing pump trajectory")
    pump_fwd, pump_bwd = solve_pump_trajectory(
        x,
        signal_indices,
        pump_indices,
        y0_fwd,
        y0_bwd,
        gammas_pump,
        reflections_pump,
        relations_3wm,
        relations_4wm,
        coeffs_3wm,
        coeffs_4wm,
        passes=pump_passes,
        kappa_cutoff=kappa_cutoff,
        Z0=Z0,
    )
    print("Done")

    gain_db = np.zeros(n_freq)
    squeezing_db = np.zeros(n_freq)
    squeezing_db_fixedquad = np.zeros(n_freq)
    antisqueezing_db = np.zeros(n_freq)
    qe = np.zeros(n_freq)
    qe_lossless = np.zeros(n_freq)
    qe_normalized = np.zeros(n_freq)
    added_noise = np.zeros(n_freq)
    commutator = np.zeros(n_freq)
    commutator_with_loss = np.zeros(n_freq)
    bogoliubov_residual = np.zeros(n_freq)
    loss_noise_total = np.zeros(n_freq)
    sym_err = np.zeros(n_freq)
    S0_array = np.zeros((n_freq, 4 * M, 4 * M), dtype=np.complex128)
    Sn_array = np.zeros((n_freq, 4 * M, 4 * M), dtype=np.complex128)
    sigma_out_array = np.zeros((n_freq, 4 * M, 4 * M), dtype=np.complex128)

    pump_bwd_rev = pump_bwd[::-1]
    task_args = [
        (
            i,
            kappas_array[i, signal_indices],
            alphas_array[i, signal_indices],
            reflections_array[i, signal_indices],
            Zb_array[i, signal_indices],
            kappas_array[i],
            alphas_array[i],
            signal_indices,
            pump_indices,
            relations_3wm,
            relations_4wm,
            coeffs_3wm,
            coeffs_4wm,
            kappa_cutoff,
            L,
            pump_fwd,
            pump_bwd_rev,
            x,
            n_modes,
            M,
            idler_row,
        )
        for i in range(n_freq)
    ]

    _n_workers = min(n_freq, os.cpu_count() or 1) if n_workers is None else n_workers
    print(
        f"Running frequency sweep with {_n_workers} worker(s) over {n_freq} frequencies"
    )

    if _n_workers == 1:
        results = [_process_single_freq(a) for a in task_args]
    else:
        with ProcessPoolExecutor(max_workers=_n_workers) as pool:
            results = list(pool.map(_process_single_freq, task_args))

    for result in results:
        i, obs, Sigma_bath = result
        gain_db[i] = obs["gain_db"]
        squeezing_db[i] = obs["squeezing_db"]
        squeezing_db_fixedquad[i] = obs["squeezing_db_fixedquad"]
        antisqueezing_db[i] = obs["antisqueezing_db"]
        qe[i] = obs["quantum_efficiency"]
        qe_lossless[i] = obs["quantum_efficiency_lossless"]
        qe_normalized[i] = obs["quantum_efficiency_normalized"]
        added_noise[i] = obs["added_noise_photons"]
        commutator[i] = obs["commutator"]
        commutator_with_loss[i] = obs["commutator_with_loss"]
        bogoliubov_residual[i] = obs["bogoliubov_residual"]
        loss_noise_total[i] = obs["loss_noise_total"]
        S0_array[i] = obs["S0"]
        Sn_array[i] = Sigma_bath
        sigma_out_array[i] = obs["sigma_out"]

    return dict(
        pump_fwd=pump_fwd,
        pump_bwd=pump_bwd,
        gain_db=gain_db,
        squeezing_db=squeezing_db,
        squeezing_db_fixedquad=squeezing_db_fixedquad,
        antisqueezing_db=antisqueezing_db,
        quantum_efficiency=qe,
        quantum_efficiency_lossless=qe_lossless,
        quantum_efficiency_normalized=qe_normalized,
        added_noise_photons=added_noise,
        commutator=commutator,
        commutator_with_loss=commutator_with_loss,
        bogoliubov_residual=bogoliubov_residual,
        loss_noise_total=loss_noise_total,
        S0_array=S0_array,
        Sn_array=Sn_array,
        sigma_out_array=sigma_out_array,
        signal_freqs=signal_freqs,
        ks_L=2 * alphas_array[:, signal_idx] * L,
        ki_L=2 * alphas_array[:, idler_idx] * L,
    )
