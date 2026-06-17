"""Magnus-based adjoint backward pass for the loss-noise integral."""

from __future__ import annotations

import numpy as np

from twpasolver.cmes.magnus import *


def _terminal_condition(P_L, Q_L, kappas, L, M, kappa_norm):
    """Terminal condition Y(L) for the adjoint ODE."""
    el = np.exp(1j * kappas * L)
    ML = np.block([[P_L, Q_L], [Q_L.conj(), P_L.conj()]]) * kappa_norm
    ML_inv = np.linalg.pinv(ML)

    Pg_L = np.zeros((2 * M, 2 * M), dtype=np.complex128)
    Qg_L = np.zeros((2 * M, 2 * M), dtype=np.complex128)
    Pg_L[:M, :M] = np.diag(el)
    Pg_L[M:, :] = np.diag(el) @ ML_inv[M : 2 * M, : 2 * M]
    Qg_L[M:, :] = np.diag(el) @ ML_inv[M : 2 * M, 2 * M :]
    return Pg_L, Qg_L


def _assemble_Sx(Pg, Qg, kappa_norm, M):
    """Assemble the (4M, 4M) photon-basis Green's function from (Pg, Qg)."""
    TwoM = 2 * M
    Sx = np.empty((4 * M, 4 * M), dtype=np.complex128)
    Sx[:TwoM, :TwoM] = Pg
    Sx[:TwoM, TwoM:] = Qg
    Sx[TwoM:, :TwoM] = Qg.conj()
    Sx[TwoM:, TwoM:] = Pg.conj()
    return Sx * kappa_norm


def build_transfer_matrices(x, U_traj, V_traj):
    """
    Build the (N, 4M, 4M) array of Magnus transfer matrices.

    Parameters
    ----------
    x      : (N+1,) float
    U_traj : (N+1, 2M, 2M) complex
    V_traj : (N+1, 2M, 2M) complex

    Returns
    -------
    Ts : (N, 4M, 4M) complex
    """
    return pade_batch_auto(build_omegas(x, U_traj, V_traj))


def apply_transfer_matrices(Ts, M):
    """
    Extract Bogoliubov blocks (P, Q) from the accumulated product of Ts.

    Parameters
    ----------
    Ts : (N, 4M, 4M) complex
    M  : int

    Returns
    -------
    P, Q : (2M, 2M) complex
    """
    TwoM = 2 * M
    T_tot = tree_product(Ts)
    return T_tot[:TwoM, :TwoM], np.conj(T_tot[TwoM:, :TwoM])


def compute_loss_noise_adjoint(
    x, U_traj, V_traj, P_L, Q_L, alphas_diss, kappas_signal, M, Ts=None
):
    """
    Compute loss-bath noise integrals via Magnus adjoint backward pass.

    Parameters
    ----------
    x             : (n_x,) float
    U_traj        : (n_x, 2M, 2M) complex
    V_traj        : (n_x, 2M, 2M) complex
    P_L, Q_L      : (2M, 2M) complex
    alphas_diss   : (M,) float
    kappas_signal : (M,) float
    M             : int
    Ts            : (N, 4M, 4M) complex or None

    Returns
    -------
    N_loss     : (4M, 4M) real
    Sigma_bath : (4M, 4M) complex Hermitian
    """
    n_x = len(x)
    L = float(x[-1])
    TwoM = 2 * M
    FourM = 4 * M

    kZ_2M = np.r_[kappas_signal, kappas_signal]
    kZ_4M = np.r_[kZ_2M, np.conj(kZ_2M)]
    kappa_norm = np.sqrt(kZ_4M[None, :] / kZ_4M[:, None])
    alpha_4M = np.r_[alphas_diss, alphas_diss, alphas_diss, alphas_diss]
    alpha_diag = np.diag(alpha_4M)
    sqrt2a = np.sqrt(2.0 * alpha_diag)

    if Ts is None:
        Ts = build_transfer_matrices(x, U_traj, V_traj)

    Pg, Qg = _terminal_condition(P_L, Q_L, kappas_signal, L, M, kappa_norm)

    N_loss = np.zeros((FourM, FourM), dtype=np.float64)
    Sigma_bath = np.zeros((FourM, FourM), dtype=np.complex128)
    integrand_N_prev = np.zeros((FourM, FourM), dtype=np.float64)
    integrand_cov_prev = np.zeros((FourM, FourM), dtype=np.complex128)

    for ix in range(n_x - 1, -1, -1):
        Sx = _assemble_Sx(Pg, Qg, kappa_norm, M)
        integrand_N_curr = 2.0 * np.abs(Sx) ** 2 @ alpha_diag
        W = Sx @ sqrt2a
        integrand_cov_curr = W @ W.conj().T

        if ix < n_x - 1:
            dx = x[ix + 1] - x[ix]
            N_loss += 0.5 * dx * (integrand_N_prev + integrand_N_curr)
            Sigma_bath += 0.25 * dx * (integrand_cov_prev + integrand_cov_curr)

        integrand_N_prev = integrand_N_curr
        integrand_cov_prev = integrand_cov_curr

        if ix > 0:
            T = Ts[ix - 1]
            T_UU = T[:TwoM, :TwoM]
            T_UV = T[:TwoM, TwoM:]
            T_VU = T[TwoM:, :TwoM]
            T_VV = T[TwoM:, TwoM:]
            Pg, Qg = Pg @ T_UU + Qg @ T_VU, Pg @ T_UV + Qg @ T_VV

    return N_loss, 0.5 * (Sigma_bath + Sigma_bath.conj().T)
