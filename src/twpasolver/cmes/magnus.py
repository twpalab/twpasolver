"""Fast 4th-order Magnus propagator for the Bogoliubov ODE."""

from __future__ import annotations

import numpy as np

_S3_6 = np.sqrt(3.0) / 6.0
_S3_12 = np.sqrt(3.0) / 12.0


def build_omegas(x, U_traj, V_traj):
    """Build all N Magnus Ω matrices in one vectorized pass."""
    TwoM = U_traj.shape[-1]
    FourM = 2 * TwoM
    N = len(x) - 1
    h = x[1:] - x[:-1]
    hc = h[:, None, None]
    h2c = hc**2
    a1 = 0.5 - _S3_6
    a2 = 0.5 + _S3_6
    Ul = U_traj[:-1]
    Ur = U_traj[1:]
    Vl = V_traj[:-1]
    Vr = V_traj[1:]
    U1 = (1 - a1) * Ul + a1 * Ur
    V1 = (1 - a1) * Vl + a1 * Vr
    U2 = (1 - a2) * Ul + a2 * Ur
    V2 = (1 - a2) * Vl + a2 * Vr
    V1c = np.conj(V1)
    V2c = np.conj(V2)
    U1c = np.conj(U1)
    U2c = np.conj(U2)
    OU = hc * 0.5 * (U1 + U2) + h2c * _S3_12 * (
        (U1 @ U2 + V1 @ V2c) - (U2 @ U1 + V2 @ V1c)
    )
    OV = hc * 0.5 * (V1 + V2) + h2c * _S3_12 * (
        (U1 @ V2 + V1 @ U2c) - (U2 @ V1 + V2 @ U1c)
    )
    Om = np.zeros((N, FourM, FourM), dtype=np.complex128)
    Om[:, :TwoM, :TwoM] = OU
    Om[:, :TwoM, TwoM:] = OV
    Om[:, TwoM:, :TwoM] = np.conj(OV)
    Om[:, TwoM:, TwoM:] = np.conj(OU)
    return Om


def pade5_batch(As):
    """Padé[5/5] expm, accurate for ‖A‖₁ < 0.25. (N,n,n)→(N,n,n)."""
    b = [30240.0, 15120.0, 3360.0, 420.0, 30.0, 1.0]
    A2 = As @ As
    A4 = A2 @ A2
    I = np.eye(As.shape[-1], dtype=As.dtype)[None]
    U = As @ (b[5] * A4 + b[3] * A2 + b[1] * I)
    V = b[4] * A4 + b[2] * A2 + b[0] * I
    return np.linalg.solve(V - U, V + U)


def pade7_batch(As):
    """Padé[7/7] expm, accurate for ‖A‖₁ < 0.95. (N,n,n)→(N,n,n)."""
    b = [17297280.0, 8648640.0, 1995840.0, 277200.0, 25200.0, 1512.0, 56.0, 1.0]
    A2 = As @ As
    A4 = A2 @ A2
    A6 = A4 @ A2
    I = np.eye(As.shape[-1], dtype=As.dtype)[None]
    U = As @ (b[7] * A6 + b[5] * A4 + b[3] * A2 + b[1] * I)
    V = b[6] * A6 + b[4] * A4 + b[2] * A2 + b[0] * I
    return np.linalg.solve(V - U, V + U)


def pade_batch_auto(As):
    """Batched expm with auto Padé order and scaling-and-squaring."""
    norms = np.linalg.norm(As, ord=1, axis=(-2, -1)).max()
    if norms <= 0.25:
        return pade5_batch(As)
    if norms <= 0.95:
        return pade7_batch(As)
    s = int(np.ceil(np.log2(norms / 0.95)))
    Ts = pade7_batch(As / (2**s))
    for _ in range(s):
        Ts = Ts @ Ts
    return Ts


def tree_product(Ts):
    """Compute T[N-1]@...@T[0] via balanced binary tree."""
    Ms = Ts.copy()
    while len(Ms) > 1:
        n = len(Ms)
        np2 = n // 2
        products = Ms[1 : np2 * 2 : 2] @ Ms[: np2 * 2 : 2]
        Ms = np.concatenate([products, Ms[-1:]], axis=0) if n % 2 else products
    return Ms[0]


def propagate_upper_block(x, U_traj, V_traj, M, save_trajectory=False):
    """
    4th-order Magnus propagator for the Bogoliubov ODE.

    Parameters
    ----------
    x            : (N+1,) float
    U_traj       : (N+1, 2M, 2M) complex128
    V_traj       : (N+1, 2M, 2M) complex128
    M            : int
    save_trajectory : bool

    Returns
    -------
    P, Q : (2M, 2M) complex128
    snapshots : list[(P_i, Q_i)]  — only when save_trajectory=True
    """
    TwoM = 2 * M
    Omegas = build_omegas(x, U_traj, V_traj)
    Ts = pade_batch_auto(Omegas)

    if not save_trajectory:
        T_tot = tree_product(Ts)
        return T_tot[:TwoM, :TwoM], np.conj(T_tot[TwoM:, :TwoM])

    snapshots = [
        (np.eye(TwoM, dtype=np.complex128), np.zeros((TwoM, TwoM), dtype=np.complex128))
    ]
    acc = np.eye(4 * M, dtype=np.complex128)
    for T in Ts:
        acc = T @ acc
        snapshots.append((acc[:TwoM, :TwoM].copy(), np.conj(acc[TwoM:, :TwoM]).copy()))
    return acc[:TwoM, :TwoM], np.conj(acc[TwoM:, :TwoM]), snapshots
