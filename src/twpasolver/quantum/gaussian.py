"""
Gaussian-channel cascade: every element of the device as a scattering block.

A block has a left and a right port. Each port carries the weak modes in the
doubled representation ``(a_1..a_M, a_1^dag..a_M^dag)``. A block is the triple
``(S, D, C)`` acting on ``[left port; right port]``:

* ``y = S u + n``: outputs (left: backward-going, right: forward-going) from
  inputs (left: forward-going, right: backward-going) plus internal noise ``n``;
* ``D = [n, n^dag]`` (commutator of the internal noise) and
  ``C = <{n, n^dag}>/2`` (its symmetrised covariance).

Every block satisfies ``S J S^dag + D = J`` with ``J = diag(Jp, Jp)``,
``Jp = diag(I_M, -I_M)``, and the Redheffer star product preserves it, so the
whole device preserves the commutators to round-off for any step size, loss
and reflection. Blocks:

* parametric step (lossless): 4th-order Magnus + diagonal Pade (structure
  preserving), converted from transfer to scattering form;
* loss: beam splitter per mode and direction, ``t = exp(-alpha h)``, noise
  ``D = (1 - t^2) J``, ``C = (1 - t^2)(n_th + 1/2)``;
* edges: per-mode passive 2x2 scattering between the external line and the
  Bloch modes, from the connector ABCD (noise added only if lossy).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

Block = Tuple[np.ndarray, np.ndarray, np.ndarray]

_S3_6 = np.sqrt(3.0) / 6.0
_S3_12 = np.sqrt(3.0) / 12.0


# --------------------------------------------------------------------------- #
# Matrix exponential (batched diagonal Pade with scaling and squaring)
# --------------------------------------------------------------------------- #


def _pade7(As):
    b = [17297280.0, 8648640.0, 1995840.0, 277200.0, 25200.0, 1512.0, 56.0, 1.0]
    A2 = As @ As
    A4 = A2 @ A2
    A6 = A4 @ A2
    I = np.eye(As.shape[-1], dtype=As.dtype)[None]
    U = As @ (b[7] * A6 + b[5] * A4 + b[3] * A2 + b[1] * I)
    V = b[6] * A6 + b[4] * A4 + b[2] * A2 + b[0] * I
    return np.linalg.solve(V - U, V + U)


def expm_batch(As: np.ndarray) -> np.ndarray:
    """Batched ``expm`` (Pade [7/7] + scaling and squaring, accurate to ~1e-15)."""
    norm = np.linalg.norm(As, ord=1, axis=(-2, -1)).max() if len(As) else 0.0
    s = max(0, int(np.ceil(np.log2(norm / 0.95)))) if norm > 0.95 else 0
    T = _pade7(As / (2**s))
    for _ in range(s):
        T = T @ T
    return T


def magnus4_omega(K1: np.ndarray, K2: np.ndarray, h: float) -> np.ndarray:
    """4th-order Magnus exponent from the generator at the two Gauss nodes."""
    return 0.5 * h * (K1 + K2) + _S3_12 * h * h * (K2 @ K1 - K1 @ K2)


def gauss_nodes(x0: np.ndarray, h: float) -> tuple[np.ndarray, np.ndarray]:
    """The two Gauss-Legendre nodes of each step ``[x0, x0 + h]``."""
    return x0 + (0.5 - _S3_6) * h, x0 + (0.5 + _S3_6) * h


# --------------------------------------------------------------------------- #
# Blocks
# --------------------------------------------------------------------------- #


def port_J(M: int) -> np.ndarray:
    return np.r_[np.ones(M), -np.ones(M)]


def full_J(M: int) -> np.ndarray:
    j = port_J(M)
    return np.diag(np.r_[j, j]).astype(np.complex128)


def transfer_to_scattering(T: np.ndarray) -> np.ndarray:
    """
    Scattering form of transfer matrices on ``v = (a^+, a^-, a^{+*}, a^{-*})``.

    Left port: input ``F(x) = (a^+, a^{+*})``, output ``B(x) = (a^-, a^{-*})``;
    right port: input ``B(x+h)``, output ``F(x+h)``.
    """
    n4 = T.shape[-1]
    M = n4 // 4
    perm = np.r_[0:M, 2 * M : 3 * M, M : 2 * M, 3 * M : 4 * M]
    Tp = T[:, perm][:, :, perm]
    m2 = 2 * M
    TFF, TFB = Tp[:, :m2, :m2], Tp[:, :m2, m2:]
    TBF, TBB = Tp[:, m2:, :m2], Tp[:, m2:, m2:]
    iBB = np.linalg.inv(TBB)
    S = np.empty_like(Tp)
    S[:, :m2, :m2] = -iBB @ TBF
    S[:, :m2, m2:] = iBB
    S[:, m2:, :m2] = TFF - TFB @ iBB @ TBF
    S[:, m2:, m2:] = TFB @ iBB
    return S


def parametric_blocks(K_nodes1: np.ndarray, K_nodes2: np.ndarray, h: float) -> Block:
    """Lossless parametric steps (no internal noise)."""
    T = expm_batch(magnus4_omega(K_nodes1, K_nodes2, h))
    S = transfer_to_scattering(T)
    Z = np.zeros_like(S)
    return S, Z, Z.copy()


def loss_block(t: np.ndarray, n_th: np.ndarray, n: int = 1) -> Block:
    """
    Beam-splitter loss on every mode and direction, repeated ``n`` times.

    ``t`` and ``n_th`` have shape (M,): amplitude transmission and bath
    occupation of each weak mode.
    """
    M = len(t)
    m2 = 2 * M
    tt = np.r_[t, t].astype(np.complex128)
    S = np.zeros((4 * M, 4 * M), dtype=np.complex128)
    S[m2:, :m2] = np.diag(tt)  # forward: left in -> right out
    S[:m2, m2:] = np.diag(tt)  # backward: right in -> left out
    r2 = np.r_[1 - t**2, 1 - t**2]
    D = np.diag(np.r_[r2 * port_J(M), r2 * port_J(M)]).astype(np.complex128)
    nn = np.r_[n_th, n_th, n_th, n_th]
    C = np.diag(np.r_[r2, r2] * (nn + 0.5)).astype(np.complex128)
    rep = lambda A: np.repeat(A[None], n, axis=0)
    return rep(S), rep(D), rep(C)


def passive_block(P: np.ndarray, n_th: np.ndarray) -> Block:
    """
    Doubled block of a passive (beam-splitter-like) element.

    ``P`` is (2M, 2M) on ``[left modes; right modes]`` (annihilation operators
    only); outputs ordered as inputs. Noise from ``I - P P^dag`` if lossy.
    """
    M = P.shape[0] // 2
    idx_a = np.r_[0:M, 2 * M : 3 * M]  # a entries of left and right ports
    idx_c = np.r_[M : 2 * M, 3 * M : 4 * M]  # a^dag entries
    S = np.zeros((4 * M, 4 * M), dtype=np.complex128)
    S[np.ix_(idx_a, idx_a)] = P
    S[np.ix_(idx_c, idx_c)] = P.conj()
    R = np.eye(2 * M) - P @ P.conj().T
    R = 0.5 * (R + R.conj().T)
    w, V = np.linalg.eigh(R)
    R = (V * np.clip(w, 0, None)) @ V.conj().T  # remove round-off negatives
    D = np.zeros_like(S)
    C = np.zeros_like(S)
    D[np.ix_(idx_a, idx_a)] = R
    D[np.ix_(idx_c, idx_c)] = -R.conj()
    nn = np.r_[n_th, n_th]
    G = np.sqrt(nn + 0.5)
    C[np.ix_(idx_a, idx_a)] = G[:, None] * R * G[None, :]
    C[np.ix_(idx_c, idx_c)] = (G[:, None] * R * G[None, :]).conj()
    return S[None], D[None], C[None]


# --------------------------------------------------------------------------- #
# Redheffer star product with noise
# --------------------------------------------------------------------------- #


def star(A: Block, B: Block, track: list | None = None) -> Block:
    """Cascade ``A`` (left) with ``B`` (right), batched over the leading axis."""
    SA, DA, CA = A
    SB, DB, CB = B
    m2 = SA.shape[-1] // 2
    All, Alr = SA[:, :m2, :m2], SA[:, :m2, m2:]
    Arl, Arr = SA[:, m2:, :m2], SA[:, m2:, m2:]
    Bll, Blr = SB[:, :m2, :m2], SB[:, :m2, m2:]
    Brl, Brr = SB[:, m2:, :m2], SB[:, m2:, m2:]
    I = np.eye(m2, dtype=np.complex128)[None]
    X = np.linalg.inv(I - Arr @ Bll)
    Y = np.linalg.inv(I - Bll @ Arr)
    if track is not None:
        track.append(float(np.abs(X).max()))
    S = np.empty_like(SA)
    S[:, :m2, :m2] = All + Alr @ Bll @ X @ Arl
    S[:, :m2, m2:] = Alr @ Y @ Blr
    S[:, m2:, :m2] = Brl @ X @ Arl
    S[:, m2:, m2:] = Brr + Brl @ X @ Arr @ Blr
    Z = np.zeros_like(X)
    LA = np.concatenate(
        [np.concatenate([I + Z, Alr @ Bll @ X], 2), np.concatenate([Z, Brl @ X], 2)], 1
    )
    LB = np.concatenate(
        [np.concatenate([Alr @ Y, Z], 2), np.concatenate([Brl @ X @ Arr, I + Z], 2)], 1
    )
    LAh = np.swapaxes(LA.conj(), -1, -2)
    LBh = np.swapaxes(LB.conj(), -1, -2)
    D = LA @ DA @ LAh + LB @ DB @ LBh
    C = LA @ CA @ LAh + LB @ CB @ LBh
    return S, D, C


def cascade(blocks: Block, track: list | None = None) -> Block:
    """Reduce an ordered stack of blocks (left to right) by a balanced tree."""
    S, D, C = blocks
    while S.shape[0] > 1:
        n = S.shape[0]
        h = n // 2
        a = (S[0 : 2 * h : 2], D[0 : 2 * h : 2], C[0 : 2 * h : 2])
        b = (S[1 : 2 * h : 2], D[1 : 2 * h : 2], C[1 : 2 * h : 2])
        S2, D2, C2 = star(a, b, track)
        if n % 2:
            S2 = np.concatenate([S2, S[-1:]])
            D2 = np.concatenate([D2, D[-1:]])
            C2 = np.concatenate([C2, C[-1:]])
        S, D, C = S2, D2, C2
    return S, D, C


def concat(*blocks: Block) -> Block:
    """Stack blocks along the cascade axis."""
    return tuple(np.concatenate([b[j] for b in blocks]) for j in range(3))  # type: ignore


def commutator_defect(blk: Block) -> float:
    """``max |S J S^dag + D - J|`` of a single block."""
    S, D, _ = blk
    S, D = S[0], D[0]
    J = full_J(S.shape[-1] // 4)
    return float(np.abs(S @ J @ S.conj().T + D - J).max())
