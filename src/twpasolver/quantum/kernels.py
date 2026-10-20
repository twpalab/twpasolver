"""
Compiled kernels of the Gaussian cascade.

The matrices are small (4M x 4M with M the number of weak modes, typically
8 to 24), so the kernels use their own loop-based linear algebra instead of
BLAS/LAPACK calls: no per-call overhead and no nested threading inside the
``prange`` loops, which run over steps (step blocks) and over pairs of blocks
(each level of the cascade tree).

Conventions are those of :mod:`twpasolver.quantum.gaussian`: a block is
``(S, C, D)`` on ``[left port; right port]``, each port ``(a_1..a_M,
a_1^dag..a_M^dag)``, with ``S J S^dag + D = J``.
"""

from __future__ import annotations

import numba as nb
import numpy as np

# --------------------------------------------------------------------------- #
# Small dense linear algebra
# --------------------------------------------------------------------------- #


@nb.njit(cache=True, inline="always")
def _mm(A, B, out):
    n, m = A.shape
    p = B.shape[1]
    for i in range(n):
        for j in range(p):
            out[i, j] = 0.0
        for k in range(m):
            a = A[i, k]
            if a != 0.0:
                for j in range(p):
                    out[i, j] += a * B[k, j]
    return out


@nb.njit(cache=True)
def _matmul(A, B):
    return _mm(A, B, np.empty((A.shape[0], B.shape[1]), dtype=np.complex128))


@nb.njit(cache=True)
def _mmh(A, B):
    """``A @ B^dag``."""
    n, m = A.shape
    p = B.shape[0]
    out = np.zeros((n, p), dtype=np.complex128)
    for i in range(n):
        for j in range(p):
            acc = 0.0 + 0.0j
            for k in range(m):
                acc += A[i, k] * np.conj(B[j, k])
            out[i, j] = acc
    return out


@nb.njit(cache=True)
def _inv(A):
    """Gauss-Jordan inverse with partial pivoting."""
    n = A.shape[0]
    M = A.copy()
    I = np.zeros((n, n), dtype=np.complex128)
    for i in range(n):
        I[i, i] = 1.0
    for c in range(n):
        p = c
        best = np.abs(M[c, c])
        for r in range(c + 1, n):
            v = np.abs(M[r, c])
            if v > best:
                best = v
                p = r
        if p != c:
            for j in range(n):
                tmp = M[c, j]
                M[c, j] = M[p, j]
                M[p, j] = tmp
                tmp = I[c, j]
                I[c, j] = I[p, j]
                I[p, j] = tmp
        piv = 1.0 / M[c, c]
        for j in range(n):
            M[c, j] *= piv
            I[c, j] *= piv
        for r in range(n):
            if r != c:
                f = M[r, c]
                if f != 0.0:
                    for j in range(n):
                        M[r, j] -= f * M[c, j]
                        I[r, j] -= f * I[c, j]
    return I


@nb.njit(cache=True)
def _expm(A):
    """``expm`` by Pade [7/7] with scaling and squaring (diagonal Pade keeps
    the result in the group of A's Lie algebra)."""
    n = A.shape[0]
    norm = 0.0
    for j in range(n):
        s = 0.0
        for i in range(n):
            s += np.abs(A[i, j])
        if s > norm:
            norm = s
    sq = 0
    if norm > 0.95:
        sq = int(np.ceil(np.log2(norm / 0.95)))
    X = A / (2.0**sq)
    X2 = _matmul(X, X)
    X4 = _matmul(X2, X2)
    X6 = _matmul(X4, X2)
    b0, b1, b2, b3 = 17297280.0, 8648640.0, 1995840.0, 277200.0
    b4, b5, b6, b7 = 25200.0, 1512.0, 56.0, 1.0
    Uin = b7 * X6 + b5 * X4 + b3 * X2
    V = b6 * X6 + b4 * X4 + b2 * X2
    for i in range(n):
        Uin[i, i] += b1
        V[i, i] += b0
    U = _matmul(X, Uin)
    T = _matmul(_inv(V - U), V + U)
    for _ in range(sq):
        T = _matmul(T, T)
    return T


# --------------------------------------------------------------------------- #
# Generator and Magnus exponent
# --------------------------------------------------------------------------- #


@nb.njit(cache=True, parallel=True)
def generator_nb(x, pf, coeff, phase, slots, tgt, src, scale, M):
    """
    Flux-basis generator ``K(x)`` on ``(a^+, a^-, a^{+*}, a^{-*})`` at every ``x``.

    ``pf`` (n_x, 4Np+1) pump-field vectors; terms in the current basis
    (``coeff``, ``phase``, ``slots``, ``tgt``, ``src``); ``scale`` (M,)
    ``sqrt(W/omega)`` of the active modes.
    """
    nx = x.shape[0]
    n2 = 2 * M
    n4 = 4 * M
    K = np.zeros((nx, n4, n4), dtype=np.complex128)
    s_full = np.empty(n4)
    for j in range(M):
        s_full[j] = scale[j]
        s_full[M + j] = scale[j]
        s_full[2 * M + j] = scale[j]
        s_full[3 * M + j] = scale[j]
    for t in nb.prange(nx):
        top = np.zeros((n2, n4), dtype=np.complex128)
        for q in range(coeff.shape[0]):
            v = (
                coeff[q]
                * np.exp(1j * phase[q] * x[t])
                * pf[t, slots[q, 0]]
                * pf[t, slots[q, 1]]
                * pf[t, slots[q, 2]]
            )
            top[tgt[q], src[q]] += v
        for r in range(n2):
            for c in range(n4):
                top[r, c] *= s_full[r] / s_full[c]
        for r in range(n2):
            for c in range(n2):
                K[t, r, c] = top[r, c]  # K1
                K[t, r, n2 + c] = top[r, n2 + c]  # K2
                K[t, n2 + r, c] = np.conj(top[r, n2 + c])  # K2*
                K[t, n2 + r, n2 + c] = np.conj(top[r, c])  # K1*
    return K


def collocation_weights(n_q: int):
    """
    Gauss-Legendre nodes ``c``, weights ``w`` on [0, 1] and the integration
    matrix ``Q[a, b] = int_0^{c_a} l_b(t) dt`` of the Lagrange basis.
    """
    z, w = np.polynomial.legendre.leggauss(n_q)
    c = 0.5 * (z + 1.0)
    w = 0.5 * w
    Q = np.zeros((n_q, n_q))
    for b in range(n_q):
        others = np.delete(c, b)
        poly = np.poly1d(others, r=True) / np.prod(c[b] - others)
        P = np.polyint(poly)
        Q[:, b] = P(c) - P(0.0)
    return c, w, Q


@nb.njit(cache=True, parallel=True)
def magnus_omega_nb(K, w, Q, h):
    """
    Magnus exponent per step from the generator at ``n_q`` collocation nodes.

    ``K`` (N * n_q, n, n) ordered step-major. ``Omega = Omega_1 + Omega_2`` with
    ``Omega_1 = h sum_a w_a K_a`` and
    ``Omega_2 = (h^2 / 2) sum_a w_a [K_a, sum_b Q_ab K_b]``
    (Gauss collocation of ``1/2 int_0^h dt1 int_0^t1 dt2 [K(t1), K(t2)]``).
    With 2 nodes this is the classical 4th-order Magnus scheme.
    """
    nq = w.shape[0]
    n = K.shape[1]
    N = K.shape[0] // nq
    Om = np.zeros((N, n, n), dtype=np.complex128)
    for s in nb.prange(N):
        base = s * nq
        O1 = np.zeros((n, n), dtype=np.complex128)
        O2 = np.zeros((n, n), dtype=np.complex128)
        for a in range(nq):
            Ka = K[base + a]
            O1 += w[a] * Ka
            Ia = np.zeros((n, n), dtype=np.complex128)
            for b in range(nq):
                if Q[a, b] != 0.0:
                    Ia += Q[a, b] * K[base + b]
            O2 += w[a] * (_matmul(Ka, Ia) - _matmul(Ia, Ka))
        Om[s] = h * O1 + 0.5 * h * h * O2
    return Om


# --------------------------------------------------------------------------- #
# Step blocks with closed-form loss
# --------------------------------------------------------------------------- #


@nb.njit(cache=True, parallel=True)
def step_blocks_nb(Om, t, c_noise, d_noise, track_D):
    """
    ``loss(h/2) * parametric(h) * loss(h/2)`` for every step, in closed form.

    Loss blocks do not reflect, so the two star products reduce to
    ``S = L S_P L``, ``C = L S_P diag(c) S_P^dag L + diag(c)`` (same for ``D``)
    with ``L = diag(t)`` on all port entries; ``c``/``d`` are the covariance /
    commutator of the noise injected by one half-step of loss
    (``(1 - t^2)(n_th + 1/2)`` and ``(1 - t^2) J``). Also returns the largest
    symplectic defect ``|S_P J S_P^dag - J|`` of the lossless steps.

    ``Om`` (N, 4M, 4M) on ``(a^+, a^-, a^{+*}, a^{-*})``; ``t``, ``c_noise``,
    ``d_noise`` (4M,) in port order ``[left a, left a^dag, right a, right a^dag]``.
    """
    N, n, _ = Om.shape
    M = n // 4
    m2 = 2 * M
    perm = np.empty(n, dtype=np.int64)
    for j in range(M):
        perm[j] = j  # a+
        perm[M + j] = 2 * M + j  # a+*
        perm[2 * M + j] = M + j  # a-
        perm[3 * M + j] = 3 * M + j  # a-*
    Jp = np.empty(n)
    for j in range(n):
        Jp[j] = 1.0 if (j % m2) < M else -1.0
    S = np.empty((N, n, n), dtype=np.complex128)
    C = np.empty((N, n, n), dtype=np.complex128)
    D = np.empty((N if track_D else 1, n, n), dtype=np.complex128)
    defect = np.zeros(N)
    for s in nb.prange(N):
        T = _expm(Om[s])
        Tp = np.empty((n, n), dtype=np.complex128)
        for i in range(n):
            for j in range(n):
                Tp[i, j] = T[perm[i], perm[j]]
        TFF = Tp[:m2, :m2].copy()
        TFB = Tp[:m2, m2:].copy()
        TBF = Tp[m2:, :m2].copy()
        iBB = _inv(Tp[m2:, m2:].copy())
        SP = np.empty((n, n), dtype=np.complex128)
        X = _matmul(iBB, TBF)
        SP[:m2, :m2] = -X
        SP[:m2, m2:] = iBB
        SP[m2:, :m2] = TFF - _matmul(TFB, X)
        SP[m2:, m2:] = _matmul(TFB, iBB)
        # symplectic defect of the lossless step
        SJ = SP.copy()
        for j in range(n):
            for i in range(n):
                SJ[i, j] *= Jp[j]
        R = _mmh(SJ, SP)
        dmax = 0.0
        for i in range(n):
            R[i, i] -= Jp[i]
            for j in range(n):
                v = np.abs(R[i, j])
                if v > dmax:
                    dmax = v
        defect[s] = dmax
        # closed-form loss on both sides
        Sc = SP.copy()
        for j in range(n):
            for i in range(n):
                Sc[i, j] *= c_noise[j]
        Cs = _mmh(Sc, SP)
        if track_D:
            Sd = SP.copy()
            for j in range(n):
                for i in range(n):
                    Sd[i, j] *= d_noise[j]
            Ds = _mmh(Sd, SP)
        for i in range(n):
            for j in range(n):
                f = t[i] * t[j]
                S[s, i, j] = t[i] * SP[i, j] * t[j]
                C[s, i, j] = f * Cs[i, j]
                if track_D:
                    D[s, i, j] = f * Ds[i, j]
            C[s, i, i] += c_noise[i]
            if track_D:
                D[s, i, i] += d_noise[i]
    return S, C, D, defect


# --------------------------------------------------------------------------- #
# Redheffer star product and tree cascade
# --------------------------------------------------------------------------- #


@nb.njit(cache=True)
def _star(SA, CA, DA, SB, CB, DB, track_D):
    n = SA.shape[0]
    m2 = n // 2
    All = SA[:m2, :m2].copy()
    Alr = SA[:m2, m2:].copy()
    Arl = SA[m2:, :m2].copy()
    Arr = SA[m2:, m2:].copy()
    Bll = SB[:m2, :m2].copy()
    Blr = SB[:m2, m2:].copy()
    Brl = SB[m2:, :m2].copy()
    Brr = SB[m2:, m2:].copy()
    IX = -_matmul(Arr, Bll)
    for i in range(m2):
        IX[i, i] += 1.0
    X = _inv(IX)
    # Y = (I - Bll Arr)^-1 = I + Bll X Arr (push-through identity)
    BllX = _matmul(Bll, X)
    Y = _matmul(BllX, Arr)
    for i in range(m2):
        Y[i, i] += 1.0
    XArl = _matmul(X, Arl)
    AlrBllX = _matmul(Alr, BllX)
    BrlX = _matmul(Brl, X)
    AlrY = _matmul(Alr, Y)
    BrlXArr = _matmul(BrlX, Arr)
    S = np.empty((n, n), dtype=np.complex128)
    S[:m2, :m2] = All + _matmul(Alr, _matmul(Bll, XArl))
    S[:m2, m2:] = _matmul(AlrY, Blr)
    S[m2:, :m2] = _matmul(Brl, XArl)
    S[m2:, m2:] = Brr + _matmul(BrlXArr, Blr)
    LA = np.zeros((n, n), dtype=np.complex128)
    LB = np.zeros((n, n), dtype=np.complex128)
    for i in range(m2):
        LA[i, i] = 1.0
        LB[m2 + i, m2 + i] = 1.0
    LA[:m2, m2:] = AlrBllX
    LA[m2:, m2:] = BrlX
    LB[:m2, :m2] = AlrY
    LB[m2:, :m2] = BrlXArr
    C = _mmh(_matmul(LA, CA), LA) + _mmh(_matmul(LB, CB), LB)
    if track_D:
        D = _mmh(_matmul(LA, DA), LA) + _mmh(_matmul(LB, DB), LB)
    else:
        D = DA
    fb = 0.0
    for i in range(m2):
        for j in range(m2):
            v = np.abs(X[i, j])
            if v > fb:
                fb = v
    return S, C, D, fb


@nb.njit(cache=True, parallel=True)
def cascade_nb(S, C, D, track_D):
    """
    Reduce an ordered stack of blocks (left to right) by a balanced tree.

    ``D`` has the stack length if ``track_D``, else a single dummy entry.
    Returns ``S, C, D`` of the device and the largest round-trip gain
    ``max |(I - A_rr B_ll)^-1|`` met in the reduction (self-oscillation flag).
    """
    count = S.shape[0]
    fb_max = 1.0
    while count > 1:
        half = count // 2
        odd = count % 2
        n_new = half + odd
        S2 = np.empty((n_new, S.shape[1], S.shape[2]), dtype=np.complex128)
        C2 = np.empty_like(S2)
        D2 = np.empty(
            (n_new if track_D else 1, S.shape[1], S.shape[2]), dtype=np.complex128
        )
        if not track_D:
            D2[0] = D[0]
        fbs = np.zeros(half)
        for p in nb.prange(half):
            ia = 2 * p
            ib = 2 * p + 1
            da = D[ia] if track_D else D[0]
            db = D[ib] if track_D else D[0]
            s_, c_, d_, fb = _star(S[ia], C[ia], da, S[ib], C[ib], db, track_D)
            fbs[p] = fb
            S2[p] = s_
            C2[p] = c_
            if track_D:
                D2[p] = d_
        for p in range(half):
            if fbs[p] > fb_max:
                fb_max = fbs[p]
        if odd:
            S2[half] = S[count - 1]
            C2[half] = C[count - 1]
            if track_D:
                D2[half] = D[count - 1]
        S, C, D = S2, C2, D2
        count = n_new
    return S[0], C[0], D[0], fb_max
