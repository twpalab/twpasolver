"""
Quantum observables from the device block ``(S, D, C)``.

Port vectors are ``[left external port; right external port]``, each
``(a_1..a_M, a_1^dag..a_M^dag)`` over the active weak modes. All covariances
are symmetrised, ``<{v, v^dag}>/2``, so vacuum is ``1/2``.
"""

from __future__ import annotations

import numpy as np

from twpasolver.quantum.gaussian import full_J

HBAR = 1.054571817e-34
KB = 1.380649e-23


def bose(omega: np.ndarray, temperature: float) -> np.ndarray:
    """Thermal occupation at angular frequency ``omega``."""
    if temperature <= 0:
        return np.zeros_like(omega, dtype=np.float64)
    return 1.0 / np.expm1(HBAR * omega / (KB * temperature))


def port_index(M: int, port: int, mode: int, dagger: bool = False) -> int:
    """Index of ``a_mode`` (or its adjoint) of external port 0 (input) or 1 (output)."""
    return port * 2 * M + (M if dagger else 0) + mode


def quadrature_covariance(V: np.ndarray) -> np.ndarray:
    """
    Real covariance of ``(q_1..q_m, p_1..p_m)`` from the complex one of
    ``(b_1..b_m, b_1^dag..b_m^dag)``; ``q = (b + b^dag)/sqrt2``, vacuum ``1/2``.
    """
    m = V.shape[0] // 2
    I = np.eye(m)
    U = np.block([[I, I], [-1j * I, 1j * I]]) / np.sqrt(2)
    Vx = U @ V @ U.conj().T
    return 0.5 * (Vx + Vx.T).real


def log_negativity(Vx: np.ndarray) -> float:
    """Logarithmic negativity of a two-mode Gaussian state, ordering ``(q1, q2, p1, p2)``."""
    P = np.diag([1, 1, 1, -1])
    Vpt = P @ Vx @ P
    Om = np.block([[np.zeros((2, 2)), np.eye(2)], [-np.eye(2), np.zeros((2, 2))]])
    nu = np.sort(np.abs(np.linalg.eigvals(1j * Om @ Vpt)))
    return float(max(0.0, -np.log2(2 * nu[0])))


def epr_variance(Vx: np.ndarray) -> float:
    """
    ``min Var[(X_1(t1) - X_2(t2))/sqrt2]`` over both LO phases, relative to vacuum.

    ``X(t) = q cos t + p sin t``; ordering ``(q1, q2, p1, p2)``. Below 1 means
    two-mode squeezing. The variance ``w(t1, t2)^T Vx w(t1, t2)`` is evaluated
    on a vectorised grid and refined by Newton steps on the two angles.
    """

    def var_grid(a, b):
        ca, sa, cb, sb = np.cos(a), np.sin(a), np.cos(b), np.sin(b)
        w = np.stack([ca, -cb, sa, -sb])
        return np.einsum("i...,ij,j...->...", w, Vx, w)

    g = np.linspace(
        0, np.pi, 64, endpoint=False
    )  # var is pi-periodic in (t1, t2) jointly
    A, B = np.meshgrid(g, 2 * g, indexing="ij")
    vals = var_grid(A, B)
    k = np.unravel_index(np.argmin(vals), vals.shape)
    th = np.array([A[k], B[k]])
    best = vals[k]
    eps = 1e-4
    for _ in range(20):
        f0 = var_grid(*th)
        gx = (var_grid(th[0] + eps, th[1]) - var_grid(th[0] - eps, th[1])) / (2 * eps)
        gy = (var_grid(th[0], th[1] + eps) - var_grid(th[0], th[1] - eps)) / (2 * eps)
        hxx = (
            var_grid(th[0] + eps, th[1]) - 2 * f0 + var_grid(th[0] - eps, th[1])
        ) / eps**2
        hyy = (
            var_grid(th[0], th[1] + eps) - 2 * f0 + var_grid(th[0], th[1] - eps)
        ) / eps**2
        hxy = (
            var_grid(th[0] + eps, th[1] + eps)
            - var_grid(th[0] + eps, th[1] - eps)
            - var_grid(th[0] - eps, th[1] + eps)
            + var_grid(th[0] - eps, th[1] - eps)
        ) / (4 * eps**2)
        H = np.array([[hxx, hxy], [hxy, hyy]])
        try:
            step = np.linalg.solve(H, [gx, gy])
        except np.linalg.LinAlgError:
            break
        if not np.all(np.isfinite(step)) or np.linalg.eigvalsh(H).min() <= 0:
            break
        th_new = th - step
        f_new = var_grid(*th_new)
        if f_new > f0:
            break
        th = th_new
        best = min(best, f_new)
        if np.abs(step).max() < 1e-12:
            break
    return float(best)


def observables(
    S: np.ndarray,
    D: np.ndarray,
    C: np.ndarray,
    labels: list[str],
    n_in: np.ndarray,
    signal: int = 0,
    idler: int | None = None,
) -> dict[str, object]:
    """
    Observables of one frequency point.

    Parameters
    ----------
    S, D, C : (4M, 4M) device block (``D`` may be None: no commutator check)
    labels : names of the M active weak modes
    n_in : (M,) thermal occupation of the inputs (both external ports)
    signal, idler : positions of signal and idler among the active modes
    """
    M = len(labels)
    J = full_J(M)
    nn = np.r_[n_in, n_in, n_in, n_in]
    Sig_in = np.diag(nn + 0.5)
    Sig_out = S @ Sig_in @ S.conj().T + C
    Sig_out = 0.5 * (Sig_out + Sig_out.conj().T)
    comm = S @ J @ S.conj().T + D if D is not None else None

    i_in = port_index(M, 0, signal)
    o_s = port_index(M, 1, signal)
    G = abs(S[o_s, i_in]) ** 2
    N_out = Sig_out[o_s, o_s].real
    N_sig = G * (n_in[signal] + 0.5)
    added = (N_out - N_sig) / G
    eta = 0.5 * G / N_out
    eta_ideal = G / (2 * G - 1) if G > 1 else 1.0

    # Noise budget at the signal output: each external input and internal noise.
    budget: dict[str, float] = {}
    for port, name in ((0, "in"), (1, "out")):
        for m, lab in enumerate(labels):
            ia = port_index(M, port, m)
            ic = port_index(M, port, m, True)
            v = (abs(S[o_s, ia]) ** 2 + abs(S[o_s, ic]) ** 2) * (n_in[m] + 0.5)
            budget[f"{lab}@{name}"] = v / N_out
    budget["internal_loss"] = C[o_s, o_s].real / N_out

    out = {
        "gain": G,
        "gain_db": 10 * np.log10(G),
        "S21": S[o_s, i_in],
        "S11": S[port_index(M, 0, signal), i_in],
        "S22": S[o_s, port_index(M, 1, signal)],
        "S12": S[port_index(M, 0, signal), port_index(M, 1, signal)],
        "added_noise": added,
        "caves_bound": 0.5 * (1 - 1 / G) if G > 1 else 0.0,
        "quantum_efficiency": eta,
        "qe_ideal": eta_ideal,
        "qe_normalized": eta / eta_ideal,
        "noise_budget": budget,
        "commutator_signal": comm[o_s, o_s].real if comm is not None else np.nan,
        "commutator_defect": (
            float(np.abs(comm - J).max()) if comm is not None else np.nan
        ),
        "covariance_out": Sig_out[2 * M :, 2 * M :],
    }
    if idler is not None:
        o_i = port_index(M, 1, idler)
        rows = [o_s, o_i, o_s + M, o_i + M]
        Vx = quadrature_covariance(Sig_out[np.ix_(rows, rows)])
        lam = np.linalg.eigvalsh(Vx)
        out.update(
            {
                "idler_gain_db": 10
                * np.log10(abs(S[o_i, port_index(M, 0, signal, True)]) ** 2),
                "squeezing_db": 10 * np.log10(2 * lam[0]),
                "antisqueezing_db": 10 * np.log10(2 * lam[-1]),
                "epr_variance_db": 10 * np.log10(epr_variance(Vx)),
                "log_negativity": log_negativity(Vx),
                "quadrature_cov_si": Vx,
            }
        )
    return out
