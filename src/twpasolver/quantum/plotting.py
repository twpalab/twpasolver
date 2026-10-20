"""Summary plot of :func:`twpasolver.quantum.quantum_response` results."""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]


def plot_quantum_response(
    results: Sequence[dict] | dict,
    labels: Sequence[str] | None = None,
    freqs_unit: str = "GHz",
    figsize=(10, 9),
):
    """
    Six panels: gain, QE / ideal, added noise (with Caves bound), signal-idler
    EPR variance, log-negativity and commutator defect.

    ``results`` is one result dict or a list of them (e.g. lossy vs lossless,
    or two mode arrays), ``labels`` their legend entries.
    """
    if isinstance(results, dict):
        results = [results]
    labels = labels or [f"run {j}" for j in range(len(results))]
    fig, ax = plt.subplots(3, 2, figsize=figsize, sharex=True)
    panels = [
        ("gain_db", "Gain S21 (dB)", False),
        ("qe_normalized", r"$\eta/\eta_\mathrm{ideal}$", False),
        ("added_noise", "Added noise (quanta)", False),
        ("epr_variance_db", "EPR variance s-i (dB)", False),
        ("log_negativity", r"$E_N$ (ebits)", False),
        ("commutator_defect", r"max $|SJS^\dagger + D - J|$", True),
    ]
    for r, lab, c in zip(results, labels, _COLORS):
        f = r["signal_freqs"]
        for a, (key, _, log) in zip(ax.flat, panels):
            y = np.asarray(r[key], dtype=float)
            if log:
                a.semilogy(f, np.maximum(y, 1e-17), color=c, lw=2, label=lab)
            else:
                a.plot(f, y, color=c, lw=2, label=lab)
    r = results[0]
    ax[1, 0].plot(
        r["signal_freqs"],
        r["caves_bound"],
        "--",
        color="#52514e",
        lw=1.2,
        label=r"Caves bound $\frac{1}{2}(1-1/G)$",
    )
    for a, (_, ylab, _) in zip(ax.flat, panels):
        a.set_ylabel(ylab)
        a.grid(alpha=0.3)
    for a in ax[-1]:
        a.set_xlabel(f"Signal frequency [{freqs_unit}]")
    ax[0, 0].legend(fontsize=8)
    ax[1, 0].legend(fontsize=8, handles=[ax[1, 0].lines[-1]])
    fig.tight_layout()
    return ax
