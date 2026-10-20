"""
Schematic of the fixed ("edge") Bloch basis, used by ``06_edge_basis.ipynb``.

(a) Native basis: the two Bloch eigenvectors close up as the mode approaches
    the band edge and coincide there (exceptional point); W, the area they
    span, vanishes and every coupling grows as 1/W.
(b) Fixed basis: the pair taken at f_ref on the passband side stays put; the
    true eigenvectors are combinations of it, and W = W~ (v+ w- - v- w+)
    vanishes through the coefficients only. W~ stays finite.
(c) Along the line: the two fixed-basis amplitudes are coupled by the Bragg
    term of K~ at every supercell, and driven by the nonlinear source.
(d) One pass of the forward-backward solver for a flagged mode.

    from edge_basis_diagram import draw
    fig = draw()
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from palette import PALETTE

FWD = PALETTE["magenta"]  # forward amplitude c+
BWD = PALETTE["teal"]  # backward amplitude c-
INK = PALETTE["ink"]
GREY = PALETTE["grey"]
LIGHT = PALETTE["grey_light"]


def _arrow(
    ax, p0, p1, color, lw=1.6, alpha=1.0, ls="-", style="-|>", ms=9, rad=0.0, z=3
):
    ax.add_patch(
        FancyArrowPatch(
            p0,
            p1,
            arrowstyle=style,
            mutation_scale=ms,
            color=color,
            lw=lw,
            alpha=alpha,
            linestyle=ls,
            connectionstyle=f"arc3,rad={rad}",
            zorder=z,
        )
    )


def _vector_panel(ax, title):
    ax.set_xlim(-0.3, 1.45)
    ax.set_ylim(-0.3, 1.45)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.text(-0.3, 1.45, title, ha="left", va="top", fontsize=8, color=INK)


def _panel_native(ax):
    _vector_panel(ax, "(a) native Bloch pair at $f$")
    base = np.deg2rad(45)
    for chi, a in ((80, 0.25), (50, 0.45), (22, 0.7)):
        h = np.deg2rad(chi) / 2
        _arrow(ax, (0, 0), (np.cos(base + h), np.sin(base + h)), FWD, alpha=a)
        _arrow(ax, (0, 0), (np.cos(base - h), np.sin(base - h)), BWD, alpha=a)
    # at the edge: one standing wave
    _arrow(ax, (0, 0), (1.05 * np.cos(base), 1.05 * np.sin(base)), INK, lw=2.0)
    ax.text(
        1.0 * np.cos(base) + 0.02,
        1.0 * np.sin(base) + 0.1,
        "edge:\none vector",
        fontsize=7,
        color=INK,
        ha="left",
        va="bottom",
    )
    ax.text(0.92, 0.12, r"$\psi^{-}$", color=BWD, fontsize=8)
    ax.text(-0.06, 0.86, r"$\psi^{+}$", color=FWD, fontsize=8, ha="right")
    ax.text(
        0.55,
        -0.12,
        "$W\\propto\\sin\\chi\\to0$\ncouplings $\\propto 1/W$",
        fontsize=7,
        color=INK,
        ha="center",
        va="top",
    )


def _panel_fixed(ax):
    _vector_panel(ax, r"(b) fixed pair at $f_\mathrm{ref}$")
    base = np.deg2rad(45)
    h = np.deg2rad(80) / 2
    p = np.array([np.cos(base + h), np.sin(base + h)])
    m = np.array([np.cos(base - h), np.sin(base - h)])
    _arrow(ax, (0, 0), tuple(p), FWD, lw=2.2)
    _arrow(ax, (0, 0), tuple(m), BWD, lw=2.2)
    edge = 1.05 * np.array([np.cos(base), np.sin(base)])
    _arrow(ax, (0, 0), tuple(edge), INK, lw=1.2, ls="--")
    # decomposition of the edge vector on the fixed pair
    ax.plot([p[0] * 0.55, edge[0]], [p[1] * 0.55, edge[1]], color=GREY, lw=0.6, ls=":")
    ax.plot([m[0] * 0.55, edge[0]], [m[1] * 0.55, edge[1]], color=GREY, lw=0.6, ls=":")
    ax.text(
        p[0] + 0.05, p[1] - 0.05, r"$\tilde\psi^{+}$", color=FWD, fontsize=8, ha="left"
    )
    ax.text(m[0] + 0.03, m[1] - 0.02, r"$\tilde\psi^{-}$", color=BWD, fontsize=8)
    ax.text(
        edge[0] - 0.05,
        edge[1] + 0.06,
        "edge vector\n$v^{+}\\tilde\\psi^{+}+v^{-}\\tilde\\psi^{-}$",
        fontsize=7,
        color=INK,
        ha="center",
        va="bottom",
    )
    ax.text(
        0.55,
        -0.12,
        "$W=\\tilde W\\,(v_+w_- - v_-w_+)$\n$\\tilde W$ fixed and finite",
        fontsize=7,
        color=INK,
        ha="center",
        va="top",
    )


def _panel_line(ax):
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.4, 3.9)
    ax.axis("off")
    ax.text(
        0,
        3.9,
        r"(c) along the line: $dc/dx=\tilde K c+S(x)$",
        fontsize=8,
        color=INK,
        va="top",
    )
    y_f, y_b = 2.0, 0.9
    x0, x1 = 1.2, 8.8
    for x in np.linspace(x0, x1, 9):
        ax.plot([x, x], [0.55, 2.35], color=LIGHT, lw=0.8, zorder=1)
    _arrow(ax, (x0, y_f), (x1, y_f), FWD, lw=2.0, ms=11)
    _arrow(ax, (x1, y_b), (x0, y_b), BWD, lw=2.0, ms=11)
    ax.text(x1 + 0.1, y_f, r"$c^{+}$", color=FWD, fontsize=9, va="center")
    ax.text(
        x0 + 0.15, y_b - 0.3, r"$c^{-}$", color=BWD, fontsize=9, va="top", ha="left"
    )
    for x in (2.6, 5.0, 7.4):
        _arrow(
            ax, (x, y_f - 0.08), (x + 0.35, y_b + 0.08), GREY, lw=0.9, ms=7, rad=-0.3
        )
        _arrow(
            ax,
            (x + 0.75, y_b + 0.08),
            (x + 1.1, y_f - 0.08),
            GREY,
            lw=0.9,
            ms=7,
            rad=-0.3,
        )
    ax.text(
        5.55,
        1.45,
        r"Bragg: $\tilde K_{12},\ \tilde K_{21}$",
        fontsize=7,
        color=INK,
        ha="center",
        va="center",
        bbox=dict(fc="white", ec="none", pad=0.5),
    )
    for x in (3.4, 6.2):
        _arrow(ax, (x, 2.95), (x + 0.3, y_f + 0.1), INK, lw=0.8, ms=7, rad=0.25)
    ax.text(
        4.8, 3.0, r"$S$: pump $\times$ other modes", fontsize=7, color=INK, ha="center"
    )
    for x, lab in (
        (0.55, r"port, $\rho_\mathrm{in}$"),
        (9.45, r"port, $\rho_\mathrm{out}$"),
    ):
        ax.add_patch(
            FancyBboxPatch(
                (x - 0.45, 0.6),
                0.9,
                1.7,
                boxstyle="round,pad=0.02",
                fc="white",
                ec=GREY,
                lw=0.8,
            )
        )
        ax.text(
            x, 1.45, lab, fontsize=6.5, color=INK, ha="center", va="center", rotation=90
        )
    ax.text(
        5.0,
        0.1,
        "eigenvalues of $\\tilde K$: imaginary (passband),\n0 (edge, Jordan block), real (gap)",
        fontsize=6.5,
        color=GREY,
        ha="center",
        va="top",
    )


def _panel_pass(ax):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.0)
    ax.axis("off")
    ax.text(0, 3.0, "(d) one pass for a flagged mode", fontsize=8, color=INK, va="top")
    boxes = [
        (
            0.1,
            1.55,
            "1  forward sweep (RK)\nall modes; $c^{+}$ predicted\nwith $\\tilde K_{11}c^{+}+\\tilde K_{12}c^{-}$",
            FWD,
        ),
        (5.6, 1.55, "2  backward sweep (RK)\nall other modes;\n$c^{-}$ frozen", BWD),
        (
            5.6,
            0.1,
            "3  exact $2\\times2$ solve\nsources $S^{\\pm}$ from the sweeps;\nRiccati $R(x)$ from $x=L$",
            INK,
        ),
        (0.1, 0.1, "4  Anderson update of\nthe backward envelopes\n(next pass)", GREY),
    ]
    w, h = 4.3, 1.05
    for x, y, txt, c in boxes:
        ax.add_patch(
            FancyBboxPatch(
                (x, y), w, h, boxstyle="round,pad=0.03", fc="white", ec=c, lw=1.2
            )
        )
        ax.text(
            x + 0.15,
            y + h / 2,
            txt,
            fontsize=7.5,
            color=INK,
            va="center",
            ha="left",
            linespacing=1.25,
        )
    _arrow(ax, (0.1 + w, 1.55 + h / 2), (5.6, 1.55 + h / 2), INK, lw=1.2, ms=10)
    _arrow(ax, (5.6 + w / 2, 1.55), (5.6 + w / 2, 0.1 + h), INK, lw=1.2, ms=10)
    _arrow(ax, (5.6, 0.1 + h / 2), (0.1 + w, 0.1 + h / 2), INK, lw=1.2, ms=10)
    _arrow(ax, (0.1 + w / 2, 0.1 + h), (0.1 + w / 2, 1.55), INK, lw=1.2, ms=10)


def draw():
    fig = plt.figure(figsize=(7.0, 4.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 0.85], hspace=0.12, wspace=0.05)
    sub = gs[0, :].subgridspec(1, 3, width_ratios=[1, 1, 1.25], wspace=0.05)
    _panel_native(fig.add_subplot(sub[0]))
    _panel_fixed(fig.add_subplot(sub[1]))
    _panel_line(fig.add_subplot(sub[2]))
    _panel_pass(fig.add_subplot(gs[1, :]))
    fig.savefig("edge_basis_data/edge_basis_diagram.pdf", dpi=200, bbox_inches="tight")
    return fig


if __name__ == "__main__":
    draw().savefig(
        "edge_basis_data/edge_basis_diagram.pdf", dpi=200, bbox_inches="tight"
    )
