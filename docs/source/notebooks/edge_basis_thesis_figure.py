"""
Thesis figure ``cmes_edge_basis.pdf``: full-band gain of the V3 holotype with and without
the fixed edge basis.

(a) splitting |sinh(gamma d)| of the signal-dependent modes, coloured by sector;
(b) signal gain with the fixed edge basis over the band, with the windows where a mode
    is near a stopband edge;
(c) zoom across the p2s edge on a fine grid: native Bloch basis against fixed edge basis.

Device, mode set and solver settings are those of ``06_edge_basis.ipynb``
(``edge_basis_sweep``), on uniform signal grids. The sweeps are cached in
``edge_basis_data/cmes_edge_basis_data.npz`` (band) and
``edge_basis_data/cmes_edge_basis_window.npz`` (zoom); pass ``--recompute`` to rerun them. The figure
is written to the thesis (``tesi_phd/figures/ch6``, next to the twpasolver repository) when
that folder exists, otherwise to ``edge_basis_data``:

    python edge_basis_thesis_figure.py [--recompute]
"""

import logging
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.chdir(HERE)  # palette.py loads the style sheet by relative path
import edge_basis_sweep as E  # noqa: E402
from palette import PALETTE  # noqa: E402

from twpasolver.cmes.solver_config import FBSolverConfig  # noqa: E402
from twpasolver.logger import log  # noqa: E402

THESIS = HERE.parents[3] / "tesi_phd" / "figures" / "ch6"
OUT = (THESIS if THESIS.is_dir() else HERE / "edge_basis_data") / "cmes_edge_basis.pdf"
CACHE = HERE / "edge_basis_data" / "cmes_edge_basis_data.npz"
CACHE_ZOOM = HERE / "edge_basis_data" / "cmes_edge_basis_window.npz"
F_GRID = np.round(np.arange(3.0, 13.2 + 1e-9, 0.002), 6)  # GHz; above: signal stopband
F_ZOOM = np.round(np.arange(9.56, 9.66 + 1e-9, 0.0005), 6)  # GHz, across the p2s edge
SECTOR_COLORS = [PALETTE["purple"], PALETTE["gold"], PALETTE["magenta"]]
BAND_ALPHA = 0.18
INK, GREY = PALETTE["ink"], PALETTE["grey"]
EDGE = PALETTE["purple"]  # fixed edge basis
NATIVE = PALETTE["ink"]  # native Bloch basis


def sweep(recompute=True, grid=F_GRID, cache=CACHE):
    """Gain on ``grid`` in the native basis and with the fixed edge basis (cached)."""
    if cache.exists() and not recompute:
        return dict(np.load(cache))
    log.setLevel(logging.ERROR)
    an = E.make_analysis()
    out = {"fs": grid}
    for name, threshold in E.RUNS.items():
        r = an.gain(
            signal_freqs=grid,
            Is0=E.IS0,
            pump=E.FP,
            model="bloch",
            mode_array_config="ext2",
            fb_config=FBSolverConfig(**E.FB, edge_basis_threshold=threshold),
        )
        out[f"{name}_gain"] = np.asarray(r["gain_db"], dtype=float)
        out[f"{name}_converged"] = np.asarray(r["bloch"]["converged"])
        print(f"{name} done", flush=True)
    np.savez(cache, **out)
    return out


def shade(ax, wins):
    for m, ws in wins.items():
        color = SECTOR_COLORS[E.SECTOR_OF[m]]
        for a, b, _ in ws:
            if m == "s":
                ax.axvspan(
                    a, b, fc="none", ec=GREY, hatch="////", lw=0, alpha=0.5, zorder=0
                )
            else:
                ax.axvspan(a, b, color=color, alpha=BAND_ALPHA, lw=0, zorder=0)


def main(recompute=False):
    d = sweep(recompute)
    z = sweep(recompute, grid=F_ZOOM, cache=CACHE_ZOOM)
    fs, g_edge = d["fs"], d["edge_gain"]

    an = E.make_analysis()
    fd = np.arange(F_GRID[0], F_GRID[-1] + 0.05, 2e-3)
    split = E.mode_splitting(an.twpa, fd)
    wins = {m: E.windows(fd, split[m]) for m in E.WEAK}
    fz = np.arange(F_ZOOM[0], F_ZOOM[-1] + 1e-9, 1e-4)
    f_edge = fz[np.argmin(E.mode_splitting(an.twpa, fz)["p2s"])]

    fig = plt.figure(figsize=(4.95, 3.7))
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.6, 1.0],
        height_ratios=[1.0, 1.55],
        hspace=0.08,
        wspace=0.38,
    )
    ax_s = fig.add_subplot(gs[0, 0])
    ax_g = fig.add_subplot(gs[1, 0], sharex=ax_s)
    ax_z = fig.add_subplot(gs[:, 1])
    for a in (ax_s, ax_g):
        shade(a, wins)

    # (a) splitting: the Bloch modes stop propagating at every edge
    style = {"s": "-", "ps": "-", "p2s": "-", "i": "--", "pi": "--", "p2i": "--"}
    for m in E.WEAK:
        ax_s.semilogy(
            fd, split[m], style[m], color=SECTOR_COLORS[E.SECTOR_OF[m]], lw=0.8, label=m
        )
    ax_s.axhline(E.E_WINDOW, color=GREY, lw=0.7, ls=":")
    ax_s.set_ylim(5e-3, 2)
    ax_s.set_ylabel(r"$|\sinh\gamma d|$")
    ax_s.tick_params(labelbottom=False)
    ax_s.legend(
        ncol=6,
        fontsize=5.5,
        loc="lower center",
        frameon=False,
        bbox_to_anchor=(0.5, 1.0),
        handlelength=1.5,
        columnspacing=0.8,
    )

    # (b) gain over the band, fixed edge basis
    ax_g.plot(fs, g_edge, color=EDGE, lw=0.4)
    ax_g.set_ylim(-10, 35)
    ax_g.set_xlim(F_GRID[0], F_GRID[-1])
    ax_g.set_ylabel("Signal gain (dB)")
    ax_g.set_xlabel("Signal frequency (GHz)")
    # zoom area on top of the curve: the frequency range of (c), its gain range
    in_z = (fs >= F_ZOOM[0]) & (fs <= F_ZOOM[-1])
    g_lo, g_hi = np.nanmin(g_edge[in_z]) - 1.0, np.nanmax(g_edge[in_z]) + 1.0
    ax_g.add_patch(
        plt.Rectangle(
            (F_ZOOM[0] - 0.05, g_lo),
            F_ZOOM[-1] - F_ZOOM[0] + 0.1,
            g_hi - g_lo,
            fill=False,
            ec=INK,
            lw=0.8,
            zorder=5,
        )
    )
    ax_g.text(
        F_ZOOM[0] + 0.05,
        g_hi + 0.6,
        "(c)",
        fontsize=6.5,
        color=INK,
        ha="center",
        va="bottom",
        zorder=5,
    )

    # (c) zoom across the p2s edge: native basis against fixed edge basis
    ax_z.plot(
        z["fs"], z["edge_gain"], color=EDGE, lw=1.0, label="fixed edge basis", zorder=3
    )
    ax_z.plot(
        z["fs"],
        z["none_gain"],
        ".",
        color=NATIVE,
        ms=3.6,
        label="native basis",
        zorder=4,
    )
    # stretches where the native iteration diverges (no value to draw)
    bad = ~z["none_converged"] | ~np.isfinite(z["none_gain"])
    if bad.any():
        idx = np.flatnonzero(bad)
        for run in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1):
            ax_z.axvspan(
                z["fs"][run[0]],
                z["fs"][run[-1]],
                color=GREY,
                alpha=0.25,
                lw=0,
                zorder=0,
            )
        ax_z.text(
            z["fs"][idx[-1]] + 0.0015,
            0.985,
            "native basis\ndiverges",
            transform=ax_z.get_xaxis_transform(),
            ha="left",
            va="top",
            fontsize=6,
            color=INK,
        )
    ax_z.axvline(f_edge, color=SECTOR_COLORS[2], lw=0.8, ls="-")  # solid, as p2s in (a)
    ax_z.text(
        f_edge + 0.002,
        0.02,
        "$p_2s$ edge",
        transform=ax_z.get_xaxis_transform(),
        color=SECTOR_COLORS[2],
        fontsize=6.5,
        va="bottom",
    )
    ax_z.set_xlim(F_ZOOM[0], F_ZOOM[-1])
    # headroom above the ripple for the label of the divergence band
    z_lo = np.nanmin(z["edge_gain"])
    z_hi = max(np.nanmax(z["edge_gain"]), np.nanmax(z["none_gain"]))
    ax_z.set_ylim(z_lo - 0.3, z_hi + 0.25 * (z_hi - z_lo))
    ax_z.set_xticks([9.56, 9.61, 9.66])
    ax_z.set_xlabel("Signal frequency (GHz)")
    ax_z.set_ylabel("Signal gain (dB)")
    ax_z.legend(
        fontsize=6, loc="upper center", frameon=False, bbox_to_anchor=(0.5, 1.13)
    )

    for a, letter in ((ax_s, "a"), (ax_g, "b"), (ax_z, "c")):
        a.text(
            -0.2 if a is not ax_z else -0.3,
            1.0,
            f"({letter})",
            transform=a.transAxes,
            va="top",
            ha="right",
            fontsize=8,
        )
    fig.align_ylabels([ax_s, ax_g])
    fig.savefig(OUT, bbox_inches="tight")
    fig.savefig(
        HERE / "edge_basis_data" / "cmes_edge_basis_preview.png",
        dpi=200,
        bbox_inches="tight",
    )


if __name__ == "__main__":
    main(recompute="--recompute" in sys.argv)
