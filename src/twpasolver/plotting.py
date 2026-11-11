"""Collection of plot functions."""

from typing import Any, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from twpasolver.mathutils import I_to_dBm


def plot_mode_currents(
    x: np.ndarray,
    I_tuples: np.ndarray,
    mode_names: List[str],
    pump_freq: Optional[float] = None,
    freqs_unit: str = "GHz",
    figsize: tuple = (9, 5),
    log_scale: bool = True,
    line_styles: Optional[List[str]] = None,
    colors: Optional[List[str]] = None,
    title: Optional[str] = None,
    ylim: Optional[tuple] = None,
    lw=2,
) -> Axes:
    """
    Plot the mean current of each mode over frequency as a function of cell number.

    Args:
        x: Positions in unit cells, shape (n_x,).
        I_tuples: Mode currents, shape (n_freq, n_modes, n_x).
        mode_names: Legend labels.
        log_scale: Plot in dBm (True) or in A (False).
    """
    n_modes = I_tuples.shape[1]

    mean_currents = np.mean(np.abs(I_tuples), axis=0)

    fig, ax = plt.subplots(figsize=figsize)

    if line_styles is None:
        line_styles = ["-", "--", "-.", ":", "-", "--", "-."] * 3

    palette: Sequence[Any] = (
        colors
        if colors is not None
        else plt.get_cmap("tab10")(np.linspace(0, 1, n_modes)).tolist()
    )

    for i in range(n_modes):
        if log_scale:
            currents_db = I_to_dBm(mean_currents[i])
            ax.plot(
                x,
                currents_db,
                label=mode_names[i],
                linestyle=line_styles[i % len(line_styles)],
                color=palette[i % len(palette)],
                lw=lw,
            )
        else:
            ax.plot(
                x,
                mean_currents[i],
                label=mode_names[i],
                linestyle=line_styles[i % len(line_styles)],
                color=palette[i % len(palette)],
                lw=lw,
            )

    ax.set_xlabel("Cell Number")

    if log_scale:
        ax.set_ylabel(f"Current (dBm)")
    else:
        ax.set_ylabel("Current (A)")

    if title:
        ax.set_title(title)
    else:
        pump = "N/A" if pump_freq is None else f"{np.round(pump_freq, 3)} {freqs_unit}"
        ax.set_title(f"Mean Mode Currents vs Cell Number\nPump Frequency: {pump}")

    if ylim:
        ax.set_ylim(ylim)

    ax.legend(bbox_to_anchor=(1.2, 1))
    ax.grid(True, alpha=0.3)

    return ax


def plot_response(
    freqs: np.ndarray,
    s21_db: np.ndarray,
    k_star: np.ndarray,
    pump_freq: Optional[float] = None,
    freqs_unit: str = "GHz",
    figsize: tuple[float, float] = (5.0, 6.0),
    **plot_kwargs,
) -> Axes:
    """Plot ``|S21|`` in dB and the wavenumber k* (rad) versus frequency.

    Args:
        pump_freq (Optional[float]): If given, marked with a vertical line.
        freqs_unit (str): Unit label for the frequency axis.
        **plot_kwargs: Passed to ``ax.plot``.
    """
    _, ax = plt.subplots(2, sharex=True, figsize=figsize)
    ax[0].plot(freqs, s21_db, **plot_kwargs)
    ax[0].set_ylabel("$|S_{21}|$ [dB]")
    ax[1].plot(freqs, k_star, **plot_kwargs)
    ax[1].set_xlabel(f"Frequency [{freqs_unit}]")
    ax[1].set_ylabel("$k^*$ [rad]")
    if pump_freq:
        ax[0].axvline(pump_freq, c="black", ls="--", zorder=0, lw=2)
        ax[1].axvline(pump_freq, c="black", ls="--", zorder=0, lw=2)
    return ax


def plot_gain(
    freqs: np.ndarray,
    gain_db: np.ndarray,
    freqs_unit: str = "GHz",
    **plot_kwargs,
) -> Axes:
    """Plot gain in dB versus frequency.

    Args:
        freqs_unit (str): Unit label for the frequency axis.
        **plot_kwargs: Passed to ``ax.plot``.
    """
    plt.figure()
    ax = plt.axes()
    ax.plot(freqs, gain_db, **plot_kwargs)
    ax.set_xlabel(f"Frequency [{freqs_unit}]")
    ax.set_ylabel("Gain [dB]")
    return ax


def plot_phase_matching(
    pump_freqs: np.ndarray,
    signal_freqs: np.ndarray,
    delta_pm: np.ndarray,
    freqs_unit: str = "GHz",
    thin=1,
    log_abs=True,
    **plot_kwargs,
) -> Axes:
    """Plot the phase mismatch map over pump and signal frequency.

    Args:
        delta_pm (np.ndarray): Phase mismatch, shape ``(len(signal_freqs), len(pump_freqs))``.
        freqs_unit (str): Unit label for the frequency axes.
        thin (int): Plot every ``thin``-th point along both axes.
        log_abs (bool): Plot ``log|delta_pm|``.
        **plot_kwargs: Passed to ``ax.pcolor``.
    """
    plt.figure()
    ax = plt.axes()
    z_label = r"$\Delta_\beta$"
    if log_abs:
        delta_pm = np.log(np.abs(delta_pm))
        z_label = "log|" + z_label + "|"

    m = ax.pcolor(
        pump_freqs[::thin],
        signal_freqs[::thin],
        delta_pm[::thin, ::thin],
        **plot_kwargs,
    )
    c = plt.colorbar(m, ax=ax)
    ax.set_xlabel(f"Pump frequency [{freqs_unit}]")
    ax.set_ylabel(f"Signal frequency [{freqs_unit}]")
    c.set_label(z_label)
    return ax
