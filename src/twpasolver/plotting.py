"""Collection of plot functions."""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def plot_response(
    freqs: np.ndarray,
    s21_db: np.ndarray,
    k_star: np.ndarray,
    pump_freq: Optional[float] = None,
    freqs_unit: str = "GHz",
    figsize: tuple[float, float] = (5.0, 6.0),
    **plot_kwargs,
):
    """Plot response of twpa."""
    _, ax = plt.subplots(2, sharex=True, figsize=figsize)
    ax[0].plot(freqs, s21_db, **plot_kwargs)
    ax[0].set_ylabel("$|S_{21}|$ [dB]")
    ax[1].plot(freqs, k_star, **plot_kwargs)
    ax[1].set_xlabel("Frequency " + freqs_unit)
    ax[1].set_ylabel("$k^*$ [rad]")
    if pump_freq:
        ax[0].axvline(pump_freq, c="black", ls="--", zorder=0, lw=2)
        ax[1].axvline(pump_freq, c="black", ls="--", zorder=0, lw=2)
    return ax


def plot_gain(
    freqs: np.ndarray,
    gain_db: np.ndarray,
    freqs_unit: str = "GHz",
    ax: Optional[plt.Axes] = None,
    **plot_kwargs,
) -> plt.Axes:
    """Plot gain in dB."""
    if ax is None:
        plt.figure()
        ax = plt.axes()
    ax.plot(freqs, gain_db, **plot_kwargs)
    ax.set_xlabel("Frequency " + freqs_unit)
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
) -> plt.Axes:
    """Plot gain in dB."""
    plt.figure()
    ax = plt.axes()
    z_label = r"$\Delta_\beta$"
    if log_abs:
        delta_pm = np.log(np.abs(delta_pm))
        z_label = "log|" + z_label + "|"

    m = ax.pcolor(
        pump_freqs[::thin],
        signal_freqs[::thin],
        delta_pm[::thin][::thin],
        **plot_kwargs,
    )
    c = plt.colorbar(m, ax=ax)
    ax.set_xlabel("Pump frequency " + freqs_unit)
    ax.set_ylabel("Signal frequency " + freqs_unit)
    c.set_label(z_label)
    return ax
