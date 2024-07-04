"""Collection of plot functions."""

# mypy: ignore-errors
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes


def plot_response(
    freqs: np.ndarray,
    s21_db: np.ndarray,
    k_star: np.ndarray,
    pump_freq: Optional[float] = None,
    freqs_unit: str = "GHz",
    figsize: tuple[float, float] = (5.0, 6.0),
    **plot_kwargs,
) -> Axes:
    """Plot response of twpa.

    Args:
        freqs (np.ndarray): Array of frequency values.
        s21_db (np.ndarray): S21 parameter in dB.
        k_star (np.ndarray): k* parameter.
        pump_freq (Optional[float]): Frequency of the pump signal.
        freqs_unit (str): Unit of the frequency values.
        figsize (tuple[float, float]): Size of the figure.
        **plot_kwargs: Additional keyword arguments for the plot.
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
    """Plot gain in dB.

    Args:
        freqs (np.ndarray): Array of frequency values.
        gain_db (np.ndarray): Gain values in dB.
        freqs_unit (str): Unit of the frequency values.
        **plot_kwargs: Additional keyword arguments for the plot.
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
    """Plot phase matching.

    Args:
        pump_freqs (np.ndarray): Array of pump frequency values.
        signal_freqs (np.ndarray): Array of signal frequency values.
        delta_pm (np.ndarray): Phase mismatch values.
        freqs_unit (str): Unit of the frequency values.
        thin (int): Thinning factor for plotting.
        log_abs (bool): Whether to plot the logarithm of the absolute values.
        **plot_kwargs: Additional keyword arguments for the plot.
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
