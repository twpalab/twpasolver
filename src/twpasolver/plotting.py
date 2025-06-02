"""Collection of plot functions."""

# mypy: ignore-errors
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from twpasolver.mathutils import I_to_dBm


def plot_mode_currents(
    gain_result: Dict[str, Any],
    mode_names: Optional[List[str]] = None,
    figsize: tuple = (9, 5),
    log_scale: bool = True,
    line_styles: Optional[List[str]] = None,
    colors: Optional[List[str]] = None,
    title: Optional[str] = None,
    ylim: Optional[tuple] = None,
) -> Axes:
    """
    Plot the mean current of each mode in dB as a function of cell number.

    Args:
        gain_result: Dictionary output from the gain analysis function
        mode_names: Optional list of mode names for labeling (auto-detected if not provided)
        figsize: Figure size tuple (width, height)
        log_scale: Whether to plot in dB scale (True) or linear scale (False)
        line_styles: Optional list of line styles for different modes
        colors: Optional list of colors for different modes
        title: Custom title for the plot
        ylim: Optional y-axis limits as (min, max)

    Returns:
        The matplotlib figure object
    """
    # Extract data from gain result
    x = gain_result["x"]
    model = gain_result.get("model", "minimal_3wm")

    # Determine the dimensionality and mode names based on the model
    if model == "minimal_3wm":
        # Standard 3WM model with pump, signal, idler
        I_triplets = gain_result["I_triplets"]
        n_freqs = I_triplets.shape[0]
        n_modes = I_triplets.shape[1]

        # Use default mode names for standard model
        if mode_names is None:
            mode_names = ["Pump", "Signal", "Idler"]

        # Compute mean current across all frequencies
        mean_currents = np.mean(np.abs(I_triplets), axis=0)

    else:  # general model
        # General model with arbitrary number of modes
        I_triplets = gain_result["I_triplets"]
        n_freqs = I_triplets.shape[0]
        n_modes = I_triplets.shape[1]

        # Extract mode names from the result if available
        if mode_names is None and "mode_info" in gain_result:
            mode_info = gain_result["mode_info"]
            reverse_mode_map = mode_info.get("reverse_mode_map", {})
            if reverse_mode_map:
                mode_names = [
                    reverse_mode_map.get(i, f"Mode {i}") for i in range(n_modes)
                ]
            else:
                mode_names = [f"Mode {i}" for i in range(n_modes)]
        elif mode_names is None:
            mode_names = [f"Mode {i}" for i in range(n_modes)]

        # Compute mean current across all frequencies
        mean_currents = np.mean(np.abs(I_triplets), axis=0)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Define line styles and colors if not provided
    if line_styles is None:
        line_styles = ["-", "--", "-.", ":", "-", "--", "-."] * 3

    if colors is None:
        # Use a colormap for many modes
        colors = plt.cm.tab10(np.linspace(0, 1, n_modes))

    # Plot each mode
    for i in range(n_modes):
        if log_scale:
            # Convert to dB
            currents_db = I_to_dBm(mean_currents[i])
            ax.plot(
                x,
                currents_db,
                label=mode_names[i],
                linestyle=line_styles[i % len(line_styles)],
                color=colors[i % len(colors)],
            )
        else:
            # Linear scale
            ax.plot(
                x,
                mean_currents[i],
                label=mode_names[i],
                linestyle=line_styles[i % len(line_styles)],
                color=colors[i % len(colors)],
            )

    # Set labels and title
    ax.set_xlabel("Cell Number")

    if log_scale:
        ax.set_ylabel(f"Current (dBm)")
    else:
        ax.set_ylabel("Current (A)")

    if title:
        ax.set_title(title)
    else:
        pump_freq = gain_result.get("pump_freq", "N/A")
        ax.set_title(
            f"Mean Mode Currents vs Cell Number\nPump Frequency: {pump_freq} GHz"
        )

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
