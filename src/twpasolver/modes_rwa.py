"""Mode array description with automatic RWA solver."""

import itertools as it
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from math import factorial
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from sympy import symbols


class RWAAnalyzer:
    """Analyzer class for a set of coupled modes."""

    def __init__(self, modes: List[str], relations: List[List[str]]):
        """
        Initialize the RWA analyzer with modes and their relations.

        Args:
            modes: List of mode names (e.g., ["p", "s", "i", "c", "d", "u", "c2"])
            relations: List of relations [result, expression] where expression can be
                      any combination of terms with + and - (e.g., ["c2", "c+c"])
        """
        self.modes = modes
        self.modes_ext_str = modes + [f"-{m}" for m in modes]
        self.modes_symbolic = [symbols(m) for m in modes]
        self.modes_extended = self.modes_symbolic + [-m for m in self.modes_symbolic]
        self.mode_to_idx = {mode: idx for idx, mode in enumerate(modes)}
        self._relations = relations
        self.relations_idx = self._convert_relations_to_indices()
        self.modes_subs = self._compute_substitutions()

    @property
    def relations(self):
        """Getter for mode relations."""
        return self._relations

    def update_relations(relations: List[List[str]]):
        """Update mode relations."""
        self._relations = relations
        self.relations_idx = self._convert_relations_to_indices()
        self.modes_subs = self._compute_substitutions()

    def _parse_expression(self, expr: str) -> tuple:
        """
        Parse an expression string into a list of indices with proper signs.

        Args:
            expr: String expression like "p+s-i"

        Returns:
            Tuple of indices with signs indicating conjugation
        """
        expr = expr.replace(" ", "")
        if not expr.startswith("-"):
            expr = "+" + expr

        indices = []
        current_term = ""
        sign = 1

        for char in expr:
            if char in ["+", "-"]:
                if current_term:
                    idx = self.mode_to_idx[current_term]
                    indices.append(sign * idx)
                current_term = ""
                sign = 1 if char == "+" else -1
            else:
                current_term += char

        if current_term:
            idx = self.mode_to_idx[current_term]
            indices.append(sign * idx)

        return tuple(indices)

    def _convert_relations_to_indices(self) -> List[Tuple[int, tuple]]:
        """Convert string relations to index-based relations."""
        index_relations = []
        for result, expr in self.relations:
            result_idx = self.mode_to_idx[result]
            input_indices = self._parse_expression(expr)
            index_relations.append((result_idx, input_indices))
        return index_relations

    def _compute_substitutions(self) -> List:
        """Compute all mode substitutions based on the relations."""
        modes_subs = deepcopy(self.modes_symbolic)

        for rel in self.relations_idx:
            subs_rel = sum(
                [
                    int(np.sign(rel_idx + 0.5)) * modes_subs[abs(rel_idx)]
                    for rel_idx in rel[1]
                ]
            )

            for idx, m in enumerate(modes_subs):
                modes_subs[idx] = m.subs(modes_subs[rel[0]], subs_rel)

        return modes_subs

    def _calculate_coefficient(self, terms: List[str]) -> float:
        """Calculate coefficient based on term repetitions."""
        coeff = 1
        if len(set(terms)) != len(terms):
            for repetitions in Counter(terms).values():
                coeff = coeff / factorial(repetitions)
        return coeff

    def find_rwa_terms(self, power: int = 3) -> List[Tuple]:
        """
        Find all valid RWA terms of given power.

        Args:
            power: Order of the interaction (default: 3 for three-wave mixing)

        Returns:
            List of tuples (mode_idx, combination, mode_name, rhs_terms, coefficient)
        """
        modes_subs_extended = self.modes_subs + [-m for m in self.modes_subs]
        rwa_terms = []

        for comb in it.combinations_with_replacement(
            range(len(self.modes_extended)), power
        ):
            terms = [modes_subs_extended[i] for i in comb]
            sum_terms = sum(terms)

            for j, mode in enumerate(self.modes_subs):
                if sum_terms == mode:
                    rhs = [self.modes_ext_str[k] for k in comb]
                    coeff = self._calculate_coefficient(rhs)
                    rwa_terms.append((j, comb, self.modes[j], rhs, coeff))

        return sorted(rwa_terms, key=lambda x: x[0])

    def print_rwa_terms(self, terms: List[Tuple]) -> None:
        """Pretty print the RWA terms with their coefficients."""
        for term in terms:
            mode_name = term[2]
            rhs_terms = term[3]
            coeff = term[4]
            print(f"{mode_name} = {rhs_terms} {coeff}")
        print(f"\nTotal matches: {len(terms)}")


@dataclass
class Mode:
    """
    Represents a single optical mode with its physical properties.

    Attributes:
        label: Mode identifier (e.g., 'p' for pump)
        frequency: Angular frequency (ω)
        direction: 1 for forward, -1 for backward
        gamma: Reflection coefficient
        k: Wavenumber (calculated from frequency)
    """

    label: str
    direction: int = 1  # 1 for forward, -1 for backward
    frequency: Optional[float] = None
    k: Optional[float] = None
    gamma: Optional[Union[float, complex]] = 1.0

    def __post_init__(self):
        """Initialize derived quantities and validate inputs."""
        # Ensure direction is ±1
        if abs(self.direction) != 1:
            raise ValueError("Direction must be either 1 (forward) or -1 (backward)")
        if self.k is not None:
            self.k *= self.direction

    def __eq__(self, other):
        """Compare modes based on their physical properties."""
        if not isinstance(other, Mode):
            return False
        return (
            self.frequency == other.frequency
            and self.direction == other.direction
            and self.gamma == other.gamma
            and self.k == other.k
        )

    def __hash__(self):
        """Get hash based on immutable properties."""
        return hash((self.label, str(self.frequency), self.direction))

    def __str__(self):
        """Get string representation showing direction and label."""
        direction_str = "→" if self.direction == 1 else "←"
        return f"{direction_str}{self.label}"

    def __repr__(self):
        """Get representation of class."""
        return (
            f'Mode("{self.label}", freq={self.frequency}, '
            f"dir={self.direction}, gamma={self.gamma}, k={self.k})"
        )


class ParameterInterpolator:
    """
    Interpolates kappa and gamma values based on frequency.

    Handles both real and complex values.
    """

    def __init__(
        self,
        frequencies: np.ndarray,
        kappas: np.ndarray,
        gammas: np.ndarray,
        kind: str = "cubic",
    ):
        """
        Initialize interpolators for kappa and gamma.

        Args:
            frequencies: Array of frequency points
            kappas: Array of coupling coefficients
            gammas: Array of reflection coefficients
            kind: Interpolation method ('linear', 'cubic', etc.)
        """
        # Validate input arrays
        if not (len(frequencies) == len(kappas) == len(gammas)):
            raise ValueError("All input arrays must have the same length")
        if len(frequencies) < 2:
            raise ValueError("Need at least 2 points for interpolation")
        self.orig_freqs = frequencies
        self.orig_kappas = kappas
        self.orig_gammas = gammas
        self.freq_min = np.min(frequencies)
        self.freq_max = np.max(frequencies)

        # For kappa interpolation
        if np.iscomplexobj(kappas):
            self.kappa_real = interp1d(
                frequencies,
                np.real(kappas),
                kind=kind,
                bounds_error=False,
                fill_value=(kappas[0].real, kappas[-1].real),
            )
            self.kappa_imag = interp1d(
                frequencies,
                np.imag(kappas),
                kind=kind,
                bounds_error=False,
                fill_value=(kappas[0].imag, kappas[-1].imag),
            )
        else:
            self.kappa_real = interp1d(
                frequencies,
                kappas,
                kind=kind,
                bounds_error=False,
                fill_value=(kappas[0], kappas[-1]),
            )
            self.kappa_imag = None

        # For gamma interpolation
        if np.iscomplexobj(gammas):
            self.gamma_real = interp1d(
                frequencies,
                np.real(gammas),
                kind=kind,
                bounds_error=False,
                fill_value=(gammas[0].real, gammas[-1].real),
            )
            self.gamma_imag = interp1d(
                frequencies,
                np.imag(gammas),
                kind=kind,
                bounds_error=False,
                fill_value=(gammas[0].imag, gammas[-1].imag),
            )
        else:
            self.gamma_real = interp1d(
                frequencies,
                gammas,
                kind=kind,
                bounds_error=False,
                fill_value=(gammas[0], gammas[-1]),
            )
            self.gamma_imag = None

    def get_parameters(self, frequency: float) -> Tuple[complex, complex]:
        """
        Get interpolated kappa and gamma for a given frequency.

        Args:
            frequency: Frequency point for interpolation

        Returns:
            Tuple of (kappa, gamma) at the requested frequency
        """
        if frequency < self.freq_min or frequency > self.freq_max:
            print(
                f"Warning: Frequency {frequency} is outside the interpolation range "
                f"[{self.freq_min}, {self.freq_max}]. Using endpoint values."
            )

        # Interpolate kappa
        if self.kappa_imag is not None:
            kappa = complex(self.kappa_real(frequency), self.kappa_imag(frequency))
        else:
            kappa = self.kappa_real(frequency)

        # Interpolate gamma
        if self.gamma_imag is not None:
            gamma = complex(self.gamma_real(frequency), self.gamma_imag(frequency))
        else:
            gamma = self.gamma_real(frequency)

        return kappa, gamma

    def plot_interpolation(self, num_points: int = 1000) -> None:
        """
        Plot the interpolation results for visualization.

        Args:
            num_points: Number of points for plotting
        """
        import matplotlib.pyplot as plt

        # Generate frequency points for plotting
        freq_plot = np.linspace(self.freq_min, self.freq_max, num_points)
        kappas_plot = []
        gammas_plot = []

        # Get interpolated values
        for f in freq_plot:
            k, g = self.get_parameters(f)
            kappas_plot.append(k)
            gammas_plot.append(g)

        kappas_plot = np.array(kappas_plot)
        gammas_plot = np.array(gammas_plot)

        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

        # Get original points for kappa
        orig_kappas = self.orig_kappas
        orig_freqs = self.orig_freqs

        # Plot kappa
        if self.kappa_imag is not None:
            ax1.plot(freq_plot, np.real(kappas_plot), "b-", label="Interpolation")
            ax1.plot(orig_freqs, np.real(orig_kappas), "bo", label="Original Points")
            ax2.plot(freq_plot, np.imag(kappas_plot), "r-", label="Interpolation")
            ax2.plot(orig_freqs, np.imag(orig_kappas), "ro", label="Original Points")
        else:
            ax1.plot(freq_plot, kappas_plot, "b-", label="Interpolation")
            ax1.plot(orig_freqs, orig_kappas, "bo", label="Original Points")
            ax2.text(
                0.5,
                0.5,
                "No Imaginary Part",
                horizontalalignment="center",
                transform=ax2.transAxes,
            )

        # Get original points for gamma
        orig_gammas = self.orig_gammas

        # Plot gamma
        if self.gamma_imag is not None:
            ax3.plot(freq_plot, np.real(gammas_plot), "b-", label="Interpolation")
            ax3.plot(orig_freqs, np.real(orig_gammas), "bo", label="Original Points")
            ax4.plot(freq_plot, np.imag(gammas_plot), "r-", label="Interpolation")
            ax4.plot(orig_freqs, np.imag(orig_gammas), "ro", label="Original Points")
        else:
            ax3.plot(freq_plot, gammas_plot, "b-", label="Interpolation")
            ax3.plot(orig_freqs, orig_gammas, "bo", label="Original Points")
            ax4.text(
                0.5,
                0.5,
                "No Imaginary Part",
                horizontalalignment="center",
                transform=ax4.transAxes,
            )

        # Labels and titles
        ax1.set_title("κ (Real Part)")
        ax2.set_title("κ (Imaginary Part)")
        ax3.set_title("γ (Real Part)")
        ax4.set_title("γ (Imaginary Part)")

        for ax in (ax1, ax2, ax3, ax4):
            ax.set_xlabel("Frequency")
            ax.grid(True)
            ax.legend()

        plt.tight_layout()
        plt.show()


class ModeArray:
    """Manages a collection of modes, their relations, and parameter interpolation."""

    def __init__(
        self,
        modes: List[Mode],
        relations: List[List[str]],
        freq_data: np.ndarray,
        kappa_data: np.ndarray,
        gamma_data: np.ndarray,
    ):
        """
        Initialize mode array with modes, relations and interpolation data.

        Args:
            modes: List of Mode objects
            relations: List of [result, expression] pairs for mode relationships
            freq_data: Array of frequency points for interpolation
            kappa_data: Array of kappa values corresponding to frequencies
            gamma_data: Array of gamma values corresponding to frequencies
        """
        self.modes = {mode.label: mode for mode in modes}
        self.relations = relations

        # Initialize RWA analyzer for mode relations
        self.analyzer = RWAAnalyzer(list(self.modes.keys()), relations)

        # Initialize parameter interpolator
        self.interpolator = ParameterInterpolator(freq_data, kappa_data, gamma_data)

        # Validate initial state
        self._validate_modes()

    def _validate_modes(self):
        """Ensure all modes referenced in relations exist."""
        for result, expr in self.relations:
            if result not in self.modes:
                raise ValueError(f"Mode {result} in relations not found")

            # Check all modes in expression exist
            expr_modes = {
                m.strip("+-")
                for m in expr.replace(" ", "").replace("(", "").replace(")", "")
                if m.strip("+-")
            }
            for mode in expr_modes:
                if mode not in self.modes:
                    raise ValueError(f"Mode {mode} in relations not found")

    def _propagate_frequencies(
        self, updated_freqs: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Propagate frequency updates through relations.

        Args:
            updated_freqs: Dictionary of mode labels to their new frequencies

        Returns:
            Dictionary of all mode labels to their computed frequencies
        """
        # Start with existing frequencies
        frequencies = {label: mode.frequency for label, mode in self.modes.items()}
        frequencies.update(updated_freqs)

        # Propagate through relations until no changes
        changed = True
        while changed:
            changed = False
            for rel_idx, term_indices in self.analyzer.relations_idx:
                result_mode = list(self.modes.keys())[rel_idx]

                # Skip if we don't have all required frequencies
                term_modes = [list(self.modes.keys())[abs(idx)] for idx in term_indices]
                if not all(frequencies.get(mode) is not None for mode in term_modes):
                    continue

                # Calculate new frequency from relation
                new_freq = sum(
                    int(np.sign(idx + 0.5))
                    * frequencies[list(self.modes.keys())[abs(idx)]]
                    for idx in term_indices
                )

                # Update if different
                if frequencies.get(result_mode) != new_freq:
                    frequencies[result_mode] = new_freq
                    changed = True

        return frequencies

    def update_frequencies(self, new_frequencies: Dict[str, float]) -> None:
        """
        Update mode frequencies and interpolate corresponding parameters.

        Args:
            new_frequencies: Dictionary mapping mode labels to new frequencies
        """
        # Validate input modes exist
        unknown_modes = set(new_frequencies.keys()) - set(self.modes.keys())
        if unknown_modes:
            raise ValueError(f"Unknown modes in frequency update: {unknown_modes}")

        # Propagate frequencies through relations
        all_frequencies = self._propagate_frequencies(new_frequencies)

        # Update modes with new frequencies and interpolated parameters
        for label, freq in all_frequencies.items():
            if freq is not None:
                mode = self.modes[label]
                mode.frequency = freq
                # Get interpolated parameters
                kappa, gamma = self.interpolator.get_parameters(abs(freq))
                mode.gamma = gamma
                mode.k = kappa * mode.direction

    def get_mode(self, label: str) -> Mode:
        """Get mode by label."""
        return self.modes[label]

    def print_modes(self):
        """Print current state of all modes."""
        for label, mode in self.modes.items():
            print(f"{label}:")
            print(f"  Frequency: {mode.frequency}")
            print(f"  Direction: {mode.direction}")
            print(f"  k: {mode.k}")
            print(f"  gamma: {mode.gamma}")
            print()

    def plot_k_comparison(
        self, freq_range: Optional[Tuple[float, float]] = None, num_points: int = 1000
    ):
        """
        Plot k values for all modes and reference data.

        Args:
            freq_range: Optional tuple of (min_freq, max_freq) to plot
                       If None, uses the full data range
            num_points: Number of points for plotting the data curves
        """
        if freq_range is None:
            freq_min = min(self.interpolator.orig_freqs)
            freq_max = max(self.interpolator.orig_freqs)
        else:
            freq_min, freq_max = freq_range

        # Create figure and axis
        plt.figure(figsize=(9, 5))
        ax = plt.gca()

        # Plot reference k data
        freq_points = np.linspace(freq_min, freq_max, num_points)
        k_data = np.interp(
            freq_points, self.interpolator.orig_freqs, self.interpolator.orig_kappas
        )
        ax.plot(freq_points, k_data, "k--", label="Reference k(ω)", alpha=0.5)

        # Plot current mode k values
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.modes)))
        for (label, mode), color in zip(self.modes.items(), colors):
            if mode.frequency is not None:
                # Plot point for current k value
                marker = "o" if mode.direction == 1 else "s"
                ax.scatter(
                    abs(mode.frequency),
                    mode.k,
                    color=color,
                    marker=marker,
                    s=100,
                    label=f"{label}",
                )

        # Add labels and title
        ax.set_xlabel("Frequency (ω)")
        ax.set_ylabel("Wavenumber (k)")
        ax.set_title("Mode k-vectors and Reference Dispersion")
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

        # Add text box with current relations
        relation_text = "\n".join([f"{rel[0]} = {rel[1]}" for rel in self.relations])
        plt.text(
            1.05,
            0.3,
            f"Relations:\n{relation_text}",
            transform=ax.transAxes,
            verticalalignment="center",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        plt.tight_layout()
        plt.show()
