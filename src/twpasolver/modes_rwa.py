"""Simplified Mode array implementation."""

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

from twpasolver.logger import log


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

    def get_parameters(
        self, frequency: Union[float, np.ndarray]
    ) -> Tuple[Union[complex, np.ndarray], Union[complex, np.ndarray]]:
        """
        Get interpolated kappa and gamma for a given frequency or array of frequencies.

        Args:
            frequency: Frequency point(s) for interpolation, can be a single value or array

        Returns:
            Tuple of (kappa, gamma) at the requested frequency/frequencies
        """
        # Check if we're processing a single value or array
        is_array = isinstance(frequency, np.ndarray)

        # Handle out-of-range warning
        if is_array:
            if np.any((frequency < self.freq_min) | (frequency > self.freq_max)):
                log.warn(
                    f"Warning: Some frequencies are outside the interpolation range "
                    f"[{self.freq_min}, {self.freq_max}]. Using endpoint values."
                )
        else:
            if frequency < self.freq_min or frequency > self.freq_max:
                log.warn(
                    f"Warning: Frequency {frequency} is outside the interpolation range "
                    f"[{self.freq_min}, {self.freq_max}]. Using endpoint values."
                )

        # Interpolate kappa
        if self.kappa_imag is not None:
            kappa_real = self.kappa_real(frequency)
            kappa_imag = self.kappa_imag(frequency)
            kappa = kappa_real + 1j * kappa_imag
        else:
            kappa = self.kappa_real(frequency)

        # Interpolate gamma
        if self.gamma_imag is not None:
            gamma_real = self.gamma_real(frequency)
            gamma_imag = self.gamma_imag(frequency)
            gamma = gamma_real + 1j * gamma_imag
        else:
            gamma = self.gamma_real(frequency)

        return kappa, gamma


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
    gamma: Optional[Union[float, complex]] = 0.0

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
        # self.idx_to_mode = {idm: mode for idx, mode in enumerate(modes)}
        self._relations = relations
        self.relations_idx = self._convert_relations_to_indices()
        self.modes_subs = self._compute_substitutions()

        # Cache for RWA terms
        self._rwa_terms_cache = {}

    @property
    def relations(self):
        """Getter for mode relations."""
        return self._relations

    def update_relations(self, relations: List[List[str]]):
        """Update mode relations."""
        self._relations = relations
        self.relations_idx = self._convert_relations_to_indices()
        self.modes_subs = self._compute_substitutions()
        # Clear cache when relations are updated
        self._rwa_terms_cache = {}

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
        signs = []
        current_term = ""
        sign = 1

        for char in expr:
            if char in ["+", "-"]:
                if current_term:
                    idx = self.mode_to_idx[current_term]
                    indices.append(idx)
                    signs.append(sign)
                current_term = ""
                sign = 1 if char == "+" else -1
            else:
                current_term += char

        if current_term:
            idx = self.mode_to_idx[current_term]
            indices.append(idx)
            signs.append(sign)

        return tuple(indices), tuple(signs)

    def _convert_relations_to_indices(self) -> List[Tuple[int, tuple, tuple]]:
        """Convert string relations to index-based relations."""
        index_relations = []
        for result, expr in self.relations:
            result_idx = self.mode_to_idx[result]
            input_indices, input_signs = self._parse_expression(expr)
            index_relations.append((result_idx, input_indices, input_signs))
        return index_relations

    def _compute_substitutions(self) -> List:
        """Compute all mode substitutions based on the relations."""
        apply = True
        modes_subs = deepcopy(self.modes_symbolic)
        while apply:
            modes_subs_old = deepcopy(modes_subs)
            for output, input_idxs, input_signs in self.relations_idx:
                subs_rel = sum(
                    [
                        input_signs[i] * modes_subs[abs(rel_idx)]
                        for i, rel_idx in enumerate(input_idxs)
                    ]
                )
                modes_subs[output] = modes_subs[output].subs(
                    modes_subs[output], subs_rel
                )
                if all([m == modes_subs_old[i] for i, m in enumerate(modes_subs)]):
                    apply = False

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
        # Check if cached result exists
        if power in self._rwa_terms_cache:
            return self._rwa_terms_cache[power]

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

        result = sorted(rwa_terms, key=lambda x: x[0])
        # Cache the result
        self._rwa_terms_cache[power] = result

        return result

    def print_rwa_terms(self, terms: List[Tuple]) -> None:
        """Pretty print the RWA terms with their coefficients."""
        for term in terms:
            mode_name = term[2]
            rhs_terms = term[3]
            coeff = term[4]
            print(f"{mode_name} = {rhs_terms} {coeff}")
        print(f"\nTotal matches: {len(terms)}")


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

        # Cache for computed mixing coefficients
        self._rwa_terms_3wm = None
        self._rwa_terms_4wm = None

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
            for rel_idx, term_indices, term_signs in self.analyzer.relations_idx:
                result_mode = list(self.modes.keys())[rel_idx]

                # Skip if we don't have all required frequencies
                term_modes = [list(self.modes.keys())[abs(idx)] for idx in term_indices]
                if not all(frequencies.get(mode) is not None for mode in term_modes):
                    continue

                # Calculate new frequency from relation
                new_freq = sum(
                    term_signs[i] * frequencies[list(self.modes.keys())[abs(idx)]]
                    for i, idx in enumerate(term_indices)
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

    def process_frequency_array(
        self, mode_label: str, frequencies: np.ndarray
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Process an array of frequencies for a single mode, propagating to all related modes.

        Args:
            mode_label: Label of the mode to update with array of frequencies
            frequencies: Array of frequencies to process

        Returns:
            Dictionary with mode parameters for all related modes
        """
        # Validate mode exists
        if mode_label not in self.modes:
            raise ValueError(f"Unknown mode: {mode_label}")

        # Initialize result containers
        all_frequencies = {}
        mode_params = {}

        # Process each frequency point
        for i, freq in enumerate(frequencies):
            # Update with this single frequency
            freq_results = self._propagate_frequencies({mode_label: freq})

            # Store results for each mode
            for label, value in freq_results.items():
                if label not in all_frequencies:
                    # Initialize array for this mode
                    all_frequencies[label] = np.zeros(len(frequencies))
                all_frequencies[label][i] = value

        # Get all parameters for the computed frequencies
        for label, freqs in all_frequencies.items():
            # Get direction for this mode
            mode = self.modes[label]
            direction = mode.direction

            # Get parameters for all frequencies of this mode
            kappas, gammas = self.interpolator.get_parameters(np.abs(freqs))

            # Apply direction to kappas
            if isinstance(kappas, np.ndarray):
                kappas = kappas * direction
            else:
                kappas = np.array([kappas * direction])

            # Store parameters
            mode_params[label] = {"freqs": freqs, "k": kappas, "gamma": gammas}

        return mode_params

    def get_mode(self, label: str) -> Mode:
        """Get mode by label."""
        return self.modes[label]

    def get_rwa_terms(self, power: int = 3) -> List[Tuple]:
        """
        Get RWA terms for the specified mixing order with caching.

        Args:
            power: Order of the interaction (2 for 3WM, 3 for 4WM)

        Returns:
            List of RWA terms
        """
        if power == 2:
            if self._rwa_terms_3wm is None:
                self._rwa_terms_3wm = self.analyzer.find_rwa_terms(power)
            return self._rwa_terms_3wm
        elif power == 3:
            if self._rwa_terms_4wm is None:
                self._rwa_terms_4wm = self.analyzer.find_rwa_terms(power)
            return self._rwa_terms_4wm
        else:
            # For other powers, go directly to the analyzer without caching
            return self.analyzer.find_rwa_terms(power)

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


class ModeArrayFactory:
    """Factory for creating standard ModeArray configurations."""

    @staticmethod
    def create_basic_3wm(
        freqs: np.ndarray,
        kappas: np.ndarray,
        gammas: np.ndarray,
        forward_modes: bool = True,
    ) -> ModeArray:
        """
        Create a basic 3WM ModeArray with pump, signal, and idler modes.

        Args:
            freqs: Array of frequency points for interpolation
            kappas: Array of kappa values corresponding to frequencies
            gammas: Array of gamma values corresponding to frequencies
            forward_modes: Whether to create forward (True) or backward (False) propagating modes

        Returns:
            ModeArray: Configured for basic 3WM operation
        """
        direction = 1 if forward_modes else -1
        modes = [
            Mode(label="p", direction=direction),
            Mode(label="s", direction=direction),
            Mode(label="i", direction=direction),
        ]

        relations = [["i", "p-s"]]  # Idler is pump minus signal

        return ModeArray(modes, relations, freqs, kappas, gammas)

    @staticmethod
    def create_extended_3wm(
        freqs: np.ndarray,
        kappas: np.ndarray,
        gammas: np.ndarray,
        n_pump_harmonics: int = 1,
        n_frequency_conversion: int = 1,
        n_signal_harmonics: int = 1,
        forward_modes: bool = True,
    ) -> ModeArray:
        """
        Create an extended 3WM ModeArray with pump harmonics and conversion terms.

        Args:
            freqs: Array of frequency points for interpolation
            kappas: Array of kappa values corresponding to frequencies
            gammas: Array of gamma values corresponding to frequencies
            n_pump_harmonics: Number of pump harmonics to include
            forward_modes: Whether to create forward (True) or backward (False) propagating modes

        Returns:
            ModeArray: Configured for extended 3WM operation with harmonics
        """
        direction = 1 if forward_modes else -1

        # Create basic modes
        modes = [
            Mode(label="p", direction=direction),
            Mode(label="s", direction=direction),
            Mode(label="i", direction=direction),
        ]

        # Basic relation
        relations = [["i", "p-s"]]  # Idler is pump minus signal

        for n in range(2, n_signal_harmonics + 2):
            modes.append(Mode(label=f"s{n}", direction=direction))
            relations.append([f"s{n}", "s+" * (n - 1) + "s"])
            modes.append(Mode(label=f"i{n}", direction=direction))
            relations.append([f"i{n}", "i+" * (n - 1) + "i"])

        for n in range(1, n_frequency_conversion + 1):
            if n == 1:
                modes.append(Mode(label="ps", direction=direction))  # p+s
                modes.append(Mode(label="pi", direction=direction))  # p+i
                relations.append(["ps", "p+s"])
                relations.append(["pi", "p+i"])
            else:
                modes.append(Mode(label=f"p{n}s", direction=direction))  # p+s
                modes.append(Mode(label=f"p{n}i", direction=direction))  # p+i
                relations.append([f"p{n}s", "p+" * (n - 1) + "p+s"])
                relations.append([f"p{n}i", "p+" * (n - 1) + "p+i"])

        for n in range(2, n_pump_harmonics + 2):
            modes.append(Mode(label=f"p{n}", direction=direction))
            relations.append([f"p{n}", "p+" * (n - 1) + "p"])

        return ModeArray(modes, relations, freqs, kappas, gammas)

    @staticmethod
    def create_custom(
        freqs: np.ndarray,
        kappas: np.ndarray,
        gammas: np.ndarray,
        mode_labels: List[str],
        mode_directions: List[int],
        relations: List[List[str]],
    ) -> ModeArray:
        """
        Create a custom ModeArray with user-defined modes and relations.

        Args:
            freqs: Array of frequency points for interpolation
            kappas: Array of kappa values corresponding to frequencies
            gammas: Array of gamma values corresponding to frequencies
            mode_labels: List of mode labels
            mode_directions: List of mode directions (1 for forward, -1 for backward)
            relations: List of relations between modes

        Returns:
            ModeArray: Custom configured ModeArray
        """
        if len(mode_labels) != len(mode_directions):
            raise ValueError(
                "mode_labels and mode_directions must have the same length"
            )

        modes = [
            Mode(label=label, direction=direction)
            for label, direction in zip(mode_labels, mode_directions)
        ]

        return ModeArray(modes, relations, freqs, kappas, gammas)
