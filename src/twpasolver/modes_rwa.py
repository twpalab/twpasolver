"""Classes to represent frequency modes, their properties and the relations between them."""

import itertools as it
from collections import Counter, defaultdict, deque
from copy import deepcopy
from dataclasses import dataclass
from math import factorial
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.interpolate import interp1d
from sympy import symbols

from twpasolver.logger import log


class ParameterInterpolator:
    """
    Interpolates kappa, gamma, and alpha values based on frequency.

    Always returns real kappa and alpha, complex gamma.
    """

    def __init__(
        self,
        frequencies: np.ndarray,
        kappas: np.ndarray,
        gammas: np.ndarray,
        alphas: np.ndarray,
        kind: str = "cubic",
    ):
        """
        Initialize interpolators for kappa, gamma, and alpha.

        Args:
            frequencies: Array of frequency points
            kappas: Array of coupling coefficients (real)
            gammas: Array of reflection coefficients (complex)
            alphas: Array of attenuation coefficients (real)
            kind: Interpolation method ('linear', 'cubic', etc.)
        """
        # Validate input arrays
        if not (len(frequencies) == len(kappas) == len(gammas) == len(alphas)):
            raise ValueError(
                "Frequencies, kappas, gammas, and alphas must have the same length"
            )
        if len(frequencies) < 2:
            raise ValueError("Need at least 2 points for interpolation")

        self.orig_freqs = frequencies
        self.orig_kappas = kappas
        self.orig_gammas = gammas
        self.orig_alphas = alphas
        self.freq_min = np.min(frequencies)
        self.freq_max = np.max(frequencies)

        # For kappa interpolation (real only)
        self.kappa_interp = interp1d(
            frequencies,
            np.real(kappas),
            kind=kind,
            bounds_error=False,
            fill_value=(np.real(kappas[0]), np.real(kappas[-1])),
        )

        # For gamma interpolation (complex)
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

        # For alpha interpolation (real only)
        self.alpha_interp = interp1d(
            frequencies,
            np.real(alphas),
            kind=kind,
            bounds_error=False,
            fill_value=(np.real(alphas[0]), np.real(alphas[-1])),
        )

    def get_parameters(
        self, frequency: Union[float, np.ndarray]
    ) -> Tuple[
        Union[float, np.ndarray], Union[complex, np.ndarray], Union[float, np.ndarray]
    ]:
        """
        Get interpolated kappa, gamma, and alpha for a given frequency or array of frequencies.

        Args:
            frequency: Frequency point(s) for interpolation, can be a single value or array

        Returns:
            Tuple of (kappa, gamma, alpha) at the requested frequency/frequencies
            kappa and alpha are real, gamma is complex
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

        # Interpolate kappa (real only)
        kappa = self.kappa_interp(frequency)

        # Interpolate gamma (complex)
        if self.gamma_imag is not None:
            gamma_real = self.gamma_real(frequency)
            gamma_imag = self.gamma_imag(frequency)
            gamma = gamma_real + 1j * gamma_imag
        else:
            gamma = self.gamma_real(frequency)

        # Interpolate alpha (real only)
        alpha = self.alpha_interp(frequency)

        return kappa, gamma, alpha


@dataclass
class Mode:
    """
    Represents a single mode with its physical properties.

    Args:
        label: Mode identifier (e.g., 'p' for pump)
        frequency: Mode frequency
        direction: 1 for forward, -1 for backward
        gamma: Reflection coefficient
        k: Wavenumber (calculated from frequency)
        alpha: Attenuation constant
    """

    label: str
    direction: int = 1  # 1 for forward, -1 for backward
    frequency: Optional[float] = None
    k: Optional[float] = None
    gamma: Optional[Union[float, complex]] = 0.0
    alpha: Optional[float] = 0.0

    def __post_init__(self):
        """Initialize derived quantities and validate inputs."""
        # Ensure direction is ±1
        if abs(self.direction) != 1:
            raise ValueError("Direction must be either 1 (forward) or -1 (backward)")
        if self.k is not None:
            self.k = self.k * self.direction

    def __eq__(self, other):
        """Compare modes based on their physical properties."""
        if not isinstance(other, Mode):
            return False
        return (
            self.frequency == other.frequency
            and self.direction == other.direction
            and self.gamma == other.gamma
            and self.k == other.k
            and self.alpha == other.alpha
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
            f"dir={self.direction}, gamma={self.gamma}, k={self.k}, alpha={self.alpha})"
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
        self._relations = relations
        self.relations_idx = self._convert_relations_to_indices()
        self.modes_subs = self._compute_substitutions()

        # Cache for RWA terms
        self._rwa_terms_cache: Dict[int, List[Tuple[Any, ...]]] = {}

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
        coeff = 1.0
        if len(set(terms)) != len(terms):
            for repetitions in Counter(terms).values():
                coeff = coeff / factorial(repetitions)
        return coeff

    def find_rwa_terms(self, power: int = 3) -> List[Tuple[Any, ...]]:
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

    def print_rwa_terms(self, terms: List[Tuple[Any, ...]]) -> None:
        """Pretty print the RWA terms with their coefficients."""
        for term in terms:
            mode_name = term[2]
            rhs_terms = term[3]
            coeff = term[4]
            print(f"{mode_name} = {rhs_terms} {coeff}")
        print(f"\nTotal matches: {len(terms)}")


class ModeArray:
    """Class representing a list of modes and frequency relations between them."""

    def __init__(
        self,
        modes: List[Mode],
        relations: List[List[str]],
        interpolator: ParameterInterpolator,
    ):
        """
        Initialize mode array with modes, relations and interpolator.

        Args:
            modes: List of Mode objects
            relations: List of [result, expression] pairs for mode relationships
            interpolator: ParameterInterpolator instance for getting mode parameters
        """
        self.modes = {mode.label: mode for mode in modes}
        self.relations = relations
        self.interpolator = interpolator

        self.analyzer = RWAAnalyzer(list(self.modes.keys()), relations)

        # Build symbolic expressions for efficient frequency propagation
        self._build_symbolic_expressions()

        # Cache for computed mixing coefficients
        self._rwa_terms_3wm: Optional[List[Tuple[Any, ...]]] = None
        self._rwa_terms_4wm: Optional[List[Tuple[Any, ...]]] = None

        # Validate initial state
        self._validate_modes()

    def _parse_expression_for_propagation(self, expr: str) -> List[Tuple[str, int]]:
        """Parse expression into list of (mode, coefficient) pairs for frequency propagation."""
        expr = expr.replace(" ", "")
        if not expr.startswith(("+", "-")):
            expr = "+" + expr

        terms = []
        current_term = ""
        sign = 1

        for char in expr:
            if char in ["+", "-"]:
                if current_term:
                    terms.append((current_term, sign))
                current_term = ""
                sign = 1 if char == "+" else -1
            else:
                current_term += char

        if current_term:
            terms.append((current_term, sign))

        return terms

    def _build_dependency_graph(self):
        """Build dependency graph for relation analysis."""
        self.dependency_graph = defaultdict(set)
        self.reverse_deps = defaultdict(set)

        for result, expr in self.relations:
            terms = self._parse_expression_for_propagation(expr)
            for mode, _ in terms:
                self.dependency_graph[mode].add(result)
                self.reverse_deps[result].add(mode)

    def _topological_sort(self) -> List[str]:
        """Perform topological sort of dependencies."""
        in_degree = defaultdict(int)

        # Calculate in-degrees
        for mode in self.modes.keys():
            in_degree[mode] = len(self.reverse_deps[mode])

        # Start with modes that have no dependencies
        queue = deque([mode for mode in self.modes.keys() if in_degree[mode] == 0])
        result = []

        while queue:
            current = queue.popleft()
            result.append(current)

            # Update in-degrees of dependent modes
            for dependent in self.dependency_graph[current]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

        return result

    def _build_symbolic_expressions(self):
        """
        Build symbolic expressions for each mode in terms of independent modes.

        This enables O(n) frequency propagation instead of iterative solving.
        """
        # Build dependency graph
        self._build_dependency_graph()

        # Find independent modes (those not defined by relations)
        self.independent_modes = set(self.modes.keys())
        for result, _ in self.relations:
            self.independent_modes.discard(result)

        # Initialize symbolic expressions
        self.symbolic_expressions = {}

        # Independent modes have simple expressions
        for mode in self.independent_modes:
            self.symbolic_expressions[mode] = {mode: 1.0}

        # Build expressions for dependent modes using topological order
        topo_order = self._topological_sort()

        for mode in topo_order:
            if mode in self.symbolic_expressions:
                continue  # Already processed (independent mode)

            # Find the relation that defines this mode
            for result, expr in self.relations:
                if result == mode:
                    terms = self._parse_expression_for_propagation(expr)
                    self.symbolic_expressions[mode] = {}

                    for dep_mode, coeff in terms:
                        if dep_mode in self.symbolic_expressions:
                            # Add contribution from dependency
                            for base_mode, base_coeff in self.symbolic_expressions[
                                dep_mode
                            ].items():
                                if base_mode in self.symbolic_expressions[mode]:
                                    self.symbolic_expressions[mode][base_mode] += (
                                        coeff * base_coeff
                                    )
                                else:
                                    self.symbolic_expressions[mode][base_mode] = (
                                        coeff * base_coeff
                                    )
                        else:
                            # This should not happen with proper topological sort
                            log.warning(
                                f"Dependency {dep_mode} not found when processing {mode}"
                            )
                    break

        log.info(
            f"Built symbolic expressions for {len(self.symbolic_expressions)} modes"
        )
        log.info(f"Independent modes: {self.independent_modes}")

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

    def _propagate_frequencies_symbolic(
        self, updated_freqs: Dict[str, float]
    ) -> Dict[str, Optional[float]]:
        """
        Symbolic frequency propagation - O(n) complexity.

        Args:
            updated_freqs: Dictionary of mode labels to their new frequencies

        Returns:
            Dictionary of all mode labels to their computed frequencies
        """
        # Start with current frequencies
        frequencies: Dict[str, Optional[float]] = {
            label: mode.frequency for label, mode in self.modes.items()
        }
        frequencies.update(updated_freqs)

        # Evaluate each mode using its symbolic expression
        for mode, expression in self.symbolic_expressions.items():
            if mode in updated_freqs:
                continue  # Already set by user

            # Check if all base modes are available
            all_available = True
            new_freq = 0.0

            for base_mode, coeff in expression.items():
                if frequencies.get(base_mode) is None:
                    all_available = False
                    break
                new_freq += coeff * frequencies[base_mode]

            if all_available:
                frequencies[mode] = new_freq

        return frequencies

    def update_frequencies(self, new_frequencies: Dict[str, float]) -> None:
        """
        Update mode frequencies using optimized symbolic propagation.

        Args:
            new_frequencies: Dictionary mapping mode labels to new frequencies
        """
        # Validate input modes exist
        unknown_modes = set(new_frequencies.keys()) - set(self.modes.keys())
        if unknown_modes:
            raise ValueError(f"Unknown modes in frequency update: {unknown_modes}")

        # Use symbolic propagation (much faster than iterative)
        all_frequencies = self._propagate_frequencies_symbolic(new_frequencies)

        # Update modes with new frequencies and interpolated parameters
        for label, freq in all_frequencies.items():
            if freq is not None:
                mode = self.modes[label]
                mode.frequency = freq
                # Get interpolated parameters (always returns kappa, gamma, alpha)
                kappa, gamma, alpha = self.interpolator.get_parameters(abs(freq))
                mode.alpha = alpha  # type: ignore
                mode.gamma = gamma  # type: ignore
                mode.k = kappa * mode.direction  # type: ignore

    def update_base_data(self, interpolator: ParameterInterpolator) -> None:
        """
        Update the base data interpolator and refresh all mode parameters.

        Args:
            interpolator: New ParameterInterpolator instance
        """
        self.interpolator = interpolator

        # Refresh all mode parameters with current frequencies
        current_freqs = {
            label: mode.frequency
            for label, mode in self.modes.items()
            if mode.frequency is not None
        }
        if current_freqs:
            self.update_frequencies(current_freqs)

    def process_frequency_array(
        self, mode_label: str, frequencies: np.ndarray
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Process an array of frequencies for a single mode, propagating to all related modes.

        Uses symbolic expressions for optimal performance.

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
        mode_params: Dict[str, Dict[str, np.ndarray]] = {}

        # For each mode, compute its frequency array using symbolic expressions
        for mode, expression in self.symbolic_expressions.items():
            if mode_label not in expression:
                indep_mode = self.modes[mode]
                mode_params[mode] = {
                    "freqs": np.full(len(frequencies), indep_mode.frequency),
                    "k": np.full(len(frequencies), indep_mode.k),
                    "gamma": np.full(len(frequencies), indep_mode.gamma),
                    "alpha": np.full(len(frequencies), indep_mode.alpha),
                }
                continue

            # Compute frequency array for this mode
            mode_freqs = frequencies * expression[mode_label]

            # Add contributions from other independent modes if they have values
            for base_mode, coeff in expression.items():
                if (
                    base_mode != mode_label
                    and self.modes[base_mode].frequency is not None
                ):
                    mode_freqs += coeff * self.modes[base_mode].frequency

            # Get direction for this mode
            direction = self.modes[mode].direction

            # Get parameters for all frequencies of this mode (always returns kappa, gamma, alpha)
            kappas, gammas, alphas = self.interpolator.get_parameters(
                np.abs(mode_freqs)
            )

            # Apply direction to kappas
            if isinstance(kappas, np.ndarray):
                kappas = kappas * direction
            else:
                kappas = np.array([kappas * direction])

            # Store parameters
            mode_params[mode] = {
                "freqs": mode_freqs,
                "k": kappas,
                "gamma": gammas,  # type: ignore
                "alpha": alphas,  # type: ignore
            }

        return mode_params

    def plot_mode_relations(
        self,
        figsize: Tuple[float, float] = (12, 8),
        node_size: int = 3000,
        font_size: int = 12,
        show_frequencies: bool = False,
        show_directions: bool = True,
        layout: str = "hierarchical",
    ) -> None:
        """
        Plot the relationships between modes as a directed graph.

        Args:
            figsize: Figure size (width, height)
            node_size: Size of mode nodes
            font_size: Font size for labels
            show_frequencies: Whether to show current frequencies on nodes
            show_directions: Whether to show mode propagation directions
            layout: Graph layout algorithm ('spring', 'circular', 'hierarchical')
        """
        # Create directed graph
        G = nx.DiGraph()

        # Add nodes for all modes
        for mode_label, mode in self.modes.items():
            node_attrs = {
                "frequency": mode.frequency,
                "direction": mode.direction,
                "is_independent": mode_label in self.independent_modes,
            }
            G.add_node(mode_label, **node_attrs)

        # Add edges based on relations
        edge_labels: dict[Tuple[str, str], str] = {}
        for result, expr in self.relations:
            terms = self._parse_expression_for_propagation(expr)
            for dep_mode, coeff in terms:
                if coeff > 0:
                    edge_color = "blue"
                    edge_style = "-"
                else:
                    edge_color = "red"
                    edge_style = "--"

                G.add_edge(
                    dep_mode,
                    result,
                    weight=abs(coeff),
                    color=edge_color,
                    style=edge_style,
                )

                # Create edge label
                coeff_str = f"+{coeff}" if coeff > 0 else str(coeff)
                if (dep_mode, result) in edge_labels:
                    edge_labels[(dep_mode, result)] += f", {coeff_str}"
                else:
                    edge_labels[(dep_mode, result)] = coeff_str

        # Create figure
        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=figsize, gridspec_kw={"width_ratios": [3, 1]}
        )

        # Choose layout
        if layout == "hierarchical":
            pos = self._hierarchical_layout(G)
        elif layout == "circular":
            pos = nx.circular_layout(G)
        else:  # spring layout
            pos = nx.spring_layout(G, k=2, iterations=50)

        # Node colors based on type
        node_colors = []
        for node in G.nodes():
            if G.nodes[node]["is_independent"]:
                node_colors.append("lightgreen")  # Independent modes
            else:
                node_colors.append("lightblue")  # Dependent modes

        # Draw nodes
        nx.draw_networkx_nodes(
            G, pos, node_color=node_colors, node_size=node_size, ax=ax1
        )

        # Draw edges with different styles
        edge_colors = [G.edges[edge]["color"] for edge in G.edges()]
        edge_styles = [G.edges[edge]["style"] for edge in G.edges()]

        # Draw positive and negative edges separately for different styles
        pos_edges = [(u, v) for u, v, d in G.edges(data=True) if d["color"] == "blue"]
        neg_edges = [(u, v) for u, v, d in G.edges(data=True) if d["color"] == "red"]

        if pos_edges:
            nx.draw_networkx_edges(
                G,
                pos,
                edgelist=pos_edges,
                edge_color="blue",
                style="-",
                arrows=True,
                arrowsize=20,
                ax=ax1,
            )
        if neg_edges:
            nx.draw_networkx_edges(
                G,
                pos,
                edgelist=neg_edges,
                edge_color="red",
                style="--",
                arrows=True,
                arrowsize=20,
                ax=ax1,
            )

        # Node labels
        if show_frequencies and show_directions:
            labels = {}
            for mode_label, mode in self.modes.items():
                direction_str = "→" if mode.direction == 1 else "←"
                freq_str = f"{mode.frequency:.1f}" if mode.frequency else "None"
                labels[mode_label] = f"{mode_label}\n{direction_str}\n({freq_str})"
        elif show_frequencies:
            labels = {}
            for mode_label, mode in self.modes.items():
                freq_str = f"{mode.frequency:.1f}" if mode.frequency else "None"
                labels[mode_label] = f"{mode_label}\n({freq_str})"
        elif show_directions:
            labels = {}
            for mode_label, mode in self.modes.items():
                direction_str = "→" if mode.direction == 1 else "←"
                labels[mode_label] = f"{mode_label}\n{direction_str}"
        else:
            labels = {mode: mode for mode in G.nodes()}

        nx.draw_networkx_labels(G, pos, labels, font_size=font_size, ax=ax1)

        # Edge labels (coefficients)
        nx.draw_networkx_edge_labels(
            G, pos, edge_labels, font_size=font_size - 2, ax=ax1
        )

        ax1.set_title("Mode Relationships", fontsize=font_size + 2, fontweight="bold")
        ax1.axis("off")

        # Legend
        legend_elements = [
            mpatches.Patch(color="lightgreen", label="Independent modes"),
            mpatches.Patch(color="lightblue", label="Dependent modes"),
            plt.Line2D([0], [0], color="blue", lw=2, label="Positive contribution"),
            plt.Line2D(
                [0],
                [0],
                color="red",
                lw=2,
                linestyle="--",
                label="Negative contribution",
            ),
        ]
        ax1.legend(handles=legend_elements, loc="upper right")

        # Relations text in second subplot
        ax2.axis("off")
        ax2.set_title("Relations", fontsize=font_size + 1, fontweight="bold")

        relations_text = "Symbolic Expressions:\n\n"
        for mode, expression in self.symbolic_expressions.items():
            expr_parts = []
            for base_mode, coeff in expression.items():
                if coeff == 1:
                    expr_parts.append(base_mode)
                elif coeff == -1:
                    expr_parts.append(f"-{base_mode}")
                else:
                    expr_parts.append(f"{coeff}*{base_mode}")

            expr_str = " + ".join(expr_parts).replace(" + -", " - ")
            relations_text += f"{mode} = {expr_str}\n"

        relations_text += f"\nIndependent: {', '.join(sorted(self.independent_modes))}"

        ax2.text(
            0.05,
            0.95,
            relations_text,
            transform=ax2.transAxes,
            fontsize=font_size - 1,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

        plt.tight_layout()
        plt.show()

    def _hierarchical_layout(self, G: nx.DiGraph) -> Dict[str, Tuple[float, float]]:
        """Create hierarchical layout with independent modes at bottom."""
        levels = {}

        # Independent modes at level 0
        for mode in self.independent_modes:
            levels[mode] = 0

        # Assign levels based on dependencies
        max_level = 0
        for mode in nx.topological_sort(G):
            if mode not in levels:
                # Find maximum level of predecessors
                pred_levels = [levels.get(pred, 0) for pred in G.predecessors(mode)]
                levels[mode] = max(pred_levels, default=0) + 1
                max_level = max(max_level, levels[mode])

        # Create positions
        level_counts: dict[int, int] = defaultdict(int)
        level_positions: dict[int, int] = defaultdict(int)

        # Count nodes per level
        for level in levels.values():
            level_counts[level] += 1

        pos: dict[str, tuple[float, float]] = {}
        for mode, level in levels.items():
            x = level_positions[level] - (level_counts[level] - 1) / 2
            y = max_level - level  # Flip so independent modes are at bottom
            pos[mode] = (x, y)
            level_positions[level] += 1

        return pos

    def get_mode(self, label: str) -> Mode:
        """Get mode by label."""
        return self.modes[label]

    def get_rwa_terms(self, power: int = 3) -> List[Tuple[Any, ...]]:
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
            independence = " (independent)" if label in self.independent_modes else ""
            print(f"{label}{independence}:")
            print(f"  Frequency: {mode.frequency}")
            print(f"  Direction: {mode.direction}")
            print(f"  k: {mode.k}")
            print(f"  gamma: {mode.gamma}")
            print(f"  alpha: {mode.alpha}")
            print()

    def print_symbolic_expressions(self):
        """Print the symbolic expressions for all modes."""
        print("Symbolic Expressions:")
        print("=" * 50)
        for mode, expression in self.symbolic_expressions.items():
            expr_parts = []
            for base_mode, coeff in expression.items():
                if coeff == 1:
                    expr_parts.append(base_mode)
                elif coeff == -1:
                    expr_parts.append(f"-{base_mode}")
                else:
                    expr_parts.append(f"{coeff}*{base_mode}")

            expr_str = " + ".join(expr_parts).replace(" + -", " - ")
            independence = " (independent)" if mode in self.independent_modes else ""
            print(f"{mode}{independence} = {expr_str}")


class ModeArrayFactory:
    """Factory for creating standard ModeArray configurations."""

    @staticmethod
    def create_basic_3wm(
        base_data: Dict[str, Any],
        forward_modes: bool = True,
    ) -> ModeArray:
        """
        Create a basic 3WM ModeArray with pump, signal, and idler modes.

        Args:
            base_data: Dictionary containing 'freqs', 'k', 'gammas', and 'alpha' arrays
            forward_modes: Whether to create forward (True) or backward (False) propagating modes

        Returns:
            ModeArray: Configured for basic 3WM operation
        """
        # Extract required arrays from base_data
        freqs = base_data["freqs"]
        kappas = base_data["k"]
        gammas = base_data["gammas"]
        alphas = base_data["alpha"]

        # Create interpolator
        interpolator = ParameterInterpolator(freqs, kappas, gammas, alphas)

        direction = 1 if forward_modes else -1
        modes = [
            Mode(label="p", direction=direction),
            Mode(label="s", direction=direction),
            Mode(label="i", direction=direction),
        ]

        relations = [["i", "p-s"]]  # Idler is pump minus signal

        return ModeArray(modes, relations, interpolator)

    @staticmethod
    def create_extended_3wm(
        base_data: Dict[str, Any],
        n_pump_harmonics: int = 1,
        n_frequency_conversion: int = 1,
        n_signal_harmonics: int = 1,
        n_sidebands: int = 1,
        forward_modes: bool = True,
    ) -> ModeArray:
        """
        Create an extended 3WM ModeArray with pump harmonics and conversion terms.

        Args:
            base_data: Dictionary containing 'freqs', 'k', 'gammas', and 'alpha' arrays
            n_pump_harmonics: Number of pump harmonics to include
            n_frequency_conversion: Number of frequency conversion terms
            n_signal_harmonics: Number of signal and idler harmonics
            forward_modes: Whether to create forward (True) or backward (False) propagating modes

        Returns:
            ModeArray: Configured for extended 3WM operation with harmonics
        """
        # Extract required arrays from base_data
        freqs = base_data["freqs"]
        kappas = base_data["k"]
        gammas = base_data["gammas"]
        alphas = base_data["alpha"]

        # Create interpolator
        interpolator = ParameterInterpolator(freqs, kappas, gammas, alphas)

        direction = 1 if forward_modes else -1

        # Create basic modes
        modes = [
            Mode(label="p", direction=direction),
            Mode(label="s", direction=direction),
            Mode(label="i", direction=direction),
        ]

        # Basic relation
        relations = [["i", "p-s"]]  # Idler is pump minus signal

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

        for n in range(2, n_signal_harmonics + 2):
            modes.append(Mode(label=f"s{n}", direction=direction))
            relations.append([f"s{n}", "s+" * (n - 1) + "s"])
            modes.append(Mode(label=f"i{n}", direction=direction))
            relations.append([f"i{n}", "i+" * (n - 1) + "i"])

        # for n in range(2, n_sidebands + 2):
        #     modes.append(Mode(label=f"s{n}p", direction=direction))
        #     modes.append(Mode(label=f"i{n}p", direction=direction))
        #     relations.append([f"s{n}p", "s+s-p"])
        #     relations.append([f"i{n}p", "i+i-p"])

        return ModeArray(modes, relations, interpolator)

    @staticmethod
    def create_custom(
        base_data: Dict[str, Any],
        mode_labels: List[str],
        mode_directions: List[int],
        relations: List[List[str]],
    ) -> ModeArray:
        """
        Create a custom ModeArray with user-defined modes and relations.

        Args:
            base_data: Dictionary containing 'freqs', 'k', 'gammas', and 'alpha' arrays
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

        # Extract required arrays from base_data
        freqs = base_data["freqs"]
        kappas = base_data["k"]
        gammas = base_data["gammas"]
        alphas = base_data["alpha"]

        # Create interpolator
        interpolator = ParameterInterpolator(freqs, kappas, gammas, alphas)

        modes = [
            Mode(label=label, direction=direction)
            for label, direction in zip(mode_labels, mode_directions)
        ]

        return ModeArray(modes, relations, interpolator)
