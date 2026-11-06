"""
``modes_rwa`` module.

Mode Management and Rotating Wave Approximation Analysis
========================================================

Tools for managing complex mode relationships in TWPA analysis, enabling
simulation of extended coupled mode equation (CME) systems with arbitrary
numbers of modes beyond basic pump-signal-idler configuration.

Key Components
--------------

**ModeArray**: Organizes related modes with automatic frequency propagation,
parameter interpolation, and dependency graph management.

**RWAAnalyzer**: Identifies valid 3-wave and 4-wave mixing terms that satisfy
the Rotating Wave Approximation for any given set of mode relationships.

**ModeArrayFactory**: Factory methods for creating standard and custom mode
configurations including pump harmonics, frequency conversion terms, and harmonics.

**ParameterInterpolator**: Interpolation of mode parameters (kappa, gamma, alpha)
from base TWPA circuit data across frequency ranges.

Capabilities
------------

**RWA Term Selection**:
    - Automatic discovery of valid 3WM and 4WM mixing processes
    - Coefficient calculation including factorial corrections for repeated modes
    - Caching of RWA terms for performance

**Mode Relationships**:
    - Support for arbitrary frequency relationships between modes
    - Forward and backward propagating modes
    - Pump harmonics (p, p2, p3, ...)
    - Frequency conversion terms (p+s, p+i, ...)
    - Signal/idler harmonics (s2, i2, ...)

**Symbolic Frequency Propagation**:
    - O(n) frequency updates using pre-computed symbolic expressions
    - Automatic identification of independent vs. dependent modes
    - Topological sorting of mode dependencies
    - Visualization of the mode relations with a graph

**Parameter Management**:
    - Easy interpolation of mode parameters from base circuit data

Examples
--------
See Tutorial 4 (:ref:`tutorials`) for complete examples covering:

- Basic 3WM mode array creation and visualization
- Extended mode arrays with pump harmonics and frequency conversion
- Custom mode configurations for specific processes
- RWA term analysis and CME integration

The tutorial demonstrates the progression from simple mode relationships to
complex multi-mode systems used in realistic TWPA simulations.

Performance Notes
-----------------
- Symbolic frequency propagation enables O(n) scaling, useful when iterating over arrays.
- RWA term caching eliminates redundant calculations during CME solving
- Numba-compatible data structures for integration with fast CME solvers
- Efficient interpolation supports faster parameter sweeps
"""

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
from matplotlib.figure import Figure
from scipy.interpolate import CubicSpline
from sympy import symbols

from twpasolver.logger import log


class ParameterInterpolator:
    """
    Interpolates an arbitrary set of parameters over a sorted frequency array.

    Parameters are passed as ``params={name: array}``. Each array may be:

    * **1D** ``(n_freq,)`` — real or complex, interpolated directly.
    * **2D** ``(n, n_freq)`` — every row interpolated along the frequency axis
      (one vectorised spline per parameter); returns ``(n, n_freq_interp)``.

    The base frequency array must be sorted (ascending). Uses ``CubicSpline``
    for fast, accurate interpolation (~20× faster than ``interp1d`` with cubic).

    A ``get_parameters`` compatibility shim returns ``(k, gamma, alpha, Zb)``
    to preserve the existing call-sites unchanged.
    """

    def __init__(self, frequencies: np.ndarray, params: Dict[str, np.ndarray]):
        """
        Args:
            frequencies: Sorted 1-D array of base frequency points.
            params: Mapping of parameter name to array of shape ``(n_freq,)``
                    or ``(n, n_freq)``. Complex arrays are handled transparently.
        """
        self.frequencies = np.asarray(frequencies)
        self._splines: Dict[str, Any] = {}

        for name, values in params.items():
            values = np.asarray(values)
            is_2d = values.ndim == 2
            is_complex = np.iscomplexobj(values)

            def _spl(arr: np.ndarray) -> CubicSpline:
                # axis=-1: a 2-D array is splined row by row in one call.
                return CubicSpline(self.frequencies, arr, axis=-1)

            if is_2d:
                self._splines[name] = (
                    "2d",
                    _spl(np.real(values)),
                    _spl(np.imag(values)) if is_complex else None,
                )
            elif is_complex:
                self._splines[name] = (
                    "1c",
                    _spl(np.real(values)),
                    _spl(np.imag(values)),
                )
            else:
                self._splines[name] = ("1r", _spl(np.real(values)))

    def interpolate(
        self, frequencies: Union[float, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        Interpolate all parameters at the requested frequency/frequencies.

        Args:
            frequencies: Scalar or array of frequencies.

        Returns:
            Dict mapping each parameter name to its interpolated array.
            2-D parameters return shape ``(n, len(frequencies))``.
        """
        freqs = np.atleast_1d(np.asarray(frequencies, dtype=float))
        out: Dict[str, np.ndarray] = {}
        for name, spec in self._splines.items():
            t = spec[0]
            if t == "1r":
                out[name] = spec[1](freqs)
            elif t == "1c":
                out[name] = spec[1](freqs) + 1j * spec[2](freqs)
            else:  # "2d"
                real_rows = spec[1](freqs)
                if spec[2] is not None:
                    out[name] = real_rows + 1j * spec[2](freqs)
                else:
                    out[name] = real_rows
        return out

    def get_parameters(
        self, frequency: Union[float, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compatibility shim — returns ``(k, gamma, alpha, Zb)`` tuple.

        Expects params dict to contain keys ``'k'``, ``'gamma'``, ``'alpha'``, ``'Zb'``.
        """
        p = self.interpolate(frequency)
        return p["k"], p["gamma"], p["alpha"], p["Zb"]


@dataclass
class Mode:
    """
    Represents a single mode with its physical properties.

    Every mode stands for both its +x and -x traveling waves; the propagation
    direction of injected waves is chosen by the solver inputs
    (``initial_amplitudes_fwd`` / ``initial_amplitudes_bwd``), not by the mode.

    Args:
        label: Mode identifier (e.g., 'p' for pump)
        frequency: Mode frequency
        gamma: Reflection coefficient
        k: Wavenumber (calculated from frequency)
        alpha: Attenuation constant
    """

    label: str
    frequency: Optional[float] = None
    k: Optional[float] = None
    gamma: Optional[Union[float, complex]] = 0.0
    alpha: Optional[float] = 0.0
    Zb: Optional[float] = 0.0

    def __str__(self):
        """Get string representation (the mode label)."""
        return self.label

    def __repr__(self):
        """Get representation of class."""
        return (
            f'Mode("{self.label}", freq={self.frequency}, '
            f"gamma={self.gamma}, k={self.k}, alpha={self.alpha}, Zb={self.Zb})"
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

        self._rwa_terms_cache: Dict[int, List[Tuple[Any, ...]]] = {}

    @property
    def relations(self):
        """Getter for mode relations."""
        return self._relations

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
        graph = nx.DiGraph(
            (i, output) for output, inputs, _ in self.relations_idx for i in inputs
        )
        if not nx.is_directed_acyclic_graph(graph):
            cycle = [self.modes[i] for i, _ in nx.find_cycle(graph)]
            raise ValueError(f"Mode relations are cyclic: {' -> '.join(cycle)}")
        modes_subs = deepcopy(self.modes_symbolic)
        # Without relations every mode is independent; the loop below would
        # never reach its exit condition.
        apply = bool(self.relations_idx)
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
        self._rwa_terms_cache[power] = result

        return result


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

        self._build_symbolic_expressions()

        self._rwa_terms_3wm: Optional[List[Tuple[Any, ...]]] = None
        self._rwa_terms_4wm: Optional[List[Tuple[Any, ...]]] = None

        self._compute_mode_graph()

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

        for mode in self.modes.keys():
            in_degree[mode] = len(self.reverse_deps[mode])

        queue = deque([mode for mode in self.modes.keys() if in_degree[mode] == 0])
        result = []

        while queue:
            current = queue.popleft()
            result.append(current)

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
        self._build_dependency_graph()

        self.independent_modes = set(self.modes.keys())
        for result, _ in self.relations:
            self.independent_modes.discard(result)

        self.symbolic_expressions = {}

        for mode in self.independent_modes:
            self.symbolic_expressions[mode] = {mode: 1.0}

        topo_order = self._topological_sort()

        for mode in topo_order:
            if mode in self.symbolic_expressions:
                continue

            for result, expr in self.relations:
                if result == mode:
                    terms = self._parse_expression_for_propagation(expr)
                    self.symbolic_expressions[mode] = {}

                    for dep_mode, coeff in terms:
                        if dep_mode in self.symbolic_expressions:
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
        frequencies: Dict[str, Optional[float]] = {
            label: mode.frequency for label, mode in self.modes.items()
        }
        frequencies.update(updated_freqs)

        for mode, expression in self.symbolic_expressions.items():
            if mode in updated_freqs:
                continue  # Already set by user

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
        unknown_modes = set(new_frequencies.keys()) - set(self.modes.keys())
        if unknown_modes:
            raise ValueError(f"Unknown modes in frequency update: {unknown_modes}")

        all_frequencies = self._propagate_frequencies_symbolic(new_frequencies)

        for label, freq in all_frequencies.items():
            if freq is not None:
                mode = self.modes[label]
                mode.frequency = freq
                kappa, gamma, alpha, Zb = self.interpolator.get_parameters(abs(freq))
                mode.alpha = alpha  # type: ignore
                mode.gamma = gamma  # type: ignore
                mode.k = kappa  # type: ignore
                mode.Zb = Zb  # type: ignore

    def update_base_data(self, interpolator: ParameterInterpolator) -> None:
        """
        Update the base data interpolator and refresh all mode parameters.

        Args:
            interpolator: New ParameterInterpolator instance
        """
        self.interpolator = interpolator

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
        if mode_label not in self.modes:
            raise ValueError(f"Unknown mode: {mode_label}")

        mode_params: Dict[str, Dict[str, np.ndarray]] = {}

        for mode, expression in self.symbolic_expressions.items():
            if mode_label not in expression:
                # Mode is independent of the swept frequency — broadcast its scalar
                # attributes across the frequency axis. For 2D attributes (n, 1) → (n, n_freq).
                indep_mode = self.modes[mode]
                n_freq = len(frequencies)

                def _broadcast(val):
                    val = np.asarray(val)
                    if val.ndim == 1 and val.size == 1:
                        # scalar stored as (1,) by atleast_1d in get_parameters
                        return np.full(n_freq, val.item())
                    if val.ndim == 2:
                        return np.broadcast_to(val, (val.shape[0], n_freq)).copy()
                    return np.full(n_freq, val)

                mode_params[mode] = {
                    "freqs": np.full(n_freq, indep_mode.frequency),
                    "k": _broadcast(indep_mode.k),
                    "gamma": _broadcast(indep_mode.gamma),
                    "alpha": _broadcast(indep_mode.alpha),
                    "Zb": _broadcast(indep_mode.Zb),
                }
                continue

            mode_freqs = frequencies * expression[mode_label]

            for base_mode, coeff in expression.items():
                if (
                    base_mode != mode_label
                    and self.modes[base_mode].frequency is not None
                ):
                    mode_freqs += coeff * self.modes[base_mode].frequency

            kappas, gammas, alphas, Zbs = self.interpolator.get_parameters(
                np.abs(mode_freqs)
            )

            kappas = np.asarray(kappas)

            mode_params[mode] = {
                "freqs": mode_freqs,
                "k": kappas,
                "gamma": gammas,
                "alpha": alphas,
                "Zb": Zbs,
            }

        return mode_params

    def get_dependent_modes(self, mode_label="s"):
        dg = self.dependency_graph
        allmodes = dg[mode_label]
        for mode in dg[mode_label]:
            if dg[mode]:
                allmodes = allmodes.union(self.get_dependent_modes(mode))
        return allmodes

    def _compute_mode_graph(self):
        G = nx.DiGraph()

        for mode_label, mode in self.modes.items():
            node_attrs = {
                "frequency": mode.frequency,
                "is_independent": mode_label in self.independent_modes,
            }
            G.add_node(mode_label, **node_attrs)
        self.mode_graph = G

    def plot_mode_relations(
        self,
        figsize: Tuple[float, float] = (12, 8),
        node_size: int = 3000,
        font_size: int = 12,
        show_frequencies: bool = False,
        layout: str = "hierarchical",
    ) -> Figure:
        """
        Plot the relationships between modes as a directed graph.

        Args:
            figsize: Figure size (width, height)
            node_size: Size of mode nodes
            font_size: Font size for labels
            show_frequencies: Whether to show current frequencies on nodes
            layout: Graph layout algorithm ('spring', 'circular', 'hierarchical')

        Returns:
            Figure: The matplotlib figure containing the plot.
        """
        G = self.mode_graph

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

                coeff_str = f"+{coeff}" if coeff > 0 else str(coeff)
                if (dep_mode, result) in edge_labels:
                    edge_labels[(dep_mode, result)] += f", {coeff_str}"
                else:
                    edge_labels[(dep_mode, result)] = coeff_str

        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=figsize, gridspec_kw={"width_ratios": [3, 1]}
        )

        if layout == "hierarchical":
            pos = self._hierarchical_layout(G)
        elif layout == "circular":
            pos = nx.circular_layout(G)
        else:
            pos = nx.spring_layout(G, k=2, iterations=50)

        node_colors = []
        for node in G.nodes():
            if G.nodes[node]["is_independent"]:
                node_colors.append("lightgreen")  # Independent modes
            else:
                node_colors.append("lightblue")  # Dependent modes

        nx.draw_networkx_nodes(
            G, pos, node_color=node_colors, node_size=node_size, ax=ax1
        )

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

        if show_frequencies:
            labels = {}
            for mode_label, mode in self.modes.items():
                freq_str = f"{mode.frequency:.1f}" if mode.frequency else "None"
                labels[mode_label] = f"{mode_label}\n({freq_str})"
        else:
            labels = {mode: mode for mode in G.nodes()}

        nx.draw_networkx_labels(G, pos, labels, font_size=font_size, ax=ax1)

        nx.draw_networkx_edge_labels(
            G, pos, edge_labels, font_size=font_size - 2, ax=ax1
        )

        ax1.set_title("Mode Relationships", fontsize=font_size + 2, fontweight="bold")
        ax1.axis("off")

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
        return fig

    def _hierarchical_layout(self, G: nx.DiGraph) -> Dict[str, Tuple[float, float]]:
        """Create hierarchical layout with independent modes at bottom."""
        levels = {}

        for mode in self.independent_modes:
            levels[mode] = 0

        max_level = 0
        for mode in nx.topological_sort(G):
            if mode not in levels:
                pred_levels = [levels.get(pred, 0) for pred in G.predecessors(mode)]
                levels[mode] = max(pred_levels, default=0) + 1
                max_level = max(max_level, levels[mode])

        level_counts: dict[int, int] = defaultdict(int)
        level_positions: dict[int, int] = defaultdict(int)

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
            return self.analyzer.find_rwa_terms(power)

    def print_modes(self):
        """Print current state of all modes."""
        for label, mode in self.modes.items():
            independence = " (independent)" if label in self.independent_modes else ""
            print(f"{label}{independence}:")
            print(f"  Frequency: {mode.frequency}")
            print(f"  k: {mode.k}")
            print(f"  gamma: {mode.gamma}")
            print(f"  alpha: {mode.alpha}")
            print(f"  Zb: {mode.Zb}")
            print()


class ModeArrayFactory:
    """Factory for creating standard ModeArray configurations."""

    @staticmethod
    def _interpolator_from_data(
        base_data: Dict[str, Any],
        interpolator: Optional["ParameterInterpolator"] = None,
    ) -> "ParameterInterpolator":
        """Build a ParameterInterpolator from a standard base_data dict.

        If ``interpolator`` is given it is returned unchanged, so several mode
        arrays built from the same base data can share one interpolator.
        """
        if interpolator is not None:
            return interpolator
        return ParameterInterpolator(
            base_data["freqs"],
            {
                "k": base_data["k"],
                "gamma": base_data["reflection_coeff"],
                "alpha": base_data["alpha"],
                "Zb": base_data["Zb"],
            },
        )

    @staticmethod
    def create_basic(
        base_data: Dict[str, Any],
        three_wave: bool = True,
        interpolator: Optional[ParameterInterpolator] = None,
    ) -> ModeArray:
        """
        Create a basic 3WM ModeArray with pump, signal, and idler modes.

        Args:
            base_data: Dictionary containing 'freqs', 'k', 'gammas', and 'alpha' arrays
            three_wave: Whether to include 3WM or 4WM mode relations
            interpolator: Optional shared ParameterInterpolator (built from
                ``base_data`` when omitted)

        Returns:
            ModeArray: Configured for basic 3WM operation
        """
        interpolator = ModeArrayFactory._interpolator_from_data(base_data, interpolator)

        modes = [Mode(label="p"), Mode(label="s"), Mode(label="i")]
        if three_wave:
            relations = [["i", "p-s"]]
        else:
            relations = [["i", "p+p-s"]]
        return ModeArray(modes, relations, interpolator)

    @staticmethod
    def create_extended_3wm(
        base_data: Dict[str, Any],
        n_pump_harmonics: int = 1,
        n_frequency_conversion: int = 1,
        n_signal_harmonics: int = 1,
        n_sidebands: int = 1,
        interpolator: Optional[ParameterInterpolator] = None,
    ) -> ModeArray:
        """
        Create an extended 3WM ModeArray with pump harmonics and conversion terms.

        Args:
            base_data: Dictionary containing 'freqs', 'k', 'gammas', and 'alpha' arrays
            n_pump_harmonics: Number of pump harmonics to include
            n_frequency_conversion: Number of frequency conversion terms
            n_signal_harmonics: Number of signal and idler harmonics
            interpolator: Optional shared ParameterInterpolator (built from
                ``base_data`` when omitted)

        Returns:
            ModeArray: Configured for extended 3WM operation with harmonics
        """
        interpolator = ModeArrayFactory._interpolator_from_data(base_data, interpolator)

        modes = [Mode(label="p"), Mode(label="s"), Mode(label="i")]

        relations = [["i", "p-s"]]

        for n in range(1, n_frequency_conversion + 1):
            if n == 1:
                modes.append(Mode(label="ps"))
                modes.append(Mode(label="pi"))
                relations.append(["ps", "p+s"])
                relations.append(["pi", "p+i"])
            else:
                modes.append(Mode(label=f"p{n}s"))
                modes.append(Mode(label=f"p{n}i"))
                relations.append([f"p{n}s", "p+" * (n - 1) + "p+s"])
                relations.append([f"p{n}i", "p+" * (n - 1) + "p+i"])

        for n in range(2, n_pump_harmonics + 2):
            modes.append(Mode(label=f"p{n}"))
            relations.append([f"p{n}", "p+" * (n - 1) + "p"])

        for n in range(2, n_signal_harmonics + 2):
            modes.append(Mode(label=f"s{n}"))
            relations.append([f"s{n}", "s+" * (n - 1) + "s"])
            modes.append(Mode(label=f"i{n}"))
            relations.append([f"i{n}", "i+" * (n - 1) + "i"])

        return ModeArray(modes, relations, interpolator)

    @staticmethod
    def create_custom(
        base_data: Dict[str, Any],
        mode_labels: List[str],
        relations: List[List[str]],
        interpolator: Optional[ParameterInterpolator] = None,
    ) -> ModeArray:
        """
        Create a custom ModeArray with user-defined modes and relations.

        Args:
            base_data: Dictionary containing 'freqs', 'k', 'gammas', and 'alpha' arrays
            mode_labels: List of mode labels
            relations: List of relations between modes
            interpolator: Optional shared ParameterInterpolator (built from
                ``base_data`` when omitted)

        Returns:
            ModeArray: Custom configured ModeArray
        """
        interpolator = ModeArrayFactory._interpolator_from_data(base_data, interpolator)

        modes = [Mode(label=label) for label in mode_labels]

        return ModeArray(modes, relations, interpolator)
