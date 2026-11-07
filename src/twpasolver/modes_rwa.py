"""
Modes, frequency relations and rotating-wave-approximation (RWA) terms.

A :class:`ModeSet` lists the modes of a coupled-mode-equation (CME) system,
beyond pump, signal and idler (pump harmonics ``p2``, conversion terms ``ps``,
signal/idler harmonics ``s2``, ...), and the relations fixing their frequencies.
A relation such as ``["i", "p-s"]`` sets a mode to a sum or difference of other
modes, so every mode frequency is an integer combination of the independent
tones (the modes no relation defines, any number of them). The RWA terms of a
given order are the products of modes and conjugates whose combinations add up
to a mode, found exactly on these integer vectors.

Examples
--------
See Tutorial 4 (:ref:`tutorials`) for standard and custom mode sets and RWA
term analysis.
"""

import itertools as it
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from math import factorial
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.figure import Figure


def _parse(expr: str) -> list[tuple[str, int]]:
    """``"p+p-s"`` -> ``[("p", 1), ("p", 1), ("s", -1)]``."""
    expr = expr.replace(" ", "")
    terms, sign, name = [], 1, ""
    for ch in ("" if expr.startswith(("+", "-")) else "+") + expr + "+":
        if ch in "+-":
            if name:
                terms.append((name, sign))
            sign, name = (1 if ch == "+" else -1), ""
        else:
            name += ch
    return terms


@dataclass(frozen=True)
class ModeSet:
    """
    Modes of a CME system and the frequency relations between them.

    Args:
        labels: Mode names; their order is the order of the CME amplitudes.
        relations: ``[result, expression]`` pairs, e.g. ``["i", "p-s"]``; an
            expression is a sum or difference of mode names (repetitions allowed,
            ``"p+p-s"``). A mode is defined by at most one relation; relations must
            not be cyclic.

    Attributes:
        independent: Modes not defined by any relation (the independent tones).
        matrix: Integer coefficients of every mode over ``independent``, shape
            (n_modes, n_independent).
    """

    labels: Tuple[str, ...]
    relations: Tuple[Tuple[str, str], ...] = ()
    independent: Tuple[str, ...] = field(init=False)
    matrix: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        """Normalise the inputs, validate the relations and build ``matrix``."""
        labels = tuple(self.labels)
        relations = tuple((str(r), str(e)) for r, e in self.relations)
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "relations", relations)
        if len(set(labels)) != len(labels):
            raise ValueError(f"Duplicate mode labels in {labels}.")
        defined: dict[str, list[tuple[str, int]]] = {}
        for result, expr in relations:
            if result in defined:
                raise ValueError(
                    f"Mode '{result}' is defined by more than one relation."
                )
            terms = _parse(expr)
            for name in [result] + [n for n, _ in terms]:
                if name not in labels:
                    raise KeyError(
                        f"Unknown mode '{name}' in relation {result} = {expr}."
                    )
            defined[result] = terms
        independent = tuple(m for m in labels if m not in defined)
        unit = np.eye(len(independent), dtype=np.int64)
        vectors: dict[str, np.ndarray] = {}

        def vector(m: str, path: tuple[str, ...]) -> np.ndarray:
            if m in vectors:
                return vectors[m]
            if m in path:
                raise ValueError(
                    "Mode relations are cyclic: "
                    + " -> ".join(path[path.index(m) :] + (m,))
                )
            if m in defined:
                vec = sum(
                    (s * vector(n, path + (m,)) for n, s in defined[m]),
                    np.zeros(len(independent), dtype=np.int64),
                )
            else:
                vec = unit[independent.index(m)]
            vectors[m] = vec
            return vec

        object.__setattr__(self, "independent", independent)
        object.__setattr__(
            self, "matrix", np.array([vector(m, ()) for m in labels], dtype=np.int64)
        )

    @classmethod
    def basic(cls, three_wave: bool = True) -> "ModeSet":
        """Pump, signal and idler, with ``i = p - s`` (3WM) or ``i = 2p - s`` (4WM)."""
        return cls(("p", "s", "i"), (("i", "p-s" if three_wave else "p+p-s"),))

    @classmethod
    def extended_3wm(
        cls,
        n_pump_harmonics: int = 1,
        n_frequency_conversion: int = 1,
        n_signal_harmonics: int = 1,
    ) -> "ModeSet":
        """
        3WM pump, signal and idler with pump harmonics and conversion terms.

        Args:
            n_pump_harmonics: Number of pump harmonics (p2, p3, ...).
            n_frequency_conversion: Number of up-conversion pairs (ps/pi, p2s/p2i, ...).
            n_signal_harmonics: Number of signal and idler harmonics (s2/i2, ...).
        """
        labels = ["p", "s", "i"]
        relations = [("i", "p-s")]
        for n in range(1, n_frequency_conversion + 1):
            s, i = ("ps", "pi") if n == 1 else (f"p{n}s", f"p{n}i")
            labels += [s, i]
            relations += [(s, "p+" * (n - 1) + "p+s"), (i, "p+" * (n - 1) + "p+i")]
        for n in range(2, n_pump_harmonics + 2):
            labels.append(f"p{n}")
            relations.append((f"p{n}", "p+" * (n - 1) + "p"))
        for n in range(2, n_signal_harmonics + 2):
            labels += [f"s{n}", f"i{n}"]
            relations += [
                (f"s{n}", "s+" * (n - 1) + "s"),
                (f"i{n}", "i+" * (n - 1) + "i"),
            ]
        return cls(tuple(labels), tuple(relations))

    def index(self, label: str) -> int:
        """Position of ``label`` among the modes."""
        return self.labels.index(label)

    def frequencies(self, tones: Mapping[str, Union[float, np.ndarray]]) -> np.ndarray:
        """
        Frequencies of every mode, shape (n_modes, ...), from the independent tones.

        Args:
            tones: Frequency of every independent mode, scalars or arrays that
                broadcast together; extra keys raise.
        """
        missing = set(self.independent) - set(tones)
        extra = set(tones) - set(self.independent)
        if missing or extra:
            raise ValueError(
                f"Give exactly the independent tones {self.independent}; "
                f"missing {sorted(missing)}, not independent {sorted(extra)}."
            )
        f = np.broadcast_arrays(
            *(np.asarray(tones[m], float) for m in self.independent)
        )
        return np.tensordot(self.matrix, np.stack(f), axes=1)

    def rwa_terms(self, power: int) -> list[tuple[Any, ...]]:
        """
        RWA terms with ``power`` mode factors (2 for 3WM, 3 for 4WM).

        Returns:
            Tuples ``(mode_idx, combination, mode_name, rhs_terms, coefficient)``
            sorted by ``mode_idx``: the term ``prod(combination)`` drives mode
            ``mode_idx``; indices in ``combination`` at or above the number of
            modes are conjugated modes; ``coefficient`` is ``1 / prod(n_r!)`` over
            repeated factors.
        """
        return _rwa_terms(self.labels, self.relations, power)

    def dependents(self, label: str) -> set[str]:
        """Modes whose relations depend, directly or not, on ``label``."""
        graph: dict[str, set[str]] = defaultdict(set)
        for result, expr in self.relations:
            for name, _ in _parse(expr):
                graph[name].add(result)
        out: set[str] = set()
        todo = [label]
        while todo:
            for m in graph[todo.pop()] - out:
                out.add(m)
                todo.append(m)
        return out

    def expression(self, label: str) -> str:
        """Frequency of ``label`` over the independent tones, e.g. ``"p - s"``."""
        parts = []
        for name, c in zip(self.independent, self.matrix[self.index(label)]):
            if c:
                mag = "" if abs(c) == 1 else f"{abs(c)}*"
                parts.append(("- " if c < 0 else "+ ") + mag + name)
        text = " ".join(parts) or "0"
        return text[2:] if text.startswith("+ ") else text

    def __str__(self) -> str:
        """One line per mode with its frequency over the independent tones."""
        return "\n".join(
            f"{m} = {self.expression(m)}"
            + (" (independent)" if m in self.independent else "")
            for m in self.labels
        )

    def plot_relations(
        self,
        tones: Optional[Mapping[str, float]] = None,
        figsize: Tuple[float, float] = (12, 8),
        node_size: int = 3000,
        font_size: int = 12,
        layout: str = "hierarchical",
    ) -> Figure:
        """
        Plot the relations between modes as a directed graph.

        Args:
            tones: Independent tones; if given, nodes show the mode frequencies.
            figsize: Figure size (width, height).
            node_size: Size of mode nodes.
            font_size: Font size for labels.
            layout: Graph layout ('hierarchical', 'circular' or 'spring').
        """
        G = nx.DiGraph()
        for m in self.labels:
            G.add_node(m)
        edge_labels: dict[Tuple[str, str], str] = {}
        for result, expr in self.relations:
            for dep, coeff in _parse(expr):
                G.add_edge(dep, result, color="blue" if coeff > 0 else "red")
                c = f"+{coeff}" if coeff > 0 else str(coeff)
                key = (dep, result)
                edge_labels[key] = (
                    f"{edge_labels[key]}, {c}" if key in edge_labels else c
                )

        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=figsize, gridspec_kw={"width_ratios": [3, 1]}
        )
        if layout == "hierarchical":
            pos = self._hierarchical_layout(G)
        elif layout == "circular":
            pos = nx.circular_layout(G)
        else:
            pos = nx.spring_layout(G, k=2, iterations=50)

        node_colors = [
            "lightgreen" if m in self.independent else "lightblue" for m in G.nodes()
        ]
        nx.draw_networkx_nodes(
            G, pos, node_color=node_colors, node_size=node_size, ax=ax1
        )
        for color, style in (("blue", "-"), ("red", "--")):
            edges = [(u, v) for u, v, d in G.edges(data=True) if d["color"] == color]
            if edges:
                nx.draw_networkx_edges(
                    G,
                    pos,
                    edgelist=edges,
                    edge_color=color,
                    style=style,
                    arrows=True,
                    arrowsize=20,
                    ax=ax1,
                )
        labels = {m: m for m in G.nodes()}
        if tones is not None:
            freqs = self.frequencies(tones)
            labels = {m: f"{m}\n({freqs[self.index(m)]:.1f})" for m in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, font_size=font_size, ax=ax1)
        nx.draw_networkx_edge_labels(
            G, pos, edge_labels, font_size=font_size - 2, ax=ax1
        )
        ax1.set_title("Mode Relationships", fontsize=font_size + 2, fontweight="bold")
        ax1.axis("off")
        ax1.legend(
            handles=[
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
            ],
            loc="upper right",
        )
        ax2.axis("off")
        ax2.set_title("Relations", fontsize=font_size + 1, fontweight="bold")
        ax2.text(
            0.05,
            0.95,
            f"Frequencies:\n\n{self}",
            transform=ax2.transAxes,
            fontsize=font_size - 1,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )
        fig.tight_layout()
        return fig

    def _hierarchical_layout(self, G: nx.DiGraph) -> Dict[str, Tuple[float, float]]:
        """Independent modes at the bottom, every mode one level above its inputs."""
        levels = {m: 0 for m in self.independent}
        for m in nx.topological_sort(G):
            if m not in levels:
                levels[m] = (
                    max((levels.get(p, 0) for p in G.predecessors(m)), default=0) + 1
                )
        max_level = max(levels.values(), default=0)
        counts = Counter(levels.values())
        placed: dict[int, int] = defaultdict(int)
        pos: dict[str, tuple[float, float]] = {}
        for m, level in levels.items():
            pos[m] = (placed[level] - (counts[level] - 1) / 2, max_level - level)
            placed[level] += 1
        return pos


@lru_cache(maxsize=None)
def _rwa_terms(
    labels: Tuple[str, ...], relations: Tuple[Tuple[str, str], ...], power: int
) -> list[tuple[Any, ...]]:
    """RWA terms of a mode set (see :meth:`ModeSet.rwa_terms`), cached per mode set."""
    V = ModeSet(labels, relations).matrix
    ext = np.vstack([V, -V])
    names = list(labels) + [f"-{m}" for m in labels]
    terms = []
    for comb in it.combinations_with_replacement(range(len(ext)), power):
        total = ext[list(comb)].sum(axis=0)
        for j in range(len(labels)):
            if np.array_equal(total, V[j]):
                rhs = [names[k] for k in comb]
                coeff = 1.0
                for r in Counter(rhs).values():
                    coeff /= factorial(r)
                terms.append((j, comb, labels[j], rhs, coeff))
    return sorted(terms, key=lambda t: t[0])


def as_mode_set(modes: Union[str, ModeSet, Sequence[Any]]) -> ModeSet:
    """A ``ModeSet`` from itself or the names ``"basic_3wm"`` / ``"basic_4wm"``."""
    if isinstance(modes, ModeSet):
        return modes
    if modes == "basic_3wm":
        return ModeSet.basic()
    if modes == "basic_4wm":
        return ModeSet.basic(three_wave=False)
    raise ValueError(
        f"Unknown modes {modes!r}: give a ModeSet, 'basic_3wm' or 'basic_4wm'."
    )
