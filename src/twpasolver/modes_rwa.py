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

    def __post_init__(self) -> None:
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
        n_signal_harmonics: int = 0,
        n_sideband_conversion: int = 0,
    ) -> "ModeSet":
        """
        3WM pump, signal and idler with pump harmonics, conversion terms and sidebands.

        Every signal order ``m`` gives the harmonic ``m s`` and its 3WM idler
        ``p - m s``, labelled ``s{m}`` and ``i{m}``, and their conversion chain
        ``n p + m s`` and ``(n + 1) p - m s``, labelled ``p{n}s{m}`` and
        ``p{n}i{m}`` (``m`` and ``n`` are left out when 1). Order ``m = 1`` is the
        signal and idler themselves: ``s``, ``i``, ``ps``, ``pi``, ``p2s``, ...; order
        2 gives ``s2`` (2s), ``i2`` (p - 2s), ``ps2`` (p + 2s), ``pi2`` (2p - 2s), ...

        The sidebands (``m >= 2``) are second or higher order in the signal: they
        leave the small-signal gain unchanged and matter in compression. The idler
        ``p - m s`` crosses 0 Hz at ``s = p / m``, so keep the signals on one side of
        it (a mode below the frequency grid raises).

        Args:
            n_pump_harmonics: Number of pump harmonics (p2, p3, ...).
            n_frequency_conversion: Largest pump order ``n`` of the conversion chain
                of the signal and idler (ps/pi, p2s/p2i, ...).
            n_signal_harmonics: Number of sideband orders ``m = 2, 3, ...`` (s2/i2,
                s3/i3, ...).
            n_sideband_conversion: Largest pump order ``n`` of the conversion chain
                of every sideband order (ps2/pi2, p2s2/p2i2, ...).
        """

        def chain(m: int, n_max: int) -> list[tuple[str, str]]:
            """Modes of signal order ``m``: harmonic, idler, then their conversions."""
            sm, im = ("s", "i") if m == 1 else (f"s{m}", f"i{m}")
            modes = [] if m == 1 else [(sm, "+".join("s" * m))]
            modes.append((im, "p-" + "-".join("s" * m)))
            for n in range(1, n_max + 1):
                pn = "p" if n == 1 else f"p{n}"
                modes += [(pn + sm, "p+" * n + sm), (pn + im, "p+" * n + im)]
            return modes

        relations = chain(1, n_frequency_conversion)
        relations += [
            (f"p{n}", "+".join("p" * n)) for n in range(2, n_pump_harmonics + 2)
        ]
        for m in range(2, n_signal_harmonics + 2):
            relations += chain(m, n_sideband_conversion)
        labels = ("p", "s", *(label for label, _ in relations))
        return cls(labels, tuple(relations))

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
        figsize: Optional[Tuple[float, float]] = None,
        node_size: int = 3000,
        font_size: int = 12,
        layout: str = "hierarchical",
    ) -> Figure:
        """
        Plot the relations between modes as a directed graph.

        Args:
            tones: Independent tones; if given, nodes show the mode frequencies.
            figsize: Figure size (width, height); None sizes it to the graph.
            node_size: Size of mode nodes.
            font_size: Font size for labels.
            layout: Graph layout ('hierarchical', 'circular' or 'spring').
        """
        G = nx.DiGraph()
        G.add_nodes_from(self.labels)
        for result, expr in self.relations:
            coeffs: Counter = Counter()
            for dep, c in _parse(expr):
                coeffs[dep] += c
            for dep, c in coeffs.items():
                if c:
                    G.add_edge(dep, result, coeff=c)

        H: nx.DiGraph = G
        if layout == "hierarchical":
            pos, H = self._hierarchical_layout(G)
        elif layout == "circular":
            pos = nx.circular_layout(G)
        else:
            pos = nx.spring_layout(G, k=2, iterations=50, seed=0)
        if figsize is None:
            # About 1.7 in per unit of the layout keeps the nodes apart.
            xs, ys = np.array(list(pos.values())).T
            figsize = (
                max(10.0, 1.7 * (np.ptp(xs) + 1) / 0.75),
                max(5.0, 2.2 * (np.ptp(ys) + 1)),
            )
        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=figsize, gridspec_kw={"width_ratios": [3, 1]}
        )

        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=list(self.labels),
            node_color=[
                "lightgreen" if m in self.independent else "lightblue"
                for m in self.labels
            ],
            edgecolors="0.3",
            node_size=node_size,
            ax=ax1,
        )
        # Waypoints (tuples) have no size, so segments join there without a gap;
        # only the segment that reaches a mode gets the arrowhead.
        nodes = list(H.nodes())
        sizes = [0 if isinstance(n, tuple) else node_size for n in nodes]
        for positive, color, style in (
            (True, "tab:blue", "-"),
            (False, "tab:red", "--"),
        ):
            for head in (True, False):
                edges = [
                    (u, v)
                    for u, v, c in H.edges(data="coeff")
                    if (c > 0) == positive and isinstance(v, tuple) != head
                ]
                if edges:
                    nx.draw_networkx_edges(
                        H,
                        pos,
                        edgelist=edges,
                        nodelist=nodes,
                        node_size=sizes,
                        edge_color=color,
                        style=style,
                        width=1.5,
                        arrows=True,
                        arrowstyle="-|>" if head else "-",
                        arrowsize=18,
                        ax=ax1,
                    )
        labels = {m: m for m in self.labels}
        if tones is not None:
            freqs = self.frequencies(tones)
            labels = {m: f"{m}\n{freqs[self.index(m)]:.3g}" for m in self.labels}
        nx.draw_networkx_labels(G, pos, labels, font_size=font_size, ax=ax1)
        # The sign is in the line style; only multiplicities need a label (on the
        # first segment of an edge).
        multiples = {
            (u, v): f"\u00d7{abs(c)}"
            for u, v, c in H.edges(data="coeff")
            if abs(c) > 1 and not isinstance(u, tuple)
        }
        if multiples:
            nx.draw_networkx_edge_labels(
                H,
                pos,
                multiples,
                font_size=font_size - 1,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"),
                ax=ax1,
            )
        ax1.set_title("Mode relations", fontsize=font_size + 2, fontweight="bold")
        ax1.axis("off")
        ax1.margins(0.12)

        ax2.axis("off")
        ax2.text(
            0.0,
            1.0,
            str(self),
            transform=ax2.transAxes,
            fontsize=font_size - 1,
            family="monospace",
            verticalalignment="top",
        )
        ax2.legend(
            handles=[
                mpatches.Patch(
                    facecolor="lightgreen", edgecolor="0.3", label="Independent"
                ),
                mpatches.Patch(
                    facecolor="lightblue", edgecolor="0.3", label="Dependent"
                ),
                plt.Line2D([0], [0], color="tab:blue", lw=2, label="Added"),
                plt.Line2D(
                    [0], [0], color="tab:red", lw=2, ls="--", label="Subtracted"
                ),
            ],
            loc="lower left",
            frameon=False,
            fontsize=font_size - 1,
        )
        fig.tight_layout()
        return fig

    def _hierarchical_layout(
        self, G: nx.DiGraph
    ) -> Tuple[Dict[Any, Tuple[float, float]], nx.DiGraph]:
        """
        Layered layout of the mode graph ``G``.

        Independent modes on top, every mode one level below its lowest input. An
        edge that skips levels is split at a waypoint ``(u, v, level)`` in every
        level in between, so that it is drawn around the modes there. Within a
        level, modes and waypoints are ordered by the mean position of their inputs.

        Returns the positions of modes and waypoints, and ``G`` with the split edges.
        """
        levels: dict[Any, int] = {m: 0 for m in self.independent}
        for m in nx.topological_sort(G):
            if m not in levels:
                levels[m] = (
                    max((levels.get(p, 0) for p in G.predecessors(m)), default=0) + 1
                )
        H = nx.DiGraph()
        H.add_nodes_from(self.labels)
        for u, v, c in G.edges(data="coeff"):
            chain = [u, *((u, v, k) for k in range(levels[u] + 1, levels[v])), v]
            for w in chain[1:-1]:
                levels[w] = w[2]
            for x, y in zip(chain, chain[1:]):
                H.add_edge(x, y, coeff=c)
        max_level = max(levels.values(), default=0)
        rows: dict[int, list[Any]] = defaultdict(list)
        for n in H.nodes():
            rows[levels[n]].append(n)
        pos: dict[Any, tuple[float, float]] = {}
        for level in sorted(rows):
            row = rows[level]
            if level:
                row.sort(key=lambda n: np.mean([pos[q][0] for q in H.predecessors(n)]))
            # A waypoint takes half the room of a mode.
            widths = np.array([0.5 if isinstance(n, tuple) else 1.0 for n in row])
            x = np.cumsum(widths) - widths / 2
            for n, xn in zip(row, x - (x[0] + x[-1]) / 2):
                pos[n] = (float(xn), max_level - level)
        return pos, H


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
