"""
Simulation of TWPAs: linear response, phase matching and gain from coupled mode equations.

:class:`TWPAnalysis` is the interface; :class:`LinearData`, :class:`GainResult` and
:class:`PhaseMatchingResult` are what it returns, :class:`Ports` describes the
connectors and :class:`GainModel` lists the gain models.
"""

from twpasolver.analysis.core import TWPAnalysis
from twpasolver.analysis.gain import GainModel
from twpasolver.analysis.linear import LinearData, linear_data
from twpasolver.analysis.ports import Ports
from twpasolver.analysis.results import GainResult, PhaseMatchingResult

__all__ = [
    "GainModel",
    "GainResult",
    "LinearData",
    "PhaseMatchingResult",
    "Ports",
    "TWPAnalysis",
    "linear_data",
]
