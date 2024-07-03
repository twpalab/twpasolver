"""Definition of class for frequency span."""

from typing import Literal, Optional

import numpy as np
from pydantic import BaseModel, Field, PrivateAttr

from twpasolver.typing import FrequencyArange, FrequencyList


class Frequencies(BaseModel):
    """
    Frequency span with variable unit.

    The span can be provided either as a list or as a tuple that will be passed to numpy.arange.
    The array is always generated using the arange tuple if provided.
    """

    freq_list: FrequencyList = Field(
        default_factory=list, description="List of frequencies"
    )
    freq_arange: Optional[FrequencyArange] = Field(
        default=None,
        description="Tuple passed to numpy.arange to construct frequency span.",
    )
    unit: Literal["Hz", "kHz", "MHz", "GHz", "THz"] = "GHz"
    _unit_multipliers = PrivateAttr(
        {"Hz": 1, "kHz": 1e3, "MHz": 1e6, "GHz": 1e9, "THz": 1e12}
    )

    @property
    def f(self) -> np.ndarray:
        """Computed frequencies array."""
        if self.freq_arange:
            freqs = np.arange(*self.freq_list)
        else:
            freqs = np.array(self.freq_list)
        return self._unit_multipliers[self.unit] * freqs
