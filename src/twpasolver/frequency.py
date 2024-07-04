"""Definition of class for frequency span."""

from typing import Literal, Optional

import numpy as np
from pydantic import Field, PrivateAttr

from twpasolver.basemodel import BaseModel
from twpasolver.typing import FrequencyArange, FrequencyList


class Frequencies(BaseModel):
    """
    Frequency span with variable unit.

    The span can be provided either as a list or as a tuple that will be passed to numpy.arange.
    The array is always generated using the arange tuple if provided.
    """

    f_list: Optional[FrequencyList] = Field(
        default=None, description="List of frequencies"
    )
    f_arange: Optional[FrequencyArange] = Field(
        default=None,
        description="Tuple passed to numpy.arange to construct frequency span.",
    )
    unit: Literal["Hz", "kHz", "MHz", "GHz", "THz"] = Field(default="GHz", repr=False)
    _unit_multipliers = PrivateAttr(
        {"Hz": 1, "kHz": 1e3, "MHz": 1e6, "GHz": 1e9, "THz": 1e12}
    )

    @property
    def f(self) -> np.ndarray:
        """Computed frequencies array."""
        if self.f_arange:
            freqs = np.arange(*self.f_arange)
        elif self.f_list:
            freqs = np.array(self.f_list)
        else:
            return np.array([])
        return self._unit_multipliers[self.unit] * freqs

    @property
    def omega(self) -> np.ndarray:
        """Computed angular frequencies array."""
        return 2 * np.pi * self.f

    @property
    def unit_multiplier(self):
        """Get multiplier of chosen unit of measure."""
        return self._unit_multipliers[self.unit]
