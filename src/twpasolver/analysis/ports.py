"""Connectors between the TWPA and the external lines."""

from typing import Any

import numpy as np
from pydantic import ConfigDict, Field, field_validator

from twpasolver.basemodel import BaseModel
from twpasolver.twoport import TwoPortModel


class Ports(BaseModel):
    """
    Connectors at the device ends and the impedance of the external lines.

    Port 1 of a connector faces the external line, port 2 the TWPA. Without
    connectors the device is joined directly to lines of impedance ``Z_ext``.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)
    input: Any = Field(
        default=None,
        description="Connector (``TwoPortModel``) at x=0; also used at x=L when "
        "``output`` is None.",
    )
    output: Any = Field(default=None, description="Connector at x=L.")
    Z_ext: float = Field(
        default=50.0,
        description="Impedance of the external lines (e.g. the fridge coaxial lines).",
    )

    @field_validator("input", "output", mode="before")
    @classmethod
    def _accept_model_as_is(cls, v):
        """Pass TwoPortModel instances through without Pydantic coercion."""
        if v is None or isinstance(v, TwoPortModel):
            return v
        raise ValueError(f"A port must be a TwoPortModel instance, got {type(v)}.")

    def tables(self, freqs_hz: np.ndarray) -> tuple[np.ndarray, ...]:
        """
        Boundary tables of the forward-backward solver at the mode frequencies.

        Args:
            freqs_hz: Mode frequencies in Hz, shape (n_freq, n_modes).

        Returns:
            ``Zout_in``, ``Zout_out``: impedance ``(D Ze + B) / (C Ze + A)`` seen by
            the device at x=0 and x=L, shape (n_freq, n_modes); ``abcd_in``,
            ``abcd_out``: ``[A, B, C, D]`` of the connectors, shape
            (n_freq, n_modes, 4). Without a connector the tables are the identity
            and ``Zout = Ze``.
        """
        Ze = self.Z_ext
        out = []
        for model in (
            self.input,
            self.output if self.output is not None else self.input,
        ):
            if model is None:
                Zout = np.full(freqs_hz.shape, complex(Ze), dtype=np.complex128)
                abcd = np.zeros((*freqs_hz.shape, 4), dtype=np.complex128)
                abcd[..., 0] = abcd[..., 3] = 1.0
            else:
                m = np.asarray(
                    model.get_abcd(freqs_hz.ravel())
                )  # pylint: disable=no-member
                abcd = m.reshape(*freqs_hz.shape, 4).astype(np.complex128)
                A, B, C, D = (abcd[..., k] for k in range(4))
                Zout = (D * Ze + B) / (C * Ze + A)
            out.append((Zout, abcd))
        return out[0][0], out[1][0], out[0][1], out[1][1]
