"""
Models module.

One-port components (resistors, capacitors, inductors, stubs), transmission lines,
TWPA cells and two-port arrays, including the TWPA model.

Imports
-------

One-port models:
  * Resistance
  * Capacitance
  * Inductance
  * Stub
  * OnePortArray

TWPA cells:
  * LCLfBaseCell
  * StubBaseCell

Two-port arrays and models:
  * TWPA
  * AnyModel
  * TwoPortArray

Utility functions:
  * compose

Usage
-----
Networks are built by composing simpler components and expose impedance and
frequency-response calculations. See the :ref:`tutorials` page for more examples.

Example
-------
.. code-block:: python

    import numpy as np
    from twpasolver.models.oneport import Resistance, Capacitance
    from twpasolver.models.twoportarrays import TwoPortArray

    # Resistor and capacitor in series
    resistor = Resistance(R=50)
    capacitor = Capacitance(C=1e-12)
    network = TwoPortArray(cells=[resistor, capacitor])

    freqs = np.linspace(1e6, 1e9, 1000)
    abcd_params = network.single_abcd(freqs)
"""

# isort: skip_file
from twpasolver.models.oneport import (
    Resistance,
    Capacitance,
    Inductance,
    Stub,
    OnePortArray,
)
from twpasolver.models.transmission_lines import LosslessTL
from twpasolver.models.twpa_cells import LCLfBaseCell, StubBaseCell
from twpasolver.models.twoportarrays import TWPA, AnyModel, TwoPortArray
from twpasolver.models.compose import compose
