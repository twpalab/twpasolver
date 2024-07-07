"""
Models module.

This module provides a collection of models for simulating and analyzing various RF (radio frequency)
components and networks. It includes models for one-port components such as resistors, capacitors,
inductors, and stubs, as well as more complex two-port models and TWPAs (Traveling Wave Parametric Amplifiers).

Imports
-------

- One-port models:
  - Resistance
  - Capacitance
  - Inductance
  - Stub
  - OnePortArray

- TWPA cells:
  - LCLfBaseCell
  - StubBaseCell

- Two-port arrays and models:
  - TWPA
  - AnyModel
  - TwoPortArray

- Utility functions:
  - compose

Usage
-----
These models can be used to construct and analyze RF networks, allowing for the study of their behavior
under different conditions. The models support various operations such as impedance calculations,
frequency response analysis, and composition of complex networks from simpler components.

Example
-------
.. code-block:: python

    from twpasolver.models.oneport import Resistance, Capacitance
    from twpasolver.models.twoportarrays import TwoPortArray

    # Define a simple network with a resistor and capacitor in series
    resistor = Resistance(R=50)
    capacitor = Capacitance(C=1e-12)
    network = TwoPortArray(cells=[resistor, capacitor])

    # Compute the network parameters at a given frequency
    freqs = np.linspace(1e6, 1e9, 1000)
    abcd_params = network.single_abcd(freqs)
"""

# !/bin/python3
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
