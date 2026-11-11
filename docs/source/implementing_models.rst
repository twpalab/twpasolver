Implementing New Models
=======================

This guide explains how users can define their own network models by extending the provided ``TwoPortModel`` and ``OnePortModel`` base classes. For detailed information about the functionalities of these classes, refer to the API documentation and the examples.

Adding a New TwoPortModel
-------------------------

To define a new two-port model by subclassing ``TwoPortModel``, implement the **single_abcd(freqs: numpy.ndarray) -> ABCDArray** method. It takes a 1-D array of frequencies in Hz and returns the ABCD matrix of a single iteration of the model (``N`` repetitions are applied by the base class).

Give the model a **name: Literal["ClassName"] = "ClassName"** field. Every subclass that declares its own ``name`` is registered when it is defined, so arrays (``TwoPortArray``, ``TWPA``) accept it as an instance or as a dict with that ``name``, for example when a model is loaded from JSON. This works for models defined after importing ``twpasolver`` too. Names must be unique; a later class with the same name replaces the earlier one. A subclass that does not declare its own ``name`` can still be used as an instance, but is saved and loaded as its parent, so it may not add fields (defining one that does raises ``TypeError``).

Example:

.. code-block:: python

  class MyTwoPortModel(TwoPortModel):
      """Custom two-port model."""

      name: Literal["MyTwoPortModel"] = "MyTwoPortModel"

      def single_abcd(self, freqs: np.ndarray) -> ABCDArray:
          """ABCD matrix of the custom model; freqs in Hz."""
          # Implement the ABCD calculation logic here
          ...

Adding a New OnePortModel
-------------------------

All network models in ``twpasolver`` are derived from the ``TwoPortModel`` class, with ``OnePortModel`` being a specialization of it. In this case, the ``single_abcd()`` method is already implemented in the class, which constructs the ABCD matrix of the circuit element by inserting it in series or in parallel into a two-port network according to the ``twoport_parallel`` attribute.

To add a new one-port model, implement the **Z(freqs: numpy.ndarray) -> numpy.ndarray** method. It takes a 1-D array of frequencies in Hz and returns the impedance of the component. ``OnePortArray`` accepts only one-port models.

Example:

.. code-block:: python

  class Resistance(OnePortModel):
      """Model of a resistor."""

      name: Literal["Resistance"] = "Resistance"
      R: NonNegativeFloat

      def Z(self, freqs: np.ndarray) -> np.ndarray:
          """Impedance of the resistor."""
          return np.full_like(freqs, self.R, dtype=float)

Nonlinear Cells for the Bloch Model
-----------------------------------

The ``bloch`` gain model needs to know which series inductors of a unit cell are nonlinear. ``LCLfBaseCell`` and ``StubBaseCell`` describe theirs; their ``nl_scale`` field (default 1) sets the nonlinear fraction of the line inductance ``L``, for example to model nonlinearity-only gratings. It affects only the nonlinear couplings of the ``bloch`` model, not the linear response or the ``simple`` model.

A custom cell supports the ``bloch`` model by defining a **bloch_nl_elements()** method that returns a list of ``(port, L)`` pairs: each nonlinear inductance ``L`` carries the current of the cell's input port (``PORT_IN``) or output port (``PORT_OUT``), both importable from ``twpasolver.models``.

.. code-block:: python

  from twpasolver.models import PORT_OUT

  class HalfNonlinearCell(LCLfBaseCell):
      """Only half of the line inductance is nonlinear."""

      def bloch_nl_elements(self):
          return [(PORT_OUT, self.L / 2)]
