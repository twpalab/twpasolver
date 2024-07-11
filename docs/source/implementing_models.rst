Implementing New Models
=======================

This guide explains how users can define their own network models by extending the provided ``TwoPortModel`` and ``OnePortModel`` base classes. For detailed information about the functionalities of these classes, refer to the API documentation and the examples.

Adding a New TwoPortModel
-------------------------

To define a new two-port model by subclassing ``TwoPortModel``, you must implement the **get_single_abcd(freqs: numpy.ndarray) -> ABCDArray** method. This method requires a 1-D array of frequencies as input and returns the ABCD matrix of a single iteration of the model as a function of frequency using an ``ABCDArray`` object.

Additionally, to make the model available for integration with ``TwoPortArray``, it is necessary to add a **name: Literal[new_model_class_name]** attribute. This attribute is used by Pydantic to discriminate between different models when constructing the array. Since all models in the library must have a unique name, it is recommended to set this attribute to the name of the implemented class.

Example:

.. code-block:: python

  class MyTwoPortModel(TwoPortModel):
      """Custom two-port model."""

      name: Literal["MyTwoPortModel"] = "MyTwoPortModel"

      def get_single_abcd(self, freqs: np.ndarray) -> ABCDArray:
          """
          Get ABCD matrix for the custom model.

          Args:
              freqs (np.ndarray): Array of frequencies.

          Returns:
              ABCDArray: ABCD matrix of the custom model.
          """
          # Implement the ABCD calculation logic here
          pass

Adding a New OnePortModel
-------------------------

All network models in ``twpasolver`` are derived from the ``TwoPortModel`` class, with ``OnePortModel`` being a specialization of it. In this case, the ``get_single_abcd()`` method is already implemented in the class, which constructs the ABCD matrix of the circuit element by inserting it in series or in parallel into a two-port network according to the ``twoport_parallel`` attribute.

To add a new one-port model, you need to provide a concrete implementation of the **Z(freqs: numpy.ndarray) -> numpy.ndarray** property. This property takes a 1-D array of frequencies as input and returns another 1-D array containing the impedance of the component.

Example:

.. code-block:: python

  class Resistance(OnePortModel):
      """Model of a resistor."""

      name: Literal["Resistance"] = "Resistance"
      R: NonNegativeFloat

      def Z(self, freqs: np.ndarray) -> np.ndarray:
          """
          Get impedance of resistor.

          Args:
              freqs (np.ndarray): Array of frequencies.

          Returns:
              np.ndarray: Impedance of the resistor.
          """
          return np.full_like(freqs, self.R)
