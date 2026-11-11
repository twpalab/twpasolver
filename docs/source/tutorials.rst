.. _tutorials:


Tutorials
=========

The tutorials build up from circuit models to the gain of complete devices:

1. **Circuit models**: components, their ABCD and S-parameters, and how they combine.
2. **Building a TWPA**: unit cells, the nonlinearity and the linear response of a kinetic-inductance TWPA.
3. **Phase matching and gain**: ``TWPAnalysis``, the ``simple`` model, bandwidth, sweeps and compression.
4. **Generalized coupled mode equations**: mode sets, the ``bloch`` model, reflections and connectors.
5. **Other TWPAs and segmented devices**: a resonantly phase-matched 4WM line, fabrication variations and a
   Floquet-mode TWPA.

The notebooks are in the `docs/source/notebooks <https://github.com/twpalab/twpasolver/tree/main/docs/source/notebooks>`_
directory of the repository.

.. toctree::
   :maxdepth: 4

   notebooks/01_models
   notebooks/02_twpas
   notebooks/03_analysis
   notebooks/04_generalized_cmes
   notebooks/05_other_twpas
