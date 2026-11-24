Welcome to twpasolver's documentation!
======================================

``twpasolver`` simulates traveling-wave parametric amplifiers (TWPAs): circuit models of the line, linear response, phase matching, and gain from coupled mode equations (CMEs).

.. note:: This release simulates **kinetic-inductance TWPAs (KI-TWPAs)**. The cells of the line can be any linear
   circuit, but the only nonlinearity is that of a kinetic inductance. Josephson-junction and SNAIL TWPAs are not
   supported yet.

.. caution:: ``twpasolver`` is in beta: the API may still change between minor versions (see the changelog in the repository).

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   tutorials
   physics
   implementing_models

.. toctree::
    :maxdepth: 2
    :caption: Api reference

    api-reference/modules


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
