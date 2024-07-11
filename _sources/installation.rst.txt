.. _installation:

Installation
============

Clone the repository with:

.. code-block:: bash

  git clone https://github.com/twpalab/twpasolver

then to install the package in normal mode:

.. code-block:: bash

  pip install .

Use poetry to install the latest version in developer mode, remember to also
install the pre-commits!

.. code-block:: bash

  poetry install --with docs,analysis
  pip install pre-commit
  pre-commit install
