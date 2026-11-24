.. _installation:

Installation
============

Install the latest release from PyPI (Python 3.10 or newer):

.. code-block:: bash

  pip install twpasolver

The CME solvers are compiled with Numba and cached to disk the first time they run. Compile them once after
installing (a few minutes), instead of during your first analysis:

.. code-block:: bash

  twpasolver-warmup

Development install
-------------------

Clone the repository and install it with Poetry, together with the documentation and analysis tools and the
pre-commit hooks:

.. code-block:: bash

  git clone https://github.com/twpalab/twpasolver
  cd twpasolver
  poetry install --with docs,analysis
  pre-commit install

Building the documentation executes the tutorial notebooks and needs `pandoc <https://pandoc.org>`_:

.. code-block:: bash

  make -C docs html
