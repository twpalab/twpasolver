.. _installation:

Installation
============

Clone the repository with:

.. code-block:: bash

  git clone project_url 

then to install it in normal mode:

.. code-block:: bash

  pip install .

Use poetry to install the latest version in developer mode, remember to also
install the pre-commits!

.. code-block:: bash

  poetry install --with docs,analysis
  pre-commit install
