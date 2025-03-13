Installation Guide
=================

Requirements
-----------

Before installing ``langesim-optim``, ensure you have the following prerequisites:

* Python 3.9 or higher
* Poetry package manager

Dependencies
-----------

The package requires the following main dependencies:

* PyTorch 2.0.0
* NumPy
* Matplotlib
* SciPy

All dependencies will be automatically installed by Poetry.

Installation Steps
----------------

1. Clone the repository:

   .. code-block:: bash

      git clone <repository-url>
      cd langesim-optim

2. Install using Poetry:

   .. code-block:: bash

      poetry install

   This will create a virtual environment and install all required dependencies.

Development Installation
----------------------

For development, you can install additional dependencies:

.. code-block:: bash

   poetry install --with dev,test

This will install additional packages like:

* pytest (for testing)
* black (for code formatting)
* Sphinx (for documentation) 