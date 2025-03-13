Welcome to langesim_optim's documentation!
=====================================

``langesim_optim`` is a Python library for running differentiable Langevin simulations of a Brownian particle subjected to time-dependent potentials with trainable parameters. This library enables optimization of equilibration time and work through gradient-based methods.

Features
--------

- **Differentiable Simulations**: Built on PyTorch for end-to-end differentiable Langevin dynamics
- **Flexible Force Models**:
  - Harmonic forces with variable stiffness
  - Harmonic forces with variable center
  - Combined variable stiffness and center forces
  - Harmonic forces with stiffness and center given by polynomial functions of time
- **Optimization Objectives**:
  - Equilibration time optimization
  - Work optimization
  - Custom loss functions
- **Training Utilities**:
  - Built-in training loops
  - Progress tracking
  - Visualization tools

Installation
-----------

To install the package, make sure you have Poetry installed and run:

.. code-block:: bash

   git clone https://github.com/GabrielTellez/langesim_optim.git
   cd langesim_optim
   poetry install

Quick Start
----------

Here's a simple example of running a simulation with a variable stiffness harmonic force:

.. code-block:: python

   import torch
   from langesim_optim.simulator_forces import (
       Simulator,
       VariableStiffnessHarmonicForce
   )
   from langesim_optim.utilities import train_loop
   from langesim_optim.loss_functions import loss_fn_k

   # Create a force with variable stiffness
   force = VariableStiffnessHarmonicForce(
       kappai=1.0,  # Initial stiffness
       kappaf=2.0,  # Final stiffness
       tf=1.0,      # Final time
       steps=100    # Number of interpolation points
   )

   # Create a simulator
   sim = Simulator(
       dt=0.001,           # Time step
       tot_steps=1000,     # Total simulation steps
       noise_scaler=1.0,   # Noise strength
       force=force         # Force object
   )

   # Set up optimizer
   optimizer = torch.optim.SGD(sim.parameters(), lr=0.001)

   # Run training loop
   losses = train_loop(
       epochs=100,
       sim=sim,
       tot_sims=1000,
       optimizer=optimizer,
       loss_fn=loss_fn_k,
       kf=2.0,
       ki=1.0
   )

Loss Functions
-------------

The library provides loss functions for two main optimization objectives:

1. **Optimize Equilibration Time**
   These loss functions help find protocols that minimize the time needed for the system to reach equilibrium:

   - ``loss_fn_k``: Compares characteristic functions of final distributions to optimize equilibration time
   - ``loss_fn_variance``: Optimizes by comparing the final variance of particle positions
   - ``loss_fn_mean``: Optimizes by comparing the final average position of particles

2. **Optimize Work**
   These loss functions focus on minimizing the work done on the system:

   - ``loss_fn_work``: Optimizes the total work done on the system during the protocol
   - ``loss_fn_eq_work``: Optimizes both equilibration time and work simultaneously

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   modules/utilities
   modules/force_harmonic_polynomial
   modules/interpolator
   modules/loss_functions
   modules/simulator_forces

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search` 