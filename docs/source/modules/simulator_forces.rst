Simulator and Forces Module
=======================

The ``simulator_forces`` module provides the core simulation functionality and force implementations for Langevin dynamics.

.. currentmodule:: langesim_optim.simulator_forces

Base Classes
----------

.. autoclass:: BaseForce
   :members:
   :special-members: __init__

.. autoclass:: BaseHarmonicForce
   :members:
   :special-members: __init__
   :show-inheritance:

Simulator
--------

.. autoclass:: Simulator
   :members:
   :special-members: __init__

Force Implementations
------------------

.. autoclass:: VariableStiffnessHarmonicForce
   :members:
   :special-members: __init__
   :show-inheritance:

.. autoclass:: VariableCenterHarmonicForce
   :members:
   :special-members: __init__
   :show-inheritance:

.. autoclass:: VariableStiffnessCenterHarmonicForce
   :members:
   :special-members: __init__
   :show-inheritance:

Helper Functions
-------------

.. autofunction:: validate_init_interpolation_list

Example Usage
-----------

Here's an example of how to use the simulator with a harmonic force:

.. code-block:: python

    import torch
    from langesim_optim.simulator_forces import (
        Simulator,
        VariableStiffnessHarmonicForce
    )
    
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
    
    # Run simulation
    x0 = torch.randn(1000)  # Initial positions
    xf = sim(x0)           # Final positions 