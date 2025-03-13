Utilities Module
===============

The ``utilities`` module provides essential functions for training, visualization, and analysis of Langevin simulations.

.. currentmodule:: langesim_optim.utilities

Training Functions
----------------

.. autofunction:: train_loop

.. autofunction:: train_loop_snapshots

Visualization Functions
---------------------

.. autofunction:: plot_test_hist

.. autofunction:: plot_protocols

Helper Functions
--------------

.. autofunction:: k_from_sim

Example Usage
-----------

Here's a basic example of how to use the training loop:

.. code-block:: python

    import langesim_optim as lso
    from langesim_optim.utilities import train_loop
    
    # Create a simulator
    sim = lso.Simulator(...)
    
    # Set up optimizer
    optimizer = torch.optim.Adam(sim.parameters(), lr=0.001)
    
    # Train the model
    losses = train_loop(
        epochs=100,
        sim=sim,
        tot_sims=1000,
        optimizer=optimizer,
        loss_fn=your_loss_function,
        ki=1.0,
        kf=2.0
    ) 