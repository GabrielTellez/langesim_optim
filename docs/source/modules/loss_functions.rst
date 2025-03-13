Loss Functions Module
==================

The ``loss_functions`` module provides various loss functions for training Langevin dynamics simulations.

.. currentmodule:: langesim_optim.loss_functions

Basic Functions
-------------

.. autofunction:: gaussian

.. autofunction:: FT_pdf

.. autofunction:: char_fn

Loss Functions
------------

.. autofunction:: loss_fn_k

.. autofunction:: loss_fn_variance

.. autofunction:: loss_fn_mean

.. autofunction:: loss_fn_grad_k

.. autofunction:: loss_fn_control_k_vars

.. autofunction:: loss_fn_work

.. autofunction:: loss_fn_eq_work

Example Usage
-----------

Here's an example of how to use the loss functions:

.. code-block:: python

    import torch
    from langesim_optim.loss_functions import loss_fn_k, gaussian
    
    # Create some final positions
    xf = torch.randn(1000)
    
    # Calculate loss for a specific stiffness
    kf = 2.0
    loss = loss_fn_k(
        xf=xf,
        kf=kf,
        cf=0.0,  # Center
        scale=5.0,
        kFsteps=1000
    )
    
    # Calculate Gaussian distribution
    x = torch.linspace(-3, 3, 100)
    pdf = gaussian(x, var=1.0, center=0.0) 