Interpolator Module
=================

The ``interpolator`` module provides functionality for linear interpolation between a set of values over a time interval.

.. currentmodule:: langesim_optim.interpolator

Functions
--------

.. autofunction:: interpolate

.. autofunction:: make_interpolator

Classes
-------

.. autoclass:: Interpolator
   :members:
   :special-members: __init__

Example Usage
-----------

Here's an example of how to use the interpolator:

.. code-block:: python

    from langesim_optim.interpolator import Interpolator, make_interpolator
    
    # Create an interpolator object
    interp = Interpolator(
        yi=0.0,        # Initial value
        yf=1.0,        # Final value
        ti=0.0,        # Initial time
        tf=1.0,        # Final time
        ylist=[0.2, 0.5, 0.8],  # Values at intermediate points
        continuous=True  # Make the interpolation continuous
    )
    
    # Get interpolated value at a specific time
    t = 0.5
    y = interp(t)
    
    # Alternatively, use the make_interpolator function
    interp_func = make_interpolator(
        yi=0.0,
        yf=1.0,
        ti=0.0,
        tf=1.0,
        ylist=[0.2, 0.5, 0.8]
    )
    y = interp_func(t) 