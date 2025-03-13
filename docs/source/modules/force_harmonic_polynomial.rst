Force Harmonic Polynomial Module
============================

The ``force_harmonic_polynomial`` module implements a harmonic oscillator force with variable stiffness represented as a polynomial function of time.

.. currentmodule:: langesim_optim.force_harmonic_polynomial

Classes
-------

.. autoclass:: VariableStiffnessHarmonicForcePolynomial
   :members:
   :special-members: __init__
   :show-inheritance:

Example Usage
-----------

Here's an example of how to create and use a polynomial harmonic force:

.. code-block:: python

    import langesim_optim as lso
    
    # Create a polynomial force with initial coefficients
    force = lso.force_harmonic_polynomial.VariableStiffnessHarmonicForcePolynomial(
        kappai=1.0,  # Initial stiffness
        kappaf=2.0,  # Final stiffness
        tf=1.0,      # Final time
        coef_list=[0.0, 0.0, 1.0],  # Quadratic polynomial coefficients
        continuous=True,  # Enforce continuity at endpoints
        normalized=True   # Use normalized time t/tf
    )
    
    # Get stiffness at a specific time
    t = 0.5
    k = force.kappa(t) 