# FILEPATH: /hpcfs/home/fisica/gtellez/langesim_optim/tests/test_langesim_optim.py

import torch
from langesim_optim import (
    train_loop,
    Simulator,
    VariableHarmonicForce,
    loss_fn_k,
    device,
)
from torch.optim import SGD
import numpy as np
from scipy.optimize import fsolve


def test_train_loop_output_type_length():
    epochs = 5
    ki = 1.0
    kf = 2.0
    dt = 0.01
    tf = 0.1
    tot_steps = int(tf / dt)
    tot_sims = 100
    lr = 1.0

    force = VariableHarmonicForce(kappai=ki, kappaf=kf, tf=tf, steps=3)
    sim = Simulator(dt=dt, tot_steps=tot_steps, force=force, device=device)

    optimizer = SGD(params=sim.parameters(), lr=lr)
    loss_fn = loss_fn_k

    lossi = train_loop(
        epochs=epochs,
        sim=sim,
        tot_sims=tot_sims,
        ki=ki,
        kf=kf,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
    )
    # Test case 1: Test the output type of the function
    assert isinstance(lossi, list)

    # Test case 2: Test the output length of the function
    assert len(lossi) == epochs


def test_train_loop_TSP():
    """Test a train loop on the two-step protocol (TSP): only one value of k is learned"""

    def theo_var(t, ko, ki):
        """Theoretical variance"""
        return np.exp(-2.0 * ko * t) / ki + (1.0 - np.exp(-2.0 * ko * t)) / ko

    def eq_to_solve(k, ki, kf, tf):
        return 1 / kf - theo_var(tf, k, ki)

    def ktheo(ki, kf, tf):
        return fsolve(eq_to_solve, kf, args=(ki, kf, tf))[0]

    epochs = 20
    ki = 0.5
    kf = 1.0
    dt = 0.001
    tf = 0.100
    tot_steps = int(tf / dt)
    tot_sims = 100_000

    lr = 100.0
    force = VariableHarmonicForce(kappai=ki, kappaf=kf, tf=tf, k=[kf], continuous=False)
    sim = Simulator(force=force, dt=dt, tot_steps=tot_steps, device=device)

    optimizer = torch.optim.SGD(params=sim.parameters(), lr=lr)
    loss_fn = loss_fn_k

    lossi = train_loop(
        epochs=epochs,
        sim=sim,
        tot_sims=tot_sims,
        ki=ki,
        kf=kf,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
    )

    loss_mean = np.mean(lossi[-5:])
    assert loss_mean < 1e-4, f"{loss_mean=} > 1e-4"
    ktheo_val = ktheo(ki, kf, tf)
    klearned = sim.force.k[0].item()
    assert np.isclose(klearned, ktheo_val, rtol=1e-2), f"{klearned=} != {ktheo_val=}"
