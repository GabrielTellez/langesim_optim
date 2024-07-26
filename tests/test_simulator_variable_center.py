import torch
from langesim_optim import Simulator, VariableCenterHarmonicForce, device
import pytest
import numpy as np


# Test : variable center TSP
@pytest.mark.parametrize("ci,co", [(0.0, 4.0), (5.0, 1.0)])
def test_jump_center(ci, co):
    """Test that if the center jumps from ci to co at t=0, the average position
    evolves according to
    <x(t)> = (ci - cm) exp(-k t) + cm
    """
    tot_steps = 1000
    dt = 0.001
    tot_sims = 10_000
    ki = 1.0

    def theo_pos(t):
        """Theoretical average position"""
        return (ci - co) * np.exp(-ki * t) + co

    tf = dt * tot_steps

    force = VariableCenterHarmonicForce(
        centeri=ci, centerf=co, tf=tf, center_list=[co], kappa0=ki, continuous=False
    )
    sim = Simulator(dt=dt, tot_steps=tot_steps, force=force)

    x0 = torch.randn(tot_sims, device=device) * ki**-0.5 + ci
    xf = sim(x0)
    posteo = theo_pos(tf)
    posnum = xf.mean().item()
    varnum = xf.var().item()
    varteo = 1 / ki
    assert posteo == pytest.approx(posnum, rel=5e-2), f"{posteo=} != {posnum=}"
    assert varteo == pytest.approx(varnum, rel=5e-2), f"{varteo=} != {varnum=}"
