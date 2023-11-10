import torch
from langesim_optim import Simulator, VariableStiffnessHarmonicForce, device
import pytest
import numpy as np


# Test : Ornstein - Uhlenbeck process
@pytest.mark.parametrize("ki,ko", [(1.0, 4.0), (0.5, 1.0)])
def test_jump_k(ki, ko):
    """Test that if the stifness jumps from ki to ko at t=0, the variance
    evolves according to
    <x(t)^2> = exp(-2ko t)/ki + (1-exp(-2ko t))/ko
    """
    tot_steps = 1000
    dt = 0.001
    tot_sims = 10_000

    def theo_var(t):
        """Theoretical variance"""
        return np.exp(-2.0 * ko * t) / ki + (1.0 - np.exp(-2.0 * ko * t)) / ko

    tf = dt * tot_steps

    force = VariableStiffnessHarmonicForce(kappai=ki, kappaf=ko, tf=tf, k=[ko], continuous=False)
    sim = Simulator(dt=dt, tot_steps=tot_steps, force=force)

    x0 = torch.randn(tot_sims, device=device) * ki**-0.5
    xf = sim(x0)
    varteo = theo_var(tf)
    varnum = xf.var().item()
    assert varteo == pytest.approx(varnum, rel=5e-2)
