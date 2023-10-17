import torch
from langesim_optim import Simulator, BaseForce
import pytest
import numpy as np

# Mock BaseForce class for testing purposes
class ZeroForce(BaseForce):
    def force(self, x, t):
        return torch.zeros_like(x)

def test_simulator_forward_shape_output():
    # Instantiate the Simulator class
    simulator = Simulator(tot_steps=10)

    # Define initial conditions for testing
    x0 = torch.tensor([1.0, 2.0, 3.0])

    # Perform forward pass
    result = simulator(x0)

    # Check if the result has the correct shape
    assert result.shape == x0.shape

def test_simulator_with_compute_work_heat_shape_output():
    # Instantiate the Simulator class with compute_work_heat=True
    simulator = Simulator(tot_steps=10, compute_work_heat=True, force=ZeroForce())

    # Define initial conditions for testing
    x0 = torch.tensor([1.0, 2.0, 3.0])

    # Perform forward pass
    final_positions, work, heat = simulator(x0)

    # Check if the result has the correct shape
    assert final_positions.shape == x0.shape
    assert work.shape == x0.shape
    assert heat.shape == x0.shape

def test_simulator_diffusion_variances():
    """Test the simulator with zero force: simple diffusion should have <x^2> = 2Dt."""
    sim = Simulator(tot_steps=10_000, force=ZeroForce())
    N = 10_000
    D = 1.0
    x0 = torch.zeros(N)
    x1 = sim(x0)
    x2 = sim(x1)
    x3 = sim(x2)
    dt = sim.dt.item()
    tot_steps = sim.tot_steps.item()
    variances = [x1.var().item(), x2.var().item(), x3.var().item()]
    times = dt * tot_steps * np.array( [1.0, 2.0, 3.0] )
    for idx, t in enumerate(times):
        assert variances[idx] == pytest.approx(2*D*t, rel=5e-2), f"<x**2> != 2Dt at {t=}, {idx=}"

def test_simulator_diffusion_average_position_dont_move():
    """Test the simulator with zero force: simple diffusion should have <x(t)> = x_initial."""
    sim = Simulator(tot_steps=10_000, force=ZeroForce())
    N = 10_000
    D = 1.0
    x0 = 10.0*torch.ones(N)
    x1 = sim(x0)
    x2 = sim(x1)
    x3 = sim(x2)
    dt = sim.dt.item()
    tot_steps = sim.tot_steps.item()
    variances = [x1.mean().item(), x2.mean().item(), x3.mean().item()]
    times = dt * tot_steps * np.array( [1.0, 2.0, 3.0] )
    for idx, t in enumerate(times):
        assert variances[idx] == pytest.approx(10.0, rel=5e-2), f"<x(t))> != 10.0 at {t=}, {idx=}"


