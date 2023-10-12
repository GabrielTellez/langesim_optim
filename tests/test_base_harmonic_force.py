import torch
from langesim_optim import BaseHarmonicForce

def test_kappa_default_value():
    force = BaseHarmonicForce()
    t = torch.tensor(0.0)
    assert force.kappa(t) == 0.0

def test_center_default_value():
    force = BaseHarmonicForce()
    t = torch.tensor(0.0)
    assert force.center(t) == 0.0

def test_force():
    force = BaseHarmonicForce()
    x = torch.tensor(1.0)
    t = torch.tensor(0.0)
    expected_result = -force.kappa(t) * (x - force.center(t))
    assert force.force(x, t) == expected_result

def test_potential():
    force = BaseHarmonicForce()
    x = torch.tensor(1.0)
    t = torch.tensor(0.0)
    expected_result = 0.5 * force.kappa(t) * (x - force.center(t))**2
    assert force.potential(x, t) == expected_result

