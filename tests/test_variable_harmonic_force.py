import torch
from langesim_optim import VariableHarmonicForce

def test_kappa_initial_value():
    kappai = 0.0
    kappaf = 1.0
    tf = 1.0
    steps = 3
    force = VariableHarmonicForce(kappai, kappaf, tf, steps=steps)
    t = torch.tensor(0.0)
    assert force.kappa(t) == kappai

def test_kappa_final_value():
    kappai = 0.0
    kappaf = 1.0
    tf = 1.0
    steps = 3
    force = VariableHarmonicForce(kappai, kappaf, tf, steps=steps)
    t = torch.tensor(1.0)
    assert force.kappa(t) == kappaf

def test_kappa_interpolation_2points():
    kappai = 1.0
    kappaf = 2.0
    tf = 1.0
    kl = [1.0, 2.0]
    force = VariableHarmonicForce(kappai, kappaf, tf, k=kl, continuous=True)
    t = torch.tensor(0.5)
    expected_result = 1.5
    assert force.kappa(t) == expected_result
    t = torch.tensor(0.1)
    expected_result = 1.0
    assert force.kappa(t) == expected_result
    t = torch.tensor(0.9)
    expected_result = 2.0
    assert force.kappa(t) == expected_result

def test_kappa_1points():
    kappai = 1.0
    kappaf = 1.0
    tf = 1.0
    kl = [2.0]
    force = VariableHarmonicForce(kappai, kappaf, tf, k=kl, continuous=True)
    t = torch.tensor(0.0)
    expected_result = kappai
    assert force.kappa(t) == expected_result
    t = torch.tensor(0.25)
    expected_result = 1.5
    assert force.kappa(t) == expected_result
    t = torch.tensor(0.75)
    expected_result = 1.5
    assert force.kappa(t) == expected_result   
    t = torch.tensor(1.0)
    expected_result = kappaf
    assert force.kappa(t) == expected_result

def test_kappa_non_continuous():
    kappai = 1.0
    kappaf = 2.0
    tf = 1.0
    kl=[3.0]
    force = VariableHarmonicForce(kappai, kappaf, tf, k=kl, continuous=False)
    t = torch.tensor(0.0)
    expected_result = kappai
    assert force.kappa(t) == expected_result
    t = torch.tensor(0.01)
    expected_result = 3.0
    assert force.kappa(t) == expected_result
    t = torch.tensor(1.0)
    expected_result = kappaf
    assert force.kappa(t) == expected_result
    t = torch.tensor(0.99)
    expected_result = 3.0
    assert force.kappa(t) == expected_result
