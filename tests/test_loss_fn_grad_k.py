import torch
from langesim_optim import (
    loss_fn_grad_k,
    Simulator,
    VariableHarmonicForce,
    device,
    loss_fn_k,
    loss_fn_control_k_vars,
)


def test_loss_fn_grad_k_cst():
    # Test case 1: Test the output shape of the function
    ki = 1.0
    kf = 1.0
    tf = 1.0
    k = [1.0] * 5
    force = VariableHarmonicForce(kappai=ki, kappaf=kf, tf=tf, k=k, continuous=True)
    sim = Simulator(force=force)
    loss = loss_fn_grad_k(ki, kf, sim)
    assert loss.shape == ()

    # Test case 2: Test the output value of the function with k constant
    assert torch.allclose(loss, torch.tensor(0.0), rtol=1e-05, atol=1e-08)


def test_loss_fn_grad_k_linear():
    ki = 1.0
    kf = 1.0
    tf = 1.0
    k = list(range(1, 5))
    force = VariableHarmonicForce(kappai=ki, kappaf=kf, tf=tf, k=k, continuous=False)
    sim = Simulator(force=force)
    loss = loss_fn_grad_k(ki, kf, sim)

    # Test case 3: Test the output value of the function with k linear
    assert torch.allclose(loss, torch.tensor(1.0), rtol=1e-05, atol=1e-08)


def test_loss_fn_control_k_vars():
    xf = torch.randn(100_000, device=device)
    kf = 1.0
    ki = 1.0
    tf = 1.0
    k = list(range(1, 5))
    force = VariableHarmonicForce(kappai=ki, kappaf=kf, tf=tf, k=k, continuous=False)
    sim = Simulator(force=force)
    blend = 1e-1
    loss = loss_fn_control_k_vars(xf, kf, ki, sim, blend=blend)
    # Test case 1: Test the output shape of the function
    assert loss.shape == ()
    loss_value = loss_fn_k(xf, kf)
    assert loss_value.item() < 5e-5
    # Test case 2: Test the output value of the function for linear k: should be
    # 1.0*blend=1e-1 (loss_fn_k is negligible of order 1e-5)
    assert torch.allclose(loss, torch.tensor(1e-1), rtol=1e-02, atol=1e-02)
