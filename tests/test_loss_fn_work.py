import torch
from langesim_optim import loss_fn_work, Simulator, device, loss_fn_eq_work, loss_fn_k


def test_loss_fn_work():
    # Test case 1: Test the output shape of the function

    sim = Simulator()
    sim.w = torch.tensor([1.0] * 10)

    loss = loss_fn_work(sim=sim)
    assert loss.shape == ()

    # Test case 2: Test the output value of the function

    assert torch.allclose(loss, torch.tensor(1.0), rtol=1e-05, atol=1e-08)


def test_loss_fn_eq_work():
    xf = torch.randn(100_000, device=device)
    kf = 1.0
    ki = 1.0
    sim = Simulator()
    sim.w = torch.tensor([1.0] * 10)
    blend = 1e-1
    loss = loss_fn_eq_work(xf=xf, kf=kf, sim=sim, blend=blend)
    # Test case 1: Test the output shape of the function
    assert loss.shape == ()

    # Test case 2: Test the output value of the function

    loss = loss_fn_eq_work(xf=xf, kf=kf, sim=sim, blend=blend)
    loss_fn_k_value = loss_fn_k(xf=xf, kf=kf)
    assert loss_fn_k_value.item() < 5e-5
    assert torch.allclose(loss, torch.tensor(1e-1), rtol=1e-02, atol=1e-02)
