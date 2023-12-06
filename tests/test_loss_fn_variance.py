import torch
from langesim_optim import loss_fn_variance, device


def test_loss_fn_variance():
    xf = torch.randn(100_000)
    kf = 1.0
    ki = 1.0
    sim = None
    loss = loss_fn_variance(xf=xf, kf=kf)
    # Test case 1: Test the output shape of the function
    assert loss.shape == ()
    # Test case 2: Test the output value of the function
    assert torch.allclose(loss, torch.tensor(0.0), rtol=1e-05, atol=1e-04)
