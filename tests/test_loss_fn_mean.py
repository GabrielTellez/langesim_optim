import torch
from langesim_optim import loss_fn_mean, device


def test_loss_fn_mean():
    cf = 10.0
    xf = 10.0 * torch.randn(100_000) + cf
    ci = 1.0
    sim = None
    loss = loss_fn_mean(xf=xf, cf=cf)
    # Test case 1: Test the output shape of the function
    assert loss.shape == ()
    # Test case 2: Test the output value of the function
    assert torch.allclose(loss, torch.tensor(0.0), rtol=1e-05, atol=1e-03)

