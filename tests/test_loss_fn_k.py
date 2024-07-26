import torch
import pytest
from langesim_optim import (
    loss_fn_k,
    device,
    char_fn,
    FT_pdf,
)


# Define test cases for the loss_fn_k function
def test_loss_fn_k():
    # Set up mock data and parameters
    xf = torch.tensor([1.0, 2.0, 3.0, 4.0], device=device)
    kf = 1.0
    ki = 1.0

    scale = 5.0
    kFsteps = 1000
    x_steps = 10000

    # Call the function
    loss_value = loss_fn_k(
        xf=xf,
        kf=kf,
        ki=ki,
        device=device,
        scale=scale,
        kFsteps=kFsteps,
        x_steps=x_steps,
    )

    # Assertions
    assert isinstance(loss_value, torch.Tensor)
    assert loss_value.item() >= 0  # Loss value should be non-negative


def test_loss_fn_gaussian():
    """Test characteristic function of a normal distribution is a gaussian using
    the loss function loss_fn_k"""
    var = 3.0
    kf = 1.0 / var
    xf = var**0.5 * torch.randn(100_000, device=device)

    loss_value = loss_fn_k(xf=xf, kf=kf)

    assert loss_value.item() < 5e-5


def test_loss_fn_gaussian_noncentered():
    """Test characteristic function of a normal distribution with non zero
    center is a gaussian using the loss function loss_fn_k"""
    var = 2.0
    kf = 1.0 / var
    cf = 12.0
    xf = var**0.5 * torch.randn(100_000, device=device) + cf

    loss_value = loss_fn_k(xf=xf, kf=kf, cf=cf)

    assert loss_value.item() < 5e-5
