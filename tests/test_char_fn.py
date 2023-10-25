import torch
from langesim_optim import char_fn, gaussian, device
import pytest
import numpy as np


# Define test cases for the char_fn function
@pytest.mark.parametrize("kf, scale, kFsteps", [(1.0, 5.0, 1000), (0.5, 3.0, 500)])
def test_char_fn(kf, scale, kFsteps):
    # Generate sample data
    xf = torch.tensor([1.0, 2.0, 3.0, 4.0], device=device)

    # Call the function
    char_P_k, kFs = char_fn(xf, kf, scale, kFsteps, device)

    # Assertions
    assert isinstance(char_P_k, torch.Tensor)
    assert isinstance(kFs, torch.Tensor)
    assert char_P_k.shape == (kFsteps,)
    assert kFs.shape == (kFsteps,)
    assert torch.allclose(
        char_P_k[0], torch.tensor(1.0 + 0.0j)
    )  # Characteristic function at K=0 should be 1 (normalization)
    assert torch.all(kFs >= 0)  # kFs should be non-negative


def test_char_fn_gaussian():
    """Characteristic function of a normal distribution is a gaussian."""
    var = 3.0
    kf = 1.0 / var
    xf = var**0.5 * torch.randn(100_000, device=device)

    char_fn_gaussian, kFs = char_fn(xf, 1.0 / var, kFsteps=100, device=device)
    FT_gaussian_theo = (1.0 + 0.0j) * (2 * np.pi * kf) ** 0.5 * gaussian(kFs, kf)
    assert torch.allclose(char_fn_gaussian, FT_gaussian_theo, rtol=1e-2, atol=1e-2)
