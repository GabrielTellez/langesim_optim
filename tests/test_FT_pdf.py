import torch
from langesim_optim import FT_pdf, device
import pytest
import numpy as np


# Mock PDF function for testing purposes
def mock_pdf(x):
    var = 1.0
    return (2.0 * torch.tensor(np.pi) * var) ** -0.5 * torch.exp(
        -(x**2) / (2.0 * var)
    )


def test_FT_pdf_with_mock_pdf():
    # Test case for FT_pdf with a mock PDF function

    # Define parameters for testing
    kf = torch.tensor(1.0)
    scale = 5.0
    steps = 1000

    # Call FT_pdf function with the mock_pdf function
    kFsteps = 50
    result, kFs = FT_pdf(
        mock_pdf, kf, kFsteps=kFsteps, scale=scale, steps=steps, device=device
    )

    # Check if the result and kFs have the correct shapes
    assert result.shape == kFs.shape
    assert kFs.size(dim=0) == kFsteps


def test_FT_pdf_with_custom_pdf():
    # Test case for FT_pdf with a custom PDF function

    # Define custom PDF function for testing purposes
    def custom_pdf(x, a, b):
        return a * torch.exp(-b * x**2)

    # Define parameters for testing
    kf = torch.tensor(1.0)
    scale = 5.0
    steps = 1000
    a = torch.tensor(2.0)
    b = torch.tensor(3.0)

    # Call FT_pdf function with the custom PDF function and additional arguments
    result, kFs = FT_pdf(
        custom_pdf, kf, scale=scale, steps=steps, args=(a, b), device=device
    )

    # Check if the result and kFs have the correct shapes
    assert result.shape == kFs.shape


# Run tests by running `pytest` in the terminal where this test file is located.


## TO DO: Fourier transform of a gaussian is a gaussian
