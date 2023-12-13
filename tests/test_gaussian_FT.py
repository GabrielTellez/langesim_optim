import torch
from langesim_optim import FT_pdf, gaussian, device
import pytest
import numpy as np


def test_gaussian_FT():
    """Test that the Fourier transform of a gaussian is a gaussian"""

    var = 2.0
    kf = 1.0 / var
    FT_gaussian, kFs = FT_pdf(gaussian, kf, center=0.0, args={"var": var, "center": 0.0}, device=device)
    FT_gaussian_theo = (1.0 + 0.0j) * (2 * np.pi * kf) ** 0.5 * gaussian(kFs, kf)
    assert torch.allclose(FT_gaussian, FT_gaussian_theo, rtol=1e-3, atol=1e-4)

def test_gaussian_FT_noncentered():
    """Test that the Fourier transform of a gaussian is a gaussian with shifted center"""

    var = 3.0
    center = 10.0
    kf = 1.0 / var
    FT_gaussian, kFs = FT_pdf(pdf=gaussian, kf=kf, center=center, args={"var": var, "center": center}, device=device)
    FT_gaussian_theo = torch.exp(-0.5*kFs**2 * var + 1.0j * kFs * center)
    assert torch.allclose(FT_gaussian, FT_gaussian_theo, rtol=1e-3, atol=1e-4)
