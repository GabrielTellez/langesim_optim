import torch
import numpy as np
from langesim_optim import gaussian
import pytest


def test_gaussian():
    # Test case 1: check if the function returns the correct value for x=0 and var=1
    x = torch.tensor(0.0)
    var = torch.tensor(1.0)
    result = gaussian(x, var)
    expected_result = torch.tensor(1.0 / (2.0 * np.pi) ** 0.5)
    assert torch.allclose(result, expected_result)

    # Test case 2: check if the function returns 0 when x is far from 0 and var=1
    x = torch.tensor(100.0)
    var = torch.tensor(1.0)
    result = gaussian(x, var)
    assert torch.allclose(result, torch.tensor(0.0))

    # Test case 3: check if the function raises an error for negative variance
    with pytest.raises(ValueError):
        x = torch.tensor(0.0)
        var = torch.tensor(-1.0)
        gaussian(x, var)


# Run the test by running `pytest` in the terminal where this test file is located.
