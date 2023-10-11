import torch
import pytest
from langesim_optim import BaseForce  
from unittest.mock import patch

@pytest.fixture
def base_force():
    return BaseForce()

def test_force_returns_zero(base_force):
    x = torch.tensor(1.0)
    t = torch.tensor(2.0)
    result = base_force.force(x, t)
    assert result == torch.tensor(0.0)

def test_potential_returns_zero(base_force):
    x = torch.tensor(1.0)
    t = torch.tensor(2.0)
    result = base_force.potential(x, t)
    assert result == torch.tensor(0.0)

def test_forward_call_return_force(base_force):
    x = torch.tensor(3.14)
    t = torch.tensor(5.0)
    result1 = base_force(x, t)
    result2 = base_force.force(x, t)
    assert result1 == result2

def test_forward_calls_force_method(base_force):
    x = torch.tensor(1.0)
    t = torch.tensor(2.0)
    with patch.object(BaseForce, 'force', return_value=torch.tensor(42.0)) as mock_force:
        output = base_force(x, t)
        mock_force.assert_called_once_with(x, t)
        assert output == torch.tensor(42.0)