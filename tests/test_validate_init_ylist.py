import pytest
from langesim_optim import validate_init_interpolation_list

def test_ylist_steps_mismatch():
    # Test case: Test that a ValueError is raised when the length of ylist is not equal to steps
    yi = 1.0
    yf = 2.0
    steps = 3
    ylist = [1.5, 1.75]  # length of ylist is not equal to steps
    with pytest.raises(ValueError):
        _ = validate_init_interpolation_list(yi, yf, ylist=ylist, steps=steps)
  

def test_ylist_steps_none():
    # Test case: Test that a ValueError is raised when ylist and steps are not provided
    yi = 1.0
    yf = 2.0
    with pytest.raises(ValueError):
        _ = validate_init_interpolation_list(yi, yf)
  