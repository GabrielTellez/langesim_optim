import numpy as np
from langesim_optim import make_interpolator


def test_make_interpolator():
    # Test case 1: Test the output type of the function
    yi = 1.0
    yf = 2.0
    ti = 0.0
    tf = 1.0
    ylist = [1.5, 1.75]
    interpolator = make_interpolator(yi, yf, ti, tf, ylist)
    assert callable(interpolator)

    # Test case 2: Test the output values of the function at the boundaries
    assert interpolator(ti) == yi
    assert interpolator(tf) == yf

    # Test case 3: Test the output values of the function at the intermediate points
    assert interpolator(ti + (tf - ti) / 3) == ylist[0]
    assert interpolator(ti + 2 * (tf - ti) / 3) == ylist[1]

    # Test case 4: Test the function with continuous=False
    interpolator = make_interpolator(yi, yf, ti, tf, ylist, continuous=False)
    assert interpolator(ti) == yi
    assert interpolator(tf) == yf
    assert np.isclose(
        interpolator(ti + (tf - ti) / 3), ylist[0] + (ylist[1] - ylist[0]) / 3
    )
    assert np.isclose(
        interpolator(ti + 2 * (tf - ti) / 3), ylist[0] + 2 * (ylist[1] - ylist[0]) / 3
    )
