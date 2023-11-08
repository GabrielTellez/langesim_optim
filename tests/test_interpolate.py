import numpy as np
from langesim_optim import make_interpolator
from langesim_optim import interpolate
from pytest import approx


def test_make_interpolator_continuous():
    # Test case 1: Test the output type of the function
    yi = 1.0
    yf = 2.0
    ti = 1.0
    tf = 3.0
    ylist = [1.5, 1.75]
    interpolator = make_interpolator(yi, yf, ti, tf, ylist)
    assert callable(interpolator)

    # Test case 2: Test the output values of the function at the boundaries
    assert interpolator(ti) == yi, f"{interpolator(ti)=} != {yi=}"
    assert interpolator(tf) == yf, f"{interpolator(tf)=} != {yf=}"

    # Test case 3: Test the output values of the function at the intermediate points
    assert interpolator(ti + (tf - ti) / 3) == ylist[0], f"{interpolator(ti + (tf - ti) / 3)=} != {ylist[0]=}"
    assert interpolator(ti + 2 * (tf - ti) / 3) == ylist[1], f"{interpolator(ti + 2 * (tf - ti) / 3)=} != {ylist[1]=}"

    # Test case 4: middle value
    assert np.isclose(interpolator(ti + (tf - ti) / 2), (ylist[0] + ylist[1]) / 2)

def test_make_interpolator_discontinuous():
    # Test case 5: Test the function with continuous=False
    yi = 1.0
    yf = 2.0
    ti = 1.0
    tf = 3.0
    ylist = [1.7, 2.75]
    interpolator = make_interpolator(yi, yf, ti, tf, ylist, continuous=False)
    assert interpolator(ti) == yi
    assert interpolator(tf) == yf
    assert np.isclose(
        interpolator(ti + (tf - ti) / 3), ylist[0] + (ylist[1] - ylist[0]) / 3
    )
    assert np.isclose(
        interpolator(ti + 2 * (tf - ti) / 3), ylist[0] + 2 * (ylist[1] - ylist[0]) / 3
    )
    assert np.isclose(interpolator(ti + (tf - ti) / 2), (ylist[0] + ylist[1]) / 2)


def test_interpolator():
    ki = 0.0
    kf = 6.0
    tf = 0.1
    dt = 0.01
    tot_steps = int(tf / dt)
    k_in = list(range(1, 6))

    interpolator = make_interpolator(ki, kf, 0, tf, k_in, continuous=True)

    # Test case 3: Test the interpolator function
    assert interpolator(0) == approx(ki), f"interpolator(0) = {interpolator(0)} != {ki=}"
    assert interpolator(tf) == approx(kf), f"interpolator({tf=}) = {interpolator(tf)} != {kf=}"
    assert ki < interpolator(tf / 2) < kf, f"NOT: {ki=} < {interpolator(tf/2)=} < {kf=}"

    # Test a few values
    for t in [0.01, 0.02, 0.05, 0.09]:
        expected_value = ki + (kf - ki) * t / tf
        assert interpolator(t) == approx(
            expected_value
        ), f"interpolator({t}) = {interpolator(t)} != {expected_value}"

def test_interpolator_TSP():
    ki = 1.0
    kf = 2.0
    kl = [3.0]
    tf = 1.0

    interpolator = make_interpolator(ki, kf, 0, tf, kl, continuous=False)

    assert interpolator(0) == approx(ki)
    assert interpolator(tf) == approx(kf)
    assert interpolator(tf/2) == approx(kl[0])
    assert interpolator(tf/4) == approx(kl[0])
    assert interpolator(3*tf/4) == approx(kl[0])


    def test_interpolate_function_continuous():
        # Test case: Test the output values of the function at the boundaries
        # and middle values
        yi = 1.0
        yf = 2.0
        ti = 0.0
        tf = 1.0
        ylist = [1.5, 1.75]
        assert interpolate(ti, yi, yf, ti, tf, ylist) == yi
        assert interpolate(tf, yi, yf, ti, tf, ylist) == yf
        assert np.isclose(interpolate(ti + (tf - ti) / 3, yi, yf, ti, tf, ylist), ylist[0])
        assert np.isclose(interpolate(ti + 2 * (tf - ti) / 3, yi, yf, ti, tf, ylist), ylist[1])


def test_interpolate_discontinuous():
    # Test case: Test the output values of the function at the boundaries and intermediate points
    yi = 1.0
    yf = 2.0
    ti = 0.0
    tf = 1.0
    ylist = [1.7, 2.75]
    assert interpolate(ti, yi, yf, ti, tf, ylist, continuous=False) == yi
    assert interpolate(tf, yi, yf, ti, tf, ylist, continuous=False) == yf
    assert np.isclose(interpolate(ti + (tf - ti) / 3, yi, yf, ti, tf, ylist, continuous=False), ylist[0]+(ylist[1]-ylist[0])/3)
    assert np.isclose(interpolate(ti + 2 * (tf - ti) / 3, yi, yf, ti, tf, ylist, continuous=False), ylist[0]+2*(ylist[1]-ylist[0])/3)