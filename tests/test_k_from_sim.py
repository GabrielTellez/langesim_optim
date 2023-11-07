import torch
from langesim_optim import k_from_sim, Simulator, VariableHarmonicForce
import numpy as np
from pytest import approx


def test_k_from_sim():
    # Test case 1: Test the output type of the function
    ki = 0.0
    kf = 6.0
    tf = 0.1
    dt = 0.01
    tot_steps = int(tf / dt)
    k_in = list(range(1, 6))
    force = VariableHarmonicForce(kappai=ki, kappaf=kf, tf=tf, k=k_in, continuous=True)
    sim = Simulator(dt=dt, tot_steps=tot_steps, force=force)
    k, ki_out, kf_out, tf_out, kappa_numpy = k_from_sim(sim)
    assert isinstance(k, np.ndarray)
    assert isinstance(ki_out, float)
    assert isinstance(kf_out, float)
    assert isinstance(tf_out, float)
    assert callable(kappa_numpy)

    # Test case 2: Test the output values of the function
    assert ki_out == approx(ki)
    assert kf_out == approx(kf)
    assert tf_out == approx(tf)
    assert len(k) == len(k_in)
    assert (k == k_in).all()

    # Test case 3: Test the kappa_numpy function
    assert kappa_numpy(0) == approx(ki), f"kappa_numpy(0) = {kappa_numpy(0)} != {ki=}"
    assert kappa_numpy(tf) == approx(kf), f"kappa_numpy({tf=}) = {kappa_numpy(tf)} != {kf=}"
    assert ki < kappa_numpy(tf / 2) < kf, f"NOT: {ki=} < {kappa_numpy(tf/2)=} < {kf=}"

    # Test a few values
    for t in [0.01, 0.02, 0.05, 0.09]:
        expected_value = ki + (kf - ki) * t / tf
        assert kappa_numpy(t) == approx(
            expected_value
        ), f"kappa_numpy({t}) = {kappa_numpy(t)} != {expected_value}"


def test_k_from_sim_discontinuous():
    # Test case 1: Test the output type of the function
    ki = 0.0
    kf = 6.0
    tf = 0.1
    dt = 0.01
    tot_steps = int(tf / dt)
    k_in = list(range(1, 6))
    force = VariableHarmonicForce(kappai=ki, kappaf=kf, tf=tf, k=k_in, continuous=False)
    sim = Simulator(dt=dt, tot_steps=tot_steps, force=force)
    k, ki_out, kf_out, tf_out, kappa_numpy = k_from_sim(sim)
    assert isinstance(k, np.ndarray)
    assert isinstance(ki_out, float)
    assert isinstance(kf_out, float)
    assert isinstance(tf_out, float)
    assert callable(kappa_numpy)

    # Test case 2: Test the output values of the function
    assert ki_out == approx(ki)
    assert kf_out == approx(kf)
    assert tf_out == approx(tf)
    assert len(k) == len(k_in)

    # Test case 3: Test the kappa_numpy function
    assert kappa_numpy(0) == approx(ki)
    assert kappa_numpy(tf) == approx(kf)
    assert ki < kappa_numpy(tf / 2) < kf

    # Test a few values
    kinit = 1.0
    kfinal = 5.0
    for t in [0.01, 0.02, 0.05, 0.09]:
        expected_value = kinit + (kfinal - kinit) * t / tf
        assert kappa_numpy(t) == approx(
            expected_value
        ), f"kappa_numpy({t}) = {kappa_numpy(t)} != {expected_value}"
