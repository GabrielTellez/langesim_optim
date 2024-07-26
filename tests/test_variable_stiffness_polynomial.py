import torch
from langesim_optim import VariableStiffnessHarmonicForcePolynomial, device


def test_VariableStiffnessHarmonicForcePolynomial_initialization():
    kappai = 1.0
    kappaf = 2.0
    tf = 0.1
    coef_list = [1.0, 0.5, 0.1]

    force = VariableStiffnessHarmonicForcePolynomial(
        kappai=kappai, kappaf=kappaf, tf=tf, coef_list=coef_list
    )

    assert torch.isclose(force.kappai, torch.tensor(kappai, dtype=torch.float))
    assert torch.isclose(force.kappaf, torch.tensor(kappaf, dtype=torch.float))
    assert torch.isclose(force.tf, torch.tensor(tf, dtype=torch.float))
    assert torch.allclose(force.coef_list, torch.tensor(coef_list, dtype=torch.float))


def test_VariableStiffnessHarmonicForcePolynomial_kappa_discontinuous():
    kappai = 1.0
    kappaf = 2.0
    tf = 0.1
    coef_list = [1.0, 0.5, 0.1]

    force = VariableStiffnessHarmonicForcePolynomial(
        kappai=kappai, kappaf=kappaf, tf=tf, coef_list=coef_list, continuous=False
    )

    t = torch.tensor(0.05, dtype=torch.float)
    expected_kappa = coef_list[0] + coef_list[1] * t + coef_list[2] * t**2
    assert torch.isclose(force.kappa(t), expected_kappa)


def test_VariableStiffnessHarmonicForcePolynomial_kappa_continuous():
    kappai = 1.0
    kappaf = 2.0
    tf = 0.1
    coef_list = [3.0, 2.5, 2.1]

    force = VariableStiffnessHarmonicForcePolynomial(
        kappai=kappai, kappaf=kappaf, tf=tf, coef_list=coef_list, continuous=True
    )

    t = torch.tensor(0.1, dtype=torch.float)
    expected_kappa = torch.tensor(kappaf)
    assert torch.isclose(
        force.kappa(t), expected_kappa
    ), f"expected {expected_kappa} at t=tf, got {force.kappa(t)}"

    t = torch.tensor(0.0, dtype=torch.float)
    expected_kappa = torch.tensor(kappai)
    assert torch.isclose(
        force.kappa(t), expected_kappa
    ), f"expected {expected_kappa} at t=0, got {force.kappa(t)}"

    t = torch.tensor(0.05, dtype=torch.float)
    expected_kappa = coef_list[0] + coef_list[1] * t + coef_list[2] * t**2
    expected_kappa = expected_kappa * (t / tf) * (1 - t / tf)
    expected_kappa = expected_kappa + (kappaf - kappai) * t / tf + kappai
    assert torch.isclose(
        force.kappa(t), expected_kappa
    ), f"expected {expected_kappa} at t=0.05, got {force.kappa(t)}"


def test_VariableStiffnessHarmonicForcePolynomial_before_after():
    kappai = 1.0
    kappaf = 2.0
    tf = 0.1
    coef_list = [4.0, 3.0, 2.5, 2.1]

    force = VariableStiffnessHarmonicForcePolynomial(
        kappai=kappai, kappaf=kappaf, tf=tf, coef_list=coef_list, continuous=True
    )

    t = torch.tensor(-0.1, dtype=torch.float)
    expected_kappa = torch.tensor(kappai)
    assert torch.isclose(
        force.kappa(t), expected_kappa
    ), f"expected {expected_kappa} at t=-0.1, got {force.kappa(t)}"

    t = torch.tensor(0.2, dtype=torch.float)
    expected_kappa = torch.tensor(kappaf)
    assert torch.isclose(
        force.kappa(t), expected_kappa
    ), f"expected {expected_kappa} at t=0.2, got {force.kappa(t)}"

def test_VariableStiffnessHarmonicForcePolynomial_normalized():
    """Test normalized option: variable is (t/tf)"""
    kappai = 1.0
    kappaf = 2.0
    tf = 0.1
    coef_list = [4.0, 3.0, 2.5, 2.1]

    force = VariableStiffnessHarmonicForcePolynomial(
        kappai=kappai, kappaf=kappaf, tf=tf, coef_list=coef_list, continuous=True, normalized=True
    )

    t = torch.tensor(0.1, dtype=torch.float)
    expected_kappa = torch.tensor(kappaf)
    assert torch.isclose(
        force.kappa(t), expected_kappa
    ), f"expected {expected_kappa} at t=tf, got {force.kappa(t)}"

    t = torch.tensor(0.0, dtype=torch.float)
    expected_kappa = torch.tensor(kappai)
    assert torch.isclose(
        force.kappa(t), expected_kappa
    ), f"expected {expected_kappa} at t=0, got {force.kappa(t)}"

    t = torch.tensor(0.05, dtype=torch.float)
    s = t/tf
    expected_kappa = coef_list[0] + coef_list[1] * s + coef_list[2] * s**2 + coef_list[3] * s**3
    expected_kappa = expected_kappa * s * (1 - s)
    expected_kappa = expected_kappa + (kappaf - kappai) * s + kappai
    assert torch.isclose(
        force.kappa(t), expected_kappa
    ), f"expected {expected_kappa} at t=0.05, got {force.kappa(t)}"

    