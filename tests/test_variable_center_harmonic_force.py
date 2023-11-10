import torch
from langesim_optim import VariableCenterHarmonicForce


def test_center_initial_value():
    centeri = 1.0
    centerf = 3.0
    tf = 1.0
    steps = 4
    force = VariableCenterHarmonicForce(centeri, centerf, tf, steps=steps)
    t = torch.tensor(0.0)
    assert force.center(t) == centeri


def test_center_final_value():
    centeri = 2.0
    centerf = 5.0
    tf = 1.0
    steps = 5
    force = VariableCenterHarmonicForce(centeri, centerf, tf, steps=steps)
    t = torch.tensor(tf)
    assert force.center(t) == centerf


def test_center_interpolation_2points():
    centeri = 2.0
    centerf = 4.0
    tf = 1.0
    cl = [2.0, 4.0]
    force = VariableCenterHarmonicForce(centeri, centerf, tf, center_list=cl, continuous=True)
    t = torch.tensor(0.5)
    expected_result = 3.0
    assert force.center(t) == expected_result
    t = torch.tensor(0.1)
    expected_result = 2.0
    assert force.center(t) == expected_result
    t = torch.tensor(0.9)
    expected_result = 4.0
    assert force.center(t) == expected_result


def test_center_1points():
    centeri = 2.0
    centerf = 2.0
    middle = 6.0
    tf = 1.0
    cl = [6.0]
    force = VariableCenterHarmonicForce(centeri, centerf, tf, center_list=cl, continuous=True)
    t = torch.tensor(0.0)
    expected_result = centeri
    assert force.center(t) == expected_result
    t = torch.tensor(0.25)
    expected_result = 4.0
    assert force.center(t) == expected_result
    t = torch.tensor(0.75)
    expected_result = 4.0
    assert force.center(t) == expected_result
    t = torch.tensor(1.0)
    expected_result = centerf
    assert force.center(t) == expected_result


def test_center_TSP():
    centeri = 2.0
    centerf = 5.0
    tf = 1.0
    cl = [10.0]
    force = VariableCenterHarmonicForce(centeri, centerf, tf, center_list=cl, continuous=False)
    t = torch.tensor(0.0)
    expected_result = centeri
    assert force.center(t) == expected_result
    t = torch.tensor(0.01)
    expected_result = 10.0
    assert force.center(t) == expected_result
    t = torch.tensor(1.0)
    expected_result = centerf
    assert force.center(t) == expected_result
    t = torch.tensor(0.99)
    expected_result = 10.0
    assert force.center(t) == expected_result
