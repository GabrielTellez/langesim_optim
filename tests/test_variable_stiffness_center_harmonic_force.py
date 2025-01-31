import torch
from langesim_optim import VariableStiffnessCenterHarmonicForce

def test_stiffness_center_initial_end_values():
    kappai = 1.0
    kappaf = 3.0
    centeri = 1.5
    centerf = 10.2
    tf = 1.0
    steps = 4
    force = VariableStiffnessCenterHarmonicForce(
        kappai=kappai, 
        kappaf=kappaf, 
        centeri=centeri,
        centerf=centerf,
        tf=tf, 
        steps=steps)
    t = torch.tensor(0.0)
    assert force.kappa(t) == kappai
    assert force.center(t) == centeri
    t = torch.tensor(tf)
    assert force.kappa(t) == kappaf
    assert force.center(t) == centerf


def test_stiffness_center_interpolation_2points():
    kappai = 2.0
    kappaf = 4.0
    centeri=1.0
    centerf=3.0
    tf = 1.0
    kl = [2.0, 4.0]
    cl = [1.0, 3.0]
    force = VariableStiffnessCenterHarmonicForce(
        kappai=kappai,
        kappaf=kappaf,
        centeri=centeri,
        centerf=centerf,
        tf=tf,
        k=kl, 
        center_list=cl,
        continuous=True
    )
    t = torch.tensor(0.5)

    assert force.kappa(t) == 3.0
    assert force.center(t) == 2.0



def test_stiffness_TSP():
    kappai = 2.0
    kappaf = 5.0
    centeri = 0.2
    centerf = 0.8
    tf = 1.0
    ko = 10.0
    co = 1.0
    kl = [ko]
    cl = [co]
    force = VariableStiffnessCenterHarmonicForce(
        kappai=kappai, 
        kappaf=kappaf, 
        centeri=centeri,
        centerf=centerf,
        tf=tf, 
        k=kl, 
        center_list=cl,
        continuous=False
    )
    t = torch.tensor(0.0)
    assert force.kappa(t) == kappai
    assert force.center(t) == centeri
    t = torch.tensor(0.01)
    assert force.kappa(t) == ko
    assert force.center(t) == co
    t = torch.tensor(1.0)
    assert force.kappa(t) == kappaf
    assert force.center(t) == centerf

