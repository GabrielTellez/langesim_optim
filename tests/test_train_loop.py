import torch
from langesim_optim import (
    train_loop,
    Simulator,
    VariableStiffnessHarmonicForce,
    VariableCenterHarmonicForce,
    loss_fn_k,
    loss_fn_mean,
    device,
)
from torch.optim import SGD
import numpy as np
from scipy.optimize import fsolve


def test_train_loop_output_type_length():
    epochs = 5
    ki = 1.0
    kf = 2.0
    dt = 0.01
    tf = 0.1
    tot_steps = int(tf / dt)
    tot_sims = 100
    lr = 1.0

    force = VariableStiffnessHarmonicForce(kappai=ki, kappaf=kf, tf=tf, steps=3)
    sim = Simulator(dt=dt, tot_steps=tot_steps, force=force, device=device)

    optimizer = SGD(params=sim.parameters(), lr=lr)
    loss_fn = loss_fn_k

    lossi = train_loop(
        epochs=epochs,
        sim=sim,
        tot_sims=tot_sims,
        ki=ki,
        kf=kf,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
    )
    # Test case 1: Test the output type of the function
    assert isinstance(lossi, list)

    # Test case 2: Test the output length of the function
    assert len(lossi) == epochs


def test_train_loop_TSP():
    """Test a train loop on the two-step protocol (TSP): only one value of k is learned"""

    def theo_var(t, ko, ki):
        """Theoretical variance"""
        return np.exp(-2.0 * ko * t) / ki + (1.0 - np.exp(-2.0 * ko * t)) / ko

    def eq_to_solve(k, ki, kf, tf):
        return 1 / kf - theo_var(tf, k, ki)

    def ktheo(ki, kf, tf):
        return fsolve(eq_to_solve, kf, args=(ki, kf, tf))[0]

    epochs = 20
    ki = 0.5
    kf = 1.0
    dt = 0.001
    tf = 0.100
    tot_steps = int(tf / dt)
    tot_sims = 100_000

    lr = 100.0
    force = VariableStiffnessHarmonicForce(
        kappai=ki, kappaf=kf, tf=tf, k=[kf], continuous=False
    )
    sim = Simulator(force=force, dt=dt, tot_steps=tot_steps, device=device)

    optimizer = torch.optim.SGD(params=sim.parameters(), lr=lr)
    loss_fn = loss_fn_k

    lossi = train_loop(
        epochs=epochs,
        sim=sim,
        tot_sims=tot_sims,
        ki=ki,
        kf=kf,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
    )

    loss_mean = np.mean(lossi[-5:])
    assert loss_mean < 1e-4, f"{loss_mean=} > 1e-4"
    ktheo_val = ktheo(ki, kf, tf)
    klearned = sim.force.k[0].item()
    assert np.isclose(klearned, ktheo_val, rtol=1e-2), f"{klearned=} != {ktheo_val=}"


def test_train_loop_TSP_center_loss_fn_mean():
    """Test a train loop on the two-step protocol (TSP) for variable center:
    only one value of center is learned.
    Test using loss_fn_mean"""

    def center_theo(t, ci, cf, k):
        """Theoretical center"""
        return (cf - ci * np.exp(-k * t)) / (1 - np.exp(-k * t))

    epochs = 50
    ci = 1.0
    cf = 5.0
    dt = 0.001
    tf = 0.100
    ki = 1.0
    tot_steps = int(tf / dt)
    tot_sims = 100_000

    lr = 10.0
    force = VariableCenterHarmonicForce(
        centeri=ci, centerf=cf, tf=tf, center_list=[cf], kappa0=ki, continuous=False
    )
    sim = Simulator(dt=dt, tot_steps=tot_steps, force=force, device=device)

    optimizer = torch.optim.SGD(params=sim.parameters(), lr=lr)
    loss_fn = loss_fn_mean
    lossi = train_loop(
        epochs=epochs,
        sim=sim,
        tot_sims=tot_sims,
        ki=ki,
        kf=ki,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        ci=ci,
        cf=cf,
    )

    loss_mean = np.mean(lossi[-5:])
    assert loss_mean <= 1e-4, f"{loss_mean=} > 1e-4"
    c_theo_val = center_theo(tf, ci, cf, ki)
    c_learned = sim.force.center_list[0].item()
    assert np.isclose(
        c_learned, c_theo_val, rtol=1e-2
    ), f"{c_learned=} != {c_theo_val=}"


def test_train_loop_TSP_center_loss_fn_k():
    """Test a train loop on the two-step protocol (TSP) for variable center:
    only one value of center is learned.
    Test using loss_fn_k"""

    def center_theo(t, ci, cf, k):
        """Theoretical center"""
        return (cf - ci * np.exp(-k * t)) / (1 - np.exp(-k * t))

    epochs = 50
    ci = 1.0
    cf = 3.0  # does not converge well for large cf (ex. cf=8.0) without adjusting scale, kFs, x_steps
    dt = 0.001
    tf = 0.100
    ki = 1.0
    tot_steps = int(tf / dt)
    tot_sims = 100_000

    lr = 100.0
    force = VariableCenterHarmonicForce(
        centeri=ci, centerf=cf, tf=tf, center_list=[cf], kappa0=ki, continuous=False
    )
    sim = Simulator(dt=dt, tot_steps=tot_steps, force=force, device=device)

    optimizer = torch.optim.SGD(params=sim.parameters(), lr=lr)
    loss_fn = loss_fn_k
    lossi = train_loop(
        epochs=epochs,
        sim=sim,
        tot_sims=tot_sims,
        ki=ki,
        kf=ki,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        ci=ci,
        cf=cf,
    )

    loss_mean = np.mean(lossi[-5:])
    assert loss_mean <= 1e-4, f"{loss_mean=} > 1e-4"
    c_theo_val = center_theo(tf, ci, cf, ki)
    c_learned = sim.force.center_list[0].item()
    assert np.isclose(
        c_learned, c_theo_val, rtol=1e-2
    ), f"{c_learned=} != {c_theo_val=}"
