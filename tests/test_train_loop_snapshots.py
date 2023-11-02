import torch
from langesim_optim import (
    train_loop_snapshots,
    Simulator,
    VariableHarmonicForce,
    loss_fn_k,
    device,
)
from torch.optim import SGD
import matplotlib.pyplot as plt


def test_train_loop_snapshots():
    # Test case 1: Test the output type of the function
    epochs = 10
    ki = 1.0
    kf = 2.0
    dt = 0.01
    tf = 0.1
    tot_steps = int(tf / dt)
    tot_sims = 100
    lr = 1.0
    snapshot_step = 2

    force = VariableHarmonicForce(kappai=ki, kappaf=kf, tf=tf, steps=3)
    sim = Simulator(dt=dt, tot_steps=tot_steps, force=force, device=device)

    optimizer = SGD(params=sim.parameters(), lr=lr)
    loss_fn = loss_fn_k

    lossl, protocols, plots = train_loop_snapshots(
        epochs=epochs,
        sim=sim,
        tot_sims=tot_sims,
        ki=ki,
        kf=kf,
        tf=tf,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        snapshot_step=snapshot_step,
    )
    assert isinstance(lossl, list)
    assert all(isinstance(protocol, plt.Figure) for protocol in protocols)
    assert all(isinstance(plot, plt.Figure) for plot in plots)

    # Test case 2: Test the output length of the function
    assert len(lossl) == epochs
    assert len(protocols) == epochs // snapshot_step + 1
    assert len(plots) == epochs // snapshot_step + 1
