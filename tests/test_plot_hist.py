import torch
from langesim_optim import (
    plot_test_hist,
    Simulator,
    plot_protocols,
    VariableStiffnessHarmonicForce,
)
import matplotlib.pyplot as plt
import numpy as np


def test_plot_test_hist():
    tot_sims = 100
    ki = 0.5
    kf = 1.0
    dt = 0.01
    tf = 0.100
    tot_steps = int(tf / dt)
    sim = Simulator(dt=dt, tot_steps=tot_steps)
    fig = plot_test_hist(tot_sims=tot_sims,
    sim=sim,
    ki=ki,
    kf=kf,
    )
    # Test case 1: Test the output type of the function
    assert isinstance(fig, plt.Figure)

    # Test case 2: Test the output content of the function
    assert len(fig.axes) == 1
    assert len(fig.axes[0].lines) == 2
    assert len(fig.axes[0].patches) > 0


def test_plot_protocols():
    """Test the plot without a comparison protocol"""
    ki = 0.5
    kf = 1.0
    dt = 0.01
    tf = 0.100
    tot_steps = int(tf / dt)
    force = VariableStiffnessHarmonicForce(kappai=ki, kappaf=kf, tf=tf, steps=3)
    sim = Simulator(dt=dt, tot_steps=tot_steps, force=force)
    fig = plot_protocols(sim, ki, kf, tf, k_comp=None)
    # Test case 1: Test the output type of the function
    assert isinstance(fig, plt.Figure)

    # Test case 2: Test the output content of the function
    assert len(fig.axes) == 1
    assert len(fig.axes[0].lines) == 1
    assert fig.axes[0].get_xlabel() == "t"
    assert fig.axes[0].get_ylabel() == r"$\kappa$"
    assert fig.axes[0].get_title() == r"Stiffness $\kappa$"
    assert fig.axes[0].legend_ is not None


def test_plot_protocols_comp():
    """Test the plot with a comparison protocol"""
    ki = 0.5
    kf = 1.0
    dt = 0.01
    tf = 0.100
    tot_steps = int(tf / dt)
    force = VariableStiffnessHarmonicForce(kappai=ki, kappaf=kf, tf=tf, steps=4)
    sim = Simulator(dt=dt, tot_steps=tot_steps, force=force)
    k_comp = lambda t, tf, ki, kf: ki + (kf - ki) * t / tf
    fig = plot_protocols(sim, ki, kf, tf, k_comp=k_comp)
    assert len(fig.axes[0].lines) == 2
