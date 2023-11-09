# Utilities: train loop and plot histograms

import torch
import numpy as np
from .simulator_forces import Simulator, device
from .interpolator import Interpolator
from .loss_functions import gaussian
from typing import Optional, List
import matplotlib.pyplot as plt


def train_loop(
    epochs,
    sim: Simulator,
    tot_sims,
    ki,
    kf,
    optimizer,
    loss_fn,
    scheduler=None,
    device=device,
    init_epoch=0,
    **kwargs,  # keyword arguments for loss_fn
) -> List:
    """Trains model sim, returns the loss list.
    Perform the training loop for a given number of epochs.

    Args:
        epochs (int): The number of training epochs.
        sim (Simulator): The simulator object used for training.
        tot_sims (int): The total number of simulations.
        ki: The initial value of stiffness.
        kf: The final value of stiffness.
        optimizer: The optimizer used for training.
        scheduler: The learning rate scheduler.
        loss_fn: The loss function used for training. Takes as arguments (xf, kf, ki, sim, device, **kwargs)
        device (str, optional): The device where the training will be performed. Defaults to the 'device' constant.
        init_epoch(int, optional): initial epoch number for this round. Defaults to 0.
        **kwargs: Additional keyword arguments specific to the loss function.

    Returns:
        List: A list of the loss function values for each epoch.

    """
    lossi = []
    sim.train()
    for epoch in range(epochs):
        # Erase previous gradients
        optimizer.zero_grad()

        # forward pass
        x0 = torch.randn(tot_sims, device=device) * ki**-0.5
        if sim.compute_work_heat:
            xf, wf, qf = sim(x0)
        else:
            xf = sim(x0)

        loss = loss_fn(xf, kf, ki, sim, device=device, **kwargs)
        lossi.append(loss.item())

        # backward pass
        loss.backward()

        # Optimize by gradient descent
        optimizer.step()

        # Schedule the learning rate
        if scheduler is not None:
            scheduler.step()

        print(f"Epoch={epoch+init_epoch:<5} | loss={loss.item():.8E}")

    return lossi


def plot_test_hist(
    tot_sims: int,
    ki: float,
    kf: float,
    sim: Simulator,
    device=device,
    xrange=1.0,
    xpoints=200,
    init_label="Initial equilibrium PDF",
    expect_label="Expected final equilibrium PDF",
    final_label="Final PDF after simulation",
    title="Probability density function (PDF) of the position",
    grid=False,
    bins=200,
):
    """Plots a histogram of final PDF after a forward pass on the Simulator sim of an initial equilibrium PDF with stiffness ki.

    Args:
        tot_sims (int): total simulations to be performed.
        ki (float): initial stiffness
        kf (float): final stiffness
        sim (Simulator): simulator module
        device (str): device to run on cpu or cuda
    """
    with torch.inference_mode():
        x0 = torch.randn(tot_sims, device=device) * ki**-0.5
        if sim.compute_work_heat:
            xtest, wf, qf = sim(x0)
        else:
            xtest = sim(x0)

    fig = plt.figure()
    plt.hist(xtest.cpu().detach(), bins=bins, density=True, label=final_label)
    xs = torch.linspace(-xrange, xrange, xpoints)
    pdf_final = gaussian(xs, var=1.0 / kf).detach()
    plt.plot(xs, pdf_final, label=expect_label)
    pdf_i = gaussian(xs, var=1.0 / ki).detach()
    plt.plot(xs, pdf_i, label=init_label)
    plt.title(title)
    plt.legend(fontsize="small")
    plt.grid(grid)
    plt.xlim(-xrange, xrange)
    plt.xlabel("x")
    plt.ylabel("P(x)")
    return fig


def plot_protocols(
    sim: Simulator,
    ki,
    kf,
    tf,
    k_comp=None,
    t_steps=200,
    sim_label="Trained",
    comp_label="Theoretical",
    title=r"Stiffness $\kappa$",
    times=None,
    yrange=None,
    grid=True,
    time_ticks=None,
    y_ticks=None,
    y_ticklabels=None,
):
    if times is None:
        times = np.linspace(0, tf, t_steps)
    kappa = np.array([sim.force.kappa(t).item() for t in times])

    fig = plt.figure()
    plt.plot(times, kappa, label=sim_label)
    if k_comp is not None:
        k_comp_v = np.vectorize(k_comp)
        plt.plot(times, k_comp_v(times, tf, ki, kf), label=comp_label)

    if yrange is not None:
        plt.ylim(yrange)
    plt.xlabel("t")
    plt.ylabel(r"$\kappa$")
    plt.title(title)
    plt.legend()
    plt.grid(grid)
    if time_ticks:
        plt.xticks(time_ticks)
    if y_ticks:
        plt.yticks(y_ticks)
    if y_ticklabels:
        plt.gca().set_yticklabels(y_ticklabels)
    return fig


def train_loop_snapshots(
    epochs,
    sim: Simulator,
    tot_sims,
    ki,
    kf,
    tf,
    optimizer,
    loss_fn,
    scheduler=None,
    device=device,
    snapshot_step=1,
    xrange=1.0,
    bins=200,
    times=None,
    time_ticks=None,
    yrange=None,
    y_ticks=None,
    y_ticklabels=None,
    grid_histo=False,
    grid_protocol=True,
    **kwargs,  # keyword arguments for loss_fn
) -> List:
    """Trains model sim, returns the loss list.
    Perform the training loop for a given number of epochs.

    Args:
        epochs (int): The number of training epochs.
        sim (Simulator): The simulator object used for training.
        tot_sims (int): The total number of simulations.
        ki: The initial value of stiffness.
        kf: The final value of stiffness.
        optimizer: The optimizer used for training.
        scheduler: The learning rate scheduler.
        loss_fn: The loss function used for training. Takes as arguments (xf, kf, ki, sim, device, **kwargs)
        device (str, optional): The device where the training will be performed. Defaults to the 'device' constant.
        snapshot_step (int, optional): Makes a snapshot of the protocol and graphs of the final distribution every snapshot_step epochs. Defaults to 1.
        **kwargs: Additional keyword arguments specific to the loss function.

    Returns:
        Tuple: (lossl, protocols, plots), with
        lossl = A list of the loss function values for each epoch.
        protocols = A list of the snapshot of the protocols
        plots = A list of the plots of the final distribution at each snapshot step.

    """
    lossl = []
    protocols = [
        plot_protocols(
            sim,
            ki,
            kf,
            tf,
            k_comp=None,
            t_steps=200,
            sim_label=f"Trained at epoch=0",
            comp_label="Theoretical",
            title=r"Stiffness $\kappa$",
            times=times,
            yrange=yrange,
            grid=grid_protocol,
            y_ticks=y_ticks,
            y_ticklabels=y_ticklabels,
        )
    ]
    plots = [
        plot_test_hist(
            tot_sims,
            ki,
            kf,
            sim,
            device=device,
            xrange=xrange,
            grid=grid_histo,
            bins=bins,
        )
    ]
    tot_snapshots = int(epochs / snapshot_step)
    last_epochs = epochs % tot_snapshots
    epoch = 0
    for snapshots in range(tot_snapshots + 1):
        epochs_to_do = snapshot_step if snapshots < tot_snapshots else last_epochs
        if epochs_to_do == 0:
            break
        lossi = train_loop(
            epochs_to_do,
            sim=sim,
            tot_sims=tot_sims,
            ki=ki,
            kf=kf,
            optimizer=optimizer,
            loss_fn=loss_fn,
            scheduler=scheduler,
            device=device,
            init_epoch=snapshots * snapshot_step,
            **kwargs,
        )
        epoch += epochs_to_do
        protocol = plot_protocols(
            sim,
            ki,
            kf,
            tf,
            k_comp=None,
            t_steps=200,
            sim_label=f"Trained at epoch={epoch}",
            comp_label="Theoretical",
            title=r"Stiffness $\kappa$",
            times=times,
            yrange=yrange,
            grid=grid_protocol,
            y_ticks=y_ticks,
            y_ticklabels=y_ticklabels,
        )
        pl = plot_test_hist(
            tot_sims,
            ki,
            kf,
            sim,
            device=device,
            xrange=xrange,
            grid=grid_histo,
            bins=bins,
        )
        lossl = lossl + lossi
        plots.append(pl)
        protocols.append(protocol)

    return lossl, protocols, plots


def k_from_sim(sim: Simulator):
    """
    Extracts ki, kf, tf and k(t) from a simulator.
    """
    # Este código solo sirve para discontinuous force. Para revisar.
    k = sim.force.k.cpu().detach().numpy()
    ki = float(sim.force.kappai.cpu().detach().numpy())
    kf = float(sim.force.kappaf.cpu().detach().numpy())
    tf = float(sim.force.tf.cpu().detach().numpy())
    #    if sim.force.continuous:
    #        k = [ki] + k + [kf]

    # def kappa_numpy(t, tf=tf, ki=ki, kf=kf, k=k):
    #     """
    #     Stiffness given as an interpolation between the values given by the list k.

    #     Args:
    #         t: time to compute the stiffness

    #     Returns:
    #         float: the stiffness value at time t
    #     """

    #     # print(f"{k=}, {type(k)=}")

    #     if t <= 0.0:
    #         return ki
    #     if t >= tf:
    #         return kf

    #     N = len(k)

    #     dt = tf / (N + 1)
    #     idx = int(t / dt) - 1
    #     if idx >= 0 and idx < N - 1:
    #         t1 = (idx + 1) * dt
    #         kap = k[idx] + (k[idx + 1] - k[idx]) * (t - t1) * dt**-1
    #     else:
    #         # Interpolate at the edges between ki and kf
    #         if t >= 0.0 and t < dt:
    #             kap = ki + (k[0] - ki) * t * dt**-1
    #         if t >= N * dt and t <= tf:
    #             kap = k[N - 1] + (kf - k[N - 1]) * (t - N * dt) * dt**-1

    #     return kap

    kappa_numpy = Interpolator(
        yi=ki, yf=kf, ti=0.0, tf=tf, ylist=k, continuous=sim.force.continuous
    )

    return k, ki, kf, tf, kappa_numpy
