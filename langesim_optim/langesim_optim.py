import torch
import torch.nn as nn
from typing import Optional, List
import matplotlib.pyplot as plt
import numpy as np


device = "cuda" if torch.cuda.is_available() else "cpu"

# Simulator and Force classes


class BaseForce(nn.Module):
    """Base class for force on the particle.
    All forces should inherit from it.
    This base class returns a zero force at all times.
    """

    def __init__(self):
        super().__init__()

    def force(self, x: torch.tensor, t: torch.tensor) -> torch.tensor:
        """
        This sould be implemented by inherited classes.
        This base class returns a zero force.


        Args:
            x (torch.tensor): position where the force is applied.
            t (torch.tensor): time at which the force is applied.

        Returns:
            torch.tensor: The force.

        """
        return torch.tensor(0.0, dtype=torch.float)

    def potential(self, x: torch.tensor, t: torch.tensor) -> torch.tensor:
        """
        This sould be implemented by inherited classes.
        Although force = - grad potential, it is more efficient (speed and memory) to compute them apart analytically.
        This base class returns a zero potential.


        Args:
            x (torch.tensor): position where the potential energy is computed.
            t (torch.tensor): time at which the potential energy is computed.

        Returns:
            torch.tensor: The potential energy.

        """
        return torch.tensor(0.0, dtype=torch.float)

    def forward(self, x, t):
        return self.force(x, t)


class Simulator(nn.Module):
    """Simulator class for Langevin dynamics of a harmonic oscillator."""

    def __init__(
        self,
        dt=0.001,
        tot_steps=10_000,
        noise_scaler=1.0,
        force=BaseForce(),
        device=device,
        compute_work_heat=False,
    ):
        """Initializes the Simulator.

        Args:
            tot_sims (int, optional): total number of simulations. Defaults to 1000.
            dt (float, optional): time step. Defaults to 0.001.
            tot_steps (int, optional): total steps of each simulation. Defaults to 10000.
            noise_scaler (float, optional): brownian noise scale k_B T. Defaults to 1.0.
            force (BaseForce, optional): force excerted on the particle. Defaults to 0.0.
            device (str): device to run on (cpu or cuda).
            compute_work_heat (bool): compute work and heat during simulation. Defaults to False.
        """
        super().__init__()
        self.register_buffer("dt", torch.tensor(dt, dtype=torch.float))
        self.register_buffer("tot_steps", torch.tensor(tot_steps))
        self.register_buffer(
            "noise_scaler", torch.tensor(noise_scaler, dtype=torch.float)
        )
        self.device = device
        self.force = force
        self.compute_work_heat = compute_work_heat
        self.to(device)

    def forward(self, x0: torch.tensor) -> torch.tensor:
        """Forward pass:
        Performs in parallel N simulations with initial condition x0 where N=shape(x0)

        Args:
            x0 (torch.tensor): tensor containing the initial condition.
            Its shape should be (N,) where N is the number of parallel simulations to be run.

        Returns:
            torch.tensor: final positions of the particles, if compute_work_heat=False
            Tuple[torch.tensor, torch.tensor, torch.tensor]:  final positions of the particles, final work, final heat, if compute_work_heat=True
        """
        self.x = x0.to(self.device)
        N = x0.size(0)
        if self.compute_work_heat:
            self.w = torch.zeros_like(x0)  # work
            self.q = torch.zeros_like(x0)  # heat

        times = torch.arange(
            start=0.0,
            end=(self.tot_steps + 1.0) * self.dt,
            step=self.dt,
            dtype=torch.float,
        )
        for t_idx, t in enumerate(times[:-1]):
            xnew = (
                self.x
                + self.force(self.x, t) * self.dt
                + torch.randn(N, device=self.device)
                * (2.0 * self.dt * self.noise_scaler) ** 0.5
            )
            if self.compute_work_heat:
                next_t = times[t_idx + 1]
                self.w += self.force.potential(xnew, next_t) - self.force.potential(
                    xnew, t
                )
                self.q += self.force.potential(xnew, t) - self.force.potential(
                    self.x, t
                )
            self.x = xnew

        if self.compute_work_heat:
            return self.x, self.w, self.q
        else:
            return self.x


class BaseHarmonicForce(BaseForce):
    """
    Base class for harmonic oscillator force.
    """

    def __init__(self):
        """
        Initializes the force.
        """
        super().__init__()

    def kappa(self, t):
        """
        Stiffness of the harmonic force. This should be defined in inherited classes. By default it is 0.0.

        Args:
            t (torch.tensor): current time

        Returns:
            torch.tensor: the stiffness at time t.
        """
        return 0.0

    def center(self, t):
        """
        Center of the harmonic force. This should be defined in inherited classes. Defaults to 0.0.

        Args:
            t (torch.tensor): current time

        Returns:
            torch.tensor: the center at time t.
        """
        return 0.0

    def force(self, x: torch.tensor, t: torch.tensor) -> torch.tensor:
        """
        Computes and returns a harmonic oscillator force with time dependent
        stiffness kappa(t) and center(t)

        Args:
            x (torch.tensor): position where the force is applied.
            t (torch.tensor): time at which the force is applied.

        Returns:
            torch.tensor: The force.

        """
        return -self.kappa(t) * (x - self.center(t))

    def potential(self, x: torch.tensor, t: torch.tensor) -> torch.tensor:
        """
        Computes and returns a harmonic oscillator potential with time dependent
        stiffness kappa(t) and center(t)

        Args:
            x (torch.tensor): position where the force is applied.
            t (torch.tensor): time at which the force is applied.

        Returns:
            torch.tensor: The potential energy.

        """
        return 0.5 * self.kappa(t) * (x - self.center(t)) ** 2

class Interpolator():
    """Builds a linear interpolation function y(t) from a list of values ylist
    at times [ti, tf] and initial and final values yi, yf.
    """
    def __init__(self, yi, yf, ti, tf, ylist, continuous=True):
        """Initializes the interpolatior function y(t) from a list of values ylist
        at times [ti, tf] and initial and final values yi, yf.

        Args: 
            yi: initial value of y
            yf: final value of y
            ti: initial time
            tf: final time
            ylist (list): list of values of y at times [ti, tf]
            continuous (bool): whether the interpolator is continuous at ti and tf
        """ 
        self.yi = yi
        self.yf = yf
        self.ti = ti
        self.tf = tf
        self.continuous = continuous
        if continuous:
            self.ylist = [yi] + ylist + [yf]
        else:
            self.ylist = ylist
        self.N = len(self.ylist)
        if self.N > 1:
            self.dt = (self.tf - self.ti) / (self.N - 1)

    def __call__(self, t):
        if t <= self.ti:
            return self.yi
        if t >= self.tf:
            return self.yf
        
        if self.N == 1:
            # TSP case: return one constant value
            y = self.ylist[0]
        else:
            idx = int( (t - self.ti) / self.dt )
            if idx >= self.N - 1:
                y = self.ylist[N - 1]
            else:
                t1 = idx * self.dt + self.ti
                y = self.ylist[idx] + (self.ylist[idx + 1] - self.ylist[idx]) * (t - t1) * self.dt**-1
        
        return y

def make_interpolator(yi, yf, ti, tf, ylist, continuous=True):
    """Obsolete: refactored as class Interpolator.
    
    Builds a linear interpolation function y(t) from a list of values ylist
    at times [ti, tf] and initial and final values yi, yf.

    Args: 
        yi: initial value of y
        yf: final value of y
        ti: initial time
        tf: final time
        ylist (list): list of values of y at times [ti, tf]
        continuous (bool): whether the interpolator is continuous at ti and tf
    """ 
    def interpolator(t, yi=yi, yf=yf, ti=ti, tf=tf, ylist=ylist, continuous=continuous):
        if t <= ti:
            return yi
        if t >= tf:
            return yf

        if continuous:
            yl = [yi] + ylist + [yf]
        else:
            yl = ylist
    
        N = len(yl)
        if N == 1:
            # TSP case: only one constant value
            y = yl[0]
        else:
            dt = (tf - ti) / (N - 1)
            idx = int( (t - ti) / dt)
            if idx >= N - 1:
                y = yl[N - 1]
            else:
                t1 = idx * dt + ti
                y = yl[idx] + (yl[idx + 1] - yl[idx]) * (t - t1) * dt**-1
        
        return y
    
    return interpolator

class VariableHarmonicForce(BaseHarmonicForce):
    """
    Harmonic oscillator force with a variable stiffness that is a learnable parameter.
    The stiffness is made of linear interpolation of segments.
    The center is fixed at zero.
    To do: change to a variable center also.
    """

    def __init__(
        self,
        kappai: float,
        kappaf: float,
        tf: float,
        steps: Optional[int] = None,
        k: Optional[List] = None,
        continuous=True,
    ):
        """
        Initializes the force with a stiffness that is defined at `steps` values
        of time and linearly interpolated between.

        Args:
            kappai (float): initial stiffness
            kappaf (float): final stiffness
            tf (float): final time of the protocol
            steps (int, optional): number time steps where the stiffness value is given
            k (list, optional): the initial stiffness given by a list of `steps` values
            countinuous (bool): whether the stiffness is continuous at initial and final times,
            thus equal to kappai and kappaf.
        """
        super().__init__()

        if steps is None and k is None:
            raise ValueError(
                "Please provide initial number of `steps` or a list of initial values `k`"
            )

        self.register_buffer("kappai", torch.tensor(kappai, dtype=torch.float))
        self.register_buffer("kappaf", torch.tensor(kappaf, dtype=torch.float))
        self.register_buffer("tf", torch.tensor(tf, dtype=torch.float))
        self.continuous = continuous

        if k is None:
            if steps < 1 or not (isinstance(steps, int)):
                raise ValueError(
                    "{steps=} has to be an integer greater or equal than 1."
                )
            else:
                # start with a linear interpolation between kappai and kappaf plus some noise
                k_ini = torch.linspace(kappai, kappaf, steps) + 0.25 * (
                    kappaf - kappai
                ) * torch.randn(steps, dtype=torch.float)
                self.k = nn.parameter.Parameter(data=k_ini, requires_grad=True)
        else:
            if steps is None or steps == len(k):
                self.k = nn.parameter.Parameter(
                    data=torch.tensor(k, dtype=torch.float), requires_grad=True
                )
            else:
                raise ValueError(
                    f"List of initial values of the stiffness {k=} has to be equal to provided {steps=}"
                )

    def kappa(self, t):
        """
        Stiffness given as an interpolation between the values given by the list k.

        Args:
            t: time to compute the stiffness

        Returns:
            torch.tensor: the stiffness value at time t
        """
        if t <= torch.tensor(0.0):
            return self.kappai
        if t >= self.tf:
            return self.kappaf

        N = len(self.k)
        if self.continuous:
            dt = self.tf / (N + 1)
            idx = int(t / dt) - 1
            if idx >= 0 and idx < N - 1:
                t1 = (idx + 1) * dt
                k = self.k[idx] + (self.k[idx + 1] - self.k[idx]) * (t - t1) * dt**-1
            else:
                # Interpolate at the edges between kappai and kappaf
                if t >= 0.0 and t < dt:
                    k = self.kappai + (self.k[0] - self.kappai) * t * dt**-1
                if t >= N * dt and t <= self.tf:
                    k = (
                        self.k[N - 1]
                        + (self.kappaf - self.k[N - 1]) * (t - N * dt) * dt**-1
                    )
        else:  # non continuous: no interpolation at the edges with kappai and kappaf
            # If N == 1 dt=inf but it works because idx=0 always
            dt = self.tf / (N - 1)
            idx = int(t / dt)
            # print(f"{N=}, {dt=}, {idx=}")
            if idx >= N - 1:
                k = self.k[N - 1]
            else:
                t1 = idx * dt
                k = self.k[idx] + (self.k[idx + 1] - self.k[idx]) * (t - t1) * dt**-1

        return k


# Loss functions


def gaussian(x, var):
    if var < 0:
        raise ValueError("variance has to be positive")
    return (2.0 * torch.tensor(np.pi) * var) ** -0.5 * torch.exp(
        -(x**2) / (2.0 * var)
    )


def FT_pdf(
    pdf, kf, scale=5.0, steps=10_000, kFs=None, kFsteps=100, args=(), device=device
):
    """
    Fourier transform (FT) of a probability density function (pdf).
    Assumes that the pdf is centered at 0.0.

    Args:
        pdf (callable): the pdf whose Fourier transform is to be computed.
        kf (float): inverse of the variance of the pdf
        scale (float): how many kf's should we use to compute the FT.
        steps (float): steps of the x discretization.
        kFs (torch.tensor): range for the k values.
        kFsteps (optional, int): size for kFS if not provided.
        args (optional, tuple): other arguments taken by pdf.
        device (str): device to run on cpu or cuda.

    Returns:
        tuple(torch.tensor, torch.tensor): FT of pdf, kFs
    """

    if kFs is None:
        kFinit = 0.0
        kFend = scale * kf**0.5
        dkF = (kFend - kFinit) / kFsteps
        kFs = torch.arange(start=kFinit, end=kFend, step=dkF, device=device)
    xmax = scale * kf**-0.5
    dx = 2.0 * xmax / steps
    xs = torch.arange(start=-xmax, end=xmax, step=dx, device=device)
    kx = kFs.unsqueeze(dim=1) @ xs.unsqueeze(dim=0)
    integrand = pdf(xs, *args) * (1.0j * kx).exp()
    integral = torch.trapezoid(integrand, dx=dx, dim=1)
    return integral, kFs


def char_fn(xf, kf, scale=5.0, kFsteps=1000, device=device):
    """
    Computes the characteristic function from samples xf.

    Args:
        xf (torch.tensor): samples of positions
        kf (float): inverse variance of x
        scale (float): how many kf's should the values of k spread.

    Returns:
        tuple(torch.tensor, torch.tensor): characteristic function, corresponding values of k
    """

    kFinit = 0.0
    kFend = scale * kf**0.5
    dkF = (kFend - kFinit) / kFsteps
    kFs = torch.arange(start=kFinit, end=kFend, step=dkF, device=device)
    kx = kFs.unsqueeze(dim=1) @ xf.unsqueeze(dim=0)
    expikx = (1.0j * kx).exp()
    char_P_k = torch.mean(expikx, dim=1)
    return char_P_k, kFs


def loss_fn_k(
    xf,
    kf,
    ki=1.0,
    sim: Simulator = None,
    device=device,
    scale=5.0,
    kFsteps=1000,
    x_steps=10_000,
):
    """
    Loss function comparing the L2 mean square loss of the characteristic function of the pdf
    to the target normal distribution with variance 1/kf.
    ki and sim are not used, but kept optional to have the same API for all loss functions.
    """

    char_P_k, kFs = char_fn(xf, kf, scale, kFsteps, device=device)
    char_P_k_teo, _ = FT_pdf(
        pdf=gaussian,
        kf=kf,
        scale=scale,
        steps=x_steps,
        kFs=kFs,
        args=(1.0 / kf,),
        device=device,
    )

    # return torch.abs((char_P_k - char_P_k_teo)**2).mean(), char_P_k, char_P_k_teo, kFs
    return torch.abs((char_P_k - char_P_k_teo) ** 2).mean()


def loss_fn_variance(
    xf: torch.tensor, kf: float, ki: float, sim: Simulator, device=device
):
    """
    Loss function comparing square difference of theoretical variance vs computed variance of xf.
    """

    var_theo = 1.0 / kf
    var_exp = xf.var()
    return (var_exp - var_theo) ** 2


def loss_fn_grad_k(ki, kf, sim: Simulator):
    """Penalizes large variations of kappa."""
    if sim.force.continuous:
        # Include edge values kappai and kappaf
        ks = torch.cat(
            (
                torch.tensor([ki], device=device),
                sim.force.k,
                torch.tensor([kf], device=device),
            )
        )
    else:
        ks = sim.force.k
    return torch.mean(ks.diff() ** 2)


def loss_fn_control_k_vars(
    xf,
    kf,
    ki,
    sim: Simulator,
    device=device,
    scale=5.0,
    kFsteps=1_000,
    x_steps=1_000,
    blend=1e-3,
):
    """Loss function that penalizes strong variations of consecutive values of kappa."""
    loss = (1.0 - blend) * loss_fn_k(
        xf, kf, ki, sim, device=device, scale=scale, kFsteps=kFsteps, x_steps=x_steps
    ) + blend * loss_fn_grad_k(ki, kf, sim)
    return loss


def loss_fn_work(xf, kf, ki, sim: Simulator, device=device):
    """
    Loss function to minimize work with respect to lower bound Delta F.
    Note: xf is not used but needed to keep the same API for train loop.
    """
    # DeltaF = 0.5*torch.log(torch.tensor(kf * ki**-1, device=device))
    return sim.w.mean()  # - DeltaF # No se necesita DeltaF porque es constante


def loss_fn_eq_work(
    xf,
    kf,
    ki,
    sim: Simulator,
    device=device,
    scale=5.0,
    kFsteps=1_000,
    x_steps=1_000,
    blend=1e-3,
):
    """Loss function to minimize work and distance to equilibrium"""
    loss = (1.0 - blend) * loss_fn_k(
        xf, kf, ki, sim, device=device, scale=scale, kFsteps=kFsteps, x_steps=x_steps
    ) + blend * loss_fn_work(xf, ki, kf, sim)
    return loss


# Utilities: train loop and plot histograms


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
    # Este código solo sirve para continuous force. Para revisar.
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

    kappa_numpy = Interpolator(yi=ki, yf=kf, ti=0.0, tf=tf, ylist=k, continuous=sim.force.continuous)

    return k, ki, kf, tf, kappa_numpy
