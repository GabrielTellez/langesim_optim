import torch
import torch.nn as nn
from typing import Optional, List
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





