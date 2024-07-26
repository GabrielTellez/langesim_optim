import torch
import torch.nn as nn
from typing import Optional, List
import numpy as np
from .interpolator import interpolate


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


def validate_init_interpolation_list(
    yi, yf, ylist=None, steps=None, noise=0.25, ylist_name="ylist"
):
    """Validate and initialize a list of initial parameters to use for
    interpolation

    Args:
        yi (float): initial value
        yf (float): final value
        ylist (list, optional): list of values. Defaults to None.
        steps (int, optional): number of steps. Defaults to None.

    Raises:
        ValueError: If the list of values is not provided and the number of steps is not an integer greater than 1.

    Returns:
        ylist_ini: the list of values to interpolate between. To be used in nn.Parameter.

    """
    if steps is None and ylist is None:
        raise ValueError(
            "Please provide initial number of `steps` or a list of initial values for {ylist_name}"
        )
    if ylist is None:
        if steps < 1 or not (isinstance(steps, int)):
            raise ValueError("{steps=} has to be an integer greater or equal than 1.")
        else:
            # start with a linear interpolation between yi and yf plus some noise
            ylist_ini = torch.linspace(yi, yf, steps) + noise * (yf - yi) * torch.randn(
                steps, dtype=torch.float
            )
    else:
        if steps is None or steps == len(ylist):
            ylist_ini = torch.tensor(ylist, dtype=torch.float)
        else:
            raise ValueError(
                f"List of initial values of {ylist_name} {ylist=} has to be equal to provided {steps=}"
            )
    return ylist_ini


class VariableStiffnessHarmonicForce(BaseHarmonicForce):
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

        self.register_buffer("kappai", torch.tensor(kappai, dtype=torch.float))
        self.register_buffer("kappaf", torch.tensor(kappaf, dtype=torch.float))
        self.register_buffer("tf", torch.tensor(tf, dtype=torch.float))
        self.continuous = continuous
        k_ini = validate_init_interpolation_list(
            kappai, kappaf, k, steps, ylist_name="stiffness"
        )
        self.k = nn.parameter.Parameter(data=k_ini, requires_grad=True)

    def kappa(self, t):
        """
        Stiffness given as an interpolation between the values given by the list k.

        Args:
            t: time to compute the stiffness

        Returns:
            torch.tensor: the stiffness value at time t
        """
        return interpolate(
            t,
            yi=self.kappai,
            yf=self.kappaf,
            ti=0,
            tf=self.tf,
            ylist=self.k,
            continuous=self.continuous,
        )


class VariableCenterHarmonicForce(BaseHarmonicForce):
    """
    Harmonic oscillator force with a variable center that is a learnable parameter.
    """

    def __init__(
        self,
        centeri: float,
        centerf: float,
        tf: float,
        steps: Optional[int] = None,
        center_list: Optional[List] = None,
        kappa0: float = 1.0,
        continuous=True,
    ):
        """
        Initializes the force with a center that is defined at `steps` values
        of time and linearly interpolated between.

        Args:
            centeri (float): initial center
            centerf (float): final center
            tf (float): final time of the protocol
            steps (int, optional): number time steps where the center value is given
            center (list, optional): the initial center given by a list of
            `steps` values
            kappa0 (float): fixed stiffness
            countinuous (bool): whether the center is continuous at initial and final times,
            thus equal to centeri and centerf.
        """
        super().__init__()

        self.register_buffer("centeri", torch.tensor(centeri, dtype=torch.float))
        self.register_buffer("centerf", torch.tensor(centerf, dtype=torch.float))
        self.register_buffer("tf", torch.tensor(tf, dtype=torch.float))
        self.continuous = continuous
        center_ini = validate_init_interpolation_list(
            centeri, centerf, center_list, steps, ylist_name="center"
        )
        self.center_list = nn.parameter.Parameter(data=center_ini, requires_grad=True)
        self.kappa0 = kappa0

    def center(self, t):
        """
        Center given as an interpolation between the values given by the list center.

        Args:
            t: time to compute the center

        Returns:
            torch.tensor: the center value at time t
        """
        return interpolate(
            t,
            yi=self.centeri,
            yf=self.centerf,
            ti=0,
            tf=self.tf,
            ylist=self.center_list,
            continuous=self.continuous,
        )

    def kappa(self, t):
        """
        Stiffness given as a constant value kappa0.

        Args:
            t: time to compute the stiffness
        """
        return self.kappa0
