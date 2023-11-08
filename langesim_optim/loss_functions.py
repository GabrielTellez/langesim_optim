# Loss functions
from .simulator_forces import Simulator, device
import torch
import numpy as np

def gaussian(x, var=1.0, center=0.0):
    """Gaussian function with mean center and variance var."""
    if var < 0:
        raise ValueError("variance has to be positive")
    return (2.0 * torch.tensor(np.pi) * var) ** -0.5 * torch.exp(
        -((x-center)**2) / (2.0 * var)
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

