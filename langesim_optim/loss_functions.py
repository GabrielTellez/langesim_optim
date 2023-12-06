# Loss functions
from .simulator_forces import Simulator, device
import torch
import numpy as np


def gaussian(x, var=1.0, center=0.0):
    """Gaussian function with mean center and variance var."""
    if var < 0:
        raise ValueError("variance has to be positive")
    return (2.0 * torch.tensor(np.pi) * var) ** -0.5 * torch.exp(
        -((x - center) ** 2) / (2.0 * var)
    )


def FT_pdf(
    pdf, kf, scale=5.0, steps=10_000, kFs=None, kFsteps=100, args=(), device=device, **kwargs,
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
        **kwargs: extra keywords arguments ignored.

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


def char_fn(xf, kf, scale=5.0, kFsteps=1000, device=device, **kwargs):
    """
    Computes the characteristic function from samples xf.

    Args:
        xf (torch.tensor): samples of positions
        kf (float): inverse variance of x
        scale (float): how many kf's should the values of k spread.
        **kwargs: extra keywords arguments ignored.

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
    device=device,
    scale=5.0,
    kFsteps=1000,
    x_steps=10_000,
    **kwargs,
):
    """
    Loss function comparing the L2 mean square loss of the characteristic function of the pdf
    to the target normal distribution with variance 1/kf.
    Any additional keyword arguments are ignored.

    Args:
        xf (torch.Tensor): The final position of the particles.
        kf (float): The final stiffness value.
        device (torch.device, optional): The device on which to perform the computations. Defaults to the current device.
        scale (float, optional): how many kf's should the values of k spread. Defaults to 5.0.
        x_steps (float): steps of the x discretization.
        kFs (torch.tensor): range for the k values.
        kFsteps (optional, int): size for kFS if not provided.
        **kwargs: extra keywords arguments ignored.
    Returns:
        float: The MSE loss between the computed characteristic function and the target characteristic function.
    """

    char_P_k, kFs = char_fn(xf=xf, kf=kf, scale=scale, kFsteps=kFsteps, device=device)
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
    xf: torch.tensor, 
    kf: float, 
    **kwargs,
):
    """
    Loss function comparing square difference of theoretical variance 1/kf vs computed variance of xf.

    Args:
        xf (torch.Tensor): The final position of the particles.
        kf (float): The final stiffness value.
        **kwargs: extra keywords arguments ignored.

    Returns:
        float: The squared difference between the computed variance and the theoretical variance.
  
    """

    var_theo = 1.0 / kf
    var_exp = xf.var()
    return (var_exp - var_theo) ** 2

def loss_fn_mean(
    xf: torch.tensor, 
    cf: float, 
    **kwargs
):
    """
    Loss function comparing square difference of theoretical mean cf vs computed mean of xf.
    
    Args:
        xf (torch.Tensor): The final position of the particles.
        cf (float): The theoretical mean value.
        **kwargs: extra keywords arguments ignored.

    Returns:
        float: The squared difference between the computed mean and the theoretical mean.
    """

    mean_theo = cf
    mean_exp = xf.mean()
    return (mean_exp - mean_theo) ** 2

def loss_fn_grad_k(ki, kf, sim: Simulator, **kwargs):
    """
    Penalizes large variations of kappa (the stiffness of the harmonic potential).
    If the force is continuous, the function includes the edge values `ki` and `kf` in the computation.
    The function computes the difference between consecutive elements of `ks` (which represents kappa values), 
    squares these differences, and then returns the mean of these squared differences.

    Args:
        ki (float): The initial stiffness value.
        kf (float): The final stiffness value.
        sim (Simulator): The simulator object.
        **kwargs: extra keywords arguments ignored.
    Returns:
        float: The mean of the squared differences between consecutive kappa values.
"""
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
    **kwargs,
):
    """Loss function that minimizes distance to equilibrium and penalizes strong variations of consecutive values of
    kappa. This function computes a weighted sum of two loss functions: `loss_fn_k` and `loss_fn_grad_k`. 
    The weight of each loss function is determined by the `blend` parameter.

    Args:
        xf (torch.Tensor): The final position of the particles.
        kf (float): The final stiffness value.
        ki (float): The initial stiffness value.
        sim (Simulator): The simulator object.
        device (torch.device, optional): The device on which to perform the computations. Defaults to the current device.
        scale (float, optional): how many kf's should the values of k spread. Defaults to 5.0.
        x_steps (float): steps of the x discretization.
        kFs (torch.tensor): range for the k values.
        kFsteps (optional, int): size for kFS if not provided.
        blend (float, optional): The blending factor for the two loss functions. Defaults to 1e-3.
        **kwargs: extra keywords arguments ignored.

    Returns:
        float: The weighted sum of the two loss functions.
   
    """
    loss = (1.0 - blend) * loss_fn_k(
        xf=xf, kf=kf, ki=ki, device=device, scale=scale, kFsteps=kFsteps, x_steps=x_steps
    ) + blend * loss_fn_grad_k(ki=ki, kf=kf, sim=sim)
    return loss


def loss_fn_work(sim: Simulator, **kwargs):
    """
    Loss function to minimize work with respect to lower bound Delta F.

    This function returns the mean work done during the simulation, which is stored in `sim.w`. 
    The goal is to minimize this work with respect to a lower bound Delta F.

    Args:
        sim (Simulator): The simulator object, which contains the work done during the simulation.
        **kwargs: extra keywords arguments ignored.

    Returns:
        float: The mean work done during the simulation.

    """
    # DeltaF = 0.5*torch.log(torch.tensor(kf * ki**-1, device=device))
    return sim.w.mean()  # - DeltaF # No se necesita DeltaF porque es constante


def loss_fn_eq_work(
    xf,
    kf,
    sim: Simulator,
    device=device,
    scale=5.0,
    kFsteps=1_000,
    x_steps=1_000,
    blend=1e-3,
    **kwargs,
):
    """Loss function to minimize work and distance to equilibrium.
    This function computes a weighted sum of two loss functions: `loss_fn_k` and `loss_fn_work`. 
    The weight of each loss function is determined by the `blend` parameter.

    Args:
        xf (torch.Tensor): The final position of the particles.
        kf (float): The final stiffness value.
        sim (Simulator): The simulator object.
        device (torch.device, optional): The device on which to perform the computations. Defaults to the current device.
        scale (float, optional): how many kf's should the values of k spread. Defaults to 5.0.
        x_steps (float): steps of the x discretization.
        kFs (torch.tensor): range for the k values.
        kFsteps (optional, int): size for kFS if not provided.
        blend (float, optional): The blending factor for the two loss functions. Defaults to 1e-3.
        **kwargs: extra keywords arguments ignored.

    Returns:
        float: The weighted sum of the two loss functions: work and
        caracteristic function comparison to the equilibrium one.
 
"""
    loss = (1.0 - blend) * loss_fn_k(
        xf=xf, kf=kf, device=device, scale=scale, kFsteps=kFsteps, x_steps=x_steps
    ) + blend * loss_fn_work(sim=sim)
    return loss
