import torch
from .simulator_forces import BaseHarmonicForce, device


class VariableStiffnessHarmonicForcePolynomial(BaseHarmonicForce):
    """
    Harmonic oscillator force with a variable stiffness that is a learnable parameter.
    The stiffness is a polynomial function of time.
    The center is fixed at zero.
    """

    def __init__(
        self,
        kappai: float,
        kappaf: float,
        tf: float,
        coef_list: list,
        continuous: bool = False,
        normalized: bool = False,
    ):
        """
        Args:
            kappai (float): Initial stiffness.
            kappaf (float): Final stiffness.
            tf (float): final time of the protocol.
            coef_list (list): List of coefficients for the polynomial function
            of time.
            continuous (bool): If True, the stiffness is a continuous function
            of time at t=0 and t=tf.
            normalized (bool): If True, the argument is normalized to be t/tf
            instead of t.
        """
        super().__init__()
        self.register_buffer("kappai", torch.tensor(kappai, dtype=torch.float))
        self.register_buffer("kappaf", torch.tensor(kappaf, dtype=torch.float))
        self.register_buffer("tf", torch.tensor(tf, dtype=torch.float))
        powers = torch.arange(len(coef_list), dtype=torch.float)
        self.register_buffer("powers", powers)
        self.continuous = continuous
        self.normalized = normalized

        coef_list = torch.tensor(coef_list, dtype=torch.float)
        self.coef_list = torch.nn.parameter.Parameter(
            data=coef_list, requires_grad=True
        )

    def kappa(self, t):
        """
        Stiffness as a polynomial function of time.
        If continuous = False:
            kappa(t) = sum(coef_list[i] * t^i) for i in range(len(coef_list))
        If continuous = True:
            kappa(t) = kappai + (t/tf) * (kappaf - kappai)
                        + (t/tf)*(1-t/tf)* sum(coef_list[i] * t^i) for i in range(len(coef_list))
        If normalized: 
            use t/tf instead of t in the polynomial.

        Args:
            t: time to compute the stiffness
        Returns:
            torch.tensor: the stiffness value at time t
        """
        if t < 0:
            return self.kappai
        if t > self.tf:
            return self.kappaf
        s = t
        if self.normalized:
            s = t / self.tf
        sum = torch.sum(self.coef_list * s**self.powers)
        if self.continuous:
            return (
                self.kappai
                + (t / self.tf) * (self.kappaf - self.kappai)
                + (t / self.tf) * (1 - t / self.tf) * sum
            )
        else:
            return sum
