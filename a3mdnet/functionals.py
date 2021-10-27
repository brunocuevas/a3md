import torch
import math
from a3mdnet.data import load_dispersion_params


def kl_divergence(p: torch.Tensor, q: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """
    Kullback-Leibler divergence

    Implemented as dkl = int p * log(p/q) dr = H(P,Q) - H(P)

    ¡It is important to notice the assymetry!
    ¡Also important to scale input densities!

    :param p: optimizable function
    :param q: reference function
    :param w: weights
    :return: kullback-leibler divergence. Units: nats
    """
    p = p.clamp(1e-18)
    q = q.clamp(1e-18)
    dkl = (p * (p/q).log() * w).sum(1) / p.shape[0]
    return dkl.sum()


def mean_squared_error(p: torch.Tensor, q: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """
    Mean Squared Error functional

    Implemented as MSE = int q * (p - q)^2 dr = E_{q} [ (p-q)^2 ]

    :param p: optimizable function
    :param q: reference function
    :param w: weights
    :return: mean squared error. Units: (electrons/(bohr^3))^2
    """
    mse = (p * (p - q).pow(2.0) * w).sum(1) / p.shape[0]
    return mse.sum()


def dirac_exchange(p: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """

    :param p:
    :param w:
    :return:
    """
    p = p.clamp(min=1e-18)
    cx = (3.0/4.0) * math.pow((3.0/math.pi), 1.0 / 3.0)
    kd = - cx * (p.pow(4.0/3.0) * w).sum(1)
    return kd


def thomas_fermi_kinetic(p: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """

    :param p:
    :param w:
    :return:
    """
    p = p.clamp(min=1e-18)
    cf = 3 * math.pow(3 * math.pi * math.pi, 2.0/3.0) / 10.0
    ttf = cf * (p.pow(5.0/3.0) * w).sum(1)
    return ttf


def von_weizsacker_kinetic(p: torch.Tensor,  g: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return (w * g.pow(2).sum(2) / (8 * p)).sum(1)


def tkatchenko_scheffler_c6(
        z: torch.Tensor, dv: torch.Tensor, p_mol: torch.Tensor, r3_proto: torch.Tensor, h: torch.Tensor,
        w: torch.Tensor
):
    pol_ref, c6_ref, vdw = load_dispersion_params(z)
    c6_ref = c6_ref.reshape(1, -1)
    pol_ref = pol_ref.reshape(1, -1)
    r3 = dv.norm(dim=3).pow(3)
    r3_mol = (p_mol.unsqueeze(2) * w.unsqueeze(2) * h * r3).sum(1)
    c6 = c6_ref * torch.pow(r3_mol / r3_proto.clamp(min=1e-4, max=None), 2)
    pol = pol_ref * (r3_mol / r3_proto.clamp(min=1e-4, max=None))
    return c6, pol, vdw
