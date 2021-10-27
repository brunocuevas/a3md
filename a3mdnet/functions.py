import torch
from scipy.special import gammaincc, gamma, exp1
import math


def tgammainc(a: torch.Tensor, z: torch.Tensor):
    device = a.device
    dtype = a.dtype
    a_n = a.cpu().numpy()
    z_n = z.cpu().numpy()
    g1 = gammaincc(a_n, z_n)
    g2 = gamma(a_n)

    return torch.from_numpy(g1 * g2).to(dtype=dtype, device=device)


def texp1d(x: torch.Tensor):
    device = x.device
    dtype = x.dtype
    x_n = x.cpu().numpy()
    u = exp1(x_n)
    return torch.from_numpy(u).to(device=device, dtype=dtype)


def distance_vectors(sample_coords, mol_coords, labels, device, dtype=torch.float):
    """

    Takes the coordinates of the molecule and the sampling coordinates,
    and calculates the distance vectors between them. If n is the batch size,
    m the number of density samples, Ma the number of molecule coordinates;
    the result is a tensor of range 4, dimensions: n, m, Ma, 3

    :param sample_coords:
    :type sample_coords: torch.Tensor
    :param mol_coords:
    :type mol_coords: torch.Tensor
    :param labels:
    :type labels: torch.Tensor
    :param device:
    :type device: torch.device
    :param dtype:
    :return: Tensor (n, m, Ma, 3, dtype=torch.float)
    """
    # dimensions
    n = sample_coords.shape[0]
    m_sample = sample_coords.shape[1]
    m_mol = mol_coords.shape[1]
    # output tensor
    dv = torch.zeros(n, m_sample, m_mol, 3, device=device, dtype=dtype)
    if labels is None:
        labels = torch.zeros(n, m_mol, device=device, dtype=torch.long)
    # distance operation takes place by each element of the batch
    # the operation is performed by expanding the molecule and the
    # sampling
    for i in range(n):
        mask = (labels[i, :] != -1).unsqueeze(1).expand(m_mol, 3)
        sliced_mol = mol_coords[i, :, :].reshape(m_mol, 3)
        masked_mol = sliced_mol.masked_select(mask).reshape(-1, 3)
        n_ = masked_mol.size(0)
        masked_mol = masked_mol.unsqueeze(0)
        masked_mol = masked_mol.expand(m_sample, n_, 3)
        sliced_sample = sample_coords[i, :, :].unsqueeze(1)
        sliced_sample = sliced_sample.expand(m_sample, n_, 3)
        dv[i, :, :n_, :] = sliced_sample - masked_mol
    return dv


def alt_distance_vectors(x1: torch.Tensor, x2: torch.Tensor, dtype: torch.dtype, device: torch.device):
    """
    Similar to distance vectors, but avoiding label selection to speed up calculations
    """
    n = x1.shape[0]
    m_sample = x1.shape[1]
    m_mol = x2.shape[1]
    dv = torch.zeros(n, m_sample, m_mol, 3, dtype=dtype, device=device)
    for i, (x1_, x2_) in enumerate(zip(x1.split(1, dim=0), x2.split(1, dim=0))):
        x1_ = x1_.transpose(0, 1)
        x1_ = x1_.expand(m_sample, m_mol, 3)
        x2_ = x2_.expand(m_sample, m_mol, 3)
        dv[i, :, :, :] = x1_ - x2_
    return dv


def distance(dv):
    """

    Takes a (n, m, Ma, 3) tensor, and returns the p2 norm along the
    third dimensions of the tensor. The resulting tensor is
    (n, m, Ma).

    :param dv:
    :type dv: torch.Tensor
    :return:
    """
    return dv.norm(dim=3, p=2)


def expand_parameter(labels, param):
    """

    Takes a tensor containing parameters, and expands it to match the
    molecule atoms

    :param labels:
    :param param:
    :return:
    """
    labels_copy = labels.clone()
    labels_copy[labels == -1] = 0
    dtype = param.dtype
    output = torch.zeros_like(labels, dtype=dtype)

    output.masked_scatter_(
        labels != -1,
        param.index_select(0, labels_copy.flatten()).view_as(labels)[labels != - 1]
    )
    return output


def exponential_kernel(d, a, b):
    """

    Applies a*exp(-B*d)

    :param d:
    :param a:
    :param b:
    :return:
    """
    a_usq = a.unsqueeze(1)
    b_usq = b.unsqueeze(1)
    buffer = torch.exp(-d * b_usq)
    buffer = a_usq * buffer
    return buffer


def xexponential_kernel(d, a, b):
    """

    Applies u*d*exp(-g*d)

    :param d:
    :param a:
    :param b:
    :return:
    """

    buffer = torch.exp(-d * b.unsqueeze(1))
    buffer = a.unsqueeze(1) * buffer * d
    return buffer


def gen_exponential_kernel(r, a, b, p):
    """

    Applies a * (r^p) * Exp(-b * r)

    :param r:
    :param a:
    :param b:
    :param p:
    :return:
    """
    b = b.unsqueeze(1)
    p = p.unsqueeze(1)
    a = a.unsqueeze(1)
    buffer = torch.exp(-r * b)
    buffer *= r.pow(p)
    buffer *= a
    return buffer


def gen_exponential_kernel_potential(
        r: torch.Tensor, a: torch.Tensor, b: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """

    Obtains the electrostatic potential of  (r ^ p) * Exp(-b * r) * a

    """
    b = b.unsqueeze(1)
    p = p.unsqueeze(1)
    a = a.unsqueeze(1)
    r = r.clamp(min=1e-18, max=None)
    factor = 4.0 * a * b.pow(- p - 3) * math.pi
    t1 = (torch.lgamma(3 + p)).exp()
    t3 = tgammainc(3 + p, b * r)

    t2 = (b * tgammainc(2 + p, b * r))
    return - factor * (t2 + (t1 - t3)/r)


def gen_exponential_kernal_moments(n: int, a: torch.Tensor, b: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """

    """
    b = b.unsqueeze(1)
    p = p.unsqueeze(1)
    a = a.unsqueeze(1)
    q = p + 3 + n
    b = b.clamp(min=1e-4)
    m = torch.lgamma(q).exp() * b.pow(-q) * a
    return math.pi * 4.0 * m


def gen_exponential_kernel_der(r, a: torch.Tensor, b: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    b = b.unsqueeze(1)
    p = p.unsqueeze(1)
    a = a.unsqueeze(1)
    term1 = torch.exp(-r * b) * p * r.clamp(min=1e-18).pow(-2 + p)
    term2 = torch.exp(-r * b) * r.clamp(min=1e-18).pow(-1 + p)
    return a * (term1 + term2)


def gaussian_kernel(z, alpha):
    """

    Applies exp(-alpha * (z**2))

    :param z:
    :param alpha:
    :return:
    """
    buffer = torch.exp(-alpha.unsqueeze(1) * z.pow(2.0))
    return buffer


def _legendre_polynomial1(x):
    return x


def _legendre_polynomial2(x):
    return (3 * x.pow(2.0) - 1) / 2


def _legendre_polynomial3(x):
    return (5 * x.pow(3.0) - 3 * x) / 2


def legendre_polynomial(z, angular_momentum):
    """

    Applies

    :param z:
    :param angular_momentum:
    :return:
    """
    if angular_momentum == 1:
        leg_pol = _legendre_polynomial1
    elif angular_momentum == 2:
        leg_pol = _legendre_polynomial2
    elif angular_momentum == 3:
        leg_pol = _legendre_polynomial3
    else:
        raise NotImplementedError('spherical harmonics l > 3 not implemented')

    return leg_pol(z)


def legendre_polynomial_derivative(z, angular_momentum):
    if angular_momentum == 1:
        return 1
    elif angular_momentum == 2:
        return 6 * z
    elif angular_momentum == 3:
        return 15 * (z ** 2) - 3


def spherical_harmonic(dv, basis_vector, angular_momentum):
    basis_vector = basis_vector / basis_vector.norm(dim=3).clamp(min=1e-18).unsqueeze(3)
    cos = (dv * basis_vector).sum(dim=3)
    cos = cos / dv.norm(dim=3).clamp(min=1e-18, max=None)
    sph = legendre_polynomial(cos, angular_momentum)
    return sph


def cosine_derivative(dv, basis_vector):
    term1 = dv / dv.norm(dim=3)
    term2 = (dv * basis_vector).sum(dim=3).unsqueeze(3) * dv
    return term1 - term2


def spherical_harmonic_gradient(dv, basis_vector, angular_momentum):
    basis_vector = basis_vector / basis_vector.norm(dim=3).clamp(min=1e-18).unsqueeze(3)
    cos = (dv * basis_vector).sum(dim=3)
    cos = cos / dv.norm(dim=3).clamp(min=1e-18, max=None)
    cos_der = cosine_derivative(dv, basis_vector)
    sph = legendre_polynomial_derivative(cos, angular_momentum)
    return sph * cos_der


def spherical_harmonic_basis(dv, r, angular_momentum):
    basis = torch.eye(3).to(dv.device)
    ux = spherical_harmonic(dv, basis[0, :], angular_momentum=angular_momentum)
    uy = spherical_harmonic(dv, basis[1, :], angular_momentum=angular_momentum)
    uz = spherical_harmonic(dv, basis[2, :], angular_momentum=angular_momentum)
    u = torch.stack((ux, uy, uz), dim=3)
    return u


def gto_kernel(
        r: torch.Tensor, rv: torch.Tensor,
        a: torch.Tensor, px: torch.Tensor, py: torch.Tensor, pz: torch.Tensor
):
    rx, ry, rz = torch.split(rv, 1, dim=3)
    rx = rx.squeeze(3).pow(px)
    ry = ry.squeeze(3).pow(py)
    rz = rz.squeeze(3).pow(pz)
    a = a.unsqueeze(0).unsqueeze(1)
    buffer = (-r * a).exp()
    buffer = rx * ry * rz * buffer
    return buffer


def cosine_cutoff(x, cutoff):
    x = x.clamp(min=None, max=cutoff)
    x = (1.0 + ((x / cutoff) * math.pi).cos()) / 2.0
    return x


def becke_johnson_dumping(x: torch.Tensor, n: int):
    kfact = torch.tensor([math.factorial(i) for i in range(n)], dtype=torch.float, device=x.device).reshape(1, 1, -1)
    k = torch.arange(0, n, dtype=torch.float, device=x.device).reshape(1, 1, -1)
    xk = x.squeeze(0).unsqueeze(2).expand(x.shape[1], x.shape[2], k.size()[2])
    xk = xk.pow(k) / kfact
    return 1 - (xk.sum(2) * (-x).exp())


def exponential_anisotropic(r: torch.Tensor, a: torch.Tensor, b: torch.Tensor, p: torch.Tensor, l):
    b = b.unsqueeze(1)
    p = p.unsqueeze(1)
    a = a.unsqueeze(1)

    s = 2 - l + p
    mask1 = torch.gt(s, 0)
    mask2 = torch.eq(s, 0)
    mask3 = torch.eq(s, -1)
    v1 = pot1(mask1, r, a, b, p, l)
    v2 = pot2(mask2, r, a, b, p, l)
    v3 = pot3(mask3, r, a, b, p, l)
    return v1 + v2 + v3


def pot1(mask, r, a, b, p, l):
    r = r.clamp(min=1e-2, max=None)
    v = torch.zeros_like(r)
    n = r.shape[0]
    ma = r.shape[2]
    ms = r.shape[1]
    nf = mask.long().sum().item()
    r = r[mask.expand(n, ms, ma)].reshape(n, ms, nf)
    a = a[mask.expand(n, 1, ma)].reshape(n, 1, nf)
    b = b[mask.expand(n, 1, ma)].reshape(n, 1, nf)
    p = p[mask.expand(n, 1, ma)].reshape(n, 1, nf)
    short = r.pow(2 + p) * (b * r).pow(-3 - l - p) * (torch.lgamma(3 + l + p).exp() - tgammainc(3 + l + p, b * r))
    long_factor = r.pow(l) * b.pow(-2)
    long_term1 = (b.pow(l - p) * torch.lgamma(2 - l + p).exp())
    long_term2 = (r.pow(-l + p)) * (b * r).pow(l-p)*(tgammainc(2 - l + p, b * r) - torch.lgamma(2 - l + p).exp())
    long = long_factor * (long_term1 + long_term2)
    pot = - a * (short + long)
    v[mask.expand(n, ms, ma)] = pot.flatten()
    return v


def pot2(mask, r, a, b, p, l):
    r = r.clamp(min=1e-2, max=None)
    v = torch.zeros_like(r)
    n = r.shape[0]
    ma = r.shape[2]
    ms = r.shape[1]
    nf = mask.long().sum().item()
    r = r[mask.expand(n, ms, ma)].reshape(n, ms, nf)
    a = a[mask.expand(n, 1, ma)].reshape(n, 1, nf)
    b = b[mask.expand(n, 1, ma)].reshape(n, 1, nf)
    p = p[mask.expand(n, 1, ma)].reshape(n, 1, nf)
    short = r.pow(2 + p) * (b * r).pow(-3 - l - p) * (torch.lgamma(3 + l + p).exp() - tgammainc(3 + l + p, b * r))
    long = (r ** l) * texp1d(b*r)
    v[mask.expand(n, ms, ma)] = -(a * (short + long)).flatten()
    return v


def pot3(mask, r, a, b, p, l):
    r = r.clamp(min=1e-2, max=None)
    v = torch.zeros_like(r)
    n = r.shape[0]
    ma = r.shape[2]
    ms = r.shape[1]
    nf = mask.long().sum().item()
    r = r[mask.expand(n, ms, ma)].reshape(n, ms, nf)
    a = a[mask.expand(n, 1, ma)].reshape(n, 1, nf)
    b = b[mask.expand(n, 1, ma)].reshape(n, 1, nf)
    p = p[mask.expand(n, 1, ma)].reshape(n, 1, nf)
    short = r.pow(2 + p) * (b * r).pow(-3 - l - p) * (torch.lgamma(3 + l + p).exp() - tgammainc(3 + l + p, b * r))
    long = (r ** l) * ((-b*r).exp() * r.pow(-1) - b * texp1d(b * r))
    v[mask.expand(n, ms, ma)] = -(a * (short + long)).flatten()
    return v


def nuclear_coulomb(z, r):
    dv = distance(alt_distance_vectors(r, r, dtype=r.dtype, device=r.device)).clamp(min=1e-18, max=None).pow(-1)
    z_a = z.unsqueeze(2)
    z_b = z.unsqueeze(1)
    coulomb = z_a * z_b * dv * (1 - torch.eye(z.shape[1]).unsqueeze(0).to(r.device)) / 2
    return coulomb.sum((1, 2))


def nuclear_potential(dv, z):
    distance_op = z * distance(dv).clamp(min=1e-18, max=None).pow(-1)
    coulomb = (distance_op * z).sum(2)
    return coulomb


def intermolecular_nuclear_repulsion(ra, rb, za, zb):
    dv = distance(alt_distance_vectors(rb, ra, dtype=ra.dtype, device=rb.device)).clamp(min=1e-18, max=None).pow(-1)
    za = za.unsqueeze(1)
    zb = zb.unsqueeze(2)
    coulomb = za * zb * dv
    return coulomb.sum((1, 2))


def dispersion(
        c6a: torch.Tensor, c6b: torch.Tensor, alpha_a: torch.Tensor, alpha_b: torch.Tensor,
        vdw_a: torch.Tensor, vdw_b: torch.Tensor, ra: torch.Tensor, rb: torch.Tensor
):
    r = distance(alt_distance_vectors(rb, ra, dtype=ra.dtype, device=rb.device)).clamp(min=1e-18, max=None)
    vdw_a = vdw_a.unsqueeze(1)
    vdw_b = vdw_b.unsqueeze(2)
    x = 2 * r / (vdw_a + vdw_b)
    bjd6 = becke_johnson_dumping(x, 6).squeeze(0)
    bjd8 = becke_johnson_dumping(x, 8).squeeze(0)

    alpha_ab = alpha_b.unsqueeze(2) / alpha_a.unsqueeze(1)
    alpha_ba = alpha_a.unsqueeze(1) / alpha_b.unsqueeze(2)

    c6a = c6a.unsqueeze(1).clamp(min=1e-4)
    c6b = c6b.unsqueeze(2).clamp(min=1e-4)

    c6: torch.Tensor = 1.5 * c6a * c6b / (alpha_ab * c6a + alpha_ba * c6b)
    c8: torch.Tensor = (3 / 2) * c6 * math.sqrt(2)

    r6 = r.pow(-6)
    r8 = r.pow(-8)

    dispersion_c6 = - (c6 * bjd6 * r6)
    dispersion_c8 = - (c8 * bjd8 * r8)
    return dispersion_c6.sum((1, 2)), dispersion_c8.sum((1, 2))
