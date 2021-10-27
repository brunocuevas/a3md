from a3mdnet.data import convert_label2atomicnumber
from a3mdnet.functions import distance, distance_vectors
from a3mdnet.functions import gen_exponential_kernel, gen_exponential_kernel_der
from a3mdnet.functions import gen_exponential_kernel_potential, gen_exponential_kernal_moments, exponential_anisotropic
from a3mdnet.functions import expand_parameter, becke_johnson_dumping
from a3mdnet.functions import spherical_harmonic
from a3mdnet.data import AMDParameters
from a3mdnet import ELEMENT2SYMBOL, SYMBOL2NN
from torch import nn
import torch
import math


class GenAMD(nn.Module):

    def __init__(self, parameters: AMDParameters, table=ELEMENT2SYMBOL):  # device: torch.device, dtype: torch.dtype):
        """
        GenAMD
        ======

        Generalized AMD model (only isotropic functions)

        Parameters
        ----------
        parameters: AMDParameters
            processed table with the coefficients, exponents and powers of the function set
        table: 
            arbitrary labelling of the atoms


        """
        super(GenAMD, self).__init__()
        self.pars = parameters
        # self.device = device
        # self.dtype = dtype
        self.table = table
        apd, bpd, ppd, zpd = self.build(parameters.keep_frozen())
        amd, bmd, pmd, _ = self.build(parameters.remove_frozen())

        self.a = nn.Parameter(torch.cat([apd, amd], dim=1), requires_grad=False)
        self.b = nn.Parameter(torch.cat([bpd, bmd], dim=1), requires_grad=False)
        self.p = nn.Parameter(torch.cat([ppd, pmd], dim=1), requires_grad=False)
        self.z = nn.Parameter(zpd, requires_grad=False)

        self.max_molecular_functions = self.pars.remove_frozen().get_maxfunctions()
        self.max_protomolecule_functions = self.pars.keep_frozen().get_maxfunctions()
        self.max_fun = self.max_protomolecule_functions + self.max_molecular_functions
        self.nelements = self.pars.get_nelements()

    def proto(self, z):
        """
        proto
        =====

        Returns pro-molecular density



        """
        n = z.shape[0]
        ma = z.shape[1]
        proto_c = torch.ones(n, ma, self.max_protomolecule_functions, dtype=self.a.dtype, device=z.device)
        mol_c = torch.zeros(n, ma, self.max_molecular_functions, dtype=self.a.dtype, device=z.device)
        c = torch.cat([proto_c, mol_c], dim=2)
        return c

    def molecular(self, c, z):
        n = z.shape[0]
        ma = z.shape[1]
        proto_c = torch.ones(n, ma, self.max_protomolecule_functions, dtype=c.dtype, device=c.device)
        c = torch.cat([proto_c, c], dim=2)
        return c

    def deformation(self, c, z):
        n = z.shape[0]
        ma = z.shape[1]
        proto_c = torch.zeros(n, ma, self.max_protomolecule_functions, dtype=c.dtype, device=c.device)
        c = torch.cat([proto_c, c], dim=2)
        return c

    def build(self, pars):
        """
        Reads powers, coefficients and exponents from a2mdparameters file
        :return:
        """
        max_funs = pars.get_maxfunctions()
        n_species = pars.get_nelements()

        p = torch.zeros(n_species, max_funs, dtype=torch.float)
        b = torch.zeros(n_species, max_funs, dtype=torch.float)
        a = torch.zeros(n_species, max_funs, dtype=torch.float)
        z = torch.zeros(n_species, dtype=torch.float)
        for element, i in self.table.items():
            if element == -1:
                continue
            symbol = ELEMENT2SYMBOL[element]
            z[i] = float(element)
            for j, fun in enumerate(pars.iter_element(symbol)):

                a[i, j] = fun['A']
                b[i, j] = fun['B']
                p[i, j] = fun['P']

        return a, b, p, z

    def __charge(self, coefficients: torch.Tensor, labels: torch.Tensor):
        integrals = self.__integral(labels)
        charges = (coefficients * integrals).sum(2)
        return charges

    def __integral(self, labels: torch.Tensor):
        """
        evaluates functions integrals

        :param labels:
        :return:
        """
        integrals = torch.zeros(
            labels.size()[0], labels.size()[1], self.max_fun,
            dtype=torch.float, device=labels.device
        )
        split_p = self.p.split(1, dim=1)
        split_b = self.b.split(1, dim=1)
        split_a = self.a.split(1, dim=1)
        for i, (p, b, a) in enumerate(zip(split_p, split_b, split_a)):
            p_s = expand_parameter(labels, p)
            b_s = expand_parameter(labels, b)
            a_s = expand_parameter(labels, a)
            f = (3 + p_s).to(torch.float).lgamma().exp()  # this is a factorial
            integrals[:, :, i] = a_s * f * b_s.clamp(0.1, None).pow(- 3 - p_s) * 4 * math.pi

        return integrals

    def __density(
            self,
            dv: torch.Tensor,
            coefficients: torch.Tensor,
            labels: torch.Tensor
    ):
        """
        evaluates density at coordinates

        :param dv:
        :param coefficients:
        :param labels:
        :return:
        """
        r = distance(dv)
        density = torch.zeros_like(r)
        c_splitted = coefficients.split(1, dim=2)
        p_splitted = self.p.split(1, dim=1)
        b_splitted = self.b.split(1, dim=1)
        a_splitted = self.a.split(1, dim=1)
        for i, (c_s, p, b, a) in enumerate(zip(c_splitted, p_splitted, b_splitted, a_splitted)):
            p_s = expand_parameter(labels, p)
            a_s = expand_parameter(labels, a)
            b_s = expand_parameter(labels, b)
            k = gen_exponential_kernel(r, a_s, b_s, p_s)
            c_s = c_s.transpose(1, 2)
            density += c_s * k
        return density.sum(2)

    def __moments(self, coefficients, labels, n):
        batch = labels.shape[0]
        m = labels.shape[1]
        p_splitted = self.p.split(1, dim=1)
        b_splitted = self.b.split(1, dim=1)
        a_splitted = self.a.split(1, dim=1)
        c_splitted = coefficients.split(1, dim=2)
        moments = torch.zeros(batch, m, device=coefficients.device, dtype=coefficients.dtype)
        for j, (c_s, p, b, a) in enumerate(zip(c_splitted, p_splitted, b_splitted, a_splitted)):
            p_s = expand_parameter(labels, p)
            a_s = expand_parameter(labels, a)
            b_s = expand_parameter(labels, b)
            k = gen_exponential_kernal_moments(n, a_s, b_s, p_s)
            moments += (c_s.transpose(1, 2) * k).squeeze(1)

        return moments

    def __potential(
            self,
            dv: torch.Tensor,
            coefficients: torch.Tensor,
            labels: torch.Tensor
    ):
        r = distance(dv)
        potential = torch.zeros_like(r)
        c_splitted = coefficients.split(1, dim=2)
        p_splitted = self.p.split(1, dim=1)
        b_splitted = self.b.split(1, dim=1)
        a_splitted = self.a.split(1, dim=1)
        for i, (c_s, p, b, a) in enumerate(zip(c_splitted, p_splitted, b_splitted, a_splitted)):
            p_s = expand_parameter(labels, p)
            a_s = expand_parameter(labels, a)
            b_s = expand_parameter(labels, b)
            k = gen_exponential_kernel_potential(r, a_s, b_s.clamp(min=1e-1), p_s)
            c_s = c_s.transpose(1, 2)
            potential += c_s * k

        return potential.sum(2)

    def __gradient(
            self,
            dv: torch.Tensor,
            coefficients: torch.Tensor,
            labels: torch.Tensor
    ):
        r = distance(dv)
        derivative = torch.zeros(r.shape[0], r.shape[1], 3).to(dv.device)
        c_splitted = coefficients.split(1, dim=2)
        p_splitted = self.p.split(1, dim=1)
        b_splitted = self.b.split(1, dim=1)
        a_splitted = self.a.split(1, dim=1)
        for i, (c_s, p, b, a) in enumerate(zip(c_splitted, p_splitted, b_splitted, a_splitted)):
            p_s = expand_parameter(labels, p)
            a_s = expand_parameter(labels, a)
            b_s = expand_parameter(labels, b)
            k = gen_exponential_kernel_der(r, a_s, b_s.clamp(min=1e-1), p_s).unsqueeze(3)
            c_s = c_s.transpose(1, 2).unsqueeze(3)
            derivative += (c_s * k * dv).sum(2)

        return derivative

    def protodensity(self, dv: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        c = self.proto(z)
        return self.__density(dv, c, z)

    def moldensity(self, dv: torch.Tensor, c: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        c = self.molecular(c, z)
        return self.__density(dv, c, z)

    def defdensity(self, dv: torch.Tensor, c: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        c = self.deformation(c, z)
        return self.__density(dv, c, z)

    def protopotential(self, dv: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        c = self.proto(z)
        return self.__potential(dv, c, z)

    def molpotential(self, dv: torch.Tensor, c: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        c = self.molecular(c, z)
        return self.__potential(dv, c, z)

    def defpotential(self, dv: torch.Tensor, c: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        c = self.deformation(c, z)
        return self.__potential(dv, c, z)

    def protocharge(self, z: torch.Tensor) -> torch.Tensor:
        c = self.proto(z)
        return self.__charge(c, z)

    def molcharge(self, c: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        c = self.molecular(c, z)
        return self.__charge(c, z)

    def defcharge(self, c: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        c = self.deformation(c, z)
        return self.__charge(c, z)

    def protointegrals(self, z):
        pix, _ = torch.split(
            self.__integral(z),
            (self.max_protomolecule_functions, self.max_molecular_functions), dim=2
        )
        return pix

    def molintegrals(self, z):
        return self.__integral(z)

    def defintegrals(self, z):
        _, dix = torch.split(
            self.__integral(z),
            (self.max_protomolecule_functions, self.max_molecular_functions), dim=2
        )
        return dix

    def protomoments(self, z, n):
        c = self.proto(z)
        return self.__moments(c, z, n)

    def molmoments(self, c, z, n):
        c = self.molecular(c, z)
        return self.__moments(c, z, n)

    def defmoments(self, c, z, n):
        c = self.deformation(c, z)
        return self.__moments(c, z, n)

    def protogradient(self, dv, z):
        c = self.proto(z)
        return self.__gradient(dv, c, z)

    def molgradient(self, dv, c, z):
        c = self.molecular(c, z)
        return self.__gradient(dv, c, z)

    def defgradient(self, dv, c, z):
        c = self.deformation(c, z)
        return self.__gradient(dv, c, z)

    def protone(self, z, centers):
        c = self.proto(z)
        return self.__ne(c, z, centers)

    def __ne(
        self, coefficients: torch.Tensor,
        labels: torch.Tensor,
        centers: torch.Tensor
    ):
        """
        provides nuclear - electron potential
        """
        dv = distance_vectors(
            centers,
            centers,
            labels, device=coefficients.device, dtype=coefficients.dtype
        )
        electronic = self.__potential(dv, coefficients, labels)
        z = expand_parameter(labels, self.z)

        return (electronic * z).sum(1)

    def nuclear_potential(self, dv, labels):
        r = distance(dv).pow(-1.0)
        z = expand_parameter(labels, self.z)
        v = z.unsqueeze(1) * r
        return v.sum(2)

    def nuclear_nuclear(self, z, centers):
        dv = distance_vectors(centers, centers, labels=z, device=z.device, dtype=centers.dtype)
        mask = 1 - torch.eye(centers.shape[1], dtype=centers.dtype, device=centers.device).unsqueeze(0)
        inv = dv.norm(dim=3).clamp(min=1e-18).pow(-1) * mask
        z = expand_parameter(z, self.z)
        za = z.unsqueeze(1)
        zb = z.unsqueeze(2)
        return (za * zb * inv).sum([1, 2]) / 2

    def __electrostatic_nuclear_nuclear(
        self, dv: torch.Tensor, za: torch.Tensor, zb: torch.Tensor
    ):
        x = distance(dv)
        qza = convert_label2atomicnumber(label=za).to(dv.device).reshape(1, -1)
        qzb = convert_label2atomicnumber(label=zb).to(dv.device).reshape(-1, 1)
        nuclear_nuclear = qza * qzb * (x.pow(-1))
        return nuclear_nuclear.sum()

    def __electrostatic_nuclear_electron(
        self, dv: torch.Tensor, za: torch.Tensor, zb: torch.Tensor, ca: torch.Tensor, cb: torch.Tensor
    ):
        va = self.__potential(dv, ca, za)
        vb = self.__potential(dv.transpose(1, 2), cb, zb)
        qza = convert_label2atomicnumber(label=za).to(dv.device)
        qzb = convert_label2atomicnumber(label=zb).to(dv.device)
        nuclear_electron = (qza * vb).sum() + (qzb * va).sum()
        return nuclear_electron.sum()

    def __electrostatic_electron_electron(
        self, dv: torch.Tensor, za: torch.Tensor, zb: torch.Tensor,
            ca: torch.Tensor, cb: torch.Tensor
    ):

        x: torch.Tensor = distance(dv).squeeze(0)
        x = x.repeat_interleave(self.max_fun, dim=0)
        x = x.repeat_interleave(self.max_fun, dim=1)

        ba = self.b.index_select(0, index=za[za != -1]).reshape(1, -1)
        bb = self.b.index_select(0, index=zb[zb != -1]).reshape(-1, 1)
        aa = self.a.index_select(0, index=za[za != -1]).reshape(1, -1)
        ab = self.a.index_select(0, index=zb[zb != -1]).reshape(-1, 1)
        qa = aa * ba.clamp(min=1e-4).pow(-3.0) * 8.0 * math.pi
        qb = ab * bb.clamp(min=1e-4).pow(-3.0) * 8.0 * math.pi
        v_upper = self.__electrostatic_electron_electron_formula(qa, qb, ba, bb + 5e-2, x)
        v_lower = self.__electrostatic_electron_electron_formula(qa, qb, ba, bb - 5e-2, x)
        v = (v_upper + v_lower) / 2.0
        return cb.reshape(1, -1) @ v @ ca.reshape(-1, 1)

    @staticmethod
    def __electrostatic_electron_electron_factors(ba, bb):
        f11 = bb.pow(6.0) * 2
        f12 = bb.pow(4.0) * ba.pow(2.0) * 6
        f21 = ba * bb.pow(6.0)
        f22 = ba.pow(3.0) * bb.pow(4.0)
        f1 = (f11 - f12)
        f2 = (f21 - f22)
        div = ((ba.pow(2.0) - bb.pow(2.0)).pow(3.0) + 1e-18).pow(-1)
        return f1, f2, div

    def __electrostatic_electron_electron_formula(self, qa, qb, ba, bb, x):

        f1a, f2a, diva = self.__electrostatic_electron_electron_factors(ba, bb)
        f1b, f2b, divb = self.__electrostatic_electron_electron_factors(bb, ba)

        xpa = (-x * ba).exp() * diva
        xpb = (-x * bb).exp() * divb

        coulomb_fctor = (qa * qb) * (x * 2).pow(-1)
        right_term = (f1a + f2a * x) * xpa
        left_term = (f1b + f2b * x) * xpb
        overlap_factor = 2 + right_term + left_term

        return coulomb_fctor * overlap_factor

    def __electrostatic_interaction(self, dv, za, zb, ca, cb):
        nuc_nuc = self.__electrostatic_nuclear_nuclear(dv, za, zb)
        nuc_elec = self.__electrostatic_nuclear_electron(dv, za, zb, ca, cb)
        elec_elec = self.__electrostatic_electron_electron(dv, za, zb, ca, cb)
        return nuc_nuc + nuc_elec + elec_elec

    def mol_electrostatic_interaction(self, dv, za, zb, ca, cb):
        ca = self.molecular(ca, za)
        cb = self.molecular(cb, zb)
        return self.__electrostatic_interaction(dv, za, zb, ca, cb)

    def proto_electrostatic_interaction(self, dv, za, zb):
        ca = self.proto(za)
        cb = self.proto(zb)
        return self.__electrostatic_interaction(dv, za, zb, ca, cb)

    def def_electrostatic_interaction(self, dv, za, zb, ca, cb):
        ca = self.deformation(ca, za)
        cb = self.deformation(cb, zb)
        return self.__electrostatic_interaction(dv, za, zb, ca, cb)

    def mol_overlap_interaction(self, dv, za, zb, ca, cb):
        ca = self.molecular(ca, za)
        cb = self.molecular(cb, zb)
        return self.__overlap(dv, za, zb, ca, cb)

    def proto_overlap_interaction(self, dv, za, zb):
        ca = self.proto(za)
        cb = self.proto(zb)
        return self.__overlap(dv, za, zb, ca, cb)

    def def_overlap_interaction(self, dv, za, zb, ca, cb):
        ca = self.deformation(ca, za)
        cb = self.deformation(cb, zb)
        return self.__overlap(dv, za, zb, ca, cb)

    def __overlap(
        self, dv: torch.Tensor, za: torch.Tensor, zb: torch.Tensor,
            ca: torch.Tensor, cb: torch.Tensor
    ):

        x: torch.Tensor = distance(dv).squeeze(0)
        x = x.repeat_interleave(self.max_fun, dim=0)
        x = x.repeat_interleave(self.max_fun, dim=1)

        ba = self.b.index_select(0, index=za.clamp(min=0, max=None).flatten())
        ba = ba * (torch.ne(za, -1)).unsqueeze(2).expand(1, za.shape[1], self.max_fun)
        ba = ba.reshape(1, -1)

        bb = self.b.index_select(0, index=zb.clamp(min=0, max=None).flatten())
        bb = bb * (torch.ne(zb, -1)).unsqueeze(2).expand(1, zb.shape[1], self.max_fun)
        bb = bb.reshape(-1, 1)

        aa = self.a.index_select(0, index=za.clamp(min=0, max=None).flatten())
        aa = aa * (torch.ne(za, -1)).unsqueeze(2).expand(1, za.shape[1], self.max_fun)
        aa = aa.reshape(1, -1)

        ab = self.a.index_select(0, index=zb.clamp(min=0, max=None).flatten())
        ab = ab * (torch.ne(zb, -1)).unsqueeze(2).expand(1, zb.shape[1], self.max_fun)
        ab = ab.reshape(-1, 1)

        v_upper = self.__overlap_formula(aa, ab, ba, bb + 5e-2, x)
        v_lower = self.__overlap_formula(aa, ab, ba, bb - 5e-2, x)
        v = (v_upper + v_lower) / 2.0
        return cb.reshape(1, -1) @ v @ ca.reshape(-1, 1)

    @staticmethod
    def __overlap_formula(aa, ab, ba, bb, x):

        f1a = x * ba.pow(2.0) * bb
        f2a = -(x * bb.pow(3.0))
        f3a = 4 * ba * bb

        f1b = -(x * bb.pow(2.0) * ba)
        f2b = (x * ba.pow(3.0))
        f3b = -(4 * ba * bb)

        xpa = (- ba * x).exp()
        xpb = (- bb * x).exp()

        div = ((ba.pow(2.0) - bb.pow(2.0)).pow(3.0) + 1e-24).pow(-1)
        fact = aa * ab * 8.0 * math.pi * x.pow(-1)
        dumping_a = (f1a + f2a + f3a) * xpa
        dumping_b = (f1b + f2b + f3b) * xpb
        return div * (dumping_a + dumping_b) * fact

    def mol_dispersion_interaction(self, dv, za, zb, ca, cb):

        r = distance(dv)

        r6 = r.pow(-6).squeeze(0)
        r8 = r.pow(-8).squeeze(0)

        pol_a_ref, c6a_ref, vdw_a = self.load_dispersion_params(za)
        pol_b_ref, c6b_ref, vdw_b = self.load_dispersion_params(zb)

        c6a_ref = torch.tensor(c6a_ref, dtype=dv.dtype, device=dv.device).reshape(1, -1)
        c6b_ref = torch.tensor(c6b_ref, dtype=dv.dtype, device=dv.device).reshape(-1, 1)
        vdw_a = torch.tensor(vdw_a, dtype=dv.dtype, device=dv.device).reshape(1, -1)
        vdw_b = torch.tensor(vdw_b, dtype=dv.dtype, device=dv.device).reshape(-1, 1)

        x = 2 * r / (vdw_a + vdw_b)

        # m1a_proto = self.protomoments(za, -1).reshape(-1, 1)
        # m1b_proto = self.protomoments(zb, -1).reshape(1, -1)

        # x = 2 * r / (m1a_proto + m1b_proto)
        bjd6 = becke_johnson_dumping(x, 6).squeeze(0)
        bjd8 = becke_johnson_dumping(x, 8).squeeze(0)

        m3a_proto = self.protomoments(za, 3).reshape(1, -1)
        m3b_proto = self.protomoments(zb, 3).reshape(-1, 1)

        m3a_mol = self.molmoments(ca, za, 3).reshape(1, -1)
        m3b_mol = self.molmoments(cb, zb, 3).reshape(-1, 1)

        m2a_proto = self.protomoments(za, 2).reshape(1, -1)
        m2b_proto = self.protomoments(zb, 2).reshape(-1, 1)

        m4a_proto = self.protomoments(za, 4).reshape(1, -1)
        m4b_proto = self.protomoments(zb, 4).reshape(-1, 1)

        pol_a_ref = torch.tensor(pol_a_ref, dtype=m3a_mol.dtype, device=m3a_mol.device).reshape(1, -1)
        pol_b_ref = torch.tensor(pol_b_ref, dtype=m3b_mol.dtype, device=m3b_mol.device).reshape(-1, 1)

        pol_a = pol_a_ref * (m3a_mol / m3a_proto)
        pol_b = pol_b_ref * (m3b_mol / m3b_proto)

        pol_ab = pol_b / pol_a
        pol_ba = pol_a / pol_b

        c6a = c6a_ref * (m3a_mol / m3a_proto).pow(2.0)
        c6b = c6b_ref * (m3b_mol / m3b_proto).pow(2.0)

        c6 = c6a * c6b / (pol_ab * c6a + pol_ba * c6b)
        c8 = (3/2) * c6 * ((m4a_proto / m2a_proto) + (m4b_proto / m2b_proto)).sqrt()

        dispersion_c6 = - (c6 * bjd6 * r6).sum()
        dispersion_c8 = - (c8 * bjd8 * r8).sum()
        return dispersion_c6, dispersion_c8

    def proto_dispersion_interaction(self, dv, za, zb):

        r = distance(dv)

        r6 = r.pow(-6).squeeze(0)
        r8 = r.pow(-8).squeeze(0)

        pol_a_ref, c6a_ref, vdw_a = self.load_dispersion_params(za)
        pol_b_ref, c6b_ref, vdw_b = self.load_dispersion_params(zb)

        c6a_ref = torch.tensor(c6a_ref, dtype=dv.dtype, device=dv.device).reshape(1, -1)
        c6b_ref = torch.tensor(c6b_ref, dtype=dv.dtype, device=dv.device).reshape(-1, 1)
        vdw_a = torch.tensor(vdw_a, dtype=dv.dtype, device=dv.device).reshape(1, -1)
        vdw_b = torch.tensor(vdw_b, dtype=dv.dtype, device=dv.device).reshape(-1, 1)

        x = 2 * r / (vdw_a + vdw_b)

        # m1a_proto = self.protomoments(za, -1).reshape(-1, 1)
        # m1b_proto = self.protomoments(zb, -1).reshape(1, -1)

        # x = 2 * r / (m1a_proto + m1b_proto)
        bjd6 = becke_johnson_dumping(x, 6).squeeze(0)
        bjd8 = becke_johnson_dumping(x, 8).squeeze(0)

        pol_a = torch.tensor(pol_a_ref, dtype=dv.dtype, device=dv.device).reshape(1, -1)
        pol_b = torch.tensor(pol_b_ref, dtype=dv.dtype, device=dv.device).reshape(-1, 1)

        pol_ab = pol_b / pol_a
        pol_ba = pol_a / pol_b

        c6a = c6a_ref
        c6b = c6b_ref

        c6 = c6a * c6b / (pol_ab * c6a + pol_ba * c6b)
        c8 = (3/2) * c6 * math.sqrt(2)

        dispersion_c6 = - (c6 * bjd6 * r6).sum()
        dispersion_c8 = - (c8 * bjd8 * r8).sum()
        return dispersion_c6, dispersion_c8

    def hirshfeld_partition(self, dv, z):
        """
        evaluates density at coordinates

        :param dv:
        :param z:
        :return:
        """
        r = distance(dv)
        proto_density = torch.zeros_like(r)
        c_proto = self.proto(z)
        c_proto_splitted = c_proto.split(1, dim=2)
        p_splitted = self.p.split(1, dim=1)
        b_splitted = self.b.split(1, dim=1)
        a_splitted = self.a.split(1, dim=1)
        for i, (cp_s, p, b, a) in enumerate(zip(c_proto_splitted, p_splitted, b_splitted, a_splitted)):
            p_s = expand_parameter(z, p)
            a_s = expand_parameter(z, a)
            b_s = expand_parameter(z, b)
            k = gen_exponential_kernel(r, a_s, b_s, p_s)
            cp_s = cp_s.transpose(1, 2)
            proto_density += cp_s * k

        total_proto_density = proto_density.sum(2).unsqueeze(2).clamp(min=1e-18)
        w = proto_density / total_proto_density
        return w

    def __hirshfeld_partition(self, dv, c, z):
        """
        evaluates density at coordinates

        :param dv:
        :param c:
        :param z:
        :return:
        """
        r = distance(dv)
        density = torch.zeros_like(r)
        c_splitted = c.split(1, dim=2)
        p_splitted = self.p.split(1, dim=1)
        b_splitted = self.b.split(1, dim=1)
        a_splitted = self.a.split(1, dim=1)
        for i, (cp_s, p, b, a) in enumerate(zip(c_splitted, p_splitted, b_splitted, a_splitted)):
            p_s = expand_parameter(z, p)
            a_s = expand_parameter(z, a)
            b_s = expand_parameter(z, b)
            k = gen_exponential_kernel(r, a_s, b_s, p_s)
            cp_s = cp_s.transpose(1, 2)
            density += cp_s * k

        total_density = density.sum(2).unsqueeze(2).clamp(min=1e-18)
        w = density / total_density
        return w

    def proto_hpartition(self, dv, z):
        c = self.proto(z)
        return self.__hirshfeld_partition(dv, c, z)

    def mol_hpartition(self, dv, c, z):
        c = self.molecular(c, z)
        return self.__hirshfeld_partition(dv, c, z)


class WaveFunctionDensity(nn.Module):

    def __init__(self):
        from a3mdnet import WFN_SYMMETRY_INDEX
        from a3mdnet import functions
        super(WaveFunctionDensity, self).__init__()
        self.sym_map = nn.Parameter(torch.tensor(WFN_SYMMETRY_INDEX, dtype=torch.long), requires_grad=False)
        self.gto = functions.gto_kernel

    def expand_symmetry(self, symmetry_index: torch.Tensor):
        symmetry_index_f = torch.flatten(symmetry_index)
        symmetry = self.sym_map.index_select(0, symmetry_index_f)
        symmetry = symmetry.reshape(symmetry_index.shape[0], 3)
        px, py, pz = symmetry.float().split(dim=1, split_size=1)
        px.squeeze_(1).unsqueeze_(0).unsqueeze_(1)
        py.squeeze_(1).unsqueeze_(0).unsqueeze_(1)
        pz.squeeze_(1).unsqueeze_(0).unsqueeze_(1)
        return px, py, pz

    def primitives(self, dv: torch.Tensor, centers: torch.Tensor, exp: torch.Tensor, sym: torch.Tensor):
        nps = torch.sum(torch.ne(centers, -1)).item()
        centers = centers[:nps]
        exp = exp[:nps]
        sym = sym[:nps]
        r = dv.pow(2.0).sum(2)
        r = r[:, centers.view(-1)].unsqueeze(0)
        dv = dv[:, centers.view(-1)].unsqueeze(0)
        px, py, pz = self.expand_symmetry(sym)
        p = self.gto(r, dv, exp, px, py, pz)
        return p, nps

    def density(
            self, dv: torch.Tensor, centers: torch.Tensor, exp: torch.Tensor,
            sym: torch.Tensor, dm: torch.Tensor
    ):
        n = dv.shape[0]
        ms = dv.shape[1]
        p = torch.zeros(n, ms, dtype=dv.dtype, device=dv.device)
        for i in range(n):
            basis, nps = self.primitives(dv[i], centers[i], exp[i], sym[i])
            basis = basis.squeeze(0)
            p[i] = (basis * (basis @ dm[i, :nps, :nps])).sum(1)
        return p

    def density_from_dict(self, dv, u):
        return self.density(
            dv, centers=u['primitive_centers'], exp=u['exponents'],
            sym=u['symmetry_indices'], dm=u['density_matrix']
        )


class HarmonicGenAMD(nn.Module):

    def __init__(
            self, parameters: AMDParameters,
            table=ELEMENT2SYMBOL, max_angular_moment=3, k=1
    ):
        """
        Generalized Analitically Modelled Density
        :param parameters:

        """
        super(HarmonicGenAMD, self).__init__()
        self.pars = parameters
        self.table = table
        amd, bmd, pmd, zmd = self.build(parameters.remove_frozen())

        self.a = nn.Parameter(amd.repeat(1, k), requires_grad=False)
        self.b = nn.Parameter(bmd.repeat(1, k), requires_grad=False)
        self.p = nn.Parameter(pmd.repeat(1, k), requires_grad=False)
        self.z = nn.Parameter(zmd.repeat(1, k), requires_grad=False)

        self.n_radial = self.pars.remove_frozen().get_maxfunctions()
        self.n_angular = max_angular_moment
        self.max_fun = self.n_radial * self.n_angular * 3
        self.nelements = self.pars.get_nelements()

    def build(self, pars):
        """
        Reads powers, coefficients and exponents from a2mdparameters file
        :return:
        """
        max_funs = pars.get_maxfunctions()
        n_species = pars.get_nelements()

        p = torch.zeros(n_species, max_funs, dtype=torch.float)
        b = torch.zeros(n_species, max_funs, dtype=torch.float)
        a = torch.zeros(n_species, max_funs, dtype=torch.float)
        z = torch.zeros(n_species, dtype=torch.float)
        for element, i in self.table.items():
            if element == -1:
                continue
            symbol = ELEMENT2SYMBOL[element]
            z[i] = float(element)
            for j, fun in enumerate(pars.iter_element(symbol)):

                a[i, j] = fun['A']
                b[i, j] = fun['B']
                p[i, j] = fun['P']

        return a, b, p, z

    def density(
            self,
            dv: torch.Tensor,
            coefficients: torch.Tensor,
            labels: torch.Tensor
    ):
        """
        evaluates density at coordinates

        :param dv:
        :param coefficients:
        :param labels:
        :return:
        """
        r = distance(dv)
        qi = [1, 2, 3] * self.max_fun

        coefficients = coefficients.transpose(2, 1)
        # coefficients = torch.ones_like(coefficients)
        # coefficients[:, :, :, 2] = 0.0
        density = torch.zeros_like(r)
        c_splitted = coefficients.split(1, dim=1)
        p_splitted = self.p.repeat_interleave(self.n_angular, dim=1).split(1, dim=1)
        b_splitted = self.b.repeat_interleave(self.n_angular, dim=1).split(1, dim=1)
        a_splitted = self.a.repeat_interleave(self.n_angular, dim=1).split(1, dim=1)

        for i, (c_s, p, b, a) in enumerate(zip(c_splitted, p_splitted, b_splitted, a_splitted)):
            p_s = expand_parameter(labels, p)
            a_s = expand_parameter(labels, a)
            b_s = expand_parameter(labels, b)
            p = gen_exponential_kernel(r, a_s, b_s, p_s)
            q = spherical_harmonic(dv, c_s, qi[i])
            q = c_s.norm(dim=3) * q
            k = (p * q)
            density += k

        return density.sum(2)

    def potential(
            self,
            dv: torch.Tensor,
            coefficients: torch.Tensor,
            labels: torch.Tensor
    ):
        r = distance(dv)
        potential = torch.zeros_like(r)
        coefficients = coefficients.transpose(2, 1)
        qi = [1, 2, 3] * self.max_fun
        c_splitted = coefficients.split(1, dim=1)
        p_splitted = self.p.repeat_interleave(self.n_angular, dim=1).split(1, dim=1)
        b_splitted = self.b.repeat_interleave(self.n_angular, dim=1).split(1, dim=1)
        a_splitted = self.a.repeat_interleave(self.n_angular, dim=1).split(1, dim=1)

        for i, (c_s, p, b, a) in enumerate(zip(c_splitted, p_splitted, b_splitted, a_splitted)):
            p_s = expand_parameter(labels, p)
            a_s = expand_parameter(labels, a)
            b_s = expand_parameter(labels, b)
            p = exponential_anisotropic(r, a_s, b_s.clamp(min=1e-1), p_s, qi[i])
            q = spherical_harmonic(dv, c_s, qi[i]) * (4 * math.pi / ((2 * qi[i]) + 1))
            q = c_s.norm(dim=3) * q
            k = p * q
            potential += k

        return potential.sum(2)
