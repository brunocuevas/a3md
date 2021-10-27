from a3mdnet.graph import NodeConvolve, NodePool, EdgePool, TopKEdges, MolecularEmbedding, MessagePassing
from a3mdnet.graph import TopKAttentionEdgePool
from a3mdnet.norm import QPChargeNormalization
from a3mdnet.density_models import GenAMD, HarmonicGenAMD
from a3mdnet.modules import TranslateAtomicSymbols
from a3mdnet.functions import alt_distance_vectors
from a3mdnet.functionals import thomas_fermi_kinetic, dirac_exchange
from a3mdnet.functions import nuclear_coulomb
from torch import nn
import torch


class IDNNx(nn.Module):
    def __init__(
        self, n_species, embedding_dim, convolve_net, pool_net, distances, widths, update_rate,
        n_convolutions=3, decay=0.5
    ):
        super(IDNNx, self).__init__()
        self.emb = MolecularEmbedding(n_species=n_species, embedding_dim=embedding_dim)
        self.conv = NodeConvolve(distances=distances, widths=widths, net=convolve_net, update_ratio=update_rate)
        self.pool = NodePool(net=pool_net)
        self.n = n_convolutions
        self.decay = decay

    def forward(self, x):
        x = self.emb(x)
        for i in range(self.n):
            x = self.conv(x, decay=(self.decay**i))
        c = self.pool(x)[1]
        return c


class DDNNx(nn.Module):
    def __init__(
            self, n_species, embedding_dim, distances, widths,
            convolve_net, edge_net, pool_net,
            rc, update_rate, n_convolutions, decay
    ):
        super(DDNNx, self).__init__()
        self.emb = MolecularEmbedding(n_species=n_species, embedding_dim=embedding_dim)
        self.conv = NodeConvolve(distances=distances, widths=widths, net=convolve_net, update_ratio=update_rate)
        self.pool = NodePool(net=pool_net)
        self.edge = EdgePool(rc=rc, net=edge_net)
        self.n = n_convolutions
        self.decay = decay

    def forward(self, x):
        x = self.emb(x)
        for i in range(self.n):
            x = self.conv(x, decay=(self.decay**i))
        c_iso = self.pool(x)[1]
        c_aniso = self.edge(x)[2]
        c_aniso = c_aniso.sum(2)
        return c_iso, c_aniso


class TDNNx(nn.Module):
    def __init__(
            self, n_species, embedding_dim, distances, widths,
            convolve_net, edge_net, pool_net,
            rc, update_rate, n_convolutions, k, decay
    ):
        super(TDNNx, self).__init__()
        self.emb = MolecularEmbedding(n_species=n_species, embedding_dim=embedding_dim)
        self.conv = NodeConvolve(distances=distances, widths=widths, net=convolve_net, update_ratio=update_rate)
        self.pool = NodePool(net=pool_net)
        self.edge = TopKEdges(k=k, rc=rc, net=edge_net)
        self.n = n_convolutions
        self.decay = decay

    def forward(self, x):
        x = self.emb(x)
        for i in range(self.n):
            x = self.conv(x, decay=(self.decay**i))
        c_iso = self.pool(x)[1]
        # c_aniso = self.edge(x)[2]
        c_aniso = self.edge(x)[2]
        return c_iso, c_aniso.reshape(c_aniso.shape[0], c_aniso.shape[1], -1, c_aniso.shape[4])


class ADNNx(nn.Module):
    def __init__(
            self, n_species, embedding_dim, distances, widths,
            convolve_net, edge_net, pool_net, attention_net,
            rc, update_rate, n_convolutions, k, decay
    ):
        super(ADNNx, self).__init__()
        self.emb = MolecularEmbedding(n_species=n_species, embedding_dim=embedding_dim)
        self.conv = NodeConvolve(distances=distances, widths=widths, net=convolve_net, update_ratio=update_rate)
        self.pool = NodePool(net=pool_net)
        self.edge = TopKAttentionEdgePool(k=k, rc=rc, pool_net=edge_net, attention_net=attention_net)
        self.n = n_convolutions
        self.decay = decay

    def forward(self, x):
        x = self.emb(x)
        for i in range(self.n):
            x = self.conv(x, decay=(self.decay**i))
        c_iso = self.pool(x)[1]
        c_aniso = self.edge(x)[2]
        return c_iso, c_aniso.reshape(c_aniso.shape[0], c_aniso.shape[1], -1, c_aniso.shape[4])


class AMPDNN(nn.Module):
    def __init__(
            self, n_species, embedding_dim, distances, widths,
            convolve_net, pool_net, edge_net, attention_net, n_convolutions,
            rc, update_rate, decay
    ):
        super(AMPDNN, self).__init__()
        self.emb = MolecularEmbedding(n_species=n_species, embedding_dim=embedding_dim)
        self.conv = NodeConvolve(
            distances=distances, widths=widths,
            net=convolve_net, update_ratio=update_rate
        )
        self.pool = NodePool(net=pool_net)
        self.edge = TopKAttentionEdgePool(rc=rc, pool_net=edge_net, attention_net=attention_net)
        self.n = n_convolutions
        self.decay = decay

    def forward(self, x):
        x = self.emb(x)
        for i in range(self.n):
            x = self.conv(x, decay=(self.decay**i))
        c_iso = self.pool(x)[1]
        c_aniso = self.edge(x)[2]
        return c_iso, c_aniso.sum(2)


class MPDNN(nn.Module):
    def __init__(
            self, n_species, embedding_dim, distances, widths,
            convolve_net, update_net, pool_net, edge_net, n_convolutions,
            rc
    ):
        super(MPDNN, self).__init__()
        self.emb = MolecularEmbedding(n_species=n_species, embedding_dim=embedding_dim)
        self.conv = MessagePassing(
            distances=distances, widths=widths,
            convolve_net=convolve_net, update_net=update_net
        )
        self.pool = NodePool(net=pool_net)
        self.edge = EdgePool(rc=rc, net=edge_net)
        self.n = n_convolutions

    def forward(self, x):
        x = self.emb(x)
        for i in range(self.n):
            x = self.conv(x)
        c_iso = self.pool(x)[1]
        c_aniso = self.edge(x)[2]
        return c_iso, c_aniso.sum(2)


class TMPDNN(nn.Module):
    def __init__(
            self, n_species, embedding_dim, distances, widths,
            convolve_net, update_net, pool_net, edge_net, n_convolutions,
            rc, k
    ):
        super(TMPDNN, self).__init__()
        self.emb = MolecularEmbedding(n_species=n_species, embedding_dim=embedding_dim)
        self.conv = MessagePassing(
            distances=distances, widths=widths,
            convolve_net=convolve_net, update_net=update_net
        )
        self.pool = NodePool(net=pool_net)
        self.edge = TopKEdges(rc=rc, net=edge_net, k=k)
        self.n = n_convolutions

    def forward(self, x):
        x = self.emb(x)
        for i in range(self.n):
            x = self.conv(x)
        c_iso = self.pool(x)[1]
        c_aniso = self.edge(x)[2]
        return c_iso, c_aniso.sum(2)


class MPINN(nn.Module):
    def __init__(
        self, n_species, embedding_dim, distances, widths,
        convolve_net, update_net, pool_net, n_convolutions,
    ):
        super(MPINN, self).__init__()
        self.emb = MolecularEmbedding(n_species=n_species, embedding_dim=embedding_dim)
        self.conv = MessagePassing(
            distances=distances, widths=widths,
            convolve_net=convolve_net, update_net=update_net
        )
        self.pool = NodePool(net=pool_net)
        self.n = n_convolutions

    def forward(self, x):
        x = self.emb(x)
        for i in range(self.n):
            x = self.conv(x)
        c_iso = self.pool(x)[1]
        return c_iso


class AMDnet(nn.Module):
    def __init__(self, predictor, protodensity_model: GenAMD, density_model: GenAMD, table):
        super(AMDnet, self).__init__()
        self.predictor = predictor
        self.norm = QPChargeNormalization()
        self.proto = protodensity_model
        self.density = density_model
        self.translate = TranslateAtomicSymbols(table)

    def protodensity(self, dv, z):
        z, r = self.translate.forward((z, None))
        return self.proto.protodensity(dv, z)

    def forward(self, dv, z, r, q):
        charge = (torch.gt(z, 0).long() * z).sum(1).unsqueeze(1).float() - q
        z, r = self.translate.forward((z, r))
        charge = charge - torch.sum(self.proto.protocharge(z), dim=1, keepdim=True)
        c_iso = self.predictor.forward((z, r))
        integral_iso = self.density.defintegrals(z)
        c_iso = self.norm.forward(c_iso, integrals=integral_iso, charges=charge)
        density = self.density.moldensity(dv, c_iso, z)
        density = density + self.proto.protodensity(dv, z)
        return density, c_iso * torch.gt(integral_iso, 0.0).float()

    def masked_forward(self, dv, z: torch.Tensor, r, q, mask):
        charge = (torch.gt(z, 0).long() * z).sum(1).unsqueeze(1).float() - q
        z, r = self.translate.forward((z, r))
        charge = charge - torch.sum(self.proto.protocharge(z), dim=1, keepdim=True)
        mask, _ = self.translate.forward((mask, None))
        c_iso = self.predictor.forward((z, r))
        integral_iso = self.density.defintegrals(z)
        c_iso = self.norm.forward(c_iso, integrals=integral_iso, charges=charge)
        density = self.density.moldensity(dv, c_iso, mask)
        density = density + self.proto.protodensity(dv, z)
        return density, (c_iso * integral_iso)

    def potential(self, dv, z, r, q):
        charge = (torch.gt(z, 0).long() * z).sum(1).unsqueeze(1).float() - q
        z, r = self.translate.forward((z, r))
        charge = charge - torch.sum(self.proto.protocharge(z), dim=1, keepdim=True)
        c_iso = self.predictor.forward((z, r))
        integral_iso = self.density.defintegrals(z)
        c_iso = self.norm.forward(c_iso, integrals=integral_iso, charges=charge)
        electron_potential = self.density.molpotential(dv, c_iso, z)
        electron_potential = electron_potential + self.proto.protopotential(dv, z)
        return electron_potential, (c_iso * integral_iso)

    def masked_potential(self, dv, z, r, q, mask):
        charge = (torch.gt(z, 0).long() * z).sum(1).unsqueeze(1).float() - q
        z, r = self.translate.forward((z, r))
        charge = charge - torch.sum(self.proto.protocharge(z), dim=1, keepdim=True)
        mask, _ = self.translate.forward((mask, None))
        c_iso = self.predictor.forward((z, r))
        integral_iso = self.density.defintegrals(z)
        c_iso = self.norm.forward(c_iso, integrals=integral_iso, charges=charge)
        electron_potential = self.density.molpotential(dv, c_iso, mask)
        electron_potential = electron_potential + self.proto.protopotential(dv, z)
        return electron_potential, (c_iso * integral_iso)

    def energy(self, dv, w, z, r, q):
        cm = alt_distance_vectors(r, r, device=r.device, dtype=r.dtype)
        v, _ = self.potential(dv, z, r, q)
        p, _ = self.forward(dv, z, r, q)
        vnn = nuclear_coulomb(z, r)
        ktf = thomas_fermi_kinetic(p, w)
        vxc = dirac_exchange(p, w)
        vee = -(v * w * p).sum(1) / 2
        vne = (self.potential(cm, z, r, q)[0] * z).sum(1)
        return vnn + ktf + vxc + vee + vne


class A3MDnet(nn.Module):

    def __init__(
            self, predictor, protodensity_model: GenAMD, density_model: GenAMD, deformation_model: HarmonicGenAMD,
            table
    ):
        super(A3MDnet, self).__init__()
        self.predictor = predictor
        self.norm = QPChargeNormalization()
        self.density = density_model
        self.proto = protodensity_model
        self.deformation = deformation_model
        self.translate = TranslateAtomicSymbols(table)

    def protodensity(self, dv, z):
        z, r = self.translate.forward((z, None))
        return self.proto.protodensity(dv, z)

    def forward(self, dv, z, r, q):
        charge = (torch.gt(z, 0).long() * z).sum(1).unsqueeze(1).float() - q
        z, r = self.translate.forward((z, r))
        charge = charge - torch.sum(self.proto.protocharge(z), dim=1, keepdim=True)
        c_iso, c_aniso = self.predictor.forward((z, r))
        integral_iso = self.density.defintegrals(z)
        c_iso = self.norm.forward(c_iso, integrals=integral_iso, charges=charge)
        density = self.density.moldensity(dv, c_iso, z)
        density = density + self.deformation.density(dv, c_aniso, z)
        density = density + self.proto.protodensity(dv, z)
        return density, (c_iso * integral_iso, c_aniso)

    def masked_forward(self, dv, z: torch.Tensor, r, q, mask):
        charge = mask.clamp(min=0).sum(1).unsqueeze(1).float() - q
        z, r = self.translate.forward((z, r))
        mask, _ = self.translate.forward((mask, None))
        charge = charge - torch.sum(self.proto.protocharge(mask), dim=1, keepdim=True)
        c_iso, c_aniso = self.predictor.forward((z, r))
        integral_iso = self.density.defintegrals(mask)
        c_iso = self.norm.forward(c_iso, integrals=integral_iso, charges=charge)
        density = self.density.moldensity(dv, c_iso, mask)
        density = density + self.deformation.density(dv, c_aniso, mask)
        density = density + self.proto.protodensity(dv, mask)
        return density, (c_iso * integral_iso, c_aniso)

    def potential(self, dv, z, r, q):
        charge = (torch.gt(z, 0).long() * z).sum(1).unsqueeze(1).float() - q
        z, r = self.translate.forward((z, r))
        charge = charge - torch.sum(self.proto.protocharge(z), dim=1, keepdim=True)
        c_iso, c_aniso = self.predictor.forward((z, r))
        integral_iso = self.density.defintegrals(z)
        c_iso = self.norm.forward(c_iso, integrals=integral_iso, charges=charge)
        electron_potential = self.density.molpotential(dv, c_iso, z)
        electron_potential = electron_potential + self.deformation.potential(dv, c_aniso, z)
        electron_potential = electron_potential + self.proto.protopotential(dv, z)
        return electron_potential, (c_iso, c_aniso)

    def masked_potential(self, dv, z, r, q, mask):
        charge = mask.clamp(min=0).sum(1).unsqueeze(1).float() - q
        z, r = self.translate.forward((z, r))
        mask, _ = self.translate.forward((mask, None))
        charge = charge - torch.sum(self.proto.protocharge(mask), dim=1, keepdim=True)
        c_iso, c_aniso = self.predictor.forward((z, r))
        integral_iso = self.density.defintegrals(mask)
        c_iso = self.norm.forward(c_iso, integrals=integral_iso, charges=charge)
        electron_potential = self.density.molpotential(dv, c_iso, mask)
        electron_potential = electron_potential + self.deformation.potential(dv, c_aniso, mask)
        electron_potential = electron_potential + self.proto.protopotential(dv, mask)
        return electron_potential, (c_iso, c_aniso)

    def energy(self, dv, w, z, r, q):
        cm = alt_distance_vectors(r, r, device=r.device, dtype=r.dtype)
        v, _ = self.potential(dv, z, r, q)
        p, _ = self.forward(dv, z, r, q)
        vnn = nuclear_coulomb(z, r)
        ktf = thomas_fermi_kinetic(p, w)
        vxc = dirac_exchange(p, w)
        vee = -(v * w * p).sum(1) / 2
        vne = (self.potential(cm, z, r, q)[0] * z).sum(1)
        return vnn + ktf + vxc + vee + vne
