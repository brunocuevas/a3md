import math
import torch
from a3mdnet.functions import alt_distance_vectors, distance
from torch import nn as nn


class MolecularEmbedding(nn.Module):

    def __init__(self, n_species, embedding_dim):
        super(MolecularEmbedding, self).__init__()
        self.map = nn.Embedding(num_embeddings=n_species + 1, embedding_dim=embedding_dim)

    def forward(self, x):
        z, r = x
        mask = z > -1
        return z, r, self.map(z + 1) * mask.unsqueeze(2).float()


class NodeConvolve(nn.Module):
    def __init__(self, net, distances, widths, update_ratio=0.1):
        super(NodeConvolve, self).__init__()
        self.net = net
        self.distances = nn.Parameter(
            torch.tensor(distances, dtype=torch.float)
        )
        self.widths = nn.Parameter(
            torch.tensor(widths, dtype=torch.float)
        )
        self.n_functions = len(widths)
        self.update_ratio = update_ratio

    def forward(self, x, decay=1):
        r: torch.Tensor
        h: torch.Tensor
        z: torch.Tensor
        z, r, h = x
        n = z.shape[0]
        ma = z.shape[1]
        dv = alt_distance_vectors(x1=r, x2=r, dtype=r.dtype, device=r.device)
        d = distance(dv)
        d = d.unsqueeze(1).expand(n, self.n_functions, ma, ma)
        mu = self.distances.clone().reshape(1, self.n_functions, 1, 1)
        sigma = self.widths.clone().reshape(1, self.n_functions, 1, 1)
        mask = torch.gt(z, -1).float()
        mask = (mask.unsqueeze(1) * mask.unsqueeze(2)).unsqueeze(1).float()
        d = mask * (-(d - mu).pow(2.0) / sigma).exp() * 5.0
        u = (d @ h.unsqueeze(1)).transpose(1, 2)
        u = u.reshape(n, ma, -1)
        v = torch.cat((h, u), dim=2)
        v = self.net.forward(v)
        h = h + (v * self.update_ratio * decay * torch.gt(z, -1).unsqueeze(2).float())
        return z, r, h


class MessagePassing(nn.Module):
    def __init__(self, convolve_net, update_net, distances, widths, update_ratio=0.1):
        super(MessagePassing, self).__init__()
        self.convolve_net = convolve_net
        self.update_net = update_net
        self.distances = nn.Parameter(
            torch.tensor(distances, dtype=torch.float)
        )
        self.widths = nn.Parameter(
            torch.tensor(widths, dtype=torch.float)
        )
        self.n_functions = len(widths)
        self.update_ratio = update_ratio

    def forward(self, x):
        r: torch.Tensor
        h: torch.Tensor
        z: torch.Tensor
        z, r, h = x
        n = z.shape[0]
        ma = z.shape[1]
        dv = alt_distance_vectors(x1=r, x2=r, dtype=r.dtype, device=r.device)
        d = distance(dv)
        d = d.unsqueeze(1).expand(n, self.n_functions, ma, ma)
        mu = self.distances.clone().reshape(1, self.n_functions, 1, 1)
        sigma = self.widths.clone().reshape(1, self.n_functions, 1, 1)
        mask = torch.gt(z, -1).float()
        mask = (mask.unsqueeze(1) * mask.unsqueeze(2)).unsqueeze(1).float()
        d = mask * (-(d - mu).pow(2.0) / sigma).exp() * 5.0

        # u = (d @ h.unsqueeze(1)).transpose(1, 2)
        # u = u.reshape(n, ma, -1)
        # m = torch.cat((h, u), dim=2)
        # m = self.convolve_net.forward(m)

        hn = h.unsqueeze(1).unsqueeze(3).expand(n, 1, ma, ma, -1)
        d = d.unsqueeze(4)
        m = (d * hn).transpose(1, 3).reshape(n, ma, ma, -1)
        hp = h.unsqueeze(2).expand(n, ma, ma, -1)
        m = torch.cat([hp, m], dim=3)
        m = self.convolve_net.forward(m)
        m = (m * mask.transpose(1, 3)).sum(2)
        m = m.reshape(n * ma, -1)
        h = h.reshape(n * ma, -1)
        h = self.update_net(m, h)
        # h = self.update_net(m)
        h = h.reshape(n, ma, -1)
        return z, r, h


class NodePool(nn.Module):
    def __init__(self, net: nn.Module):
        super(NodePool, self).__init__()
        self.net = net

    def forward(self, x):
        r: torch.Tensor
        h: torch.Tensor
        z: torch.Tensor
        z, r, h = x
        m = self.net.forward(h)
        return z, m


class EdgePool(nn.Module):

    def __init__(self, rc, net):
        super(EdgePool, self).__init__()
        self.net = net
        self.rc = rc

    def cutoff_function(self, r):

        return 0.5 * ((math.pi * r.clamp(min=None, max=self.rc) / self.rc).cos() + 1)

    def forward(self, x):
        r: torch.Tensor
        h: torch.Tensor
        z: torch.Tensor
        z, r, h = x
        n = z.shape[0]
        ma = z.shape[1]
        nfs = h.shape[2]
        dv = alt_distance_vectors(x1=r, x2=r, dtype=r.dtype, device=r.device)
        d = distance(dv)
        device = d.device
        cf = (
                self.cutoff_function(d) -
                torch.eye(ma, ma, dtype=torch.float).unsqueeze(0).expand(n, ma, ma).to(device)
        ).unsqueeze(3)
        mask = torch.gt(z, -1).float()
        mask = (mask.unsqueeze(1) * mask.unsqueeze(2))
        mask = mask - torch.eye(ma).unsqueeze(0).to(device)
        mask = mask.unsqueeze(3)
        cf = cf * mask
        u1 = h.unsqueeze(2).expand(n, ma, ma, nfs)
        u2 = h.unsqueeze(1).expand(n, ma, ma, nfs)
        u = torch.cat([u1, u2], dim=3)
        e = cf * u
        dv = (dv / d.unsqueeze(3).clamp(min=1e-4)).unsqueeze(3)
        c = self.net(e) * mask
        c = c.unsqueeze(4) * dv
        return z, r, c


class TopKEdges(nn.Module):
    def __init__(self, rc, k, net):
        super(TopKEdges, self).__init__()
        self.k = k
        self.net = net
        self.rc = rc

    def cutoff_function(self, r):

        return 0.5 * ((math.pi * r.clamp(min=None, max=self.rc) / self.rc).cos() + 1)

    def forward(self, x):
        z = x[0]
        r = x[1]
        h = x[2]
        n = z.shape[0]
        ma = z.shape[1]
        nfs = h.shape[2]

        # u1 = h.unsqueeze(2).expand(n, ma, ma, nfs)
        # u2 = h.unsqueeze(1).expand(n, ma, ma, nfs)
        # u = torch.cat([u1, u2], dim=3)

        dv = alt_distance_vectors(x1=r, x2=r, dtype=r.dtype, device=r.device)
        d = distance(dv)
        device = d.device
        cf = (
                self.cutoff_function(d) -
                torch.eye(ma, ma, dtype=torch.float).unsqueeze(0).expand(n, ma, ma).to(device)
        )
        mask = torch.gt(z, -1).float()
        mask = (mask.unsqueeze(1) * mask.unsqueeze(2))
        mask = mask - torch.eye(ma).unsqueeze(0).to(device)
        cf = cf * mask
        cf, index = torch.topk(cf, k=self.k, dim=2)
        index_feats = index.unsqueeze(3).expand(n, ma, self.k, nfs)
        index_dv = index.unsqueeze(3).expand(n, ma, self.k, 3)
        h2 = h.unsqueeze(1).expand(n, ma, ma, nfs)
        h1 = h.unsqueeze(2).expand(n, ma, self.k, nfs)
        h2 = h2.gather(2, index_feats) * cf.unsqueeze(3)
        u = torch.cat([h1, h2], dim=3)
        dv = dv.gather(2, index_dv)
        dv = (dv / dv.norm(dim=3).unsqueeze(3).clamp(min=1e-4)).unsqueeze(3)
        c = self.net(u)
        c = c.unsqueeze(4) * dv
        return z, r, c


class TopKAttentionEdgePool(nn.Module):

    def __init__(self, rc, pool_net, attention_net, k=4):
        super(TopKAttentionEdgePool, self).__init__()
        self.pool_net = pool_net
        self.attention_net = attention_net
        self.rc = rc
        self.k = k

    def cutoff_function(self, r):
        return 0.5 * ((math.pi * r.clamp(min=None, max=self.rc) / self.rc).cos() + 1)

    def attention_function(self, u, mask):
        g = self.attention_net(u) * mask
        g = g.exp() - (1 - mask)
        attention = g / g.sum(2).unsqueeze(2).clamp(min=1e-8)
        return attention

    def forward(self, x):
        r: torch.Tensor
        h: torch.Tensor
        z: torch.Tensor
        z, r, h = x
        n = z.shape[0]
        ma = z.shape[1]
        nfs = h.shape[2]
        dv = alt_distance_vectors(x1=r, x2=r, dtype=r.dtype, device=r.device)
        d = distance(dv)
        device = d.device
        cf = (
                self.cutoff_function(d) -
                torch.eye(ma, ma, dtype=torch.float).unsqueeze(0).expand(n, ma, ma).to(device)
        ).unsqueeze(3)
        mask = torch.gt(z, -1).float()
        mask = (mask.unsqueeze(1) * mask.unsqueeze(2))
        mask = mask - torch.eye(ma).unsqueeze(0).to(device)
        mask = mask.clamp(min=0.0)
        mask = mask.unsqueeze(3)
        cf = cf * mask
        u1 = h.unsqueeze(2).expand(n, ma, ma, nfs)
        u2 = h.unsqueeze(1).expand(n, ma, ma, nfs) * cf
        u = torch.cat([u1, u2], dim=3)
        attention = self.attention_function(u, mask)
        att, index = torch.topk(attention, k=self.k, dim=2)
        index_feats = index.expand(n, ma, self.k, nfs)
        index_dv = index.expand(n, ma, self.k, 3)
        h2 = h.unsqueeze(1).expand(n, ma, ma, nfs)
        h1 = h.unsqueeze(2).expand(n, ma, self.k, nfs)
        # cf = cf.gather(2, index)
        h2 = h2.gather(2, index_feats)  # * cf
        u = torch.cat([h1, h2], dim=3) * att
        dv = dv.gather(2, index_dv)
        dv = (dv / dv.norm(dim=3).unsqueeze(3).clamp(min=1e-4)).unsqueeze(3)
        c = self.pool_net(u)
        c = c.unsqueeze(4) * dv
        return z, r, c
