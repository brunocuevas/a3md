import torch
import torch.nn as nn
from a3mdnet.data import map_an2labels


class SymFeats(nn.Module):
    def __init__(self, parameters=None):
        """
        Important: using bohr as input unit
        """
        from torchani.aev import AEVComputer
        from torchani.neurochem import Constants
        from a3mdnet import AEV_PARAMETERS
        from a3mdnet import bohr2ang
        super(SymFeats, self).__init__()
        if parameters is None:
            parameters = AEV_PARAMETERS
        # noinspection PyArgumentList
        self.aev = AEVComputer(
            **Constants(filename=parameters)
        )
        self.conversion = bohr2ang

    def forward(self, *x):
        labels_tensor = x[0]
        coords_tensor = x[1]

        coords_tensor = coords_tensor * self.conversion
        species, aevs = self.aev((labels_tensor, coords_tensor))
        return species, aevs


class LabelEmbed(nn.Module):

    def __init__(self, embedding_size=None, n_embeddings=None, source=None, operation='place', freeze=True):

        super(LabelEmbed, self).__init__()
        if source is not None:
            self.map = nn.Embedding.from_pretrained(source, freeze=freeze)
        else:
            self.map = nn.Embedding(num_embeddings=n_embeddings, embedding_dim=embedding_size)
        self.operation = operation

    def forward(self, x):
        z, u = x
        v = self.map(z + 1)
        if self.operation == 'sum':
            v = v + u
            return z, v
        elif self.operation == 'place':
            return z, v
        else:
            raise RuntimeError("use either sum or place")


class TranslateAtomicSymbols(nn.Module):
    def __init__(self, translation_table):
        super(TranslateAtomicSymbols, self).__init__()
        self.translation_table = translation_table

    def forward(self, x):
        z, r = x
        new_z = map_an2labels(z, table=self.translation_table)
        return new_z, r


class LocalMap(nn.Module):

    def __init__(self, map_list, output_size):
        super(LocalMap, self).__init__()
        self.map_list = map_list
        self.output_size = output_size

    def forward(self, z, x):

        z_ = torch.flatten(z)
        float_dtype = x.dtype
        device = x.device
        z_unique = z_.unique()
        if z_unique[0].item() == -1:
            z_unique = z_unique[1:]
        z_unique, _ = z_unique.sort()
        x = x.flatten(0, 1)

        # Declaring output tensor
        y = torch.zeros(z_.size()[0], self.output_size, dtype=float_dtype, device=device)

        # Iteration through elements
        for i, n in enumerate(z_unique):
            mask_input = torch.eq(z_, n)
            x_ = x.index_select(0, mask_input.nonzero().squeeze())
            mask_output = mask_input.unsqueeze(1).expand(mask_input.shape[0], self.output_size)
            y.masked_scatter_(mask_output, self.map_list[n.item()](x_))

        y = y.reshape(z.size()[0], z.size()[1], -1)
        return z, y


class PairMap(nn.Module):

    def __init__(self, map_list, output_size):
        super(PairMap, self).__init__()
        self.map_list = map_list
        self.output_size = output_size

    def forward(self, z, r, x, t: torch.Tensor):

        device = z.device
        float_dtype = x.dtype
        int_dtype = z.dtype

        n_batch = t.size()[0]
        n_bonds = t.size()[1]
        n_atoms = x.size()[1]

        n_batch_arange = torch.arange(n_batch, dtype=int_dtype, device=device) * n_atoms

        t_f = t.flatten(0, 1)
        x_f = x.flatten(0, 1)
        z_f = z.flatten(0, 1).unsqueeze(1)

        # Updating values on C to match the new indexing
        t_mask = t_f[:, 0] != -1
        t_mask_x = t_mask.unsqueeze(1).expand(t_f.size(0), 2 * x_f.size(1))
        t_mask_z = t_mask.unsqueeze(1).expand(t_f.size(0), 2)
        t_f = t_f + n_batch_arange.reshape(-1, 1).repeat(1, n_bonds).reshape(-1, 1).repeat(
            1, 2)

        # Creating combinations of the features and the elements

        x_c = torch.zeros(t_f.size(0), 2 * x_f.size(1), dtype=float_dtype, device=device)
        z_c = torch.ones(t_f.size(0), 2, dtype=int_dtype, device=device) * -1

        x_c.masked_scatter_(
            t_mask_x,
            torch.cat(
                (
                    x_f.index_select(dim=0, index=t_f[t_mask, 0]),
                    x_f.index_select(dim=0, index=t_f[t_mask, 1]),
                ), dim=1
            )
        )

        z_c.masked_scatter_(
            t_mask_z,
            torch.cat(
                (
                    z_f.index_select(dim=0, index=t_f[t_mask, 0]),
                    z_f.index_select(dim=0, index=t_f[t_mask, 1]),
                ), dim=1
            )
        )

        nn_index = z_c[:, 0].clamp(min=0) * 5  # Hardcoded
        nn_index += z_c[:, 1].clamp(min=0)
        nn_index = nn_index + z_c[:, 0].clamp(max=0)
        nn_index = nn_index + z_c[:, 1].clamp(max=0)
        nn_index = nn_index.clamp(min=-1)
        present_nn_index = nn_index.unique(sorted=True)
        present_nn_index = present_nn_index[1:]

        y = torch.zeros(x_c.size()[0], self.output_size, dtype=float_dtype, device=device)

        for i in present_nn_index:
            mask_input = torch.eq(nn_index, i)
            x_ = x_c.index_select(0, mask_input.nonzero().squeeze())
            mask_output = mask_input.unsqueeze(1).expand(mask_input.shape[0], self.output_size)
            y.masked_scatter_(mask_output, self.map_list[i](x_).squeeze())

        y = y.reshape(n_batch, n_bonds, self.output_size)

        index = (torch.arange(n_batch).to(device) * n_atoms).unsqueeze(1).expand(n_batch, n_bonds)
        t1, t2 = t.split(1, dim=2)
        mask = torch.ne(t1, -1)
        t1 = (t1.clamp(min=0).squeeze(2) + index).flatten()
        t2 = (t2.clamp(min=0).squeeze(2) + index).flatten()
        r = r.reshape(-1, 3)
        v1 = torch.index_select(r, 0, t1)
        v2 = torch.index_select(r, 0, t2)
        v = v2 - v1
        v = v.reshape(n_batch, n_bonds, 3)
        v = v / v.norm(dim=2).unsqueeze(2).clamp(min=1e-12)
        v = v * mask
        return z, x, t, y, v
