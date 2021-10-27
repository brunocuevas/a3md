import numpy as np
from torch.utils import data
import torch
import json
from pathlib import Path
from typing import List
from a3mdnet.functions import alt_distance_vectors, distance
from torch.nn.utils.rnn import pad_sequence
from a3mdnet import ELEMENT2NN, NN2ELEMENT, get_symbol, get_atomic_number, VDW_PARAMS
from typing import Union, Dict
import h5py as h5


def map_an2labels(atomic_numbers: torch.Tensor, table=ELEMENT2NN):
    """
    map_an2labels
    ===
    map atomic numbers to labels

    Parameters
    ----------
    atomic_numbers: torch.Tensor
        Tensor containing atomic numbers for a [n, Ma] set of atoms
    table: Union[List, Dict]
        List or Dict containing the map between atomic numbers and arbitrary labels
    
    Returns
    -------
    torch.Tensor
        Tensor containing translated labels of the [n, Ma] atoms

    Examples
    --------

    >>> a = torch.tensor([[6, 1, 1, 1]]) # Methane molecule
    >>> u = map_an2labels(a)

    """
    u = torch.flatten(atomic_numbers).tolist()
    u = [table[i] for i in u]
    u = torch.tensor(u, device=atomic_numbers.device, dtype=atomic_numbers.dtype)
    return u.resize_as(atomic_numbers)


def convert_targets2tensor(
        targets, device=torch.device('cpu'), dtype=torch.float
):
    return torch.tensor(targets, device=device, dtype=dtype).reshape(-1, 1)


def convert_label2atomicnumber(label):
    u = [[NN2ELEMENT[i] for i in j] for j in label.tolist()]
    return torch.tensor(u, dtype=torch.float)


def match_old_fun_names(fun):
    if fun['bond'] is not None and fun['support_type'] == "AG":
        if fun['params']['Psi'] == 0:
            return "aniso", 0
        elif fun['params']['Psi'] == 1:
            return "aniso", 1
        else:
            raise IOError("old parametrization does not allow Psi value {:d}".format(fun['parameters']['Psi']))
    elif fun['bond'] is None:
        if fun['support_type'] == 'ORC':
            return "core", None
        elif fun['support_type'] == 'ORCV':
            return "iso", 0
        elif fun['support_type'] == 'ORV':
            return "iso", 1
        elif fun['support_type'] == 'OR':
            return "iso", 0
        else:
            raise IOError("unknown support type {:s}".format(fun['support_type']))
    else:
        raise IOError("can not understand function {:s}".format(json.dumps(fun)))


def load_dispersion_params(z: torch.Tensor):
    symbols = [get_symbol(i) for i in torch.flatten(z).tolist()]
    polarizability = []
    c6 = []
    vdwr = []
    u = VDW_PARAMS
    for s in symbols:
        if s is None:
            polarizability.append(1000.0)
            c6.append(0)
            vdwr.append(1000.0)
        else:
            polarizability.append(u['_VALUES'][s]['polarizability'])
            c6.append(u['_VALUES'][s]['c6'])
            vdwr.append(u['_VALUES'][s]['vdwr'])
    polarizability = torch.tensor(polarizability).resize_as(z).to(z.device)
    c6 = torch.tensor(c6).resize_as(z).to(z.device)
    vdwr = torch.tensor(vdwr).resize_as(z).to(z.device)
    return polarizability, c6, vdwr


def h5wfn_to_dict(cwfn, device):
    # out = dict()
    u = DatasetOutput(
        atomic_numbers=torch.tensor(cwfn['charges'][:], dtype=torch.long, device=device).unsqueeze(0),
        coordinates=torch.tensor(cwfn['coordinates'][:], device=device, dtype=torch.float).unsqueeze(0),
        charge=torch.tensor(cwfn.attrs['total_charge'], device=device, dtype=torch.float).unsqueeze(0),
        symmetry=torch.tensor(cwfn['symmetry_indices'], device=device, dtype=torch.long).unsqueeze(0),
        primitive_centers=torch.tensor(cwfn['primitive_centers'], device=device, dtype=torch.long).unsqueeze(0),
        exponents=torch.tensor(cwfn['exponents'], device=device, dtype=torch.float).unsqueeze(0),
        density_matrix=torch.tensor(cwfn['density_matrix'], device=device, dtype=torch.float).unsqueeze(0),
        natoms=None, nprimitives=None, extra_property=None
    )
    return u


def topology_from_coordinates(z, r, cutoff):
    n_batch = r.shape[0]
    n_atoms = r.shape[1]
    d = distance(alt_distance_vectors(r, r, r.dtype, r.device))
    z1 = z.unsqueeze(1).lt(0)
    z2 = z.unsqueeze(2).lt(0)
    mask = 1 - (z1 + z2 + torch.eye(n_atoms).to(d.device).unsqueeze(0)).clamp(max=1)
    top_lists = torch.where(torch.lt(d, cutoff) * mask)

    t = []

    for i in range(n_batch):

        start = top_lists[1][top_lists[0] == i].reshape(-1, 1)
        end = top_lists[2][top_lists[0] == i].reshape(-1, 1)
        t.append(torch.cat((start, end), dim=1))
    return pad_sequence(t, padding_value=-1).transpose(0, 1)


class SplitBySegment:
    def __init__(self, expand=1):
        super(SplitBySegment, self).__init__()
        self.expand = expand

    def forward(self, z, s, sq):
        mseg = sq.shape[1]
        n = z.shape[0]
        ma = z.shape[1]
        device = z.device

        labels_pre = z.clone().unsqueeze(0).expand(mseg, n, ma).reshape(mseg, -1) + 1
        labels_post = z.clone().unsqueeze(0).expand(mseg, n, ma).reshape(mseg, -1) + 1

        seg_ = (torch.arange(n, device=device, dtype=torch.long) * mseg).to(z.device)
        seg_ = seg_.unsqueeze(1).expand(n, ma)
        seg_ = (seg_ + s).flatten()
        mapback_atoms = torch.arange(n, device=device, dtype=torch.long)
        mapback_atoms = mapback_atoms.unsqueeze(1).expand(n, ma).flatten() * mseg

        for i in range(mseg):
            if (i - self.expand) > 0:
                i_min = -self.expand
            else:
                i_min = 0
            if (i + self.expand) <= (mseg - 1):
                i_max = self.expand
            else:
                i_max = 0

            mask_min = seg_ >= (mapback_atoms + i_min + i)
            mask_max = seg_ <= (mapback_atoms + i_max + i)
            mask_eq = seg_ == mapback_atoms + i
            mask = mask_max * mask_min
            labels_pre[i, :] *= mask.to(torch.long)
            labels_post[i, :] *= mask_eq.to(torch.long)

        labels_pre = labels_pre.reshape(mseg, n, ma) - 1
        labels_post = labels_post.reshape(mseg, n, ma) - 1

        return labels_pre, labels_post


def split_polymer(coordinates, atomic_numbers, segments, device, extend=1):
    #
    # coordinates = mm.get_coordinates(units='au')
    # atomic_number = mm.get_atomic_numbers()
    # segments = mm.segment_idx

    split_coords = []
    split_atomic_numbers = []
    split_mask = []
    segments = np.array(segments)
    for s in np.unique(segments):
        mask_padd = (segments >= s - extend) & (segments <= s + extend)
        mask = segments[mask_padd] == s
        split_coords.append(torch.tensor(coordinates[mask_padd, :], dtype=torch.float))
        split_atomic_numbers.append(torch.tensor(atomic_numbers[mask_padd], dtype=torch.long))
        ans = atomic_numbers[mask_padd]
        ans[~mask] = -1
        split_mask.append(torch.tensor(ans, dtype=torch.long))

    coordinates = torch.nn.utils.rnn.pad_sequence(split_coords).transpose(0, 1).to(device)
    atomic_number = torch.nn.utils.rnn.pad_sequence(split_atomic_numbers, padding_value=-1).transpose(0, 1).to(device)
    mask = torch.nn.utils.rnn.pad_sequence(split_mask, padding_value=-1).transpose(0, 1).to(device)
    return atomic_number, coordinates, mask


class DatasetOutput:
    def __init__(
        self, atomic_numbers, coordinates, charge, symmetry, primitive_centers, exponents, density_matrix,
        natoms, nprimitives, extra_property, segment=None, segment_charge=None
    ):

        self.atomic_numbers = atomic_numbers
        self.coordinates = coordinates
        self.charge = charge
        self.symmetry = symmetry
        self.primitive_centers = primitive_centers
        self.exponents = exponents
        self.density_matrix = density_matrix
        self.number_atoms = natoms
        self.number_primitives = nprimitives
        self.extra_property = extra_property
        self.segment = segment
        self.segment_charge = segment_charge
        self.attributes = [
            'atomic_numbers', 'coordinates', 'charge', 'symmetry', 'primitive_centers',
            'exponents', 'density_matrix', 'extra_property', 'segment', 'segment_charge'
        ]

    def to(self, device):
        for item in self.attributes:
            u = self.__getattribute__(item)
            if u is not None:
                self.__setattr__(item, u.to(device))


class H5MonomerDataset(data.Dataset):
    def __init__(
            self, device, dtype, ids: List[str], file: str, property_file: str = None
    ):
        """
        H5MonomerDataset
        ================

        Allows to load batches of coordinates + atom_types + topology + charge + function integrals +
        coefficients targets. It is used to train deep learning models.

        Parameters
        ----------
        device: str
            Device where to place the data. We recommend to move data to gpu outside this object
        dtype: torch.dtype
            We recommend to employ torch.float
        property_file: str 
            file from which to bring up data
        ids: List[str]
            list of entries to read from the h5 source file
        file: h5 file
            h5 file containing information
        
        Attributes
        ----------
        max_atoms: int
            number of atoms employed to pad dataset outputs
        max_primitives: int
            number of primitives employed to pad dataset outputs

        Examples
        --------
        >>> data = H5MonomerDataset.from_json(dataset_file)
        
        """
        super(H5MonomerDataset, self).__init__()
        self.ids = dict(total=ids)
        self.device = device
        self.dtype = dtype
        self.property = []
        self.labels = []
        self.coordinates = []
        self.charges = []
        self.symmetry = []
        self.centers = []
        self.exp = []
        self.natoms = []
        self.nprimitives = []
        self.dm = []
        self.property = []
        self.source = h5.File(file, 'r')
        if property_file is not None:
            with open(property_file) as f:
                self.property_source = json.load(f)
        else:
            self.property_source = None
        ids_purged = []
        for idx in range(len(self.ids['total'])):
            try:
                out = self.load(idx)
            except KeyError:
                continue
            ids_purged.append(self.ids['total'][idx])
            na, nps, x, l, q, s, cnts, xp, dm = out
            self.natoms.append(na)
            self.labels.append(l)
            self.coordinates.append(x)
            self.charges.append(q)
            self.nprimitives.append(nps)
            self.symmetry.append(s)
            self.centers.append(cnts)
            self.exp.append(xp)
            self.dm.append(dm)

            if self.property_source is not None:
                prop = self.load_property(idx)
                self.property.append(prop)

        self.ids['remaining'] = ids_purged
        self.max_atoms = max(self.natoms)
        self.max_primitives = max(self.nprimitives)
        self.source.close()

    def __len__(self):
        return len(self.ids)

    def split(self, fraction: float, shuffle=False):
        """
        split
        =====

        Splits the dataset into training and testing

        Parameters
        ----------
        fraction: float
            Fraction of the dataset to be included in the **Training** set
        shuffle: bool
            Permutes the dataset to generate training/test lists

        
        """
        n_index = len(self.ids['remaining'])
        index = [i for i in self.ids['remaining']]
        if shuffle:
            np.random.permutation(index)

        split_point = int(n_index * fraction)
        training_index = index[:split_point]
        testing_index = index[split_point:]
        self.ids['train'] = training_index
        self.ids['test'] = testing_index

    def load(self, item):
        identifier = self.ids['total'][item]
        g = self.source[identifier]

        natoms = int(g.attrs['number_centers'])
        nprims = int(g.attrs['number_primitives'])
        charges = int(g.attrs['total_charge'])

        coords = torch.tensor(g['coordinates'][:, :], device=torch.device('cpu'), dtype=self.dtype)
        atomic_numbers = torch.tensor([get_atomic_number(i.decode('UTF-8')) for i in g['atomic_symbols'][:]],
                                      dtype=self.dtype, device=torch.device('cpu'))
        # labels = convert_label2tensor(atomic_numbers, device=torch.device('cpu'))

        symmetry = torch.tensor(g['symmetry_indices'][:], dtype=torch.long, device=torch.device('cpu'))
        centers = torch.tensor(g['primitive_centers'][:], dtype=torch.long, device=torch.device('cpu'))
        exp = torch.tensor(g['exponents'][:], dtype=self.dtype, device=torch.device('cpu'))
        dm = torch.tensor(g['density_matrix'][:, :], dtype=self.dtype, device=torch.device('cpu'))
        return natoms, nprims, coords, atomic_numbers, charges, symmetry, centers, exp, dm

    def load_property(self, item):
        identifier = self.ids[item]
        return torch.tensor(self.property_source[identifier], dtype=self.dtype, device=self.device)

    def __getitem__(self, item):
        return self.pad(item)

    def pad(self, item):
        coords = self.coordinates[item]
        labels = self.labels[item]
        charge = self.charges[item]
        symmetry = self.symmetry[item]
        centers = self.centers[item]
        exp = self.exp[item]
        na = self.natoms[item]
        nps = self.nprimitives[item]
        dm = self.dm[item]

        padded_coords = torch.zeros(self.max_atoms, 3, dtype=self.dtype, device=torch.device('cpu'))
        padded_labels = torch.ones(self.max_atoms, dtype=torch.long, device=torch.device('cpu')) * -1
        padded_charge = torch.zeros(1, dtype=self.dtype, device=torch.device('cpu'))

        padded_coords[:na, :] = coords
        padded_labels[:na] = labels
        padded_charge[0] = charge
        padded_symmetry = torch.ones(self.max_primitives, dtype=torch.long, device=torch.device('cpu')) * -1
        padded_centers = torch.ones(self.max_primitives, dtype=torch.long, device=torch.device('cpu')) * -1
        padded_exponents = torch.zeros(self.max_primitives, dtype=self.dtype, device=torch.device('cpu'))

        padded_dm = torch.zeros(
            self.max_primitives, self.max_primitives, dtype=self.dtype, device=torch.device('cpu'))
        padded_symmetry[:nps] = symmetry
        padded_centers[:nps] = centers
        padded_exponents[:nps] = exp
        padded_dm[:nps, :nps] = dm

        out = [item, padded_labels, padded_coords, padded_charge]
        out += [padded_symmetry, padded_centers, padded_exponents, padded_dm]

        if self.property_source is not None:
            out += [self.property[item]]
        return out

    def epoch(self, split='remaining', shuffle=True, batch_size=32):
        """
        epoch
        =====

        Iterates the dataset

        Parameters
        ----------
        split: str
            Use either remaining, train or test
        shuffle: bool
            Permutes the datast
        batch_size: int
            Size of the epoochs, in terms of number of molecules
        
        """

        ids = self.ids[split]
        n = len(ids)
        index = np.arange(n)
        if shuffle:
            index = np.random.permutation(index)

        n_batches = (n // batch_size) + 1 * (n % batch_size > 0)
        index = np.array_split(index, n_batches)

        for batch in index:
            z_ = []
            r_ = []
            q_ = []
            sym_ = []
            cnt_ = []
            xp_ = []
            dm_ = []
            for i in batch:
                item, z, r, q, sym, cnt, xp, dm = self.pad(i)
                z_.append(z)
                r_.append(r)
                q_.append(q)
                sym_.append(sym)
                cnt_.append(cnt)
                xp_.append(xp)
                dm_.append(dm)

            yield DatasetOutput(
                atomic_numbers=torch.stack(z_, dim=0),
                coordinates=torch.stack(r_, dim=0), charge=torch.stack(q_, dim=0),
                symmetry=torch.stack(sym_, dim=0), primitive_centers=torch.stack(cnt_, dim=0),
                exponents=torch.stack(xp_, dim=0), density_matrix=torch.stack(dm_, dim=0),
                natoms=None, nprimitives=None, extra_property=None
            )

    @staticmethod
    def from_json(file, device, float_dtype):
        """
        from_json
        =========

        Parameters
        ----------
        file: str
            Json file with dataset and index entries

        device: str
        float_dtype: torch.dtype
        """
        file = Path(file)
        path = file.parent

        with open(file) as f:
            reference = json.load(f)

        source = reference['source']
        index = reference['index']

        if type(index) == str:
            with open(path / index) as f:
                index = [i.strip() for i in f.readlines()]

        return H5MonomerDataset(
            device=device, dtype=float_dtype, ids=index, file=path / source
        )


class H5PolymerDataset(H5MonomerDataset):
    def __init__(
            self, device, dtype, ids: List[str], h5file: str, jsonfile: str
    ):
        """
        H5PolymerDataset
        ---
        Allows to load batches of
            coordinates + atom_types
            + segments + segment charge
            + sym + centers + exp + density matrix
        It is used to feed wavefunctiondensity objects and to train deep learning models.

        :param device: either cuda or cpu
        :param dtype: float32 or float64 to specify either single or double precission
        :param jsonfile: stores molecular representation, like segments, atom types and segment charges
        :param ids: names of the molecules that will be read by this data loader
        :param h5file: names of the molecules that will be read by this data loader
        """

        H5MonomerDataset.__init__(
            self, device=device, dtype=dtype, ids=ids, file=h5file
        )
        self.topology_source = self.load_json(jsonfile)
        self.segments = []
        self.segment_charges = []
        self.nsegments = []
        for i in range(len(ids)):
            identifier = self.ids['remaining'][i]
            ns, seg, segq = self.load_topo_info(identifier)
            self.nsegments.append(ns)
            self.segments.append(seg)
            self.segment_charges.append(segq)
        self.max_segments = max(self.nsegments)

    @staticmethod
    def load_json(jsonfile):
        with open(jsonfile) as f:
            jsonsource = json.load(f)
        return jsonsource

    def __len__(self):
        return len(self.ids)

    def load_topo_info(self, identifier):
        mol = self.topology_source[identifier]
        mol_segments = torch.tensor(mol['segments'], dtype=torch.long, device=torch.device('cpu'))
        mol_segments_charge = torch.tensor(mol['segment_charges'], dtype=torch.long, device=torch.device('cpu'))
        n_segments = mol_segments.unique().size()[0]
        return n_segments, mol_segments, mol_segments_charge

    def pad(self, item):
        coords = self.coordinates[item]
        labels = self.labels[item]
        segments = self.segments[item]
        segments_charge = self.segment_charges[item]
        symmetry = self.symmetry[item]
        centers = self.centers[item]
        exp = self.exp[item]
        na = self.natoms[item]
        nps = self.nprimitives[item]
        ns = self.nsegments[item]
        dm = self.dm[item]

        padded_coords = torch.zeros(self.max_atoms, 3, dtype=self.dtype, device=torch.device('cpu'))
        padded_labels = torch.ones(self.max_atoms, dtype=torch.long, device=torch.device('cpu')) * -1
        padded_segments = torch.zeros(self.max_atoms, dtype=torch.long, device=torch.device('cpu'))
        padded_segment_charges = torch.zeros(self.max_segments, dtype=self.dtype, device=torch.device('cpu'))

        padded_symmetry = torch.ones(self.max_primitives, dtype=torch.long, device=torch.device('cpu')) * -1
        padded_centers = torch.ones(self.max_primitives, dtype=torch.long, device=torch.device('cpu')) * -1
        padded_exponents = torch.zeros(self.max_primitives, dtype=self.dtype, device=torch.device('cpu'))
        padded_dm = torch.zeros(self.max_primitives, self.max_primitives, dtype=self.dtype, device=torch.device('cpu'))

        padded_coords[:na, :] = coords
        padded_labels[:na] = labels

        padded_segments[:na] = segments
        padded_segment_charges[:ns] = segments_charge

        padded_symmetry[:nps] = symmetry
        padded_centers[:nps] = centers
        padded_exponents[:nps] = exp
        padded_dm[:nps, :nps] = dm

        return item, padded_labels, padded_coords, padded_segments, padded_segment_charges,  padded_symmetry, \
            padded_centers, padded_exponents, padded_dm

    @staticmethod
    def from_json(file, device, float_dtype):
        file = Path(file)
        path = file.parent

        with open(file) as f:
            reference = json.load(f)

        source = reference['source']
        index = reference['index']
        mol = reference['mol']

        if type(index) == str:
            with open(path / index) as f:
                index = [i.strip() for i in f.readlines()]

        return H5PolymerDataset(
            device=device, dtype=float_dtype, ids=index, h5file=path / source, jsonfile=path / mol
        )

    def epoch(self, split, shuffle=True, batch_size=32):

        ids = self.ids[split]
        n = len(ids)
        index = np.arange(n)
        if shuffle:
            index = np.random.permutation(index)
        if len(ids) % batch_size != 0:
            n_batches = (n // batch_size) + 1
        else:
            n_batches = n // batch_size
        index = np.array_split(index, n_batches)

        for batch in index:
            z_ = []
            r_ = []
            sq_ = []
            s_ = []
            sym_ = []
            cnt_ = []
            xp_ = []
            dm_ = []
            for i in batch:
                item, z, r, s, sq, sym, cnt, xp, dm = self.pad(i)
                z_.append(z)
                r_.append(r)
                sq_.append(sq)
                s_.append(s)
                sym_.append(sym)
                cnt_.append(cnt)
                xp_.append(xp)
                dm_.append(dm)

            yield DatasetOutput(
                atomic_numbers=torch.stack(z_, dim=0),
                coordinates=torch.stack(r_, dim=0),
                segment=torch.stack(s_, dim=0), charge=None,
                segment_charge=torch.stack(sq_, dim=0),
                symmetry=torch.stack(sym_, dim=0), primitive_centers=torch.stack(cnt_, dim=0),
                exponents=torch.stack(xp_, dim=0), density_matrix=torch.stack(dm_, dim=0),
                natoms=None, nprimitives=None, extra_property=None
            )


class AniLoader:

    def __init__(
        self, path, device: torch.device, max_heavy_atoms=5, min_heavy_atoms=1,
        max_conformations=720, normalize=False,
        remove_self_interaction=True,
        source_type: str = 'processed'
    ):

        self.path = path
        self.device = device
        self.max_heavy_atoms = max_heavy_atoms
        self.min_heavy_atoms = min_heavy_atoms
        self.max_conformations = max_conformations

        self.energy = []
        self.coordinates = []
        self.labels = []
        self.normalize = normalize
        self.remove_si = remove_self_interaction
        self.source_type = source_type
        if self.source_type == 'native':
            self.load = self.load_from_native
        elif self.source_type == 'processed':
            self.load = self.load_from_processed
        else:
            raise IOError('unknown format')

        if self.normalize:
            self.mean = None
            self.std = None

        for z, r, y in self.load():

            self.energy += y
            self.coordinates += r
            self.labels += z

        if self.normalize:
            self.mean = torch.cat(self.energy).mean()
            self.std = torch.cat(self.energy).std()

        self.n_items = len(self.energy)

    def load_from_processed(self):
        source = h5.File(self.path, 'r')
        for i, key in enumerate(source.keys()):
            try:
                mol = source[key]
            except KeyError:
                continue

            species = [i.decode('UTF-8') for i in mol['species']]
            labels = [get_atomic_number(i) for i in species]
            coordinates = mol['coordinates'][:self.max_conformations, :, :]
            energies = mol['energies'][:self.max_conformations]
            n_conformations = energies.size

            labels = torch.tensor(labels, dtype=torch.long).unsqueeze(0).expand(
                n_conformations, len(labels))
            coordinates = torch.tensor(coordinates, dtype=torch.float)
            energies = torch.tensor(energies, dtype=torch.float)

            labels = [t.squeeze(0) for t in labels.split(dim=0, split_size=1)]
            coordinates = [t.squeeze(0) for t in coordinates.split(dim=0, split_size=1)]
            energies = energies.split(dim=0, split_size=1)

            yield labels, coordinates, energies
        source.close()

    def load_from_native(self):
        for i in range(self.min_heavy_atoms, self.max_heavy_atoms):
            print('-- heavy atoms : {:d}'.format(i))
            file_path = self.path / 'ani_gdb_s{:02d}.h5'.format(i)
            source = h5.File(file_path, 'r')
            group_keys = list(source.keys())[0]
            group = source[group_keys]
            n_mols = len(group)

            for j in range(n_mols):
                key = '{:s}-{:d}'.format(group_keys, j)
                try:
                    mol = group[key]
                except KeyError:
                    continue
                species = [i.decode('UTF-8') for i in mol['species']]
                labels = [get_atomic_number(i) for i in species]
                coordinates = mol['coordinates'][:self.max_conformations, :, :]
                energies = mol['energies'][:self.max_conformations]

                n_conformations = energies.size

                labels = torch.tensor(labels, dtype=torch.long).unsqueeze(0).expand(
                    n_conformations, len(labels))
                coordinates = torch.tensor(coordinates, dtype=torch.float)
                energies = torch.tensor(energies, dtype=torch.float)

                labels = [t.squeeze(0) for t in labels.split(dim=0, split_size=1)]
                coordinates = [t.squeeze(0) for t in coordinates.split(dim=0, split_size=1)]
                energies = energies.split(dim=0, split_size=1)

                yield labels, coordinates, energies
            source.close()

    def fetch(self, item):
        return self.energy[item], self.labels[item], self.coordinates[item]

    def __getitem__(self, item):
        return self.fetch(item)

    def __len__(self):
        return self.n_items

    def epoch(self, batch_size, shuffle=True):
        index = np.arange(self.n_items)
        if shuffle:
            index = np.random.permutation(index)
        n_batches = (self.n_items // batch_size) + 1
        index = np.array_split(index, n_batches)

        for batch in index:

            energy = [self.energy[i] for i in batch]
            coordinates = [self.coordinates[i] for i in batch]
            labels = [self.labels[i] for i in batch]

            energy = pad_sequence(energy, padding_value=0.0).transpose(0, 1).to(self.device).squeeze(1)
            coordinates = pad_sequence(coordinates, padding_value=0.0).transpose(0, 1).to(self.device)
            labels = pad_sequence(labels, padding_value=-1).transpose(0, 1).to(self.device)

            if self.normalize:
                energy -= self.mean.item()
                energy /= self.std.item()

            yield energy, labels, coordinates

    def calculate_self_interaction_energy(self):
        x = []
        y = []
        for i in range(self.n_items):

            x.append(self.labels[i].bincount(minlength=4).tolist())
            y.append(self.energy[i].item())

        return y, x


class AMDParameters:

    primary_template = {
        "_NAME": str,
        "_VERSION": str,
        "_DESCRIPTION": str,
        "_MODEL": dict,
        "_MAXFUNS": int,
        "_NELEMENTS": int,
        "_UNITS": str
    }

    function_template = {
        "_NAME": str,
        "_PARAMS": dict,
        "_FROZEN": bool,
        "_CONNECT": str,
        "_TYPE": str
    }

    def __init__(self, contents):
        AMDParameters.check_fields(contents)
        self.contents = contents

    def get_maxfunctions(self):
        return self.contents['_MAXFUNS']

    def get_included_elements(self):
        return list(self.contents['_MODEL'].keys())

    def get_nelements(self):
        return int(self.contents['_NELEMENTS'])

    def get_element(self, element: str):
        try:
            return self.contents['_MODEL'][element]
        except KeyError:
            raise IOError("element {:s} is not included".format(element))

    def iter_element(self, element: str):
        try:
            funs = self.contents['_MODEL'][element]
        except KeyError:

            raise IOError("element {:s} is not included".format(element))
        for f in funs:
            yield f['_PARAMS']

    def keep_frozen(self):
        keep = dict()
        max_elements = 0
        for element, fun_list in self.contents['_MODEL'].items():
            keep[element] = []
            n_elements = 0
            for f in fun_list:
                if f['_FROZEN']:
                    keep[element].append(f)
                    n_elements += 1
            if max_elements < n_elements:
                max_elements = n_elements

        contents = self.contents.copy()
        contents['_MODEL'] = keep
        contents['_MAXFUNS'] = max_elements
        return AMDParameters(contents)

    def remove_frozen(self):
        keep = dict()
        max_elements = 0
        for element, fun_list in self.contents['_MODEL'].items():
            keep[element] = []
            n_elements = 0
            for f in fun_list:
                if not f['_FROZEN']:
                    keep[element].append(f)
                    n_elements += 1
            if max_elements < n_elements:
                max_elements = n_elements

        contents = self.contents.copy()
        contents['_MODEL'] = keep
        contents['_MAXFUNS'] = max_elements
        return AMDParameters(contents)

    @staticmethod
    def from_file(filename: Union[str, Path]):

        with open(filename) as f:

            contents = json.load(f)

        return AMDParameters(contents)

    @staticmethod
    def check_fields(contents: Dict):

        for key, item_type in AMDParameters.primary_template.items():

            try:
                item = contents[key]
            except KeyError:
                raise IOError("missing field {:s} in primary keys".format(key))

            if type(item) is not item_type:
                raise TypeError("item {:s} should be type {:s}".format(key, str(item_type)))

        model = contents['_MODEL']
        for element_symbol, element_model in model.items():
            for fun in element_model:
                for key, item_type in AMDParameters.function_template.items():

                    try:
                        item = fun[key]
                    except KeyError:
                        raise IOError("missing field {:s} within models".format(key))

                    if type(item) is not item_type:
                        raise TypeError("item {:s} should be type {:s}".format(key, str(item_type)))
        return True
