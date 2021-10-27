from rdkit import Chem
from a3mdnet.data import DatasetOutput
import torch
from a3mdnet import ang2bohr


def mol2_to_tensor(mol, device: torch.device, min_size=4):
    """

    :param mol:
    :param device:
    :param min_size:
    :rtype: DatasetOutput
    :return: number of atoms, number of bonds, coords, labels, connectivity, charges
    """
    na = mol.GetNumAtoms()

    if na < min_size:
        coords_tensor = torch.zeros(1, min_size, 3, device=device, dtype=torch.float)
        atomic_numbers = torch.ones(1, min_size, device=device, dtype=torch.long) * -1
        coords_tensor[0, :na, :] = torch.tensor(mol.GetConformer(0).GetPositions() * ang2bohr, dtype=torch.float)
        atomic_numbers[0, :na] = torch.tensor([atom.GetAtomicNum() for atom in mol.GetAtoms()], dtype=torch.long)
    else:
        coords_tensor = torch.tensor(mol.GetConformer(0).GetPositions() * ang2bohr, dtype=torch.float).unsqueeze(0)
        atomic_numbers = torch.tensor([atom.GetAtomicNum() for atom in mol.GetAtoms()], dtype=torch.long).unsqueeze(0)

    atomic_charges = torch.tensor(
        [atom.GetDoubleProp('_TriposPartialCharge') for atom in mol.GetAtoms()], dtype=torch.float, device=device
    )
    charges_tensor = atomic_charges.sum().round().unsqueeze(0)

    u = DatasetOutput(
        atomic_numbers=atomic_numbers, coordinates=coords_tensor, charge=charges_tensor, symmetry=None,
        primitive_centers=None, exponents=None, density_matrix=None, natoms=coords_tensor.shape[1],
        nprimitives=None, extra_property=None, segment=None, segment_charge=None
    )
    return u


def mol_to_tensor(mol, charge, device: torch.device, min_size=4):
    """

    :param mol:
    :param charge:
    :param device:
    :param min_size:
    :rtype: DatasetOutput
    :return: number of atoms, number of bonds, coords, labels, connectivity
    """
    na = mol.GetNumAtoms()

    if na < min_size:
        coords_tensor = torch.zeros(1, min_size, 3, device=device, dtype=torch.float)
        atomic_numbers = torch.ones(1, min_size, device=device, dtype=torch.long) * -1
        coords_tensor[0, :na, :] = torch.tensor(mol.GetConformer(0).GetPositions() * ang2bohr, dtype=torch.float)
        atomic_numbers[0, :na] = torch.tensor([atom.GetAtomicNum() for atom in mol.GetAtoms()], dtype=torch.long)
    else:
        coords_tensor = torch.tensor(mol.GetConformer(0).GetPositions() * ang2bohr, dtype=torch.float).unsqueeze(0)
        atomic_numbers = torch.tensor([atom.GetAtomicNum() for atom in mol.GetAtoms()], dtype=torch.long).unsqueeze(0)

    charges_tensor = torch.tensor(charge, device=device, dtype=torch.float).unsqueeze(0)

    u = DatasetOutput(
        atomic_numbers=atomic_numbers, coordinates=coords_tensor, charge=charges_tensor, symmetry=None,
        primitive_centers=None, exponents=None, density_matrix=None, natoms=coords_tensor.shape[1],
        nprimitives=None, extra_property=None, segment=None, segment_charge=None
    )
    return u
