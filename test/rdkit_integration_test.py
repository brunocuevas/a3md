from a3mdnet.input import mol2_to_tensor, mol_to_tensor
from a3mdnet.data import DatasetOutput
from rdkit.Chem import AllChem
import torch
import click

cli = click.Group()


@click.command()
@click.option('--device', default='cpu', type=click.Choice(['cpu', 'cuda:0']))
@click.argument('file')
def convert_mol2(file, device):
    """

    Checking Mol2 -> Tensors

    :param file: mol2 file
    :param device: either cuda:0 or cpu
    :return:
    """

    mol = AllChem.MolFromMol2File(file, removeHs=False)
    u: DatasetOutput = mol2_to_tensor(mol, torch.device(device), min_size=4)
    print(u.atomic_numbers)
    print(u.charge)
    print(u.coordinates)
    return


@click.command()
@click.option('--device', default='cpu', type=click.Choice(['cpu', 'cuda:0']))
@click.option('--charge', default=0, type=int)
@click.argument('file')
def convert_mol(file, charge, device):
    """

    Checking Mol2 -> Tensors

    :param file: mol2 file
    :param charge: molecular charge. default 0.
    :param device: either cuda:0 or cpu
    :return:
    """

    mol = AllChem.MolFromMolFile(file, removeHs=False)
    u: DatasetOutput = mol_to_tensor(mol, charge, torch.device(device), min_size=4)
    print(u.atomic_numbers)
    print(u.charge)
    print(u.coordinates)
    return


cli.add_command(convert_mol2)
cli.add_command(convert_mol)


if __name__ == '__main__':
    cli()
