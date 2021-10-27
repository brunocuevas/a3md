from a3mdnet.data import H5MonomerDataset
import torch
import click


@click.command()
@click.option('--device', type=click.Choice(['cpu', 'cuda:0']), default='cpu')
@click.option('--split', type=float, default=None)
@click.option('--batch_size', type=int, default=16)
@click.argument('dataset')
def load_dataset(dataset, device, split, batch_size):
    """
    load_dataset
    ============

    This script aims to test if a dataset can be loaded by the the library data loader

    arguments
    ---------
    - device: {"cpu", "cuda:0"}, default="cpu"
        Where to store the data. Our experience is that it's better to use CPU and
        then transfer to GPU
    - split: float, default=None
        Fraction of the dataset that will be used for test
    - batch_size: int, default=16
        Size of the batches employed
    - dataset: str
        JSON file containing paths for index and wfn.h5 files

    """
    click.echo('-- data loading test')
    device = torch.device(device)
    h5md = H5MonomerDataset.from_json(file=dataset, device=device, float_dtype=torch.float)
    click.echo('-- loading done')

    if split is not None:
        h5md.split(split)
        label = 'train'
    else:
        label = 'remaining'

    for i, u in enumerate(h5md.epoch(split=label, batch_size=batch_size, shuffle=False)):
        click.echo('-- displaying atomic numbers for the first epoch')
        print(i, u.atomic_numbers)
        break
    click.echo('-- done!')


if __name__ == '__main__':
    load_dataset()
