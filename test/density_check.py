from a3mdnet.sampling import IntegrationGrid
from a3mdnet.density_models import WaveFunctionDensity
from a3mdnet.data import H5MonomerDataset
import torch
import click


@click.command()
@click.option('--device', type=click.Choice(['cpu', 'cuda:0']), default='cpu')
@click.option('--grid', type=click.Choice(['minimal', 'xtcoarse', 'coarse']), default='minimal')
@click.option('--radial_resolution', type=int, default=15)
@click.argument('model')
@click.argument('dataset')
def check(model, dataset, device, grid, radial_resolution):
    """
    check
    =====

    Predicts the electron density using a model and then it measures the result against the 
    QM reference. The result is calculated as ABSE (absolute error divided the number of electrons)

    arguments
    ---------
    - model: str
        It should point to something that can be loaded with torch.load(model)
    - dataset: str
        JSON file
    - device: {"cpu", "cuda:0"}, default="cpu"
        Where to store the data. Our experience is that it's better to use CPU and
        then transfer to GPU
    - grid: {"minimal", "xtcoarse", "coarse"}
        Size of the angular grid
    - radial_resolution: int, default=15
        Number of radial bins
    

    """
    device = torch.device(device)
    sampler = IntegrationGrid(grid, radial_resolution).to(device)
    wfn = WaveFunctionDensity().to(device)
    data = H5MonomerDataset.from_json(dataset, device=device, float_dtype=torch.float)
    model = torch.load(model).to(device)

    for i, u in enumerate(data.epoch(batch_size=16)):
        u.to(device)
        _, dv, w = sampler.sample(u.atomic_numbers, u.coordinates)
        pred, _ = model.forward(dv, u.atomic_numbers, u.coordinates, u.charge)
        ref = wfn.density(dv, u.primitive_centers, u.exponents, u.symmetry, u.density_matrix)
        ne = (ref * w).sum(1)
        abse = 100 * ((pred - ref).abs() * w).sum(1) / ne
        print("{:12.4e}".format(abse))


if __name__ == '__main__':
    check()
