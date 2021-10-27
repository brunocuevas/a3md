import torch
from a2mdnet.sampling import IntegrationGrid
from a2mdnet.density_models import WaveFunctionDensity
from a2mdnet.data import topology_from_coordinates, H5MonomerDataset
from a2mdnet.protocols.ddnn.ddnn_zoo import process_species_str
from a2mdnet.protocols.sdnn.sdnn_zoo import declare_sdnnx
import click
import math
from a2mdio.units import bohr

cli = click.Group()


def train(
    name, model, data, opt, schd, device, epochs, radial_resolution, angular_grid, batch_size, bond_cutoff
):

    model = model.to(device)
    sampler = IntegrationGrid(grid=angular_grid, radial_resolution=radial_resolution, units=bohr).to(device)
    wfn = WaveFunctionDensity().to(device)
    print(model)
    for i in range(epochs + 1):
        test_l2 = 0.0
        with torch.no_grad():
            for u in data.epoch(split='test', shuffle=False, batch_size=batch_size):
                u.to(device)
                _, dv, w = sampler.sample(u.atomic_numbers, u.coordinates)
                topology = topology_from_coordinates(u.atomic_numbers, u.coordinates, bond_cutoff)
                pred, c = model.forward(dv, u.atomic_numbers, u.coordinates, u.charge, topology)
                ref = wfn.density(dv, u.primitive_centers, u.exponents, u.symmetry, u.density_matrix)
                test_l2 += ((ref - pred).pow(2) * w).sum()

        test_l2 = math.sqrt(test_l2 / len(data.ids['test']))
        schd.step(test_l2)
        lr = opt.param_groups[0]['lr']
        print('{:6d} {:18.6e} {:12.6e}'.format(i, test_l2, lr))

        if i % 10 == 0:
            torch.save(model, name + '_{:06d}.pt'.format(i))

        for u in data.epoch(split='train', shuffle=True, batch_size=batch_size):
            u.to(device)
            _, dv, w = sampler.sample(u.atomic_numbers, u.coordinates)
            topology = topology_from_coordinates(u.atomic_numbers, u.coordinates, bond_cutoff)
            pred, c = model.forward(dv, u.atomic_numbers, u.coordinates, u.charge, topology)
            ref = wfn.density(dv, u.primitive_centers, u.exponents, u.symmetry, u.density_matrix)
            l2 = ((ref - pred).pow(2) * w).sum()
            opt.zero_grad()
            l2.backward()
            opt.step()

        torch.save(
            {
                'model': model,
                'optimizer': opt.state_dict(),
                'scheduler': schd.state_dict()
            }, '.tmp.checkpoint.pt'
        )
    return model


@click.command()
@click.option('--species', default='H,C,N,O,S')
@click.option('--bond_cutoff', default=3.5)
@click.option('--learning_rate', default=1e-3)
@click.option('--epochs', default=200)
@click.option('--batch_size', default=16)
@click.option('--radial_resolution', default=5)
@click.option('--angular_grid', type=click.Choice(['minimal', 'xtracoarse', 'coarse']), default='minimal')
@click.option('--device', default='cuda:0')
@click.option('--train_fraction', default=0.8)
@click.option('--weight_decay', default=1e-5)
@click.argument('name')
@click.argument('dataset')
def declare_train_a2md(
    name, dataset, species, bond_cutoff, learning_rate,
    epochs, batch_size, radial_resolution, angular_grid, device, train_fraction,
    weight_decay
):
    print('-- declare and train : a2mdnet')
    species = process_species_str(species)
    device = torch.device(device)
    model = declare_sdnnx(species)
    dataset = H5MonomerDataset.from_json(dataset, device=device, float_dtype=torch.float)
    dataset.split(train_fraction, True)
    opt = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    schd = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5)
    print('-- training for {:d} epochs max'.format(epochs))
    model = train(
        name, model, dataset, opt, schd, device, epochs, radial_resolution, angular_grid, batch_size,
        bond_cutoff
    )
    torch.save(model, name)


cli.add_command(declare_train_a2md)


if __name__ == '__main__':
    cli()