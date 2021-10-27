import click
import torch
import math
import sys
from torch import nn
from a3mdnet.protocols import ddnn_zoo
from a3mdnet.data import H5MonomerDataset
from a3mdnet.sampling import IntegrationGrid
from a3mdnet.density_models import WaveFunctionDensity
from torch import optim


cli = click.Group()


def train(
    name, model: nn.Module, data: H5MonomerDataset, opt: optim.Optimizer, schd: optim.lr_scheduler,
    device: torch.device, initial_epoch=0, final_epoch=200, radial_resolution=15, angular_grid='minimal',
    batch_size=32
):

    model = model.to(device)
    sampler = IntegrationGrid(grid=angular_grid, radial_resolution=radial_resolution).to(device)
    wfn = WaveFunctionDensity().to(device)
    for i in range(initial_epoch, final_epoch + 1):
        test_l2 = 0.0
        with torch.no_grad():
            for u in data.epoch(split='test', shuffle=False, batch_size=batch_size):
                u.to(device)
                _, dv, w = sampler.sample(u.atomic_numbers, u.coordinates)
                pred, c = model.forward(dv, u.atomic_numbers, u.coordinates, u.charge)
                ref = wfn.density(dv, u.primitive_centers, u.exponents, u.symmetry, u.density_matrix)
                test_l2 += ((ref - pred).pow(2) * w).sum()

        test_l2 = math.sqrt(test_l2 / len(data.ids['test']))
        schd.step(test_l2)
        lr = opt.param_groups[0]['lr']
        click.echo('{:6d} {:18.6e} {:12.6e}'.format(i, test_l2, lr))

        if i % 10 == 0:
            torch.save(model, name + '_{:06d}.pt'.format(i))

        for u in data.epoch(split='train', shuffle=True, batch_size=batch_size):
            u.to(device)
            _, dv, w = sampler.sample(u.atomic_numbers, u.coordinates)
            pred, c = model.forward(dv, u.atomic_numbers, u.coordinates, u.charge)
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
@click.option('--model', default='mpdnn')
@click.option('--species', default='H,C,N,O,S')
@click.option('--rc', default=8.0)
@click.option('--spacing', default=1.88)
@click.option('--update_rate', default=0.1)
@click.option('--update_decay', default=0.5)
@click.option('--learning_rate', default=1e-3)
@click.option('--convolutions', default=0)
@click.option('--k', default=4)
@click.option('--epochs', default=200)
@click.option('--batch_size', default=16)
@click.option('--radial_resolution', default=5)
@click.option('--angular_grid', type=click.Choice(['minimal', 'xtracoarse', 'coarse']), default='minimal')
@click.option('--device', default='cuda:0')
@click.option('--train_fraction', default=0.8)
@click.option('--weight_decay', default=1e-5)
@click.argument('name')
@click.argument('dataset')
def declare_train_a3md(
        name, dataset, model, species, rc, spacing, update_rate,
        update_decay, learning_rate, convolutions, k,
        epochs, device, train_fraction, weight_decay,
        batch_size, radial_resolution, angular_grid
):
    """

    """
    click.echo('-- model: {:s}'.format(model))
    species = ddnn_zoo.process_species_str(species)
    declaration_dict = dict(
        species=species, rc=rc, spacing=spacing, update_rate=update_rate, update_decay=update_decay,
        convolutions=convolutions, k=k
    )
    try:
        model = ddnn_zoo.model_zoo[model](**declaration_dict)
    except KeyError:
        print('unknown model {:s}. Please use:')
        for model_name in ddnn_zoo.model_zoo.keys():
            print('\t{:s}'.format(model_name))
        sys.exit()
    click.echo(model)
    click.echo('---')
    click.echo('-- dataset: {:s}'.format(dataset))
    dataset = H5MonomerDataset.from_json(dataset, device, torch.float)
    dataset.split(train_fraction, True)
    opt = optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    schd = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5)
    model = train(
        name, model=model, data=dataset, opt=opt, schd=schd,
        initial_epoch=0, final_epoch=epochs, device=device,
        radial_resolution=radial_resolution, angular_grid=angular_grid,
        batch_size=batch_size
    )
    torch.save(model, name)


@click.command()
@click.option('--device', default='cuda:0')
@click.option('--train_fraction', default=0.8)
@click.option('--epochs', default=100)
@click.option('--batch_size', default=8)
@click.option('--radial_resolution', default=10)
@click.option('--angular_grid', type=click.Choice(['minimal', 'xtracoarse', 'coarse']), default='minimal')
@click.option('--train_fraction', default=0.8)
@click.option('--weight_decay', default=1e-5)
@click.option('--learning_rate', default=1e-3)
@click.argument('name')
@click.argument('model')
@click.argument('dataset')
def resume(
        name, model, dataset, device, train_fraction, learning_rate,
        epochs, radial_resolution, angular_grid, weight_decay, batch_size
):
    device = torch.device(device)
    model = torch.load(model)
    dataset = H5MonomerDataset.from_json(dataset, device, torch.float)

    dataset.split(train_fraction, True)
    opt = optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    schd = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5)
    model = train(
        name, model=model, data=dataset, opt=opt, schd=schd, initial_epoch=0, final_epoch=epochs, device=device,
        angular_grid=angular_grid, radial_resolution=radial_resolution, batch_size=batch_size
    )
    torch.save(model, name)


cli.add_command(declare_train_a3md)
cli.add_command(resume)


if __name__ == '__main__':
    cli()
