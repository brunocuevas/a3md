import click
from a3mdnet.density_models import GenAMD
from a3mdnet.data import AMDParameters
from a3mdnet import get_atomic_number

def process_species_str(species):
    species = species.split(',')
    species = [get_atomic_number(i) for i in species]
    species = dict((j, i) for i, j in enumerate(species))
    species[-1] = -1
    return species


@click.command()
@click.argument('NAME')
def load_parameters(name):

    p = AMDParameters.from_file(name)
    table = process_species_str('H,C,N,O,S')
    u = GenAMD(p, table=table)
    print('--done!')


if __name__ == "__main__":
    load_parameters()

