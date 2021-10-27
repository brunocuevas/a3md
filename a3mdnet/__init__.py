from pathlib import Path
import pint
from mendeleev import get_table
import json


def open_file(name):
    with open(name) as f:
        try:
            file = json.load(f)
        except json.JSONDecodeError:
            raise IOError("could not read amd params file")
        return file


ureg = pint.UnitRegistry()
ang2bohr = (1*ureg.angstrom).to(ureg.bohr).magnitude
bohr2ang = (1*ureg.bohr).to(ureg.angstrom).magnitude

LIBRARY_PATH = Path(__file__).parent
ELEMENT2NN = {-1: -1, 0: -1, 1: 0, 6: 1, 7: 2, 8: 3, 16: 4}
SYMBOL2NN = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'S': 4}
NN2ELEMENT = {0: 1, 1: 6, 2: 7, 3: 8, 4: 16}
ELEMENT2SYMBOL = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 16: 'S'}
ALLOWED_SPECIES = [1, 6, 7, 8, 16]

LEBEDEV_GRIDS = dict(
    minimal=LIBRARY_PATH / 'params/lebedev_011.txt',
    xtcoarse=LIBRARY_PATH / 'params/lebedev_017.txt',
    coarse=LIBRARY_PATH / 'params/lebedev_027.txt',
    medium=LIBRARY_PATH / 'params/lebedev_053.txt',
    tight=LIBRARY_PATH / 'params/lebedev_101.txt'
)

MODELS = dict(
    a2mdc=LIBRARY_PATH / "models/a2mdnet_coeff.pt"
)

PERIODIC_TABLE = get_table('elements')
MAP_AN2SYMBOL = PERIODIC_TABLE[['atomic_number', 'symbol']].set_index('atomic_number')
MAP_SYMBOL2AN = PERIODIC_TABLE[['atomic_number', 'symbol']].set_index('symbol')
VDW_PARAMS = open_file(LIBRARY_PATH / 'params/vanDerWaals_coefficients.json')

with open(LIBRARY_PATH / 'params/wfn_symmetry_index.json') as f:
    WFN_SYMMETRY_INDEX = json.load(f)


def get_atomic_number(symbol):
    an = MAP_SYMBOL2AN.loc[symbol]['atomic_number']
    return an


def get_symbol(atomic_number):
    atomic_number = int(atomic_number)
    if atomic_number == -1:
        return None
    s = MAP_AN2SYMBOL.loc[atomic_number]['symbol']
    return s
