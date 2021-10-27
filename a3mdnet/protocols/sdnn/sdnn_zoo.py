from torch import nn
from a2mdnet.models.sdnn import SDNNx, A2MDnet
from a2mdio.params import AMDParameters
from a2mdnet.density_models import GenAMD, GaussianGenAMD
from a2mdnet.modules import LocalMap, PairMap
from a2md import LIBRARY_PATH
from a2mdnet import AEV_PARAMETERS

def declare_sdnnx(species):
    predictor = SDNNx(
        symmetry_function_parameters=AEV_PARAMETERS.parent / "aev_polymer.params",
        common_net=LocalMap(
            nn.ModuleList(
                [
                    nn.Sequential(nn.Linear(560, 256), nn.SELU(), nn.Linear(256, 128), nn.SELU(), nn.Linear(128, 64)),
                    nn.Sequential(nn.Linear(560, 256), nn.SELU(), nn.Linear(256, 128), nn.SELU(), nn.Linear(128, 64)),
                    nn.Sequential(nn.Linear(560, 256), nn.SELU(), nn.Linear(256, 128), nn.SELU(), nn.Linear(128, 64)),
                    nn.Sequential(nn.Linear(560, 256), nn.SELU(), nn.Linear(256, 128), nn.SELU(), nn.Linear(128, 64)),
                    nn.Sequential(nn.Linear(560, 256), nn.SELU(), nn.Linear(256, 128), nn.SELU(), nn.Linear(128, 64)),
                ]
            ), output_size=64
        ),
        isotropic_net=LocalMap(
            nn.ModuleList(
                [
                    nn.Sequential(nn.Linear(64, 32), nn.Tanh(), nn.Linear(32, 8)),
                    nn.Sequential(nn.Linear(64, 32), nn.Tanh(), nn.Linear(32, 8)),
                    nn.Sequential(nn.Linear(64, 32), nn.Tanh(), nn.Linear(32, 8)),
                    nn.Sequential(nn.Linear(64, 32), nn.Tanh(), nn.Linear(32, 8)),
                    nn.Sequential(nn.Linear(64, 32), nn.Tanh(), nn.Linear(32, 8)),
                ]
            ), output_size=8
        ),
        anisotropic_net=PairMap(
            nn.ModuleList(
                [
                    nn.Sequential(nn.Linear(128, 64), nn.Tanh(), nn.Linear(64, 6)),
                    nn.Sequential(nn.Linear(128, 64), nn.Tanh(), nn.Linear(64, 6)),
                    nn.Sequential(nn.Linear(128, 64), nn.Tanh(), nn.Linear(64, 6)),
                    nn.Sequential(nn.Linear(128, 64), nn.Tanh(), nn.Linear(64, 6)),
                    nn.Sequential(nn.Linear(128, 64), nn.Tanh(), nn.Linear(64, 6)),
                    nn.Sequential(nn.Linear(128, 64), nn.Tanh(), nn.Linear(64, 6)),
                    nn.Sequential(nn.Linear(128, 64), nn.Tanh(), nn.Linear(64, 6)),
                    nn.Sequential(nn.Linear(128, 64), nn.Tanh(), nn.Linear(64, 6)),
                    nn.Sequential(nn.Linear(128, 64), nn.Tanh(), nn.Linear(64, 6)),
                    nn.Sequential(nn.Linear(128, 64), nn.Tanh(), nn.Linear(64, 6)),
                    nn.Sequential(nn.Linear(128, 64), nn.Tanh(), nn.Linear(64, 6)),
                    nn.Sequential(nn.Linear(128, 64), nn.Tanh(), nn.Linear(64, 6)),
                    nn.Sequential(nn.Linear(128, 64), nn.Tanh(), nn.Linear(64, 6)),
                    nn.Sequential(nn.Linear(128, 64), nn.Tanh(), nn.Linear(64, 6)),
                    nn.Sequential(nn.Linear(128, 64), nn.Tanh(), nn.Linear(64, 6)),
                    nn.Sequential(nn.Linear(128, 64), nn.Tanh(), nn.Linear(64, 6)),
                    nn.Sequential(nn.Linear(128, 64), nn.Tanh(), nn.Linear(64, 6)),
                    nn.Sequential(nn.Linear(128, 64), nn.Tanh(), nn.Linear(64, 6)),
                    nn.Sequential(nn.Linear(128, 64), nn.Tanh(), nn.Linear(64, 6)),
                    nn.Sequential(nn.Linear(128, 64), nn.Tanh(), nn.Linear(64, 6)),
                    nn.Sequential(nn.Linear(128, 64), nn.Tanh(), nn.Linear(64, 6)),
                    nn.Sequential(nn.Linear(128, 64), nn.Tanh(), nn.Linear(64, 6)),
                    nn.Sequential(nn.Linear(128, 64), nn.Tanh(), nn.Linear(64, 6)),
                    nn.Sequential(nn.Linear(128, 64), nn.Tanh(), nn.Linear(64, 6)),
                    nn.Sequential(nn.Linear(128, 64), nn.Tanh(), nn.Linear(64, 6)),
                ]
            ), output_size=6
        )

    )
    amd_aniso = AMDParameters.from_file(LIBRARY_PATH / r'parameters/sgb.anisotropic.gau1.0.json')
    amd_iso = AMDParameters.from_file(LIBRARY_PATH / r'parameters/expgen8.spherical.json')
    protodensity = AMDParameters.from_file(LIBRARY_PATH / r'parameters/protodensity.json')
    ggamd = GaussianGenAMD(amd_aniso, table=species)
    gamd = GenAMD(amd_iso, table=species)
    pamd = GenAMD(protodensity, table=species)
    a2md = A2MDnet(
        predictor=predictor, protodensity_model=pamd,
        density_model=gamd, deformation_model=ggamd, table=species
    )
    return a2md
