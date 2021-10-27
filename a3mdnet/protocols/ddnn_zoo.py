from torch import nn
from a3mdnet.models import ddnn
from a3mdnet.data import AMDParameters
from a3mdnet import get_atomic_number, LIBRARY_PATH
from a3mdnet.density_models import HarmonicGenAMD, GenAMD


def process_species_str(species):
    species = species.split(',')
    species = [get_atomic_number(i) for i in species]
    species = dict((j, i) for i, j in enumerate(species))
    species[-1] = -1
    return species


def declare_ddnnx(species, rc, convolutions, update_decay, update_rate, spacing, k):
    predictor = ddnn.DDNNx(
        n_species=len(species.keys()), embedding_dim=128, rc=rc,
        update_rate=update_rate, n_convolutions=convolutions, decay=update_decay,
        distances=[0, 0.9 * spacing, 1.8 * spacing, 2.7 * spacing],
        widths=[spacing, spacing, spacing, spacing],
        convolve_net=nn.Sequential(
            nn.Linear(640, 256),
            nn.Tanh(),
            nn.Linear(256, 128)
        ),
        edge_net=nn.Sequential(
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 9)
        ),
        pool_net=nn.Sequential(
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 8)
        )
    )
    amd_aniso = AMDParameters.from_file(LIBRARY_PATH / r'params/a3md_anisotropic_basis.json')
    amd_iso = AMDParameters.from_file(LIBRARY_PATH / r'params/a3md_isotropic_basis.json')
    protodensity = AMDParameters.from_file(LIBRARY_PATH / r'params/a3md_promolecule.json')
    hgamd = HarmonicGenAMD(amd_aniso, max_angular_moment=3, table=species, k=1)
    gamd = GenAMD(amd_iso, table=species)
    pamd = GenAMD(protodensity, table=species)
    a3md = ddnn.A3MDnet(
        predictor=predictor,
        density_model=gamd,
        deformation_model=hgamd,
        protodensity_model=pamd,
        table=species
    )
    return a3md


def declare_tdnnx(species, rc, convolutions, update_decay, update_rate, spacing, k):
    predictor = ddnn.TDNNx(
        n_species=len(species.keys()), embedding_dim=128, rc=rc, k=k,
        update_rate=update_rate, n_convolutions=convolutions, decay=update_decay,
        distances=[0, 0.9 * spacing, 1.8 * spacing, 2.7 * spacing],
        widths=[spacing, spacing, spacing, spacing],
        convolve_net=nn.Sequential(
            nn.Linear(640, 256),
            nn.Tanh(),
            nn.Linear(256, 128)
        ),
        edge_net=nn.Sequential(
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 9)
        ),
        pool_net=nn.Sequential(
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 8)
        )
    )
    amd_aniso = AMDParameters.from_file(LIBRARY_PATH / r'params/a3md_anisotropic_basis.json')
    amd_iso = AMDParameters.from_file(LIBRARY_PATH / r'params/a3md_isotropic_basis.json')
    protodensity = AMDParameters.from_file(LIBRARY_PATH / r'params/a3md_promolecule.json')
    hgamd = HarmonicGenAMD(amd_aniso, max_angular_moment=3, table=species, k=k)
    gamd = GenAMD(amd_iso, table=species)
    pamd = GenAMD(protodensity, table=species)
    a3md = ddnn.A3MDnet(
        predictor=predictor,
        density_model=gamd,
        deformation_model=hgamd,
        protodensity_model=pamd,
        table=species
    )
    return a3md


def declare_adnnx(species, rc, convolutions, update_decay, update_rate, spacing, k):
    predictor = ddnn.ADNNx(
        n_species=len(species.keys()), embedding_dim=128, rc=rc, k=k,
        update_rate=update_rate, n_convolutions=convolutions, decay=update_decay,
        distances=[0, 0.9 * spacing, 1.8 * spacing, 2.7 * spacing],
        widths=[spacing, spacing, spacing, spacing],
        convolve_net=nn.Sequential(
            nn.Linear(640, 256),
            nn.Tanh(),
            nn.Linear(256, 128)
        ),
        edge_net=nn.Sequential(
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 9)
        ),
        pool_net=nn.Sequential(
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 8)
        ),
        attention_net=nn.Sequential(
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
    )
    amd_aniso = AMDParameters.from_file(LIBRARY_PATH / r'params/a3md_anisotropic_basis.json')
    amd_iso = AMDParameters.from_file(LIBRARY_PATH / r'params/a3md_isotropic_basis.json')
    protodensity = AMDParameters.from_file(LIBRARY_PATH / r'params/a3md_promolecule.json')
    hgamd = HarmonicGenAMD(amd_aniso, max_angular_moment=3, table=species, k=k)
    gamd = GenAMD(amd_iso, table=species)
    pamd = GenAMD(protodensity, table=species)
    a3md = ddnn.A3MDnet(
        predictor=predictor,
        density_model=gamd,
        deformation_model=hgamd,
        protodensity_model=pamd,
        table=species
    )
    return a3md


def declare_ampdnn(species, rc, convolutions, update_decay, update_rate, spacing, k):
    predictor = ddnn.AMPDNN(
        n_species=len(species.keys()), embedding_dim=128, rc=rc,
        n_convolutions=convolutions,
        distances=[0, 0.9 * spacing, 1.8 * spacing, 2.7 * spacing],
        widths=[spacing, spacing, spacing, spacing], update_rate=update_rate, decay=update_decay,
        convolve_net=nn.Sequential(
            nn.Linear(640, 256),
            nn.Tanh(),
            nn.Linear(256, 128)
        ),
        edge_net=nn.Sequential(
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 9)
        ),
        pool_net=nn.Sequential(
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 8)
        ),
        attention_net=nn.Sequential(
            nn.Linear(256, 128),
            nn.CELU(),
            nn.Linear(128, 1)
        )
    )
    amd_aniso = AMDParameters.from_file(LIBRARY_PATH / r'params/a3md_anisotropic_basis.json')
    amd_iso = AMDParameters.from_file(LIBRARY_PATH / r'params/a3md_isotropic_basis.json')
    protodensity = AMDParameters.from_file(LIBRARY_PATH / r'params/a3md_promolecule.json')
    hgamd = HarmonicGenAMD(amd_aniso, max_angular_moment=3, table=species, k=1)
    gamd = GenAMD(amd_iso, table=species)
    pamd = GenAMD(protodensity, table=species)
    a3md = ddnn.A3MDnet(
        predictor=predictor,
        density_model=gamd,
        deformation_model=hgamd,
        protodensity_model=pamd,
        table=species
    )
    return a3md


def declare_mpdnn(species, rc, convolutions, update_decay, update_rate, spacing, k):
    predictor = ddnn.MPDNN(
        n_species=len(species.keys()), embedding_dim=128, rc=rc,
        n_convolutions=convolutions,
        distances=[0, 0.9 * spacing, 1.8 * spacing, 2.7 * spacing],
        widths=[spacing, spacing, spacing, spacing],
        convolve_net=nn.Sequential(
            nn.Linear(640, 256),
            nn.Tanh(),
            nn.Linear(256, 128)
        ),
        edge_net=nn.Sequential(
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 9)
        ),
        pool_net=nn.Sequential(
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 8)
        ),
        update_net=nn.GRUCell(128, 128)
    )
    amd_aniso = AMDParameters.from_file(LIBRARY_PATH / r'params/a3md_anisotropic_basis.json')
    amd_iso = AMDParameters.from_file(LIBRARY_PATH / r'params/a3md_isotropic_basis.json')
    protodensity = AMDParameters.from_file(LIBRARY_PATH / r'params/a3md_promolecule.json')
    hgamd = HarmonicGenAMD(amd_aniso, max_angular_moment=3, table=species, k=1)
    gamd = GenAMD(amd_iso, table=species)
    pamd = GenAMD(protodensity, table=species)
    a3md = ddnn.A3MDnet(
        predictor=predictor,
        density_model=gamd,
        deformation_model=hgamd,
        protodensity_model=pamd,
        table=species
    )
    return a3md


def declare_tmpdnn(species, rc, convolutions, update_decay, update_rate, spacing, k):
    predictor = ddnn.TMPDNN(
        n_species=len(species.keys()), embedding_dim=128, rc=rc, k=k,
        n_convolutions=convolutions,
        distances=[0, 0.9 * spacing, 1.8 * spacing, 2.7 * spacing],
        widths=[spacing, spacing, spacing, spacing],
        convolve_net=nn.Sequential(
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh()
        ),
        edge_net=nn.Sequential(
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 9)
        ),
        pool_net=nn.Sequential(
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 8)
        ),
        update_net=nn.GRUCell(128, 128)
    )
    amd_aniso = AMDParameters.from_file(LIBRARY_PATH / r'params/a3md_anisotropic_basis.json')
    amd_iso = AMDParameters.from_file(LIBRARY_PATH / r'params/a3md_isotropic_basis.json')
    protodensity = AMDParameters.from_file(LIBRARY_PATH / r'params/a3md_promolecule.json')
    hgamd = HarmonicGenAMD(amd_aniso, max_angular_moment=3, table=species, k=k)
    gamd = GenAMD(amd_iso, table=species)
    pamd = GenAMD(protodensity, table=species)
    a3md = ddnn.A3MDnet(
        predictor=predictor,
        density_model=gamd,
        deformation_model=hgamd,
        protodensity_model=pamd,
        table=species
    )
    return a3md


def declare_mpinn(species, rc, convolutions, update_decay, update_rate, spacing, k):
    predictor = ddnn.MPINN(
        n_species=len(species.keys()), embedding_dim=128,
        n_convolutions=convolutions,
        distances=[0, 0.9 * spacing, 1.8 * spacing, 2.7 * spacing],
        widths=[spacing, spacing, spacing, spacing],
        convolve_net=nn.Sequential(
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, 128)
        ),
        pool_net=nn.Sequential(
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 8)
        ),
        update_net=nn.GRU(128, 128, 1)
    )
    amd_iso = AMDParameters.from_file(LIBRARY_PATH / r'params/a3md_isotropic_basis.json')
    protodensity = AMDParameters.from_file(LIBRARY_PATH / r'params/a3md_promolecule.json')
    gamd = GenAMD(amd_iso, table=species)
    pamd = GenAMD(protodensity, table=species)
    amd = ddnn.AMDnet(
        predictor=predictor,
        density_model=gamd,
        protodensity_model=pamd,
        table=species
    )
    return amd


def declare_idnnx(species, rc, convolutions, update_decay, update_rate, spacing, k):
    predictor = ddnn.IDNNx(
        n_species=len(species.keys()), embedding_dim=128, update_rate=update_decay,
        distances=[0, 0.9 * spacing, 1.8 * spacing, 2.7 * spacing],
        widths=[spacing, spacing, spacing, spacing],
        convolve_net=nn.Sequential(
            nn.Linear(640, 256),
            nn.Tanh(),
            nn.Linear(256, 128)
        ),
        pool_net=nn.Sequential(
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 8)
        ), n_convolutions=convolutions, decay=update_decay
    )
    amd_iso = AMDParameters.from_file(LIBRARY_PATH / r'params/a3md_isotropic_basis.json')
    protodensity = AMDParameters.from_file(LIBRARY_PATH / r'params/promolecule.json')
    gamd = GenAMD(amd_iso, table=species)
    pamd = GenAMD(protodensity, table=species)
    amd = ddnn.AMDnet(
        predictor=predictor,
        density_model=gamd,
        protodensity_model=pamd,
        table=species
    )
    return amd


model_zoo = dict(
    mpdnn=declare_mpdnn,
    ampdnn=declare_ampdnn,
    mpinn=declare_mpinn,
    tmpdnn=declare_tmpdnn,
    adnn=declare_adnnx,
    tdnn=declare_tdnnx,
    ddnn=declare_ddnnx,
    idnn=declare_idnnx
)
