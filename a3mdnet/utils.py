from a3mdnet.data import map_an2labels
from a3mdnet.functionals import thomas_fermi_kinetic, von_weizsacker_kinetic, dirac_exchange
from a3mdnet.density_models import GenAMD
from a3mdnet.functions import alt_distance_vectors
import torch
from typing import List
import numpy as np


def hirshfeld_weights(dv, atomic_numbers, density_model: GenAMD):
    z = map_an2labels(atomic_numbers=atomic_numbers)
    wh = density_model.hirshfeld_partition(dv, z)
    return wh


def split_calculation(dv, split_size):
    for split_dv in torch.split(dv, split_size, 1):
        yield split_dv


def get_dft_protoenergy(model, dv, w, z, coords):
    g = model.protogradient(dv, z)
    p = model.protodensity(dv, z).clamp(min=1e-22, max=None)
    v = model.protopotential(dv, z)
    nn = model.nuclear_nuclear(z, coords)
    vne = model.protone(z, coords)
    tf = thomas_fermi_kinetic(p, w)
    xc = dirac_exchange(p, w)
    vw = von_weizsacker_kinetic(p, g, w)
    ee = (-v * w * p).sum(1) / 2
    return nn + vne + tf + xc + vw + ee


class DxGrid:

    def __init__(self, device: torch.device, dtype: torch.dtype, resolution: float, spacing: float):
        """
        DxGrid
        ======
    
        Eases representation of electron density in a volumetric format that can be read by popular chemistries
        software as Chimera, PyMol, or VMD

        Parameters
        ----------
        device: torch.Device
        dtype: torch.dtype
        resolution: float
        spacing: float

        Note
        ----
        The DX output units are Angstroms, but we employ Bohrs

        Examples
        --------
        >>> r = torch.randn(1, 10, 3)
        >>> dxg = DxGrid(
            device=torch.device('cuda:0'), dtype=torch.float, resolution=0.25, spacing=2.0
        )
        >>> x, dv, cell_info = dxg.generate_grid(r)
        >>> p = fun(dv)
        >>> dx = dxg.grid(p, **cell_info)
        >>> dx.write('foo.dx')
        >>>
        """
        from a3mdutils.volumes import Volume
        self.device = device
        self.dtype = dtype
        self.resolution = resolution
        self.spacing = spacing
        self.angstrom = 1.8897259886
        self.volume = Volume

    def generate_grid(self, coords: torch.Tensor):
        """
        generate_grid
        =============

        Returns coordinates, distances to atoms, and cell information

        Parameters 
        ----------
        coords: torch.Tensor
            [n, Ma, 3]
        
        """

        # rotcoords, basis, mean = CoordinatesSampler.principal_components(coords)
        coords = coords.squeeze(0)
        basis = torch.eye(3, 3, dtype=self.dtype, device=self.device) * (self.resolution * (1 / self.angstrom))
        box_min = coords.min(0, keepdim=False)[0] - self.spacing
        box_max = coords.max(0, keepdim=False)[0] + self.spacing
        diff = (box_max + (box_min * -1))
        dims = ((diff / self.resolution).floor() + 1).to(torch.long)

        xx = torch.arange(dims[0], dtype=self.dtype, device=self.device) * self.resolution
        yy = torch.arange(dims[1], dtype=self.dtype, device=self.device) * self.resolution
        zz = torch.arange(dims[2], dtype=self.dtype, device=self.device) * self.resolution

        xg, yg, zg = torch.meshgrid([xx, yy, zz])
        xg = xg.flatten()
        yg = yg.flatten()
        zg = zg.flatten()

        rgrid = torch.stack([xg, yg, zg], dim=1)

        rgrid += box_min
        rgrid = rgrid.reshape(-1, 3).unsqueeze(0)

        box_min *= (1 / self.angstrom)
        rv = alt_distance_vectors(rgrid, coords.unsqueeze(0), dtype=self.dtype, device=self.device)

        return rgrid, rv, dict(r0=box_min.tolist(), basis=basis.data.cpu().numpy(), dims=dims.tolist())

    def dx(self, grid: torch.Tensor, r0: List[float], basis: np.ndarray, dims: List[int]):
        """
        dx
        ==

        Parameters
        ----------
        grid: torch.Tensor
            [n, Ms], where Ms is the size of the grid
        
        r0: List[float]
            vector
        
        basis: np.ndarray
        dims: Lis[int]
        """
        grid = grid.reshape(dims[0], dims[1], dims[2]).cpu().numpy()
        return self.volume(
            filename=None, dxvalues=grid, r0=r0, basis=basis
        )