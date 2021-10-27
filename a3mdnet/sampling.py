import math
from typing import Tuple
import numpy as np
import torch
from a3mdnet import LEBEDEV_GRIDS
from a3mdnet.functions import alt_distance_vectors, distance
from torch import nn


class Sampler(nn.Module):
    def __init__(
            self
    ):
        """
        coordinates sampler
        ---
        performs a 3d coordinates sample given some initical coordinates
        """
        super(Sampler, self).__init__()

    def sample(self, z: torch.Tensor, coords: torch.Tensor):

        pass

    @staticmethod
    def principal_components(coords: torch.Tensor):
        mean = coords.mean(1, keepdim=True)
        x = coords - mean
        n = coords.size()[1]
        c = x.transpose(1, 2) @ x / n
        eig, eiv = torch.symeig(c, eigenvectors=True)
        eivp = eiv.inverse()
        x = (eivp @ x.transpose(1, 2)).transpose(1, 2)
        # x += mean
        return x, eiv, mean


class RandomBox(Sampler):

    def __init__(self, n_sample: int = 1000, spacing: float = 6.0):
        """
        Generates random points spaning the centers location plus an additional padding (spacing).
        """
        Sampler.__init__(
            self
        )
        self.n_sample = n_sample
        self.spacing = spacing

    def sample(self, z, coords):
        device = coords.device
        dtype = coords.dtype

        r = torch.rand(coords.size()[0], self.n_sample, 3, device=device, dtype=dtype)
        rotcoords, eivp, mean = Sampler.principal_components(coords)
        box_min = rotcoords.min(1, keepdim=True)[0] - self.spacing
        box_max = rotcoords.max(1, keepdim=True)[0] + self.spacing
        diff = box_max + (box_min * -1)
        r = (r * diff) + box_min
        r = (eivp @ r.transpose(1, 2)).transpose(1, 2)
        r += mean

        dv = alt_distance_vectors(r, coords, dtype=dtype, device=device)
        w = torch.ones(r.shape[0], r.shape[1], dtype=dtype, device=device)
        return r, dv, w


class RegularGrid(Sampler):

    def __init__(self, size: int, spacing: float):
        """
        Generates a grid size*size*size in which each dimension spans the coordinates dimensions and an additional
        padding space.

        The box is aligned with principal components to make sampling more efficient.
        """
        Sampler.__init__(self)
        self.size = size
        self.spacing = spacing

    def sample(self, z: torch.Tensor, coords: torch.Tensor):
        device = coords.device
        dtype = coords.dtype
        n = coords.size()[0]
        x = torch.arange(self.size, dtype=dtype, device=device) / float(self.size)
        x, y, z = torch.meshgrid([x, x, x])
        r = torch.stack([torch.flatten(x), torch.flatten(y), torch.flatten(z)], dim=1)
        r = r.unsqueeze(0).expand([n, self.size ** 3, 3])
        rotcoords, eivp, mean = Sampler.principal_components(coords)
        box_min = rotcoords.min(1, keepdim=True)[0] - self.spacing
        box_max = rotcoords.max(1, keepdim=True)[0] + self.spacing
        diff = (box_max + (box_min * -1))
        r = (r * diff) + box_min
        r = (eivp @ r.transpose(1, 2)).transpose(1, 2)
        r += mean

        dv = alt_distance_vectors(r, coords, self.dtype, self.device)
        w = torch.ones(r.shape[0], r.shape[1], device=device, dtype=dtype)
        return r, dv, w


class IntegrationGrid(Sampler):

    def __init__(
            self, grid: str = 'minimal',
            radial_resolution: int = 15, softening: int = 3, rm: float = 5.0
    ):
        """
        Integration grid
        ================

        Generates a integration grid suitable for electron density problems. To do so,
        concentrical Lebdenev spheres are placed on top of the molecule coordinates, using
        radius from Gauss-Chebysev quadrature; then weights are calculated using the
        elliptic coordinates suggested by Becke (JCP, 1988).

        Parameters
        ----------

        grid: str
            use either minimal, xtcoarse, or coarse
        radial_resolution: int
            number of bins. Values larger than 15 are ok
        softening: int
            number of softening passes. 3 is ok
        rm: float
            middle point for radiual integration. In coordinates units

        """
        Sampler.__init__(self)
        sph, sphw, ds = self.load_design(grid=grid)
        self.sphere = nn.Parameter(sph, requires_grad=False)
        self.sphere_weights = nn.Parameter(sphw, requires_grad=False)
        self.design_size = ds
        self.radial_resolution = radial_resolution
        self.softening = softening
        self.rm = rm

    @staticmethod
    def load_design(grid) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Loads one of the precalculated designs for lebdenev spheres
        :param grid:
        :return:
        """
        design = np.loadtxt(LEBEDEV_GRIDS[grid])
        design = torch.tensor(design, dtype=torch.float)
        phi, psi, weights = design.split(1, dim=1)
        phi = (phi / 180.0) * math.pi
        psi = (psi / 180.0) * math.pi
        design_size = phi.size(0)
        sphere_x = (phi.cos() * psi.sin()).flatten()
        sphere_y = (phi.sin() * psi.sin()).flatten()
        sphere_z = (psi.cos()).flatten()
        weights = weights.flatten()
        sphere = torch.stack([sphere_x, sphere_y, sphere_z], dim=1)
        return sphere, weights, design_size

    @staticmethod
    def read_padding(labels: torch.Tensor):
        n = labels.size()[0]
        padding = []
        for i in range(n):
            padding.append(labels[i, :].ge(0).sum().item())
        return padding, sum(padding)

    def sample(self, labels: torch.Tensor, coords: torch.Tensor):
        """
        sample
        ======

        Generates grid and weights on top of the coordinates
        
        Parameters
        ----------
        labels: torch.Tensor
            Employed to filter out the padding
        coords: torch.Tensor
            Use bohr cartessian coordinates

        Returns
        -------
        output_coordinates: torch.Tensor
            [n, Ms, 3]
        dv: torch.Tensor
            [n, Ms, Ma, 3]
        output_weights: torch.Tensor
            [n, Ms]
        """
        device = coords.device
        dtype = coords.dtype
        dm = distance(alt_distance_vectors(coords, coords, dtype=dtype, device=device))
        n = coords.size()[0]
        padding, max_padding = self.read_padding(labels)
        ps = max(padding)
        # Defining spherical grids
        ms, concentric_spheres, weights = self.spheric_integration_grid()
        # Defining output tensor (with padding)
        output_coordinates = torch.zeros([n, ps * ms, 3], dtype=dtype, device=device)
        output_weights = torch.zeros([n, ps * ms], dtype=dtype, device=device)
        # Subsetting real coords from paddded coords

        for i in range(n):
            coords_ = coords[i, :, :]
            coords_ = coords_[:padding[i], :]
            dm_ = dm[i, :padding[i], :padding[i]].unsqueeze(0)
            integration_grid = coords_.unsqueeze(1).expand(padding[i], ms, 3)
            integration_grid = concentric_spheres + integration_grid
            integration_grid = integration_grid.reshape(1, padding[i] * ms, 3)
            weights_ = weights.unsqueeze(1).expand(1, padding[i], ms).reshape(1, padding[i] * ms)
            # Generating list of sampling centers
            sampling_centers = torch.arange(
                padding[i], device=device, dtype=torch.long
            ).unsqueeze(0).unsqueeze(2).expand(1, padding[i], ms).reshape(1, padding[i] * ms)
            # Tesellation
            coords_ = coords_.unsqueeze(0)
            v = self.becke_tesellation(
                x=integration_grid, r=coords_, dm=dm_, i=sampling_centers
            )
            w = v * weights_
            output_coordinates[i, :ms * padding[i], :] = integration_grid
            output_weights[i, :ms * padding[i]] = w

        dv = alt_distance_vectors(output_coordinates, coords, dtype=dtype, device=device)

        return output_coordinates, dv, output_weights

    def becke_charges(self, labels, coords):
    
        device = coords.device
        dtype = coords.dtype
        dm = distance(alt_distance_vectors(coords, coords, dtype=dtype, device=device))
        n = coords.size()[0]
        padding, max_padding = self.read_padding(labels)
        ps = max(padding)
        # Defining spherical grids
        ms, concentric_spheres, weights = self.spheric_integration_grid()
        # Defining output tensor (with padding)
        output_coordinates = torch.zeros([n, ps * ms, 3], dtype=dtype, device=device)
        output_weights = torch.zeros([n, ps * ms], dtype=dtype, device=device)
        # Subsetting real coords from paddded coords
        output_centers = torch.zeros([n, ps * ms], dtype=dtype, device=device)
        for i in range(n):
            coords_ = coords[i, :, :]
            coords_ = coords_[:padding[i], :]
            dm_ = dm[i, :padding[i], :padding[i]].unsqueeze(0)
            integration_grid = coords_.unsqueeze(1).expand(padding[i], ms, 3)
            integration_grid = concentric_spheres + integration_grid
            integration_grid = integration_grid.reshape(1, padding[i] * ms, 3)
            weights_ = weights.unsqueeze(1).expand(1, padding[i], ms).reshape(1, padding[i] * ms)
            # Generating list of sampling centers
            sampling_centers = torch.arange(
                padding[i], device=device, dtype=torch.long
            ).unsqueeze(0).unsqueeze(2).expand(1, padding[i], ms).reshape(1, padding[i] * ms)
            # Tesellation
            coords_ = coords_.unsqueeze(0)
            v = self.becke_tesellation(
                x=integration_grid, r=coords_, dm=dm_, i=sampling_centers
            )
            w = v * weights_
            output_coordinates[i, :ms * padding[i], :] = integration_grid
            output_weights[i, :ms * padding[i]] = w
            output_centers[i, :ms * padding[i]] = sampling_centers

        dv = alt_distance_vectors(output_coordinates, coords, dtype=dtype, device=device)

        return output_coordinates, dv, output_weights, output_centers

    def spheric_integration_grid(
            self
    ) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """
        Generates concentric Lebdenev-Gauss Chebysev spheres
        NOTE: There is no implementation right now to avoid the padding problem!
        :return:
        """

        weights = self.sphere_weights.clone()
        sphere = self.sphere.clone()

        # Generating the radial component
        # Using Gauss-Chebysev with variable change
        #
        #       r = rm ( 1 + x ) / ( 1 - x )
        #

        i = torch.arange(1, self.radial_resolution + 1, dtype=torch.float, device=sphere.device)
        z = - (math.pi * ((2.0 * i) - 1.0) / (2.0 * self.radial_resolution)).cos()
        dr = 2.0 * self.rm * torch.pow(- z + 1, -2.0)
        r = self.rm * (1 + z) / (- z + 1)
        w = torch.sqrt(- z.pow(2.0) + 1) * dr * math.pi / self.radial_resolution
        w = r.pow(2.0) * 4.0 * math.pi * w

        # Stacking concentric spheres
        n_spheres = r.size()[0]
        r = r.unsqueeze(1).unsqueeze(2)
        sphere = sphere.unsqueeze(0).expand(n_spheres, self.design_size, 3)
        weights = weights.unsqueeze(0).expand(n_spheres, self.design_size)
        concentric_spheres = r * sphere
        concentric_spheres = concentric_spheres.reshape(-1, 3)
        concentric_spheres = concentric_spheres.unsqueeze(0)
        thinness = w.unsqueeze(1)
        weights = weights * thinness
        weights = weights.reshape(-1).unsqueeze(0)
        ms = concentric_spheres.size()[1]
        return ms, concentric_spheres, weights

    def becke_tesellation(
            self, x: torch.Tensor, r: torch.Tensor, dm: torch.Tensor,
            i: torch.Tensor
    ) -> torch.Tensor:
        """

        Generates a weight scheme to avoid overlap

        :param x: sample
        :param r: centers
        :param dm: centers distance matrix
        :param i: map sample -> center
        :return:
        """
        device = x.device
        dtype = x.dtype

        nx = x.size()[0]  # number of sample matches. It should match nx
        mx = x.size()[1]  # number of samples per molecule
        nr = r.size()[0]  # number of molecules
        mr = r.size()[1]  # number of atoms per molecule
        if nx != nr:
            raise IOError("batch dims don't not match: {:d} {:d}".format(nx, nr))

        # calculate r-r distance matrix

        # calculate x-r distance matrix
        # dim 1 will be r and dim 2 will be x
        # final dims should be [nr, mx, mr]

        x_exp = x.unsqueeze(2).expand(nr, mx, mr, 3)
        r_exp = r.unsqueeze(1).expand(nr, mx, mr, 3)
        rx_dm = torch.norm(x_exp - r_exp, dim=3)
        del x_exp, r_exp

        # calculate (x-r)-(x-r)T distance vectors

        rx_dm_expanded1 = rx_dm.unsqueeze(3).expand(nr, mx, mr, mr)
        rx_dm_expanded2 = rx_dm.unsqueeze(2).expand(nr, mx, mr, mr)
        xx_dm_expanded = dm.unsqueeze(0)
        mu = (rx_dm_expanded1 - rx_dm_expanded2) / xx_dm_expanded.clamp(min=1e-12)

        del rx_dm_expanded1, rx_dm_expanded2, xx_dm_expanded
        del dm, rx_dm
        # soft-cutoff function

        for _ in range(1, self.softening):
            mu = 0.5 * mu * (3.0 - (mu.pow(2.0)))

        mu = 0.5 * (1 - mu)

        # diagonal fill

        diag_fill = torch.eye(
            mr, mr, device=device, dtype=dtype
        ).unsqueeze(0).unsqueeze(1).expand(nr, mx, mr, mr)

        mu += (diag_fill * 0.5)

        # productory

        mu = mu.prod(dim=3)

        # mu should be now [nr, mx, mr] containing
        # the weights of each nuclei for each sampling point

        i = i.unsqueeze(2)

        v_i = torch.gather(mu, 2, i).squeeze(2)
        v_all = mu.sum(2)
        w = v_i / v_all

        # w should be now [nr, mx] containing the weight of each sample

        return w
