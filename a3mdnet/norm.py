import torch


class Normalizer:

    def __init__(self):
        pass

    def forward(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class ChargeNormalization(Normalizer):
    def __init__(self):
        super(ChargeNormalization, self).__init__()

    @staticmethod
    def forward(coefficients: torch.Tensor, integrals: torch.Tensor, charges: torch.Tensor):

        coefficients = coefficients.flatten(1)
        integrals = integrals.flatten(1)

        estimated_charge = coefficients * integrals
        estimated_charge = torch.sum(estimated_charge, dim=(1, 2))

        factor = charges / estimated_charge
        factor = factor.unsqueeze(dim=1)
        factor = factor.unsqueeze(dim=1)
        return coefficients * factor


class QPChargeNormalization(Normalizer):
    def __init__(self):
        super(QPChargeNormalization, self).__init__()

    def forward(self, coefficients: torch.Tensor, integrals: torch.Tensor, charges: torch.Tensor):
        """

        Least squares with restraints solving.
        This module allows to normalize the input coefficients so it reproduces the integral of the charge.
        To do so:

            1.  Merges the coefficients into a single tensor
            2.  Defines a linear system that represents the derivative of the lagrangian:

                    L(x, u) = (x-c)^T (x-c) - u (qx - Q)

            3. Solves this system and reshapes the output into the input shape

        :param coefficients:
        :param integrals:
        :param charges
        """

        coefficients_dims = coefficients.size()
        nbatch = coefficients.size()[0]
        natoms = coefficients.size()[1]
        niso_functions = coefficients.size()[2]

        coefficients = coefficients.reshape(nbatch, -1)
        integrals = integrals.reshape(nbatch, -1)
        nfunctions = coefficients.size()[1]

        lqoperator = (torch.eye(nfunctions, device=coefficients.device) * 2).repeat(nbatch, 1, 1)
        integrals_flat = integrals.reshape(-1, nfunctions, 1)
        lqoperator = torch.cat([lqoperator, -integrals_flat], dim=2)

        integrals_flat = integrals.reshape(-1, 1, nfunctions)
        integrals_flat = torch.cat(
            [
                integrals_flat, torch.zeros(nbatch, 1, 1, dtype=torch.float, device=coefficients.device)
            ], dim=2
        )
        lqoperator = torch.cat([lqoperator, integrals_flat], dim=1)

        coefficients_flat = coefficients.reshape(nbatch, nfunctions, 1)
        charge_flat = charges.reshape(nbatch, 1, 1)
        operator_problem = torch.cat(
            [
                2 * coefficients_flat,
                charge_flat
            ], dim=1
        )

        operator_solution, lu = torch.solve(operator_problem, lqoperator)
        solutions, lagmult = torch.split(
            operator_solution, [natoms * niso_functions, 1], dim=1
        )

        assert isinstance(solutions, torch.Tensor)
        solutions = solutions.reshape(coefficients_dims)

        return solutions


class QPSegmentChargeNormalization(Normalizer):

    def __init__(self):

        super(QPSegmentChargeNormalization, self).__init__()

    @staticmethod
    def check(
        coefficients: torch.Tensor, integrals: torch.Tensor,
        segment: torch.Tensor, charges: torch.Tensor
    ):
        nbatch = coefficients.size()[0]
        natoms = coefficients.size()[1]
        niso_functions = coefficients.size()[2]
        nsegments = charges.size()[1]
        # nsegbatch = nbatch * nsegments
        segment_r = torch.arange(nbatch, device=coefficients.device).unsqueeze(1).expand(nbatch, natoms) * nsegments
        segment_r = segment_r + segment
        segment_r = segment_r.flatten()
        coefficients = coefficients.reshape(-1, niso_functions)
        integrals = integrals.reshape(-1, niso_functions)
        charges = torch.flatten(charges)
        pred_charges = torch.zeros_like(charges)
        segments_uniques = segment_r.unique()

        for i, sg in enumerate(segments_uniques):
            mask = torch.eq(segment_r, sg)
            coefficients_sel = coefficients.index_select(0, torch.nonzero(mask).squeeze())
            integrals_sel = integrals.index_select(0, mask.nonzero().squeeze())
            pred_charges[i] = (coefficients_sel * integrals_sel).sum()

        return charges, pred_charges

    def forward(
        self, coefficients: torch.Tensor, integrals: torch.Tensor,
        segment: torch.Tensor, segment_charges: torch.Tensor
    ):

        nbatch = coefficients.size()[0]
        natoms = coefficients.size()[1]
        niso_functions = coefficients.size()[2]
        nsegments = segment_charges.size()[1]
        # nsegbatch = nbatch * nsegments
        segment_r = torch.arange(nbatch, device=coefficients.device).unsqueeze(1).expand(nbatch, natoms) * nsegments
        segment_r = segment_r + segment
        segment_r = segment_r.flatten()
        coefficients = coefficients.reshape(-1, niso_functions)
        norm_coefficients = torch.zeros_like(coefficients)
        integrals = integrals.reshape(-1, niso_functions)
        segment_charges = torch.flatten(segment_charges)
        segments_uniques = segment_r.unique()

        for i, sg in enumerate(segments_uniques):
            mask = torch.eq(segment_r, sg)
            coefficients_sel = coefficients.index_select(0, torch.nonzero(mask).squeeze())
            integrals_sel = integrals.index_select(0, torch.nonzero(mask).squeeze())
            coefficients_res = self.solve_qp_problem(
                coefficients_sel, integrals_sel, segment_charges[i]
            )
            mask_expanded = mask.unsqueeze(1).expand(natoms * nbatch, niso_functions)
            norm_coefficients.masked_scatter_(mask_expanded, coefficients_res)

        return norm_coefficients.reshape(nbatch, natoms, niso_functions)

    @staticmethod
    def solve_qp_problem(coefficients, integrals, charge):
        natoms = coefficients.size()[0]
        nfunctions = coefficients.size()[1] * natoms

        lqoperator = (torch.eye(nfunctions, device=coefficients.device) * 2)
        integrals_flat = integrals.reshape(-1, nfunctions)
        lqoperator = torch.cat([lqoperator, -integrals_flat], dim=0)

        integrals_flat = integrals.reshape(1, nfunctions)
        integrals_flat = torch.cat(
            [
                integrals_flat, torch.zeros(1, 1, dtype=torch.float, device=coefficients.device)
            ], dim=1
        )
        lqoperator = torch.cat([lqoperator, integrals_flat.transpose(0, 1)], dim=1)

        coefficients_flat = coefficients.reshape(nfunctions, 1)
        charge_flat = charge.reshape(1, 1)
        operator_problem = torch.cat(
            [
                2 * coefficients_flat,
                -charge_flat
            ], dim=0
        )

        operator_solution, lu = torch.solve(operator_problem, lqoperator)
        solutions_iso, lagmult = torch.split(
            operator_solution, [nfunctions, 1], dim=0
        )
        solutions_iso = solutions_iso.view_as(coefficients)

        return solutions_iso

    def map(self, charges: torch.Tensor, segments: torch.Tensor, nsegments: int):
        nbatch = charges.shape[0]
        natoms = charges.shape[1]
        segment_r = torch.arange(nbatch, device=charges.device).unsqueeze(1).expand(nbatch, natoms) * nsegments
        segment_r = segment_r + segments
        segment_r = segment_r.flatten()
        charges = torch.flatten(charges)
        output = torch.flatten(torch.zeros(nbatch, nsegments, dtype=torch.float, device=charges.device))
        output = output.index_add(0, index=segment_r, tensor=charges)

        return output.reshape(nbatch, nsegments)
