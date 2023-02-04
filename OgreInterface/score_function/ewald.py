from OgreInterface.score_function.scatter import scatter_add
from typing import Dict, Optional
import numpy as np
import torch.nn as nn
import torch


class EnergyEwaldError(Exception):
    pass


class EnergyEwald(nn.Module):
    """
    Compute the Coulomb energy of a set of point charges inside a periodic box.
    Only works for periodic boundary conditions in all three spatial directions and orthorhombic boxes.
    Args:
        alpha (float): Ewald alpha.
        k_max (int): Number of lattice vectors.
        charges_key (str): Key of partial charges in the input batch.
    """

    def __init__(
        self,
        alpha: float,
        k_max: int,
        cutoff: Optional[float] = None,
    ):
        super(EnergyEwald, self).__init__()

        # Get the appropriate Coulomb constant
        ke = 14.3996
        self.register_buffer("ke", torch.Tensor([ke]))

        # TODO: automatic computation of alpha
        self.register_buffer("alpha", torch.Tensor([alpha]))

        cutoff = torch.tensor(cutoff)
        self.register_buffer("cutoff", cutoff)

        # Set up lattice vectors
        self.k_max = k_max
        kvecs = self._generate_kvecs()
        self.register_buffer("kvecs", kvecs)

    def _generate_kvecs(self) -> torch.Tensor:
        """
        Auxiliary routine for setting up the k-vectors.
        Returns:
            torch.Tensor: k-vectors.
        """
        krange = torch.arange(0, self.k_max + 1, dtype=self.alpha.dtype)
        krange = torch.cat([krange, -krange[1:]])
        kvecs = torch.cartesian_prod(krange, krange, krange)
        norm = torch.sum(kvecs**2, dim=1)
        kvecs = kvecs[norm <= self.k_max**2 + 2, :]
        norm = norm[norm <= self.k_max**2 + 2]
        kvecs = kvecs[norm != 0, :]

        return kvecs

    def forward(
        self, inputs: Dict[str, torch.Tensor], z_shift: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the Coulomb energy of the periodic system.
        Args:
            inputs (dict(str,torch.Tensor)): Input batch.
            z_shift: Determines if z_shifted film positions are used
        Returns:
            dict(str, torch.Tensor): results with Coulomb energy.
        """
        if z_shift:
            z_str = "_z"
        else:
            z_str = ""

        q = inputs["partial_charges"].squeeze(-1)
        idx_m = inputs["idx_m"]

        r_ij = inputs[f"Rij{z_str}"]
        idx_i = inputs["idx_i"]
        idx_j = inputs["idx_j"]

        d_ij = torch.norm(r_ij, dim=1)

        positions = inputs[f"R{z_str}"]
        cell = inputs["cell"]

        n_atoms = q.shape[0]
        n_molecules = int(idx_m[-1]) + 1

        # Get real space and reciprocal space contributions
        y_real = self._real_space(
            q, d_ij, idx_i, idx_j, idx_m, n_atoms, n_molecules
        )
        y_reciprocal = self._reciprocal_space(
            q, positions, cell, idx_m, n_molecules
        )

        y = y_real + y_reciprocal

        return y.numpy()

    def _real_space(
        self,
        q: torch.Tensor,
        d_ij: torch.Tensor,
        idx_i: torch.Tensor,
        idx_j: torch.Tensor,
        idx_m: torch.Tensor,
        n_atoms: int,
        n_molecules: int,
    ) -> torch.Tensor:
        """
        Compute the real space contribution of the screened charges.
        Args:
            q (torch.Tensor): Partial charges.
            d_ij (torch.Tensor): Interatomic distances.
            idx_i (torch.Tensor): Indices of atoms i in the distance pairs.
            idx_j (torch.Tensor): Indices of atoms j in the distance pairs.
            idx_m (torch.Tensor): Molecular indices of each atom.
            n_atoms (int): Total number of atoms.
            n_molecules (int): Number of molecules.
        Returns:
            torch.Tensor: Real space Coulomb energy.
        """

        # Apply erfc for Ewald summation
        f_erfc = torch.erfc(torch.sqrt(self.alpha) * d_ij)
        # Combine functions and multiply with inverse distance
        f_r = f_erfc / d_ij

        f_erfc_cutoff = torch.erfc(torch.sqrt(self.alpha) * self.cutoff)
        f_r_cutoff = f_erfc_cutoff / self.cutoff

        potential_ij = q[idx_i] * q[idx_j] * (f_r - f_r_cutoff)
        # potential_ij = q[idx_i] * q[idx_j] * f_r

        if self.cutoff is not None:
            potential_ij = torch.where(
                d_ij <= self.cutoff,
                potential_ij,
                torch.zeros_like(potential_ij),
            )

        y = scatter_add(potential_ij, idx_i, dim_size=n_atoms)
        y = scatter_add(y, idx_m, dim_size=n_molecules)
        y = 0.5 * self.ke * y.squeeze(-1)

        return y

    def _reciprocal_space(
        self,
        q: torch.Tensor,
        positions: torch.Tensor,
        cell: torch.Tensor,
        idx_m: torch.Tensor,
        n_molecules: int,
    ):
        """
        Compute the reciprocal space contribution.
        Args:
            q (torch.Tensor): Partial charges.
            positions (torch.Tensor): Atom positions.
            cell (torch.Tensor): Molecular cells.
            idx_m (torch.Tensor): Molecular indices of each atom.
            n_molecules (int): Number of molecules.
        Returns:
            torch.Tensor: Real space Coulomb energy.
        """
        # extract box dimensions from cells
        recip_box = 2.0 * np.pi * torch.linalg.inv(cell).transpose(1, 2)
        v_box = torch.abs(torch.linalg.det(cell))

        if torch.any(torch.isclose(v_box, torch.zeros_like(v_box))):
            raise EnergyEwaldError("Simulation box has no volume.")

        # 1) compute the prefactor
        prefactor = 2.0 * np.pi / v_box

        # setup kvecs M x K x 3
        kvecs = torch.matmul(self.kvecs[None, :, :], recip_box)

        # Squared length of vectors M x K
        k_squared = torch.sum(kvecs**2, dim=2)

        # 2) Gaussian part of ewald sum
        q_gauss = torch.exp(-0.25 * k_squared / self.alpha)  # M x K

        # 3) Compute charge density fourier terms
        # Dot product in exponent -> MN x K, expand kvecs in MN batch structure
        kvec_dot_pos = torch.sum(kvecs[idx_m] * positions[:, None, :], dim=2)

        # charge densities MN x K -> M x K
        q_real = scatter_add(
            (q[:, None] * torch.cos(kvec_dot_pos)), idx_m, dim_size=n_molecules
        )
        q_imag = scatter_add(
            (q[:, None] * torch.sin(kvec_dot_pos)), idx_m, dim_size=n_molecules
        )
        # Compute square of density
        q_dens = q_real**2 + q_imag**2

        # Sum over k vectors -> M x K -> M
        y_ewald = prefactor * torch.sum(q_dens * q_gauss / k_squared, dim=1)

        # 4) self interaction correction -> MN
        self_interaction = torch.sqrt(self.alpha / np.pi) * scatter_add(
            q**2, idx_m, dim_size=n_molecules
        )

        # Bring everything together
        y_ewald = self.ke * (y_ewald - self_interaction)

        return y_ewald


if __name__ == "__main__":
    from generate_inputs import generate_dict
    from ase.io import read

    ew = EnergyEwald(alpha, k_max)
    ew.forward(generate_dict(atoms, cutoff))
