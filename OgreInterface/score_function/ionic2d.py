from OgreInterface.score_function.scatter import scatter_add
from typing import Dict, Optional, Tuple
import numpy as np
import torch.nn as nn
import torch
from torch.autograd import grad


class IonicPotentialError(Exception):
    pass


class IonicPotential(nn.Module):
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
        super(IonicPotential, self).__init__()

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
        krangez = torch.arange(0, 1, dtype=self.alpha.dtype)
        krange = torch.cat([krange, -krange[1:]])
        # kvecs = torch.cartesian_prod(krange, krange, krange)
        kvecs = torch.cartesian_prod(krange, krange, krangez)
        norm = torch.sum(kvecs**2, dim=1)
        kvecs = kvecs[norm <= self.k_max**2 + 2, :]
        norm = norm[norm <= self.k_max**2 + 2]
        kvecs = kvecs[norm != 0, :]

        return kvecs

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        r0_dict: Dict[Tuple[int, int, int, int], float],
        ns_dict: Dict[Tuple[int, int], float],
        z_shift: bool = False,
    ) -> Dict[str, torch.Tensor]:

        if z_shift:
            z_str = "_z"
        else:
            z_str = ""

        q = inputs["partial_charges"].squeeze(-1)
        idx_m = inputs["idx_m"]
        # ns = inputs["ns"]

        cell = inputs["cell"]
        n_atoms = q.shape[0]
        n_molecules = int(idx_m[-1]) + 1
        z = inputs["Z"]
        idx_m = inputs["idx_m"]

        idx_i_all = inputs["idx_i"]
        idx_j_all = inputs["idx_j"]
        is_film = inputs["is_film"]
        is_film_bool = torch.clone(is_film).to(dtype=bool)
        is_sub_bool = torch.logical_not(is_film_bool)

        R = inputs[f"R{z_str}"]
        R.requires_grad_()

        shift = inputs["shift"]
        shift.requires_grad_()
        shifts = torch.repeat_interleave(
            shift, repeats=inputs["n_atoms"], dim=0
        )
        shifts[is_sub_bool] *= 0.0
        shifts.requires_grad_()

        R_shift = R + shifts
        r_ij_all = R_shift[idx_j_all] - R_shift[idx_i_all] + inputs["offsets"]

        distances = torch.norm(r_ij_all, dim=1)
        in_cutoff = torch.nonzero(distances < self.cutoff, as_tuple=False)
        pair_index = in_cutoff.squeeze()
        idx_i = idx_i_all[pair_index]
        idx_j = idx_j_all[pair_index]
        offsets = inputs["offsets"][pair_index]

        is_film_i = is_film[idx_i]
        is_film_j = is_film[idx_j]

        r_ij = R_shift[idx_j] - R_shift[idx_i] + offsets

        r0_key_array = (
            torch.stack([is_film_i, is_film_j, z[idx_i], z[idx_j]], dim=1)
            .numpy()
            .astype(int)
        )
        r0_keys = list(map(tuple, r0_key_array))

        d_ij = torch.norm(r_ij, dim=1)
        q_ij = torch.abs(q[idx_i] * q[idx_j])
        # n_ij = ns[idx_i] + ns[idx_j] / 2.0
        r0_ij = torch.tensor([r0_dict[k] for k in r0_keys]).to(torch.float32)
        n_ij = torch.tensor([ns_dict[k[2:]] for k in r0_keys]).to(
            torch.float32
        )
        r0_ij = r0_ij.view(q_ij.shape)
        B_ij = (q_ij * (r0_ij ** (n_ij - 1.0))) / n_ij

        n_atoms = z.shape[0]
        n_molecules = int(idx_m[-1]) + 1

        # Get real space and reciprocal space contributions
        y_real = self._real_space(
            q, d_ij, idx_i, idx_j, idx_m, n_atoms, n_molecules
        )
        y_reciprocal = self._reciprocal_space(
            q, R_shift, cell, idx_m, n_molecules
        )
        y_born = self._born(d_ij, n_ij, B_ij)
        y_born = scatter_add(y_born, idx_i, dim_size=n_atoms)
        y_born = scatter_add(y_born, idx_m, dim_size=n_molecules)
        y_born = 0.5 * self.ke * torch.squeeze(y_born, -1)

        y_coulomb = y_real + y_reciprocal
        y_energy = y_coulomb + y_born
        # y_energy = y_real + y_reciprocal

        go = [torch.ones_like(y_energy)]
        grads = grad([y_energy], [R_shift], grad_outputs=go, create_graph=True)
        dEdR = -grads[0]
        dEdR[is_sub_bool] *= 0.0

        film_force = scatter_add(dEdR, idx_m, dim_size=n_molecules)
        film_force_norm = torch.squeeze(torch.norm(film_force, dim=1) ** 2)

        f_go = [torch.ones_like(film_force_norm)]
        film_norm_grad = grad([film_force_norm], [shift], grad_outputs=f_go)[0]
        film_norm_grad = torch.squeeze(film_norm_grad)

        R_shift.detach_()
        shifts.detach_()
        shift.detach_()

        return (
            y_energy.detach().numpy(),
            y_coulomb.detach().numpy(),
            y_born.detach().numpy(),
            film_force_norm.detach().numpy(),
            film_norm_grad.detach().numpy(),
        )

    def _born(
        self, d_ij: torch.Tensor, n_ij: torch.Tensor, B_ij: torch.Tensor
    ):
        return B_ij * ((1 / (d_ij**n_ij)) - (1 / (self.cutoff**n_ij)))

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

        # slab correction
        y_slab_correct = (
            self.ke
            * ((2 * np.pi) / v_box)
            * torch.sum(q * positions[:, -1]) ** 2
        )

        return y_ewald + y_slab_correct
        # return y_ewald


if __name__ == "__main__":
    from generate_inputs import generate_dict
    from ase.io import read

    ew = EnergyEwald(alpha, k_max)
    ew.forward(generate_dict(atoms, cutoff))
