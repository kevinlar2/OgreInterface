from OgreInterface.score_function.scatter import scatter_add
from typing import Dict, Optional, Tuple
import numpy as np
import torch.nn as nn
import torch
from torch.autograd import grad


class IonicPotentialError(Exception):
    pass


class IonicPotential2D(nn.Module):
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
        r_max: int,
        cutoff: Optional[float] = None,
    ):
        super(IonicPotential2D, self).__init__()

        # Get the appropriate Coulomb constant
        ke = 14.3996
        self.register_buffer("ke", torch.Tensor([ke]))

        # TODO: automatic computation of alpha
        self.register_buffer("alpha", torch.Tensor([alpha]))

        cutoff = torch.tensor(cutoff)
        self.register_buffer("cutoff", cutoff)

        # Set up lattice vectors
        self.k_max = k_max
        self.r_max = r_max
        kvecs = self._generate_kvecs()
        self.register_buffer("kvecs", kvecs)

    def _sum_recip_total_energy(
        self,
        kmax,
        recip_cell,
        x,
        y,
        z_distance,
        charge1,
        charge2,
        distance,
        alpha,
    ):
        sum_addition = 0
        for b0 in range(-kmax, kmax + 1):
            for b1 in range(-kmax, kmax + 1):
                if b0 == b1 == 0:
                    continue

                h = b0 * recip_cell[x] + b1 * recip_cell[y]
                h2 = torch.dot(h, h)
                d_h = torch.sqrt(h2)
                exponential1 = torch.exp(d_h * z_distance) * torch.erfc(
                    (d_h / (2 * alpha)) + alpha * z_distance
                )
                exponential2 = torch.exp(-d_h * z_distance) * torch.erfc(
                    (d_h / (2 * alpha)) - alpha * z_distance
                )
                temp1 = (
                    charge1
                    * charge2
                    * torch.cos(torch.dot(h, distance))
                    * (exponential1 + exponential2)
                    / d_h
                )
                sum_addition += temp1

        return sum_addition

    def _sum_real_total_energy(
        self,
        rmax,
        real_cell,
        x,
        y,
        charge1,
        charge2,
        distance,
        alpha,
    ):
        sum_addition = 0
        for n0 in range(-rmax, rmax + 1):
            for n1 in range(-rmax, rmax + 1):
                a = n0 * real_cell[x] + n1 * real_cell[y]
                r = distance + a
                dsq = torch.dot(r, r)
                d = torch.sqrt(dsq)
                if dsq == 0 and n0 == n1 == 0:
                    continue
                addition = charge1 * charge2 * torch.erfc(alpha * d) / d
                sum_addition += addition

        return sum_addition

    def _coulomb_energy(self, cell, recip_cell, positions, charges):
        Sum = 0
        Sum2 = 0
        pi = torch.tensor(np.pi)
        Kcell = recip_cell * 2 * pi

        ######## long range ###################
        for r1, q1 in zip(positions, charges):
            for r2, q2 in zip(positions, charges):
                r1r2 = r1 - r2
                # zij = torch.abs(r1[2] - r2[2])
                zij = r1[2] - r2[2]
                addition1 = self._sum_recip_total_energy(
                    kmax=self.k_max,
                    recip_cell=Kcell,
                    x=0,
                    y=1,
                    z_distance=zij,
                    charge1=q1,
                    charge2=q2,
                    distance=r1r2,
                    alpha=self.alpha,
                )
                Sum += addition1

        for r1, q1 in zip(positions, charges):
            for r2, q2 in zip(positions, charges):
                zij = r1[2] - r2[2]
                # zij = torch.abs(r1[2] - r2[2])
                temp = (
                    q1
                    * q2
                    * (
                        zij * torch.erf(self.alpha * zij)
                        + torch.exp(-((self.alpha * zij) ** 2))
                        / (self.alpha * torch.sqrt(pi))
                    )
                )
                Sum2 += temp

        recip_energy_1 = self.ke * pi * Sum / (2 * self.area)
        recip_energy_2 = -self.ke * pi * Sum2 / self.area
        recip_energy = recip_energy_1 + recip_energy_2

        ######### short range ###################
        Sum = 0
        for r1, q1 in zip(positions, charges):
            for r2, q2 in zip(positions, charges):
                r1r2 = r1 - r2
                addition1 = self._sum_real_total_energy(
                    rmax=self.r_max,
                    real_cell=cell,
                    x=0,
                    y=1,
                    charge1=q1,
                    charge2=q2,
                    distance=r1r2,
                    alpha=self.alpha,
                )
                Sum += addition1

        real_energy = Sum * self.ke / 2

        Sum = 0
        for i in charges:
            Sum += i * i

        self_energy = ((-self.alpha) / torch.sqrt(pi)) * Sum * self.ke
        energy = recip_energy + real_energy + self_energy

        return energy

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
        # recip_cell = inputs["recip_cell"][0] * 2 * torch.tensor(np.pi)
        self.area = torch.norm(torch.cross(cell[0][0], cell[0][1]))
        print(self.area)
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

        # # Get real space and reciprocal space contributions
        # y_real = self._real_space(
        #     q, d_ij, idx_i, idx_j, idx_m, n_atoms, n_molecules
        # )
        # y_reciprocal = self._reciprocal_space(
        #     q, R_shift, cell, idx_m, n_molecules
        # )
        y_born = self._born(d_ij, n_ij, B_ij)
        y_born = scatter_add(y_born, idx_i, dim_size=n_atoms)
        y_born = scatter_add(y_born, idx_m, dim_size=n_molecules)
        y_born = 0.5 * self.ke * torch.squeeze(y_born, -1)

        y_coulomb = self._coulomb_energy(
            cell=inputs["cell"][0],
            recip_cell=inputs["recip_cell"][0],
            positions=R_shift,
            charges=q,
        )

        # y_coulomb = y_real + y_reciprocal
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
        f_erfc = torch.erfc(self.alpha * d_ij)
        # Combine functions and multiply with inverse distance
        f_r = f_erfc / d_ij

        f_erfc_cutoff = torch.erfc(self.alpha * self.cutoff)
        f_r_cutoff = f_erfc_cutoff / self.cutoff

        potential_ij = q[idx_i] * q[idx_j] * (f_r - f_r_cutoff)

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
        r_ij: torch.Tensor,
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
        recip_box = 2.0 * np.pi * torch.linalg.inv(cell)
        # .transpose(1, 2)
        A_box = torch.norm(torch.cross(cell[0], cell[1]))

        # 1) compute the prefactor
        prefactor = 2.0 * np.pi / A_box

        # setup kvecs M x K x 3
        kvecs = torch.matmul(self.kvecs[None, :, :], recip_box)

        # Squared length of vectors M x K
        k_norms = torch.norm(kvecs, dim=2)

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
