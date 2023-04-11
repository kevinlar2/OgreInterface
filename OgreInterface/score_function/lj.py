from OgreInterface.score_function.scatter import scatter_add
from typing import Dict, Optional, Tuple
import torch.nn as nn
import torch
from torch.autograd import grad


class LJ(nn.Module):
    """
    Compute Coulomb energy from a set of point charges via direct summation. Depending on the form of the
    potential function, the interaction can be damped for short distances. If a cutoff is requested, the full
    potential is shifted, so that it and its first derivative is zero starting from the cutoff.
    Args:
        energy_unit (str/float): Units used for the energy.
        position_unit (str/float): Units used for lengths and positions.
        coulomb_potential (torch.nn.Module): Distance part of the potential.
        output_key (str): Name of the energy property in the output.
        charges_key (str): Key of partial charges in the input batch.
        use_neighbors_lr (bool): Whether to use standard or long range neighbor list elements (default = True).
        cutoff (optional, float): Apply a long range cutoff (potential is shifted to 0, default=None).
    """

    def __init__(
        self,
        cutoff: Optional[float] = None,
    ):
        super(LJ, self).__init__()

        cutoff = torch.tensor(cutoff)
        self.register_buffer("cutoff", cutoff)

    def potential_energy(
        self, d_ij: torch.Tensor, delta_e_neg: torch.Tensor, r0: torch.Tensor
    ):
        n = 6
        sigma = r0 / (2 ** (1 / n))

        return (4 * delta_e_neg) * (
            ((sigma / d_ij) ** (2 * n)) - ((sigma / d_ij) ** n)
        )

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        r0_dict: Dict[Tuple[int, int, int, int], float],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the Born repulsion energy.
        Args:
            inputs (dict(str,torch.Tensor)): Input batch.
        Returns:
            dict(str, torch.Tensor): results with Coulomb energy.
        """
        e_neg = inputs["e_negs"].squeeze(-1)
        z = inputs["Z"]

        idx_m = inputs["idx_m"]

        idx_i_all = inputs["idx_i"]
        idx_j_all = inputs["idx_j"]
        is_film = inputs["is_film"]
        is_film_bool = torch.clone(is_film).to(dtype=bool)
        is_sub_bool = torch.logical_not(is_film_bool)

        R = inputs["R"]

        shift = inputs["shift"]
        shift.requires_grad_()
        shifts = torch.repeat_interleave(
            shift, repeats=inputs["n_atoms"], dim=0
        )
        shifts[is_sub_bool] *= 0.0

        shifts.requires_grad_()
        R.requires_grad_()

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

        e_neg_ij = 1.0 + torch.abs(e_neg[idx_i] - e_neg[idx_j])
        d_ij = torch.norm(r_ij, dim=1)
        r0_ij = torch.tensor([r0_dict[k] for k in r0_keys]).to(torch.float32)
        r0_ij = r0_ij.view(e_neg_ij.shape)

        n_atoms = z.shape[0]
        n_molecules = int(idx_m[-1]) + 1

        potential_energy = self.potential_energy(
            d_ij, delta_e_neg=e_neg_ij, r0=r0_ij
        )

        # Apply cutoff if requested (shifting to zero)
        if self.cutoff is not None:
            potential_energy = torch.where(
                d_ij <= self.cutoff,
                potential_energy,
                torch.zeros_like(potential_energy),
            )

        y_energy = scatter_add(potential_energy, idx_i, dim_size=n_atoms)
        y_energy = scatter_add(y_energy, idx_m, dim_size=n_molecules)
        y_energy = torch.squeeze(y_energy, -1)

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
            film_force_norm.detach().numpy(),
            film_norm_grad.detach().numpy(),
        )
