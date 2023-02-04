from OgreInterface.score_function.scatter import scatter_add
from typing import Dict, Optional, Tuple
import torch.nn as nn
import torch


class EnergyBorn(nn.Module):
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
        super(EnergyBorn, self).__init__()

        # Get the appropriate Coulomb constant
        ke = 14.3996
        self.register_buffer("ke", torch.Tensor([ke]))

        cutoff = torch.tensor(cutoff)
        self.register_buffer("cutoff", cutoff)

    def born_potential(
        self, d_ij: torch.Tensor, n_ij: torch.Tensor, B_ij: torch.Tensor
    ):
        return B_ij * ((1 / (d_ij**n_ij)) - (1 / (self.cutoff**n_ij)))

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        r0_dict: Dict[Tuple[int, int, int, int], float],
        z_shift: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the Born repulsion energy.
        Args:
            inputs (dict(str,torch.Tensor)): Input batch.
        Returns:
            dict(str, torch.Tensor): results with Coulomb energy.
        """
        if z_shift:
            z_str = "_z"
        else:
            z_str = ""

        q = inputs["partial_charges"].squeeze(-1)
        z = inputs["Z"]

        ns = inputs["ns"]
        idx_m = inputs["idx_m"]

        r_ij = inputs[f"Rij{z_str}"]
        idx_i = inputs["idx_i"]
        idx_j = inputs["idx_j"]
        is_film = inputs["is_film"]

        r0_key_array = (
            torch.stack(
                [is_film[idx_i], is_film[idx_j], z[idx_i], z[idx_j]], dim=1
            )
            .numpy()
            .astype(int)
        )
        r0_keys = list(map(tuple, r0_key_array))

        q_ij = torch.abs(q[idx_i] * q[idx_j])
        d_ij = torch.norm(r_ij, dim=1)
        n_ij = ns[idx_i] + ns[idx_j] / 2.0
        r0_ij = torch.tensor([r0_dict[k] for k in r0_keys]).to(torch.float32)
        r0_ij = r0_ij.view(q_ij.shape)
        B_ij = q_ij * (r0_ij ** (n_ij - 1.0)) / n_ij

        n_atoms = z.shape[0]
        n_molecules = int(idx_m[-1]) + 1

        potential = self.born_potential(d_ij, n_ij, B_ij)

        # Apply cutoff if requested (shifting to zero)
        if self.cutoff is not None:
            potential = torch.where(
                d_ij <= self.cutoff, potential, torch.zeros_like(potential)
            )

        y = scatter_add(potential, idx_i, dim_size=n_atoms)
        y = scatter_add(y, idx_m, dim_size=n_molecules)
        y = 0.5 * self.ke * torch.squeeze(y, -1)

        return y.numpy()
