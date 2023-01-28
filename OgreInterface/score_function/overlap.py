from OgreInterface.score_function.scatter import scatter_add
from typing import Dict, Optional, Tuple
import numpy as np
import torch.nn as nn
import torch


class SphereOverlap(nn.Module):
    """ """

    def __init__(
        self,
        cutoff: Optional[float] = None,
    ):
        super(SphereOverlap, self).__init__()

        cutoff = torch.tensor(cutoff)
        self.register_buffer("cutoff", cutoff)
        self.pi = torch.tensor(np.pi)

    def overlap_potential_old(
        self, d_ij: torch.Tensor, r_i: torch.Tensor, r_j: torch.Tensor
    ):
        top = (
            self.pi
            * ((r_i + r_j - d_ij) ** 2)
            * (
                d_ij**2
                + (2 * d_ij * r_j)
                - (3 * r_j**2)
                + (2 * d_ij * r_i)
                + (6 * r_i * r_j)
                - (3 * r_i**2)
            )
        )
        bot = 12 * d_ij

        overlap = top / bot

        overlap = torch.where(
            d_ij >= (r_i + r_j), overlap, torch.zeros_like(overlap)
        )

        return overlap

    def overlap_potential(
        self, d_ij: torch.Tensor, r_i: torch.Tensor, r_j: torch.Tensor
    ):
        return ((r_i + r_j) / d_ij) ** 6

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        radius_dict: Dict[Tuple[int, int], float],
    ) -> Dict[str, torch.Tensor]:
        """ """
        z = inputs["Z"]
        idx_m = inputs["idx_m"]

        r_ij = inputs["Rij"]
        idx_i = inputs["idx_i"]
        idx_j = inputs["idx_j"]
        is_film = inputs["is_film"]

        r_i_key_array = (
            torch.stack([is_film[idx_i], z[idx_i]], dim=1).numpy().astype(int)
        )
        r_j_key_array = (
            torch.stack([is_film[idx_j], z[idx_j]], dim=1).numpy().astype(int)
        )
        r_i_keys = list(map(tuple, r_i_key_array))
        r_j_keys = list(map(tuple, r_j_key_array))

        d_ij = torch.norm(r_ij, dim=1)
        r_i = torch.tensor([radius_dict[k] for k in r_i_keys]).to(
            torch.float32
        )
        r_j = torch.tensor([radius_dict[k] for k in r_j_keys]).to(
            torch.float32
        )

        r_i = r_i.view(d_ij.shape)
        r_j = r_j.view(d_ij.shape)

        n_atoms = z.shape[0]
        n_molecules = int(idx_m[-1]) + 1

        potential = self.overlap_potential(d_ij, r_i, r_j)

        # Apply cutoff if requested (shifting to zero)
        if self.cutoff is not None:
            potential = torch.where(
                d_ij <= self.cutoff, potential, torch.zeros_like(potential)
            )

        y = scatter_add(potential, idx_i, dim_size=n_atoms)
        y = scatter_add(y, idx_m, dim_size=n_molecules)
        y = torch.squeeze(y, -1)

        return y.numpy()
