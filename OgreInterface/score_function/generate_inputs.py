from ase.neighborlist import neighbor_list
from typing import Dict, List
from ase import Atoms
import torch


def _atoms_collate_fn(batch):
    """
    Build batch from systems and properties & apply padding
    Args:
        examples (list):
    Returns:
        dict[str->torch.Tensor]: mini-batch of atomistic systems
    """
    elem = batch[0]
    idx_keys = {"idx_i", "idx_j", "idx_i_triples"}
    # Atom triple indices must be treated separately
    idx_triple_keys = {"idx_j_triples", "idx_k_triples"}

    coll_batch = {}
    for key in elem:
        if (key not in idx_keys) and (key not in idx_triple_keys):
            coll_batch[key] = torch.cat([d[key] for d in batch], 0)
        elif key in idx_keys:
            coll_batch[key + "_local"] = torch.cat([d[key] for d in batch], 0)

    seg_m = torch.cumsum(coll_batch["n_atoms"], dim=0)
    seg_m = torch.cat([torch.zeros((1,), dtype=seg_m.dtype), seg_m], dim=0)
    idx_m = torch.repeat_interleave(
        torch.arange(len(batch)), repeats=coll_batch["n_atoms"], dim=0
    )
    coll_batch["idx_m"] = idx_m

    for key in idx_keys:
        if key in elem.keys():
            coll_batch[key] = torch.cat(
                [d[key] + off for d, off in zip(batch, seg_m)], 0
            )

    # Shift the indices for the atom triples
    for key in idx_triple_keys:
        if key in elem.keys():
            indices = []
            offset = 0
            for idx, d in enumerate(batch):
                indices.append(d[key] + offset)
                offset += d["idx_j"].shape[0]
            coll_batch[key] = torch.cat(indices, 0)

    return coll_batch


def generate_dict(
    atoms: List[Atoms], cutoff: float, charge_dict: Dict[str, float]
) -> Dict:
    inputs_batch = []

    for at_idx, atom in enumerate(atoms):
        charges = torch.Tensor(
            [charge_dict[s] for s in atom.get_chemical_symbols()]
        )
        R = torch.from_numpy(atom.get_positions())
        cell = torch.from_numpy(atom.get_cell().array).view(-1, 3, 3)
        idx_i, idx_j, S = neighbor_list(
            "ijS", atom, cutoff, self_interaction=False
        )
        idx_i = torch.from_numpy(idx_i)
        idx_j = torch.from_numpy(idx_j)
        S = torch.from_numpy(S).to(dtype=R.dtype)
        offsets = torch.mm(S, cell.view(3, 3))
        Rij = R[idx_j] - R[idx_i] + offsets

        input_dict = {
            "n_atoms": torch.tensor([atom.get_global_number_of_atoms()]),
            "Z": torch.from_numpy(atom.get_atomic_numbers()),
            "R": R,
            "cell": cell,
            "idx_i": idx_i,
            "idx_j": idx_j,
            "Rij": Rij,
            "pbc": torch.from_numpy(atom.get_pbc()).view(-1, 3),
            "partial_charges": charges,
        }

        inputs_batch.append(input_dict)

    inputs = _atoms_collate_fn(inputs_batch)

    for k, v in inputs.items():
        if "float" in str(v.dtype):
            inputs[k] = v.to(dtype=torch.float32)
        if "idx" in k:
            inputs[k] = v.to(dtype=torch.long)

    return inputs


if __name__ == "__main__":
    from ase.build import bulk

    InAs = bulk("InAs", crystalstructure="zincblende", a=5.6)
    print(generate_dict(InAs, cutoff=10.0))