from schnetpack.data import ASEAtomsData, AtomsLoader, AtomsDataModule
import schnetpack.transform as trn
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import schnetpack as spk
import torchmetrics
import numpy as np
import time
from ase.db import connect
from ase.io import read
from typing import Dict, Optional, List
import torch
import torch.nn as nn
from os.path import join, isdir, isfile
import os


class IonicScoreFunction(spk.model.AtomisticModel):
    def __init__(
        self,
        input_modules: List[nn.Module] = None,
        postprocessors: Optional[List[trn.Transform]] = None,
        input_dtype_str: str = "float32",
        output_modules: List[nn.Module] = None,
        do_postprocessing: Optional[bool] = None,
        model_outputs: Optional[List[str]] = None,
    ):
        super().__init__(
            input_dtype_str=input_dtype_str,
            postprocessors=postprocessors,
            do_postprocessing=do_postprocessing,
        )
        self.input_modules = nn.ModuleList(input_modules)
        self.output_modules = nn.ModuleList(output_modules)
        self.model_outputs = model_outputs

        self.collect_derivatives()

        if self.model_outputs is None:
            self.collect_outputs()

    def forward(
        self, inputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        # initialize derivatives for response properties
        inputs = self.initialize_derivatives(inputs)

        for m in self.input_modules:
            inputs = m(inputs)

        for m in self.output_modules:
            inputs = m(inputs)

        inputs = self.postprocess(inputs)
        results = self.extract_outputs(inputs)

        return results


def ionic_score_function(atoms, charge_dict, cutoff=10.0, alpha=0.2, k_max=10):
    charges = np.array(
        [
            [charge_dict[s] for s in atom.get_chemical_symbols()]
            for atom in atoms
        ]
    )

    input_generator = spk.interfaces.AtomsConverter(
        neighbor_list=trn.MatScipyNeighborList(cutoff=cutoff),
    )
    inputs = input_generator(atoms=atoms)
    inputs["partial_charges"] = torch.from_numpy(charges.ravel())

    pairwise_distance = spk.atomistic.PairwiseDistances()
    ewald_energy = spk.atomistic.EnergyEwald(
        alpha=alpha,
        k_max=k_max,
        energy_unit="eV",
        position_unit="Ang",
        output_key="energy_ewald",
        use_neighbors_lr=False,
    )

    zbl_energy = spk.atomistic.ZBLRepulsionEnergy(
        energy_unit="eV",
        position_unit="Ang",
        output_key="energy_zbl",
        trainable=False,
        cutoff_fn=spk.nn.CosineCutoff(cutoff),
    )

    agg_energy = spk.atomistic.Aggregation(
        keys=["energy_zbl", "energy_ewald"], output_key="energy"
    )

    ew = IonicScoreFunction(
        input_modules=[pairwise_distance],
        output_modules=[ewald_energy, zbl_energy, agg_energy],
        model_outputs=["energy_ewald", "energy_zbl", "energy"],
        postprocessors=[
            trn.CastTo64(),
        ],
    )

    data = ew.forward(inputs)
    numpy_data = {k: data[k].numpy() for k in data}

    return (
        numpy_data["energy"],
        numpy_data["energy_ewald"],
        numpy_data["energy_zbl"],
    )
