from matscipy.calculators import Ewald
from multiprocessing import Pool
from itertools import repeat
import numpy as np


def _ewald_calculator(atoms, cutoff, charge_dict):
    charges = np.array([charge_dict[s] for s in atoms.get_chemical_symbols()])
    atoms.set_array("charge", charges)
    calc = Ewald()
    calc.set(cutoff=cutoff, verbose=False)
    atoms.set_calculator(calc)
    calc.calculate(atoms, properties="energy", system_changes=[])

    energy = atoms.get_potential_energy()

    return energy


def ewald_calculator(atoms_list, cutoff, charge_dict, n_processes):
    energies = []
    inputs = zip(atoms_list, repeat(cutoff), repeat(charge_dict))
    with Pool(n_processes) as p:
        energies = p.starmap(_ewald_calculator, inputs)

    # for atoms in atoms_list:
    #     charges = np.array([charge_dict[s] for s in atoms.get_chemical_symbols()])
    #     atoms.set_array("charge", charges)
    #     calc = Ewald()
    #     calc.set(cutoff=cutoff, verbose=False)
    #     atoms.set_calculator(calc)
    #     calc.calculate(atoms, properties="energy", system_changes=[])

    #     energies.append(atoms.get_potential_energy())

    # energies = np.array(energies)

    return np.array(energies)
