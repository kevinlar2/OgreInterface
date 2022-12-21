from matscipy.calculators.pair_potential import PairPotential
from matscipy.calculators.pair_potential.calculator import CutoffInteraction
from collections import defaultdict
from multiprocessing import Pool
from itertools import repeat
from matscipy.neighbours import neighbour_list
from itertools import product, combinations_with_replacement
from ase.data import atomic_numbers
import numpy as np


class BornRepulsion(CutoffInteraction):
    """Born Repulsion Potential"""

    def __init__(self, cutoff, r0, n):
        super().__init__(cutoff)
        self.r0 = r0
        self.n = n
        self.ke = 14.399645
        self.B = self._calc_B()

    def __call__(self, r, qi, qj):
        return np.abs(self.B * qi * qj) / r ** (self.n)

    def first_derivative(self, r, qi, qj):
        return -np.abs(self.B * qi * qj) / r ** (self.n + 1)

    def second_derivative(self, r, qi, qj):
        return np.abs(self.B * qi * qj * (self.n + 1)) / r ** (self.n + 2)

    def _calc_B(self):
        B = self.ke * (self.r0 ** (self.n - 1) / self.n)

        return B


class BornRepulsionCalculator(PairPotential):
    def __init__(self, cutoff, r0_dict={}):
        super().__init__(defaultdict(lambda: self.get_potentials))
        self.cutoff = cutoff
        self.r0_dict = {}
        self.n_dict = {
            atomic_numbers["Pb"]: 12,
            atomic_numbers["S"]: 9,
            atomic_numbers["Br"]: 10,
            atomic_numbers["Cs"]: 12,
        }

        for k in r0_dict:
            if type(k) == str:
                self.r0_dict[atomic_numbers[k]] = r0_dict[k]
            elif type(k) == int:
                self.r0_dict[k] = r0_dict[k]

    def get_potentials(self, r0=2, n=9):
        return BornRepulsion(
            cutoff=self.cutoff,
            r0=r0,
            n=n,
        )

    def set(self, **kwargs):
        super().set(**kwargs)

    def calculate(self, atoms, properties, system_changes):
        """Calculate system properties."""
        unique_numbers = np.unique(atoms.numbers)
        pairs = list(combinations_with_replacement(unique_numbers, 2))
        r0s = [(self.r0_dict[pair[0]] + self.r0_dict[pair[1]]) / 2 for pair in pairs]
        ns = [(self.n_dict[pair[0]] + self.n_dict[pair[1]]) / 2 for pair in pairs]
        f_dict = {
            pair: self.get_potentials(r0, 5) for pair, r0, n in zip(pairs, r0s, ns)
        }

        super().__init__(f_dict)
        super().calculate(atoms, properties, system_changes)


def _born_calculator(atoms, charge_dict, radius_dict, cutoff):
    charges = np.array([charge_dict[s] for s in atoms.get_chemical_symbols()])
    atoms.set_array("charge", charges)
    brc = BornRepulsionCalculator(cutoff=cutoff, r0_dict=radius_dict)
    atoms.set_calculator(brc)
    brc.calculate(atoms, properties="energy", system_changes=[])
    energy = atoms.get_potential_energy()

    return energy


def born_calculator(atoms_list, charge_dict, radius_dict, cutoff, n_processes=1):
    # if n_processes == 1:
    #     energies = []
    #     for atoms in atoms_list:
    #         charges = np.array([charge_dict[s] for s in atoms.get_chemical_symbols()])
    #         atoms.set_array("charge", charges)
    #         brc = BornRepulsionCalculator(cutoff=cutoff, r0_dict=radius_dict)
    #         atoms.set_calculator(brc)
    #         brc.calculate(atoms, properties="energy", system_changes=[])
    #         energies.append(atoms.get_potential_energy())
    # else:
    inputs = zip(atoms_list, repeat(charge_dict), repeat(radius_dict), repeat(cutoff))
    with Pool(n_processes) as p:
        energies = p.starmap(_born_calculator, inputs)

    return np.array(energies)


if __name__ == "__main__":
    # def test2():
    #     return 1
    # def test(*args):
    #     return test2()

    # f = defaultdict(lambda : test)
    # print(f['01'])

    # for x, obj in f.items():
    #     print(x)
    #     print(obj)
    from ase.io import read
    import numpy as np
    import time

    charge_dict = {
        "Pb": 2,
        "S": -2,
        "Cs": 1,
        "Br": -1,
        # "Pb": 0.3,
        # "S": -0.3,
        # "Cs": 0.15,
        # "Br": -0.15,
    }

    atoms = read("./POSCAR_opt")

    charges = np.array([charge_dict[s] for s in atoms.get_chemical_symbols()])
    atoms.set_array("charge", charges)

    s = time.time()
    brc = BornRepulsionCalculator(
        cutoff=10.0, r0_dict={"Pb": 2.4, "Br": 1.68, "Cs": 4.5, "S": 1.68}
    )
    atoms.set_calculator(brc)
    brc.calculate(atoms, properties="energy", system_changes=[])
    print(atoms.get_potential_energy())
    print(time.time() - s)
    # brc.get_nb(atoms)
