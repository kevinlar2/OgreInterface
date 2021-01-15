"""
This module will be used to construct the surfaces and interfaces used in this package.
"""
from pymatgen.core.structure import Structure
from pymatgen.io.vasp.inputs import Poscar
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from ase import Atoms
from ase.io import read
from ase.build.surfaces_with_termination import surfaces_with_termination
from ase.spacegroup import get_spacegroup
import numpy as np
from itertools import combinations


class Surface:
    """
    The surface classes generates surfaces with all possible terminations and contains 
    information about the Miller indices of the surface and the number of different
    terminations.

    Parameters:
        structure (pymatgen.core.structure.Structure or ase.Atoms): Conventional bulk structure.
        miller_index (list): Miller index of the created surfaces.
        layers (int): Number of layers generated in the surface.
        vacuum (float): Size of vacuum in Angstroms.
    """

    def __init__(
        self,
        structure,
        miller_index,
        layers,
        vacuum,
    ):
        """TODO: to be defined. """
        if type(structure) == Atoms:
            self.structure = structure
        elif type(structure) == Structure:
            self.structure = AseAtomsAdaptor.get_atoms(structure)
        else:
            raise TypeError(f"structure accepts 'pymatgen.core.structure.Structure' or 'ase.Atoms' not '{type(structure).__name__}'")

        self.miller_index = miller_index
        self.layers = layers
        self.vacuum = vacuum
        self.slabs_pmg = self._generate_slabs()
        self.slabs_ase = [AseAtomsAdaptor.get_atoms(slab) for slab in self.slabs_pmg]
        self.number_of_terminations = len(self.slabs_pmg)

    @classmethod
    def from_file(
        cls,
        filename,
        miller_index,
        layers,
        vacuum,
    ):
        structure = Structure.from_file(filename=filename)

        return cls(structure, miller_index, layers, vacuum)


    def _get_primitive_cell(self, structure):
        if type(structure) == Atoms:
            structure = AseAtomsAdaptor.get_structure(structure)

        spacegroup = SpacegroupAnalyzer(structure)
        primitive = spacegroup.get_primitive_standard_structure()

        return primitive.get_sorted_structure()


    def _generate_slabs(self):
        slabs = surfaces_with_termination(
            self.structure,
            self.miller_index,
            self.layers,
            vacuum=self.vacuum,
            return_all=True,
            verbose=False
        )


        primitive_slabs = [self._get_primitive_cell(slab) for slab in slabs]

        combos = combinations(range(len(primitive_slabs)), 2)
        same_slab_indices = []
        for combo in combos:
            if primitive_slabs[combo[0]] == primitive_slabs[combo[1]]:
                same_slab_indices.append(combo)

        to_delete = [np.min(same_slab_index) for same_slab_index in same_slab_indices]
        unique_slab_indices = [i for i in range(len(primitive_slabs)) if i not in to_delete]
        unique_primitive_slabs = [
            primitive_slabs[i].get_sorted_structure() for i in unique_slab_indices
        ]

        return unique_primitive_slabs


if __name__ == "__main__":
    s = Surface.from_file('./POSCAR_InAs_conv', miller_index=[1,1,0], layers=5, vacuum=10)
    print(s.slabs_ase[0])
    






