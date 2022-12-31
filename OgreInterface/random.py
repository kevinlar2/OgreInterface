"""
This module will be used to construct the surfaces and interfaces used in this package.
"""
from pymatgen.core.structure import Structure
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core.lattice import Lattice
from pymatgen.core.surface import SlabGenerator
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.interfaces.zsl import ZSLGenerator, reduce_vectors
from pymatgen.core.operations import SymmOp
from pymatgen.analysis.ewald import EwaldSummation
from pymatgen.transformations.standard_transformations import (
    PerturbStructureTransformation,
)

from ase import Atoms
from ase.build.general_surface import surface
from ase.build.supercells import make_supercell
from ase.neighborlist import neighbor_list
from ase.ga.startgenerator import StartGenerator


from OgreInterface.surfaces import Surface, Interface

from itertools import combinations, combinations_with_replacement
from tqdm import tqdm
import numpy as np
import random
import time
from copy import deepcopy
import copy
from functools import reduce
from typing import Union, List, Optional


class RandomInterfaceGenerator:
    """
    This class will be used to build interfaces between a given film/substate and a random crystal structure.
    """

    def __init__(
        self,
        surface_generator,
        random_comp,
        layers=2,
        natoms=24,
        supercell=[2, 2],
        strain_range=[-0.05, 0.05],
        interfacial_distance_range=[2, 3],
        vacuum=40,
        center=True,
    ):
        try:
            from pyxtal.tolerance import Tol_matrix
            from pyxtal.symmetry import Group
            from pyxtal import pyxtal
            from pyxtal.lattice import Lattice as pyxtal_Lattice
        except ImportError:
            raise ImportError(
                "pyxtal must be installed for the RandomInterfaceGenerator"
            )

        if type(surface_generator) == SurfaceGenerator:
            self.surface_generator = surface_generator
        else:
            raise TypeError(
                f"RandomInterfaceGenerator accepts 'ogre.generate.SurfaceGenerator' not '{type(surface_generator).__name__}'"
            )

        self.bulk = self.surface_generator.slabs[0].bulk_pmg
        self.natoms = natoms
        self.layers = layers
        self.random_comp = random_comp
        self.supercell = supercell
        self.strain_range = strain_range
        self.interfacial_distance_range = interfacial_distance_range
        self.vacuum = vacuum
        self.center = center

        self.crystal_system_map = {
            "triclinic": [1, 2],
            "monoclinic": [3, 15],
            "orthorhombic": [16, 74],
            "tetragonal": [75, 142],
            "trigonal": [143, 167],
            "hexagonal": [168, 194],
            "cubic": [195, 230],
        }

    def _check_possible_comp(self, group, natoms):
        elements = self.random_comp

        compositions = list(combinations_with_replacement(elements, natoms))
        compositions = [
            comp for comp in compositions if all(e in comp for e in elements)
        ]

        possible_comps = []

        for combo in compositions:
            unique_vals, counts = np.unique(combo, return_counts=True)
            passed, freedom = group.check_compatible(counts)
            if passed:
                possible_comps.append((unique_vals.tolist(), counts.tolist()))

        return possible_comps

    def _stack_interface(self, slab, random_structure):
        layers = self.layers

        interfacial_distance = random.uniform(
            self.interfacial_distance_range[0],
            self.interfacial_distance_range[1],
        )

        random_ase_slab = surface(
            random_structure,
            layers=layers,
            indices=(0, 0, 1),
            vacuum=self.vacuum,
        )
        random_slab = AseAtomsAdaptor().get_structure(random_ase_slab)

        slab_species = slab.species
        random_species = random_slab.species

        slab_frac_coords = deepcopy(slab.frac_coords)
        random_frac_coords = deepcopy(random_slab.frac_coords)

        slab_cart_coords = slab_frac_coords.dot(slab.lattice.matrix)
        random_cart_coords = random_frac_coords.dot(random_slab.lattice.matrix)

        old_matrix = deepcopy(slab.lattice.matrix)
        c = old_matrix[-1]
        c_len = np.linalg.norm(c)

        min_slab_coords = np.min(slab_frac_coords[:, -1])
        max_slab_coords = np.max(slab_frac_coords[:, -1])
        min_random_coords = np.min(random_frac_coords[:, -1])
        max_random_coords = np.max(random_frac_coords[:, -1])

        interface_c_len = np.sum(
            [
                (max_slab_coords - min_slab_coords) * c_len,
                (max_random_coords - min_random_coords) * c_len,
                self.vacuum,
                interfacial_distance,
            ]
        )

        new_c = interface_c_len * (c / c_len)

        new_matrix = np.vstack([old_matrix[:2], new_c])
        new_lattice = Lattice(matrix=new_matrix)

        slab_frac_coords = slab_cart_coords.dot(new_lattice.inv_matrix)
        slab_frac_coords[:, -1] -= slab_frac_coords[:, -1].min()

        interface_height = slab_frac_coords[:, -1].max() + (
            0.5 * interfacial_distance / interface_c_len
        )

        random_frac_coords = random_cart_coords.dot(new_lattice.inv_matrix)
        random_frac_coords[:, -1] -= random_frac_coords[:, -1].min()

        random_frac_coords[:, -1] += slab_frac_coords[:, -1].max() + (
            interfacial_distance / interface_c_len
        )

        interface = Structure(
            lattice=new_lattice,
            coords=np.vstack([slab_frac_coords, random_frac_coords]),
            species=slab_species + random_species,
            coords_are_cartesian=False,
            to_unit_cell=True,
        )
        interface.translate_sites(
            range(len(interface)), [0.0, 0.0, 0.5 - interface_height]
        )
        shift = [random.uniform(0, 1), random.uniform(0, 1), 0.0]
        interface.translate_sites(
            range(len(slab_frac_coords), len(interface)), shift
        )

        interface.sort()
        slab.sort()
        random_slab.sort()

        return interface, slab, random_slab

    def _generate_random_structure(self, slab, factor, t_factor, timeout=10):
        supercell_options = np.array(
            [
                [2, 2, 1],
                [2, 2, 1],
                [2, 1, 1],
                [1, 2, 1],
                [1, 1, 1],
                [1, 1, 1],
                [1, 1, 1],
                [1, 1, 1],
            ]
        )
        # ind = 3
        ind = random.randint(0, len(supercell_options) - 1)
        # print(ind)
        supercell = supercell_options[ind]

        if supercell.sum() == 5:
            prim_cell_natoms = (self.natoms // self.layers) // 4
        elif supercell.sum() == 4:
            prim_cell_natoms = (self.natoms // self.layers) // 2
        elif supercell.sum() == 3:
            prim_cell_natoms = self.natoms // self.layers

        surface_lattice = deepcopy(slab.lattice.matrix)
        surface_AB_plane = (1 / supercell[:2])[:, None] * surface_lattice[:2]
        surface_area = np.linalg.norm(
            np.cross(surface_AB_plane[0], surface_AB_plane[1])
        )
        surface_atom_density = len(self.bulk) / self.bulk.volume
        surface_atom_density = 0.04

        random_density = random.uniform(
            surface_atom_density - (0.1 * surface_atom_density),
            surface_atom_density + (0.1 * surface_atom_density),
        )

        random_cvec = np.array(
            [0, 0, prim_cell_natoms / (surface_area * random_density)]
        )

        random_lattice = np.vstack([surface_AB_plane, random_cvec])
        # print(np.round(random_lattice, 3))
        s = Structure(
            lattice=Lattice(random_lattice),
            coords=[[0, 0, 0]],
            species=["Ga"],
        )
        sg = SpacegroupAnalyzer(s)
        lattice_type = sg.get_crystal_system()

        struc_compat = False
        while not struc_compat:
            struc_group = Group(
                random.randint(
                    self.crystal_system_map[lattice_type][0],
                    self.crystal_system_map[lattice_type][1],
                )
            )
            possible_struc_comps = self._check_possible_comp(
                struc_group, prim_cell_natoms
            )
            if len(possible_struc_comps) > 0:
                struc_comp_ind = random.randint(
                    0, len(possible_struc_comps) - 1
                )
                struc_species, struc_numIons = possible_struc_comps[
                    struc_comp_ind
                ]
                struc_compat = True

        good_struc = False
        start_time = time.time()
        while not good_struc:
            try:
                struc = pyxtal()
                struc.from_random(
                    3,
                    struc_group,
                    species=struc_species,
                    numIons=struc_numIons,
                    lattice=pyxtal_Lattice.from_matrix(random_lattice),
                    factor=factor,
                    t_factor=t_factor,
                )
                ase_struc = struc.to_ase()
                struc_min_d = np.min(neighbor_list("d", ase_struc, cutoff=5.0))

                if struc_min_d >= 2.0 and struc_min_d <= 3:
                    good_struc = True
            except RuntimeError as e:
                print("error")
                pass

            if time.time() - start_time > timeout:
                break

        if supercell.sum() > 3:
            ase_struc = make_supercell(ase_struc, supercell * np.eye(3))

        ase_struc.wrap()

        return ase_struc, good_struc

    def generate_interface(self, factor, t_factor, timeout=30):
        slab_ind = random.randint(0, len(self.surface_generator.slabs) - 1)
        slab = deepcopy(
            self.surface_generator.slabs[slab_ind].primitive_slab_pmg
        )
        slab.apply_strain(
            [
                random.uniform(self.strain_range[0], self.strain_range[1]),
                random.uniform(self.strain_range[0], self.strain_range[1]),
                0,
            ]
        )
        slab.make_supercell(self.supercell)

        valid_struc = False
        start_time = time.time()
        while not valid_struc:
            ase_struc, valid = self._generate_random_structure(
                slab, factor, t_factor
            )
            valid_struc = valid

            if time.time() - start_time > timeout:
                break
                print("not valid")

        interface, surf, rand = self._stack_interface(slab, ase_struc)

        return interface, surf, rand


class RandomSurfaceGenerator:
    """
    This class will be used to build interfaces between a given film/substate and a random crystal structure.
    """

    def __init__(
        self,
        random_comp,
        layers=2,
        natoms_per_layer=12,
        vacuum=40,
        center=True,
        rattle=True,
    ):
        try:
            from pyxtal.tolerance import Tol_matrix
            from pyxtal.symmetry import Group
            from pyxtal import pyxtal
            from pyxtal.lattice import Lattice as pyxtal_Lattice
        except ImportError:
            raise ImportError(
                "pyxtal must be installed for the RandomInterfaceGenerator"
            )

        self.random_comp = random_comp
        self.natoms_per_layer = natoms_per_layer
        self.layers = layers
        self.vacuum = vacuum
        self.center = center
        self.rattle = rattle

        self.crystal_system_map = {
            "triclinic": [1, 2],
            "monoclinic": [3, 15],
            "orthorhombic": [16, 74],
            "tetragonal": [75, 142],
            "trigonal": [143, 167],
            "hexagonal": [168, 194],
            "cubic": [195, 230],
        }

    def _check_possible_comp(self, group, natoms):
        elements = self.random_comp

        compositions = list(combinations_with_replacement(elements, natoms))
        compositions = [
            comp for comp in compositions if all(e in comp for e in elements)
        ]

        possible_comps = []

        for combo in compositions:
            unique_vals, counts = np.unique(combo, return_counts=True)
            passed, freedom = group.check_compatible(counts)
            if passed:
                possible_comps.append((unique_vals.tolist(), counts.tolist()))

        return possible_comps

    def _generate_random_structure(self, factor, t_factor, timeout=10):
        struc_compat = False
        while not struc_compat:
            struc_group = Group(random.randint(1, 230))
            possible_struc_comps = self._check_possible_comp(
                struc_group, self.natoms_per_layer
            )
            if len(possible_struc_comps) > 0:
                struc_comp_ind = random.randint(
                    0, len(possible_struc_comps) - 1
                )
                struc_species, struc_numIons = possible_struc_comps[
                    struc_comp_ind
                ]
                struc_compat = True

        good_struc = False
        start_time = time.time()
        while not good_struc:
            try:
                struc = pyxtal()
                struc.from_random(
                    3,
                    struc_group,
                    species=struc_species,
                    numIons=struc_numIons,
                    # lattice=pyxtal_Lattice.from_matrix(random_lattice),
                    factor=factor,
                    t_factor=t_factor,
                )
                ase_struc = struc.to_ase()
                struc_min_d = np.min(neighbor_list("d", ase_struc, cutoff=5.0))

                if struc_min_d >= 2.0 and struc_min_d <= 3:
                    good_struc = True

            except RuntimeError as e:
                pass

            if time.time() - start_time > timeout:
                break

        ase_struc.wrap()

        return ase_struc, good_struc

    def generate_surface(self, factor, t_factor, timeout=30):
        valid_struc = False
        start_time = time.time()
        while not valid_struc:
            ase_struc, valid = self._generate_random_structure(
                factor, t_factor
            )
            valid_struc = valid

            if time.time() - start_time > timeout:
                break

        surface_generator = SurfaceGenerator(
            structure=ase_struc,
            miller_index=[0, 0, 1],
            layers=self.layers,
            vacuum=self.vacuum,
            generate_all=True,
            filter_ionic_slabs=False,
        )

        slab_ind = random.randint(0, len(surface_generator.slabs) - 1)
        slab = deepcopy(surface_generator.slabs[slab_ind].slab_pmg)

        if self.center:
            slab.translate_sites(range(len(slab)), [0.0, 0.0, 0.05])
            top_z = slab.frac_coords[:, -1].max()
            bot_z = slab.frac_coords[:, -1].min()
            slab.translate_sites(
                range(len(slab)), [0.0, 0.0, 0.5 - ((top_z + bot_z) / 2)]
            )

        if self.rattle:
            pertub = PerturbStructureTransformation(
                distance=0.15, min_distance=0.05
            )
            slab = pertub.apply_transformation(slab)

        return slab


class RandomBulkGenerator:
    """
    This class will be used to build interfaces between a given film/substate and a random crystal structure.
    """

    def __init__(
        self,
        random_comp,
        natoms=40,
        cell_size=11,
    ):
        self.random_comp = random_comp
        self.natoms = natoms
        self.cell_size = cell_size

    def _get_composition(self, natoms):
        elements = self.random_comp

        compositions = list(combinations_with_replacement(elements, natoms))
        compositions = [
            comp for comp in compositions if all(e in comp for e in elements)
        ]

        ind = random.randint(0, len(compositions) - 1)
        composition = compositions[ind]

        return composition

    def generate_structure(self):
        blocks = self._get_composition(self.natoms)
        unique_e, counts = np.unique(blocks, return_counts=True)

        blmin = closest_distances_generator(
            atom_numbers=[atomic_numbers[i] for i in unique_e],
            ratio_of_covalent_radii=0.9,
        )

        cell = Atoms(cell=np.eye(3) * self.cell_size, pbc=True)

        sg = StartGenerator(
            cell,
            blocks,
            blmin,
            number_of_variable_cell_vectors=0,
        )

        a = sg.get_new_candidate()
        s = AseAtomsAdaptor().get_structure(a)

        return s
