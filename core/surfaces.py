"""
This module will be used to construct the surfaces and interfaces used in this package.
"""
from pymatgen.core.structure import Structure 
from pymatgen.core.lattice import Lattice 
from pymatgen.core.surface import get_slab_regions, center_slab
from pymatgen.core.periodic_table import Element
from pymatgen.io.vasp.inputs import Poscar
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.symmetry.structure import SymmetrizedStructure
from pymatgen.analysis.substrate_analyzer import ZSLGenerator, SubstrateAnalyzer, reduce_vectors
from pymatgen.symmetry.analyzer import SymmOp
from pymatgen.analysis.molecule_structure_comparator import CovalentRadius
from ase import Atoms
from ase.io import read
from ase.build.surfaces_with_termination import surfaces_with_termination
from ase.build import make_supercell, stack
from ase.spacegroup import get_spacegroup
from ase.geometry import get_layers
from ase.build.general_surface import ext_gcd
import numpy as np
from math import gcd
from itertools import combinations, product, repeat
from pymatgen.analysis.interface import merge_slabs
from vaspvis.utils import group_layers
import surface_matching_utils as smu 
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from multiprocessing import Pool, cpu_count
import copy
import time


class Surface:
    """
    The Surface class is a container for surfaces generated with the SurfaceGenerator
    class and will be used as an input to the InterfaceGenerator class
    """

    def __init__(
        self,
        slab,
        bulk,
        miller_index,
        layers,
        vacuum,
    ):
        if type(slab) == Atoms:
            self.slab_ase = slab
            self.slab_pmg = AseAtomsAdaptor.get_structure(slab)
        elif type(slab) == Structure:
            self.slab_ase = AseAtomsAdaptor.get_atoms(slab)
            self.slab_pmg = slab
        else:
            raise TypeError(f"Structure accepts 'pymatgen.core.structure.Structure' or 'ase.Atoms' not '{type(slab).__name__}'")

        #  self.no_vac_slab_pmg = self._remove_vacuum(self.slab_pmg)
        #  self.no_vac_slab_ase = AseAtomsAdaptor.get_atoms(self.no_vac_slab_pmg)

        self.primitive_slab_pmg = self._get_primitive(self.slab_pmg)
        self.primitive_slab_ase = AseAtomsAdaptor.get_atoms(self._get_primitive(self.slab_pmg))

        if type(bulk) == Atoms:
            self.bulk_ase = bulk
            self.bulk_pmg = AseAtomsAdaptor.get_structure(bulk)
        elif type(bulk) == Structure:
            self.bulk_ase = AseAtomsAdaptor.get_atoms(bulk)
            self.bulk_pmg = bulk
        else:
            raise TypeError(f"structure accepts 'pymatgen.core.structure.Structure' or 'ase.Atoms' not '{type(bulk).__name__}'")

        self.miller_index = miller_index
        self.layers = layers
        self.vacuum = vacuum
        self.termination = self._get_termination()
        self.crystallographic_directions = self._get_crystallographic_directions()
        self.area = np.linalg.norm(np.cross(self.slab_pmg.lattice.matrix[0], self.slab_pmg.lattice.matrix[1]))

    def remove_layers(self, num_layers, top=False):
        group_inds, _ = group_layers(structure=self.slab_pmg)
        if top:
            group_inds = group_inds[::-1]
        
        to_delete = []
        for i in range(num_layers):
            to_delete.extend(group_inds[i])

        self.slab_pmg.remove_sites(to_delete)

    def _get_primitive(self, structure):
        if type(structure) == Atoms:
            structure = AseAtomsAdaptor.get_structure(structure)

        spacegroup = SpacegroupAnalyzer(structure)
        primitive = spacegroup.get_primitive_standard_structure()

        return primitive.get_sorted_structure()

    def _remove_vacuum(self, slab):
        struc = copy.deepcopy(slab)
        bot, _ = get_slab_regions(struc)[0]
        struc.translate_sites(range(len(struc)), [0,0,-bot])
        frac_coords = struc.frac_coords
        cart_coords = struc.cart_coords
        max_z = np.max(frac_coords[:,-1])
        _, top_new = get_slab_regions(struc)[0]
        matrix = copy.deepcopy(struc.lattice.matrix)
        matrix[2,2] = top_new * matrix[2,2]
        new_lattice = Lattice(matrix)
        struc.lattice = new_lattice

        for i in range(len(struc)):
            struc.sites[i].frac_coords[-1] = struc.sites[i].frac_coords[-1] / max_z
            struc.sites[i].coords[-1] = cart_coords[i,-1]

        return struc

    def _get_termination(self):
        z_layers, _ = get_layers(self.slab_ase, (0, 0, 1))
        top_layer = [i for i, val in enumerate(z_layers == max(z_layers)) if val]
        termination = np.unique(
            [self.slab_ase.get_chemical_symbols()[a] for a in top_layer]
        )

        return termination

    def _get_crystallographic_directions(self, tol=1e-10):
        """
        TODO: Figure out how to find the crystallographic directions of the a and b
        lattice vectors in the primitive slab
        """
        indices = np.asarray(self.miller_index)

        if indices.shape != (3,) or not indices.any() or indices.dtype != int:
            raise ValueError('%s is an invalid surface type' % indices)

        h, k, l = indices
        h0, k0, l0 = (indices == 0)

        if h0 and k0 or h0 and l0 or k0 and l0:  # if two indices are zero
            if not h0:
                c1, c2, c3 = [(0, 1, 0), (0, 0, 1), (1, 0, 0)]
            if not k0:
                c1, c2, c3 = [(0, 0, 1), (1, 0, 0), (0, 1, 0)]
            if not l0:
                c1, c2, c3 = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
        else:
            p, q = ext_gcd(k, l)
            a1, a2, a3 = self.bulk_ase.cell

            # constants describing the dot product of basis c1 and c2:
            # dot(c1,c2) = k1+i*k2, i in Z
            k1 = np.dot(p * (k * a1 - h * a2) + q * (l * a1 - h * a3),
                        l * a2 - k * a3)
            k2 = np.dot(l * (k * a1 - h * a2) - k * (l * a1 - h * a3),
                        l * a2 - k * a3)

            if abs(k2) > tol:
                i = -int(round(k1 / k2))  # i corresponding to the optimal basis
                p, q = p + i * l, q - i * k

            a, b = ext_gcd(p * k + q * l, h)


            c1 = (p * k + q * l, -p * h, -q * h)
            c2 = np.array((0, l, -k)) // abs(gcd(l, k))
            c3 = (b, a * p, a * q)

        return np.array([c1, c2, c3])


class Interface:

    def __init__(
        self,
        substrate,
        film,
        film_transformation,
        substrate_transformation,
        strain,
        angle_diff,
        sub_strain_frac,
        interfacial_distance,
        vacuum,
    ):
        self.substrate = substrate
        self.film = film
        self.film_transformation = film_transformation
        self.substrate_transformation = substrate_transformation
        self.strain = strain
        self.angle_diff = angle_diff
        self.sub_strain_frac = sub_strain_frac
        self.vacuum = vacuum
        self.substrate_supercell = self._prepare_slab(
            self.substrate.slab_pmg,
            self.substrate_transformation
        )
        self.film_supercell = self._prepare_slab(
            self.film.slab_pmg,
            self.film_transformation
        )
        self.interface_sl_vectors = self._get_interface_sl_vecs()
        self.interfacial_distance = interfacial_distance
        self.interface_height = None
        self.strained_sub = self._strain_and_orient_sub()
        self.strained_film = self._strain_and_orient_film()
        self.interface = self._stack_interface()

    def shift_film(self, shift):
        shift = np.array(shift)

        if shift[-1] + self.interfacial_distance < 0.5:
            raise ValueError(f"The film shift results in an interfacial distance of less than 0.5 Angstroms which is non-physical")
        
        shifted_interface = copy.copy(self.interface)
        norms = np.linalg.norm(shifted_interface.lattice.matrix, axis=1)
        frac_shift = np.divide(shift, norms)

        film_ind = np.where((shifted_interface.frac_coords[:,-1] > self.interface_height) & (shifted_interface.frac_coords[:,-1] < 0.99))[0]
        shifted_interface.translate_sites(
            film_ind,
            frac_shift,
        )
        self.interface_height += frac_shift[-1] / 2
        self.interfacial_distance += shift[-1]

        return shifted_interface

    def _get_supercell(self, slab, matrix):
        new_slab = copy.copy(slab)
        initial_slab_matrix = new_slab.lattice.matrix
        initial_reduced_vecs = reduce_vectors(
            initial_slab_matrix[0],
            initial_slab_matrix[1],
        )
        initial_reduced_lattice = Lattice(
            np.vstack([
                initial_reduced_vecs,
                initial_slab_matrix[-1],
            ])
        )
        new_slab.lattice = initial_reduced_lattice

        new_slab.make_supercell(scaling_matrix=matrix)
        supercell_slab_matrix = new_slab.lattice.matrix
        supercell_reduced_vecs = reduce_vectors(
            supercell_slab_matrix[0],
            supercell_slab_matrix[1],
        )
        supercell_reduced_lattice = Lattice(
            np.vstack([
                supercell_reduced_vecs,
                supercell_slab_matrix[-1],
            ])
        )
        new_slab.lattice = supercell_reduced_lattice

        return new_slab

    def _prepare_slab(self, slab, matrix):
        initial_slab_matrix = slab.lattice.matrix
        initial_reduced_vecs = reduce_vectors(
            initial_slab_matrix[0],
            initial_slab_matrix[1],
        )
        initial_reduced_lattice = Lattice(np.vstack([
            initial_reduced_vecs,
            initial_slab_matrix[-1],
        ]))
        initial_reduced_slab = Structure(
            lattice=initial_reduced_lattice,
            species=slab.species,
            coords=slab.cart_coords,
            to_unit_cell=True,
            coords_are_cartesian=True,
        )
        supercell_slab = copy.copy(initial_reduced_slab)
        supercell_slab.make_supercell(scaling_matrix=matrix)
        supercell_slab_matrix = supercell_slab.lattice.matrix
        supercell_reduced_vecs = reduce_vectors(
            supercell_slab_matrix[0],
            supercell_slab_matrix[1],
        )
        supercell_reduced_lattice = Lattice(np.vstack([
            supercell_reduced_vecs,
            supercell_slab_matrix[-1],
        ]))
        initial_reduced_slab = Structure(
            lattice=supercell_reduced_lattice,
            species=supercell_slab.species,
            coords=supercell_slab.cart_coords,
            to_unit_cell=True,
            coords_are_cartesian=True,
        )

        return initial_reduced_slab


    def _get_angle(self, a, b):
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
        dot_prod = np.dot(a, b)
        angles = np.arccos(dot_prod / (a_norm * b_norm))

        return angles

    def _rotate(self, vec, angle):
        rot_mat = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ])
        new_vec = np.matmul(rot_mat, vec.reshape(-1,1))

        return new_vec

    def _get_interface_sl_vecs(self):
        sub_sl_vecs = copy.copy(
            self.substrate_supercell.lattice.matrix[:2,:]
        )
        if self.sub_strain_frac == 0:
            new_sl_vecs = sub_sl_vecs

            return new_sl_vecs
        else:
            a_strain = self.strain[0]
            b_strain = self.strain[1]
            a_norm = np.linalg.norm(sub_sl_vecs[0])
            b_norm = np.linalg.norm(sub_sl_vecs[1])
            sub_angle = self._get_angle(sub_sl_vecs[0], sub_sl_vecs[1])
            new_angle = sub_angle * (1 + (self.sub_strain_frac * self.angle_diff))
            new_a = sub_sl_vecs[0] * (1 + (self.sub_strain_frac * a_strain))
            new_b = self._rotate(sub_sl_vecs[0] * (b_norm / a_norm), new_angle) * (1 + (self.sub_strain_frac * b_strain))
            new_sl_vecs = np.array([new_a, np.squeeze(new_b)])

            return new_sl_vecs

    def _get_vacuum_coords(self, structure):
        z_frac_coords = structure.frac_coords[:,-1]
        min_z = np.min(z_frac_coords)
        max_z = np.max(z_frac_coords)

        return min_z, max_z

    def _strain_and_orient_film(self):
        strained_film = copy.copy(self.film_supercell)
        new_lattice = Lattice(
            np.vstack([
                self.interface_sl_vectors,
                self.film_supercell.lattice.matrix[-1],
            ])
        )
        strained_film.lattice = new_lattice

        return strained_film

    def _strain_and_orient_film_old(self):
        strained_film = copy.copy(self.film_supercell)
        new_lattice = Lattice(
            np.vstack([
                self.interface_sl_vectors,
                self.film_supercell.lattice.matrix[-1],
            ])
        )
        strained_film.lattice = new_lattice

        bot, _ = self._get_vacuum_coords(strained_film)
        strained_film.translate_sites(
            range(len(strained_film)),
            [0,0,-(0.99 * bot)],
        )
        self.interfacial_distance -= 0.01 * bot * np.linalg.norm(new_lattice.matrix[-1])

        return strained_film

    def _strain_and_orient_sub(self):
        strained_sub = copy.copy(self.substrate_supercell)
        new_lattice = Lattice(
            np.vstack([
                self.interface_sl_vectors,
                self.substrate_supercell.lattice.matrix[-1],
            ])
        )
        strained_sub.lattice = new_lattice

        return strained_sub

    def _strain_and_orient_sub_old(self):
        strained_sub = copy.copy(self.substrate_supercell)
        new_lattice = Lattice(
            np.vstack([
                self.interface_sl_vectors,
                self.substrate_supercell.lattice.matrix[-1],
            ])
        )
        strained_sub.lattice = new_lattice

        _, top = self._get_vacuum_coords(strained_sub)
        strained_sub.translate_sites(
            range(len(strained_sub)),
            [0,0,(1 - (1.01 * top))],
        )

        return strained_sub

    def _flip_structure(self, structure):
        copy_structure = copy.copy(structure)
        operation = SymmOp.from_origin_axis_angle(
            origin=[0.5,0.5,0.5],
            axis=[1,1,0],
            angle=180,
        )
        copy_structure.apply_operation(operation, fractional=True)

        return copy_structure

    def _flip_structure_old(self, structure):
        copy_structure = copy.copy(structure)
        operation = SymmOp.reflection([0,0,1], origin=[0.5,0.5,0.5])
        structure.apply_operation(operation, fractional=False)
        flipped_structure = Structure(
            lattice=copy_structure.lattice,
            species=structure.species,
            coords=structure.cart_coords,
            to_unit_cell=True,
            coords_are_cartesian=True,
        )

        return flipped_structure

    def _get_unique_species(self):
        substrate_species = np.unique(self.substrate.bulk_pmg.species).astype(str)
        film_species = np.unique(self.film.bulk_pmg.species).astype(str)

        species_in_both = []
        for species in substrate_species:
            if species in film_species:
                species_in_both.append(species)

        return substrate_species, film_species, species_in_both

    def _stack_interface(self):
        strained_sub = self.strained_sub
        strained_film = self.strained_film
        strained_sub_coords = copy.deepcopy(strained_sub.frac_coords)
        strained_film_coords = copy.deepcopy(strained_film.frac_coords)

        min_sub_coords = np.min(strained_sub_coords[:,-1])
        max_sub_coords = np.max(strained_sub_coords[:,-1])
        min_film_coords = np.min(strained_film_coords[:,-1])
        max_film_coords = np.max(strained_film_coords[:,-1])

        sub_c_len = np.linalg.norm(strained_sub.lattice.matrix[-1])
        film_c_len = np.linalg.norm(strained_film.lattice.matrix[-1])
        interface_c_len = np.sum([
            (max_sub_coords - min_sub_coords) * sub_c_len,
            (max_film_coords - min_film_coords) * film_c_len,
            self.vacuum,
            self.interfacial_distance,
        ])

        sub_conv_factor = sub_c_len / interface_c_len
        film_conv_factor = film_c_len / interface_c_len

        strained_sub_coords[:,-1] *= sub_conv_factor
        strained_film_coords[:,-1] *= film_conv_factor

        sub_size = np.max(strained_sub_coords[:,-1]) - np.min(strained_sub_coords[:,-1])

        strained_sub_coords[:,-1] -= np.min(strained_sub_coords[:,-1])
        strained_film_coords[:,-1] -= np.min(strained_film_coords[:,-1])
        strained_film_coords[:,-1] += sub_size + (self.interfacial_distance / interface_c_len)

        interface_coords = np.r_[strained_sub_coords, strained_film_coords]
        interface_species = strained_sub.species + strained_film.species

        self.interface_height =  sub_size + ((0.5 * self.interfacial_distance) / interface_c_len)

        interface_lattice = Lattice(
            np.vstack([
                strained_sub.lattice.matrix[:2],
                (interface_c_len / np.linalg.norm(strained_sub.lattice.matrix[:,-1])) * strained_sub.lattice.matrix[:,-1] 
            ])
        )

        interface_struc = Structure(
            lattice=interface_lattice,
            species=interface_species,
            coords=interface_coords,
            to_unit_cell=True,
        )

        reduced_interface_struc = interface_struc.get_reduced_structure()
        sg = SpacegroupAnalyzer(reduced_interface_struc)
        refined_interface_struc = sg.get_conventional_standard_structure()
        primitive_interface_struc = refined_interface_struc.get_reduced_structure()
        primitive_interface_struc = primitive_interface_struc.get_primitive_structure()

        substrate_species, film_species, species_in_both = self._get_unique_species()

        if len(species_in_both) == 0:
            pass
        else:
            for i in species_in_both:
                substrate_species = np.delete(substrate_species, np.where(substrate_species == i))
                film_species = np.delete(film_species, np.where(film_species == i))

        element_array_prim = np.array([
            site.species.elements[0].symbol for site in primitive_interface_struc
        ])

        substrate_ind_prim = np.isin(element_array_prim, substrate_species)
        film_ind_prim = np.isin(element_array_prim, film_species)

        average_sub_height_prim = np.mean(primitive_interface_struc.frac_coords[substrate_ind_prim,-1])
        average_film_height_prim = np.mean(primitive_interface_struc.frac_coords[film_ind_prim,-1])

        if average_film_height_prim < average_sub_height_prim:
            primitive_interface_struc = self._flip_structure(primitive_interface_struc)
        else:
            pass

        return primitive_interface_struc 

    def _setup_for_surface_matching(self, layers_sub=2, layers_film=2):
        interface = self.interface
        interface_layer_inds, interface_layes_heights = group_layers(interface)
        dist_from_interface = interface_layes_heights - self.interface_height 
        sub_layer_inds = np.where(dist_from_interface < 0)[0]
        film_layer_inds = np.where(dist_from_interface > 0)[0]
        top_sub_inds = sub_layer_inds[-layers_sub:]
        bot_film_inds = film_layer_inds[:layers_film]
        
        sub_inds = np.concatenate([interface_layer_inds[i] for i in top_sub_inds])
        film_inds = np.concatenate([interface_layer_inds[i] for i in bot_film_inds])

        smaller_interface_lattice = np.copy(interface.lattice.matrix)
        interface_c = np.linalg.norm(interface.lattice.matrix[-1])

        sub_coords = interface.frac_coords[sub_inds]
        film_coords = interface.frac_coords[film_inds]
        min_val = np.min([sub_coords[:,-1].min(), film_coords[:,-1].min()])
        sub_coords[:,-1] -= min_val
        film_coords[:,-1] -= min_val

        sub_species = [interface.species[i] for i in sub_inds]
        film_species = [interface.species[i] for i in film_inds]

        smaller_interface_c = (np.max([sub_coords[:,-1].max(), film_coords[:,-1].max()]) * interface_c) + 4
        smaller_interface_lattice[-1] /= np.linalg.norm(smaller_interface_lattice[-1])
        smaller_interface_lattice[-1] *= smaller_interface_c
        conv_factor = interface_c / smaller_interface_c

        sub_coords[:,-1] *= conv_factor
        film_coords[:,-1] *= conv_factor

        max_val = np.max([sub_coords[:,-1].max(), film_coords[:,-1].max()])
        sub_coords[:,-1] += (1 - max_val) / 2
        film_coords[:,-1] += (1 - max_val) / 2

        sub_struc = Structure(
            lattice=Lattice(smaller_interface_lattice),
            species=sub_species,
            coords=sub_coords,
            to_unit_cell=True,
        )

        film_struc = Structure(
            lattice=Lattice(smaller_interface_lattice),
            species=film_species,
            coords=film_coords,
            to_unit_cell=True,
        )

        return sub_struc, film_struc

    def _get_voxelized_structures(self, radius_dict, grid_size=0.05):
        sub_struc, film_struc = self._setup_for_surface_matching()
        species = np.unique(
            np.array(sub_struc.species + film_struc.species, dtype=str)
        )
        sub_unit_cell = smu.generate_unit_cell_tensor(
            structure=sub_struc,
            grid_size=grid_size,
            return_plotting_info=False,
        )
        film_unit_cell = np.copy(sub_unit_cell) 
        int_unit_cell = np.copy(sub_unit_cell) 

        radii = {
            s: smu.get_radii(
                structure=sub_struc,
                radius=radius_dict[s],
                grid_size=grid_size
            ) for s in species
        }

        ellipsoids = {
            s: smu.generate_ellipsoid(radii[s]) for s in species
        }

        sub_voxel, sub_overlap = smu.append_atoms(
            structure=sub_struc,
            unit_cell=sub_unit_cell,
            ellipsoids=ellipsoids
        )

        film_voxel, film_overlap = smu.append_atoms(
            structure=film_struc,
            unit_cell=film_unit_cell,
            ellipsoids=ellipsoids
        )

        return sub_voxel, film_voxel, sub_overlap, film_overlap, sub_struc, film_struc


    def _score_calculator(
        self,
        sub_voxel,
        film_voxel,
        volume_diff,
        overlap_atoms,
    ):
        overlap_int = np.logical_and(sub_voxel, film_voxel).sum()
        empty_space = volume_diff - (overlap_atoms - (2 * overlap_int))

        relative_overlap = overlap_int / overlap_atoms
        relative_empty = empty_space / volume_diff
        #  volume_density = relative_empty - (1 - (overlap_atoms / volume_diff))

        #  print(np.round(relative_empty, 4))
        score = relative_overlap / relative_empty
        #  score = (1 + relative_overlap)**2 + relative_empty

        return overlap_int, empty_space

    def _score_calculator2(
        self,
        sub_voxel,
        film_voxel,
        interface_volume,
        atoms_volume,
    ):
        overlap_int = np.logical_and(sub_voxel, film_voxel).sum()
        #  empty_space = np.logical_xor(sub_voxel, film_voxel).sum()
        empty_space = np.logical_not(sub_voxel + film_voxel).sum()
        #  print(empty_space, empty_space2)

        relative_overlap = overlap_int / atoms_volume
        relative_empty = empty_space / interface_volume

        #  score = np.sqrt((1 / relative_overlap) + (1 / relative_empty))

        return overlap_int, empty_space

    def _score_calculator_paper(
        self,
        sub_voxel,
        film_voxel,
        cell_volume,
    ):
        interface_atoms_volume = np.logical_and(sub_voxel, film_voxel).sum()
        total_atoms_volume = sub_voxel.sum() + film_voxel.sum()
        interface_overlap_volume = np.logical_and(sub_voxel, film_voxel).sum()
        interface_empty_space = cell_volume - interface_atoms_volume

        relative_overlap = interface_overlap_volume / total_atoms_volume
        relative_empty = interface_empty_space / cell_volume

        #  score = (1 + relative_overlap)**2 + (c * relative_empty)

        return relative_overlap, relative_empty

    def _get_score(
        self,
        inds,
        sub_voxel,
        film_voxel,
        volume_diff,
        overlap_atoms,
        a_range_ind,
        b_range_ind,
    ):
        shifted_film_voxel = np.roll(film_voxel, shift=[a_range_ind[inds[0]], b_range_ind[inds[1]]], axis=(0,1))
        score = self._score_calculator(
            sub_voxel=sub_voxel,
            film_voxel=shifted_film_voxel,
            volume_diff=volume_diff,
            overlap_atoms=overlap_atoms,
        )
        return score


    def _get_score2(
        self,
        inds,
        sub_voxel,
        film_voxel,
        interface_volume,
        atoms_volume,
        a_range_ind,
        b_range_ind,
    ):
        shifted_film_voxel = np.roll(film_voxel, shift=[a_range_ind[inds[0]], b_range_ind[inds[1]]], axis=(0,1))
        score = self._score_calculator2(
            sub_voxel=sub_voxel,
            film_voxel=shifted_film_voxel,
            interface_volume=interface_volume,
            atoms_volume=atoms_volume,
        )
        return score

    def _get_score_paper(
        self,
        inds,
        sub_voxel,
        film_voxel,
        cell_volume,
        a_range_ind,
        b_range_ind,
    ):
        shifted_film_voxel = np.roll(film_voxel, shift=[a_range_ind[inds[0]], b_range_ind[inds[1]]], axis=(0,1))
        score = self._score_calculator_paper(
            sub_voxel=sub_voxel,
            film_voxel=shifted_film_voxel,
            cell_volume=cell_volume,
        )
        return score


    @property
    def _metallic_elements(self):
        elements_list = np.array(
            ['Li', 'Be', 'Na', 'Mg', 'Al', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds ', 'Rg ', 'Cn ', 'Nh', 'Fl', 'Mc', 'Lv']
        )
        return elements_list

    def _get_radii(self):
        sub_species = np.unique(
            np.array(self.substrate.bulk_pmg.species, dtype=str)
        )
        film_species = np.unique(
            np.array(self.film.bulk_pmg.species, dtype=str)
        )

        sub_elements = [Element(s) for s in sub_species]
        film_elements = [Element(f) for f in film_species]

        sub_metal = np.isin(sub_species, self._metallic_elements)
        film_metal = np.isin(film_species, self._metallic_elements)

        if sub_metal.all():
            sub_dict = {sub_species[i]: sub_elements[i].metallic_radius for i in range(len(sub_elements))}
        else:
            Xs = [e.X for e in sub_elements]
            X_diff = np.abs([c[0] - c[1] for c in combinations(Xs, 2)])
            if (X_diff >= 1.7).any():
                sub_dict = {sub_species[i]: sub_elements[i].average_ionic_radius for i in range(len(sub_elements))}
            else:
                sub_dict = {s: CovalentRadius.radius[s] for s in sub_species}

        if film_metal.all():
            film_dict = {film_species[i]: film_elements[i].metallic_radius for i in range(len(film_elements))}
        else:
            Xs = [e.X for e in film_elements]
            X_diff = np.abs([c[0] - c[1] for c in combinations(Xs, 2)])
            if (X_diff >= 1.7).any():
                film_dict = {film_species[i]: film_elements[i].average_ionic_radius for i in range(len(film_elements))}
            else:
                film_dict = {f: CovalentRadius.radius[f] for f in film_species}

        sub_dict.update(film_dict)

        return sub_dict

    def _match(
        self,
        radius_dict,
        grid_size,
        grid_density,
        a_range,
        b_range,
    ):
        voxel_output = self._get_voxelized_structures(
            radius_dict=radius_dict,
            grid_size=grid_size,
        )

        sub_voxel = voxel_output[0]
        film_voxel = voxel_output[1]
        sub_overlap = voxel_output[2]
        film_overlap = voxel_output[3]
        sub_struc = voxel_output[4]
        film_struc = voxel_output[5]
        #  smu.view_structure(sub_voxel)


        sub_top = np.where(sub_overlap > 0)[-1].max()
        sub_bot = np.where(sub_overlap > 0)[-1].min()
        sub_volume = (sub_top - sub_bot) * np.product(sub_voxel.shape[1:])

        film_top = np.where(film_overlap > 0)[-1].max()
        film_bot = np.where(film_overlap > 0)[-1].min()
        film_volume = (film_top - film_bot) * np.product(film_voxel.shape[1:])

        int_volume = (film_top - sub_bot) * np.product(film_voxel.shape[1:])

        film_int_atoms = film_voxel[:,:,film_bot:sub_top+1].sum()
        sub_int_atoms = sub_voxel[:,:,film_bot:sub_top+1].sum()

        overlap_atoms = film_int_atoms + sub_int_atoms

        volume_diff = np.abs(int_volume - (sub_volume + film_volume))

        unit_cell_size = sub_voxel.shape[0]
        abc = np.linalg.norm(sub_struc.lattice.matrix, axis=1)
        voxel_dist = abc / unit_cell_size

        a_range_s = np.array(a_range) / voxel_dist[0]
        b_range_s = np.array(b_range) / voxel_dist[1]
        
        a_range_ind = np.linspace(a_range_s[0], a_range_s[1], grid_density, dtype=int)
        b_range_ind = np.linspace(b_range_s[0], b_range_s[1], grid_density, dtype=int)

        overlap = np.zeros((len(a_range_ind), len(b_range_ind)))
        empty = np.zeros((len(a_range_ind), len(b_range_ind)))
        #  scores = np.zeros((len(a_range_ind), len(b_range_ind)))

        indices = list(product(range(len(a_range_ind)), range(len(b_range_ind))))

        inputs = zip(
            indices,
            repeat(sub_voxel),
            repeat(film_voxel),
            repeat(volume_diff),
            repeat(overlap_atoms),
            repeat(a_range_ind),
            repeat(b_range_ind),
        )

        s = time.time()
        p = Pool(cpu_count())
        sc = p.starmap(self._get_score, inputs, chunksize=5)
        p.close()
        p.join()
        e = time.time()
        #  print('Parallel =', e - s)

        for i, j in enumerate(indices):
            #  scores[j[0], j[1]] = sc[i]
            overlap[j[0], j[1]] = sc[i][0]
            empty[j[0], j[1]] = sc[i][1]

        return overlap, empty, abc

    def _match2(
        self,
        radius_dict,
        grid_size,
        grid_density,
        a_range,
        b_range,
    ):
        voxel_output = self._get_voxelized_structures(
            radius_dict=radius_dict,
            grid_size=grid_size,
        )

        sub_voxel = voxel_output[0]
        film_voxel = voxel_output[1]
        sub_overlap = voxel_output[2]
        film_overlap = voxel_output[3]
        sub_struc = voxel_output[4]
        film_struc = voxel_output[5]
        smu.view_structure(sub_voxel)

        sub_top = np.where(sub_overlap > 0)[-1].max()
        sub_bot = np.where(sub_overlap > 0)[-1].min()
        sub_volume = (sub_top - sub_bot) * np.product(sub_voxel.shape[1:])
        sub_atoms_volume = sub_voxel.sum()
        sub_voxel[:,:,:sub_bot] = True

        film_top = np.where(film_overlap > 0)[-1].max()
        film_bot = np.where(film_overlap > 0)[-1].min()
        film_volume = (film_top - film_bot) * np.product(film_voxel.shape[1:])
        film_atoms_volume = sub_voxel.sum()
        film_voxel[:,:,film_top:] = True

        int_volume = (film_top - sub_bot) * np.product(film_voxel.shape[1:])
        atoms_volume = sub_atoms_volume + film_atoms_volume

        unit_cell_size = sub_voxel.shape[0]
        abc = np.linalg.norm(sub_struc.lattice.matrix, axis=1)
        voxel_dist = abc / unit_cell_size

        a_range_s = np.array(a_range) / voxel_dist[0]
        b_range_s = np.array(b_range) / voxel_dist[1]
        
        a_range_ind = np.linspace(a_range_s[0], a_range_s[1], grid_density, dtype=int)
        b_range_ind = np.linspace(b_range_s[0], b_range_s[1], grid_density, dtype=int)

        overlap = np.zeros((len(a_range_ind), len(b_range_ind)))
        empty = np.zeros((len(a_range_ind), len(b_range_ind)))

        indices = list(product(range(len(a_range_ind)), range(len(b_range_ind))))

        inputs = zip(
            indices,
            repeat(sub_voxel),
            repeat(film_voxel),
            repeat(int_volume),
            repeat(atoms_volume),
            repeat(a_range_ind),
            repeat(b_range_ind),
        )

        s = time.time()
        p = Pool(cpu_count())
        sc = p.starmap(self._get_score2, inputs, chunksize=5)
        p.close()
        p.join()
        e = time.time()
        #  print('Parallel =', e - s)

        for i, j in enumerate(indices):
            overlap[j[0], j[1]] = sc[i][0]
            empty[j[0], j[1]] = sc[i][1]

        return overlap, empty, abc

    def _match_paper(
        self,
        radius_dict,
        grid_size,
        grid_density,
        a_range,
        b_range,
        ncores,
    ):
        voxel_output = self._get_voxelized_structures(
            radius_dict=radius_dict,
            grid_size=grid_size,
        )

        sub_voxel = voxel_output[0]
        film_voxel = voxel_output[1]
        sub_overlap = voxel_output[2]
        film_overlap = voxel_output[3]
        sub_struc = voxel_output[4]
        film_struc = voxel_output[5]

        sub_top = np.where(sub_overlap > 0)[-1].max()
        sub_bot = np.where(sub_overlap > 0)[-1].min()
        sub_volume = (sub_top - sub_bot) * np.product(sub_voxel.shape[1:])

        film_top = np.where(film_overlap > 0)[-1].max()
        film_bot = np.where(film_overlap > 0)[-1].min()
        film_volume = (film_top - film_bot) * np.product(film_voxel.shape[1:])

        int_volume = (film_top - sub_bot) * np.product(film_voxel.shape[1:])

        film_int_atoms = film_voxel[:,:,film_bot:sub_top+1].sum()
        sub_int_atoms = sub_voxel[:,:,film_bot:sub_top+1].sum()

        overlap_atoms = film_int_atoms + sub_int_atoms

        volume_diff = np.abs(int_volume - (sub_volume + film_volume))

        unit_cell_size = sub_voxel.shape[0]
        abc = np.linalg.norm(sub_struc.lattice.matrix, axis=1)
        voxel_dist = abc / unit_cell_size

        a_range_s = np.array(a_range) / voxel_dist[0]
        b_range_s = np.array(b_range) / voxel_dist[1]
        
        a_range_ind = np.linspace(a_range_s[0], a_range_s[1], grid_density, dtype=int)
        b_range_ind = np.linspace(b_range_s[0], b_range_s[1], grid_density, dtype=int)

        overlap = np.zeros((len(a_range_ind), len(b_range_ind)))
        empty = np.zeros((len(a_range_ind), len(b_range_ind)))
        #  scores = np.zeros((len(a_range_ind), len(b_range_ind)))

        indices = list(product(range(len(a_range_ind)), range(len(b_range_ind))))

        inputs = zip(
            indices,
            repeat(sub_voxel),
            repeat(film_voxel),
            repeat(int_volume),
            repeat(a_range_ind),
            repeat(b_range_ind),
        )

        #### SCALING
        s = time.time()
        #  p = Pool(cpu_count())
        p = Pool(ncores)
        sc = p.starmap(self._get_score_paper, inputs, chunksize=5)
        p.close()
        p.join()
        e = time.time()
        #  print('Parallel =', e - s)

        for i, j in enumerate(indices):
            #  scores[j[0], j[1]] = sc[i]
            overlap[j[0], j[1]] = sc[i][0]
            empty[j[0], j[1]] = sc[i][1]

        return overlap, empty, abc

    def run_surface_matching(
        self,
        a_range,
        b_range,
        custom_radius_dict=None,
        grid_size=0.075,
        grid_density=20,
        output='PES.png',
        zrange=None,
        zlevels=10,
        c=1,
        ncores=cpu_count(),
    ):
        if custom_radius_dict is None:
            radius_dict = self._get_radii()
        else:
            if type(custom_radius_dict) == dict:
                radius_dict = custom_radius_dict

        if zrange is not None:
            z_shifts = np.linspace(zrange[0], zrange[1], zlevels) 
            all_scores = []
            binding_energy = []
            for z_shift in z_shifts:
                print('Before Shift =', self.interfacial_distance)
                self.shift_film([0,0,z_shift])
                print('Shift =', self.interfacial_distance)
                scores, abc = self._match2(radius_dict, grid_size, grid_density, a_range, b_range)
                binding_energy.append(scores[int(grid_density / 2), int(grid_density / 2)])
                all_scores.append(scores / scores.max())
                self.shift_film([0,0,-z_shift])
                print('After Shift =', self.interfacial_distance)

            scores_3d = np.dstack(all_scores)

            fig1, ax1 = plt.subplots()
            ax1.plot(
                z_shifts,
                binding_energy,
            )
            fig1.savefig('be.png')
        else:
            #  scores, abc = self._match2(radius_dict, grid_size, grid_density, a_range, b_range)
            #  overlap, empty, abc = self._match(radius_dict, grid_size, grid_density, a_range, b_range)
            overlap, empty, abc = self._match_paper(radius_dict, grid_size, grid_density, a_range, b_range, ncores)


        #  overlap = (overlap - overlap.min()) / (overlap.max() - overlap.min())
        #  empty = (empty - empty.min()) / (empty.max() - empty.min())

        #  fig, ax = plt.subplots(figsize=(4,4.5), dpi=400)
        #  ax.set_xlabel(r"Fractional Shift in $\vec{a}$ Direction", fontsize=12)
        #  ax.set_ylabel(r"Fractional Shift in $\vec{b}$ Direction", fontsize=12)
#
        #  score = (1 + overlap)**2 + (c * empty)
        #  score -= score.min()
        #  score /= score.max()
#
        #  im = ax.contourf(
            #  np.linspace(a_range[0], a_range[1], grid_density) / abc[0],
            #  np.linspace(b_range[0], b_range[1], grid_density) / abc[1],
            #  score.T,
            #  cmap='jet',
            #  levels=200,
            #  norm=Normalize(vmin=score.min(), vmax=score.max()),
        #  )
#
        #  cbar = fig.colorbar(im, ax=ax, orientation='horizontal')
        #  cbar.ax.tick_params(labelsize=12)
        #  cbar.ax.locator_params(nbins=4)
        #  cbar.set_label('Score', fontsize=12)
        #  ax.tick_params(labelsize=12)
        #  fig.tight_layout()
        #  fig.savefig(output)

        
    def get_ranking_score(self, radius_dict, grid_size=0.05):
        int_struc_sub_sub, int_struc_film_sub = self._setup_for_surface_matching(
            layers_sub=4,
            layers_film=1,
        )

        int_struc_sub_film, int_struc_film_film = self._setup_for_surface_matching(
            layers_sub=1,
            layers_film=4,
        )

        species = np.unique(
            np.array(int_struc_sub_sub.species + int_struc_film_film.species, dtype=str)
        )

        int_sub_sub_unit_cell = smu.generate_unit_cell_tensor(
            structure=int_struc_sub_sub,
            grid_size=grid_size,
            return_plotting_info=False,
        ) 
        int_film_sub_unit_cell = np.copy(int_sub_sub_unit_cell) 

        int_sub_film_unit_cell = smu.generate_unit_cell_tensor(
            structure=int_struc_sub_film,
            grid_size=grid_size,
            return_plotting_info=False,
        ) 
        int_film_film_unit_cell = np.copy(int_sub_film_unit_cell) 

        radii_int_sub = {
            s: smu.get_radii(
                structure=int_struc_sub_sub,
                radius=radius_dict[s],
                grid_size=grid_size
            ) for s in species
        }

        radii_int_film = {
            s: smu.get_radii(
                structure=int_struc_film_film,
                radius=radius_dict[s],
                grid_size=grid_size
            ) for s in species
        }

        ellipsoids_int_sub = {
            s: smu.generate_ellipsoid(radii_int_sub[s]) for s in species
        }

        ellipsoids_int_film = {
            s: smu.generate_ellipsoid(radii_int_film[s]) for s in species
        }

        int_sub_sub_voxel, int_sub_sub_overlap = smu.append_atoms(
            structure=int_struc_sub_sub,
            unit_cell=int_sub_sub_unit_cell,
            ellipsoids=ellipsoids_int_sub
        )

        int_film_sub_voxel, int_film_sub_overlap = smu.append_atoms(
            structure=int_struc_film_sub,
            unit_cell=int_film_sub_unit_cell,
            ellipsoids=ellipsoids_int_sub
        )

        int_sub_film_voxel, int_sub_film_overlap = smu.append_atoms(
            structure=int_struc_sub_film,
            unit_cell=int_sub_film_unit_cell,
            ellipsoids=ellipsoids_int_film
        )

        int_film_film_voxel, int_film_film_overlap = smu.append_atoms(
            structure=int_struc_film_film,
            unit_cell=int_film_film_unit_cell,
            ellipsoids=ellipsoids_int_film
        )

        O_sub = (int_sub_sub_overlap > 1).sum() / int_sub_sub_voxel.sum()
        O_film = (int_film_film_overlap > 1).sum() / int_film_film_voxel.sum()
        O_int_sub = (int_sub_sub_overlap + int_film_sub_overlap > 1).sum() / (int_sub_sub_voxel + int_film_sub_voxel).sum()
        O_int_film = (int_sub_film_overlap + int_film_film_overlap > 1).sum() / (int_sub_film_voxel + int_film_film_voxel).sum()

        ranking_score = np.abs(O_int_sub - O_sub) - np.abs(O_int_film - O_film)

        return ranking_score



