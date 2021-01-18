"""
This module will be used to construct the surfaces and interfaces used in this package.
"""
from pymatgen.core.structure import Structure 
from pymatgen.core.lattice import Lattice 
from pymatgen.core.surface import get_slab_regions
from pymatgen.io.vasp.inputs import Poscar
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.substrate_analyzer import ZSLGenerator, SubstrateAnalyzer, reduce_vectors
from pymatgen.symmetry.analyzer import SymmOp
from ase import Atoms
from ase.io import read
from ase.build.surfaces_with_termination import surfaces_with_termination
from ase.build import make_supercell, stack
from ase.spacegroup import get_spacegroup
from ase.geometry import get_layers
from ase.build.general_surface import ext_gcd
import numpy as np
from math import gcd
from itertools import combinations
from pymatgen.analysis.interface import merge_slabs
import copy


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

        self.no_vac_slab_pmg = self._remove_vacuum(self.slab_pmg)
        self.no_vac_slab_ase = AseAtomsAdaptor.get_atoms(self.no_vac_slab_pmg)

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

        #  print(np.matmul(
            #  np.linalg.inv(self.slab_basis), structure.lattice.matrix.T).T /
            #  self.structure.cell[0][0]
        #  )


        return np.array([c1, c2, c3])


class Interface:

    def __init__(
        self,
        substrate,
        film,
        #  film_sl_vecs,
        #  sub_sl_vecs,
        #  match_area,
        #  film_vecs,
        #  sub_vecs,
        film_transformation,
        substrate_transformation,
        strain,
        angle_diff,
        sub_strain_frac=0,
        interfacial_distance=2,
        #  area_diff,
    ):
        self.substrate = substrate
        self.film = film
        #  self.film_sl_vecs = film_sl_vecs
        #  self.sub_sl_vecs = sub_sl_vecs
        #  self.match_area = match_area
        #  self.film_vecs = film_vecs
        #  self.sub_vecs = sub_vecs
        self.film_transformation = film_transformation
        self.substrate_transformation = substrate_transformation
        self.strain = strain
        self.angle_diff = angle_diff
        self.sub_strain_frac = sub_strain_frac
        #  self.area_diff = area_diff
        self.substrate_supercell = self._get_supercell(
            self.substrate.slab_pmg,
            self.substrate_transformation
        )
        self.film_supercell = self._get_supercell(
            self.film.slab_pmg,
            self.film_transformation
        )
        self.interface_sl_vectors = self._get_interface_sl_vecs()
        self.interfacial_distance = interfacial_distance
        #  self.interface = self._stack_interface()

    def _get_supercell(self, slab, matrix):
        new_slab = copy.deepcopy(slab)
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
        supercell_slab = copy.deepcopy(initial_reduced_slab)
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
        sub_sl_vecs = copy.deepcopy(
            self.substrate_supercell.lattice.matrix[:2,:]
        )
        film_sl_vecs = copy.deepcopy(
            self.film_supercell.lattice.matrix[:2,:]
        )
        if self.sub_strain_frac == 0:
            new_sl_vecs = sub_sl_vecs

            return new_sl_vecs
        else:
            a_strain = self.strain[0]
            b_strain = self.strain[1]
            a_norm = np.linalg.norm(sub_sl_vecs[0])
            b_norm = np.linalg.norm(sub_sl_vecs[1])
            #  a1_norm = np.linalg.norm(film_sl_vecs[0])
            #  b1_norm = np.linalg.norm(film_sl_vecs[1])
            sub_angle = self._get_angle(sub_sl_vecs[0], sub_sl_vecs[1])
            #  film_angle = self._get_angle(film_sl_vecs[0], film_sl_vecs[1])
            new_angle = sub_angle * (1 + (self.sub_strain_frac * self.angle_diff))
            new_a = sub_sl_vecs[0] * (1 + (self.sub_strain_frac * a_strain))
            new_b = self._rotate(sub_sl_vecs[0] * (b_norm / a_norm), new_angle) * (1 + (self.sub_strain_frac * b_strain))
            #  print('Sub a', a_norm)
            #  print('Film a', a1_norm)
            #  print('New a ', np.linalg.norm(new_a))
            #  print('Calc Strain', (a_norm - np.linalg.norm(new_a)) / (a_norm - a1_norm))
            #  print('Sub b', b_norm)
            #  print('Film b', b1_norm)
            #  print('New b ', np.linalg.norm(new_b))
            #  print('Calc Strain', (b_norm - np.linalg.norm(new_b)) / (b_norm - b1_norm))

            new_sl_vecs = np.array([new_a, np.squeeze(new_b)])

            return new_sl_vecs

    def _strain_and_orient_film(self):
        strained_film = copy.deepcopy(self.film_supercell)
        new_lattice = Lattice(
            np.vstack([
                self.interface_sl_vectors,
                strained_film.lattice.matrix[-1],
            ])
        )
        strained_film.lattice = new_lattice
        bot, _ = get_slab_regions(strained_film)[0]
        strained_film.translate_sites(
            range(len(strained_film)),
            [0,0,-(0.99 * bot)],
        )
        self.interfacial_distance -= 0.01 * bot * np.linalg.norm(new_lattice.matrix[-1])

        return strained_film

    def _strain_and_orient_sub(self):
        strained_sub = copy.deepcopy(self.substrate_supercell)
        new_lattice = Lattice(
            np.vstack([
                self.interface_sl_vectors,
                strained_sub.lattice.matrix[-1],
            ])
        )
        strained_sub.lattice = new_lattice
        _, top = get_slab_regions(strained_sub)[0]
        strained_sub.translate_sites(
            range(len(strained_sub)),
            [0,0,(1 - (1.01 * top))],
        )
        #  self.interfacial_distance -= 0.01 * (1 - top) * np.linalg.norm(new_lattice.matrix[-1])

        return strained_sub

    def _flip_structure(self, structure):
        copy_structure = copy.deepcopy(structure)
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

    def _group_layers(self, structure):
        sites = structure.sites
        zvals = np.array([site.c for site in sites])
        unique_values = np.sort(np.unique(np.round(zvals, 3)))
        diff = np.mean(np.diff(unique_values)) * 0.2

        grouped = False
        groups = []
        group_heights = []
        zvals_copy = copy.deepcopy(zvals)
        while not grouped:
            if len(zvals_copy) > 0:
                group_index = np.where(
                    np.isclose(zvals, np.min(zvals_copy), atol=diff)
                )[0]
                group_heights.append(np.min(zvals_copy))
                zvals_copy = np.delete(zvals_copy, np.where(
                    np.isin(zvals_copy, zvals[group_index]))[0])
                groups.append(group_index)
            else:
                grouped = True

        return groups

    def _stack_interface(self):
        strained_sub = self._strain_and_orient_sub()
        strained_film = self._strain_and_orient_film()
        strained_sub_coords = strained_sub.cart_coords
        strained_film_coords = strained_film.cart_coords
        max_sub_coords = np.max(strained_sub_coords[:,-1])
        film_shift = max_sub_coords + self.interfacial_distance
        strained_film_coords[:,-1] = strained_film_coords[:,-1] + film_shift
        interface_coords = np.r_[strained_sub_coords, strained_film_coords]
        interface_species = strained_sub.species + strained_film.species
        new_c_len = np.sum([
            np.linalg.norm(strained_sub.lattice.matrix[-1]),
            np.linalg.norm(strained_film.lattice.matrix[-1]),
            self.interfacial_distance,
        ])
        #  middle_height = np.sum([
            #  np.linalg.norm(strained_sub.lattice.matrix[-1]),
            #  0.5 * self.interfacial_distance,
        #  ])

        interface_lattice = Lattice(
            np.vstack([
                strained_sub.lattice.matrix[:2],
                [0, 0, new_c_len]
            ])
        )


        interface_struc = Structure(
            lattice=interface_lattice,
            species=interface_species,
            coords=interface_coords,
            to_unit_cell=True,
            coords_are_cartesian=True,
        )

        layer_inds_conv = self._group_layers(interface_struc)
        
        avg_Zs_conv = []
        for layer_ind in layer_inds_conv:
            avg_Z = np.mean([interface_struc.sites[i].species.elements[0].Z for i in layer_ind])
            avg_Zs_conv.append(avg_Z)

        sg = SpacegroupAnalyzer(interface_struc)
        interface_prim_struc = sg.get_primitive_standard_structure()

        layer_inds_prim = self._group_layers(interface_prim_struc)
        
        avg_Zs_prim = []
        for layer_ind in layer_inds_prim:
            avg_Z = np.mean([interface_prim_struc.sites[i].species.elements[0].Z for i in layer_ind])
            avg_Zs_prim.append(avg_Z)

        avg_Zs_conv = np.array(avg_Zs_conv)
        avg_Zs_prim = np.array(avg_Zs_prim)

        if np.mean(np.isclose(avg_Zs_conv, avg_Zs_prim)) < 1:
            if np.mean(np.isclose(avg_Zs_conv, np.flip(avg_Zs_prim))) == 1:
                interface_prim_struc = self._flip_structure(interface_prim_struc)
            else:
                pass
        else:
            pass

        return interface_prim_struc

        #  interface = stack(
            #  atoms1=self.substrate_supercell,
            #  atoms2=self.film_supercell,
        #  )
        #  interface = merge_slabs(
            #  substrate=self.substrate_supercell,
            #  film=self.film_supercell,
            #  slab_offset=2,
            #  x_offset=0,
            #  y_offset=0,
        #  )

        #  return interface

