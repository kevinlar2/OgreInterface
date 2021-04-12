"""
This module will be used to construct the surfaces and interfaces used in this package.
"""
from pymatgen.core.structure import Structure 
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core.lattice import Lattice
from pymatgen.io.vasp.inputs import Poscar
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.substrate_analyzer import ZSLGenerator, SubstrateAnalyzer, reduce_vectors
from pymatgen.core.surface import get_slab_regions
from ase import Atoms
from ase.io import read
from ase.build.surfaces_with_termination import surfaces_with_termination 
from ase.spacegroup import get_spacegroup
from ase.geometry import get_layers
from ase.build.general_surface import ext_gcd
import numpy as np
from math import gcd
from itertools import combinations
from surfaces import Surface, Interface
from tqdm import tqdm
import time
import copy

class SurfaceGenerator:
    """
    The SurfaceGenerator classes generates surfaces with all possible terminations and contains 
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
        if type(structure) == Atoms:
            self.structure = structure
        elif type(structure) == Structure:
            self.structure = AseAtomsAdaptor.get_atoms(structure)
        else:
            raise TypeError(f"structure accepts 'pymatgen.core.structure.Structure' or 'ase.Atoms' not '{type(structure).__name__}'")

        self.miller_index = miller_index
        self.layers = layers
        self.vacuum = vacuum
        self.slabs = self._generate_slabs()

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

    def _generate_slabs(self):
        slabs = surfaces_with_termination(
            self.structure,
            self.miller_index,
            self.layers,
            vacuum=self.vacuum,
            return_all=True,
            verbose=False
        )


        pmg_slabs = [AseAtomsAdaptor.get_structure(slab) for slab in slabs]

        combos = combinations(range(len(pmg_slabs)), 2)
        same_slab_indices = []
        for combo in combos:
            if pmg_slabs[combo[0]] == pmg_slabs[combo[1]]:
                same_slab_indices.append(combo)

        to_delete = [np.min(same_slab_index) for same_slab_index in same_slab_indices]
        unique_slab_indices = [i for i in range(len(pmg_slabs)) if i not in to_delete]
        unique_pmg_slabs = [
            pmg_slabs[i].get_sorted_structure() for i in unique_slab_indices
        ]

        surfaces = []
        for slab in unique_pmg_slabs:
            surface = Surface(
                slab=slab,
                bulk=self.structure,
                miller_index=self.miller_index,
                layers=self.layers,
                vacuum=self.vacuum,
            )
            surfaces.append(surface)


        return surfaces


class InterfaceGenerator:
    """
    This class will use the lattice matching algorithm from Zur and McGill to generate
    commensurate interface structures between two inorganic crystalline materials.
    """

    def __init__(
            self,
            substrate,
            film,
            area_tol=0.01,
            angle_tol=0.01,
            length_tol=0.01,
            max_area=500,
            interfacial_distance=2,
            sub_strain_frac=0,
            vacuum=40,
    ):
        if type(substrate) == Surface:
            self.substrate = substrate
        else:
            raise TypeError(f"InterfaceGenerator accepts 'ogre.core.Surface' not '{type(substrate).__name__}'")

        if type(film) == Surface:
            self.film = film
        else:
            raise TypeError(f"InterfaceGenerator accepts 'ogre.core.Surface' not '{type(film).__name__}'")

        self.area_tol = area_tol
        self.angle_tol = angle_tol
        self.length_tol = length_tol
        self.max_area = max_area
        self.interfacial_distance = interfacial_distance
        self.sub_strain_frac = sub_strain_frac
        self.vacuum = vacuum
        self.interface_output = self._generate_interface_props()

        if self.interface_output is None:
            pass
        else:
            [self.film_sl_vecs,
            self.sub_sl_vecs,
            self.match_area,
            self.film_vecs,
            self.sub_vecs,
            self.film_transformations,
            self.substrate_transformations] = self.interface_output
            self._film_norms = self._get_norm(self.film_sl_vecs, ein='ijk,ijk->ij')
            self._sub_norms = self._get_norm(self.sub_sl_vecs, ein='ijk,ijk->ij')
            self.strain = self._get_strain()
            self.angle_diff = self._get_angle_diff()
            self.area_diff = self._get_area_diff()
            self.area_ratio = self._get_area_ratios()
            self.substrate_areas = self._get_area(self.sub_sl_vecs[:,0], self.sub_sl_vecs[:,1])
            self.rotation_mat = self._get_rotation_mat()


    def _get_norm(self, a, ein):
        a_norm = np.sqrt(np.einsum(ein, a, a))
        
        return a_norm

    def _get_angle(self, a, b):
        ein='ij,ij->i'
        a_norm = self._get_norm(a, ein=ein)
        b_norm = self._get_norm(b,ein=ein)
        dot_prod = np.einsum('ij,ij->i', a, b)
        angles = np.arccos(dot_prod / (a_norm * b_norm))

        return angles

    def _get_area(self, a, b):
        cross_prod = np.cross(a,b)
        area = self._get_norm(cross_prod, ein='ij,ij->i')

        return area
    
    def _get_strain(self):
        a_strain = (self._film_norms[:,0] / self._sub_norms[:,0]) - 1
        b_strain = (self._film_norms[:,1] / self._sub_norms[:,1]) - 1

        return np.c_[a_strain, b_strain]

    def _get_angle_diff(self):
        sub_angles = self._get_angle(self.sub_sl_vecs[:,0], self.sub_sl_vecs[:,1])
        film_angles = self._get_angle(self.film_sl_vecs[:,0], self.film_sl_vecs[:,1])
        angle_diff = (film_angles / sub_angles) - 1

        return angle_diff

    def _get_area_diff(self):
        sub_areas = self._get_area(self.sub_sl_vecs[:,0], self.sub_sl_vecs[:,1])
        film_areas = self._get_area(self.film_sl_vecs[:,0], self.film_sl_vecs[:,1])
        area_diff = (film_areas / sub_areas) - 1

        return area_diff

    def _get_area_ratios(self):
        q = self.film_transformations[:,0,0] * self.film_transformations[:,1,1]
        p = self.substrate_transformations[:,0,0] * self.substrate_transformations[:,1,1]
        area_ratio = np.abs((p / q) - (self.film.area / self.substrate.area))

        return area_ratio

    def _get_rotation_mat(self):
        dot_prod = np.divide(
            np.einsum(
                'ij,ij->i',
                self.sub_sl_vecs[:,0],
                self.film_sl_vecs[:,0]
            ),
            np.multiply(self._sub_norms[:,0], self._film_norms[:,0])
        )

        mag_cross = np.divide(
            self._get_area(
                self.sub_sl_vecs[:,0],
                self.film_sl_vecs[:,0]
            ),
            np.multiply(self._sub_norms[:,0], self._film_norms[:,0])
        )

        rot_mat = np.c_[
            dot_prod,
            -mag_cross,
            np.zeros(len(dot_prod)),
            mag_cross,
            dot_prod,
            np.zeros(len(dot_prod)),
            np.zeros(len(dot_prod)),
            np.zeros(len(dot_prod)),
            np.ones(len(dot_prod)),
        ].reshape(-1,3,3)

        return rot_mat


    def _generate_interface_props(self):
        zsl = ZSLGenerator(
            max_area_ratio_tol=self.area_tol,
            max_angle_tol=self.angle_tol,
            max_length_tol=self.length_tol,
            max_area=self.max_area,
        )

        sa = SubstrateAnalyzer(zslgen=zsl)

        matches = sa.calculate(
            film=self.film.bulk_pmg,
            substrate=self.substrate.bulk_pmg,
            film_millers=[self.film.miller_index],
            substrate_millers=[self.substrate.miller_index],
        )

        match_list = list(matches)
        
        if len(match_list) == 0:
            return None
            "TODO: Return None if there are no matches"
            #  return [
                #  None,
                #  None,
                #  None,
                #  None,
                #  None,
                #  None,
                #  None,
            #  ]
        else:
            film_sl_vecs = np.array([match['film_sl_vecs'] for match in match_list])
            sub_sl_vecs = np.array([match['sub_sl_vecs'] for match in match_list])
            match_area = np.array([match['match_area'] for match in match_list])
            film_vecs = np.array([match['film_vecs'] for match in match_list])
            sub_vecs = np.array([match['sub_vecs'] for match in match_list])
            film_transformations = np.array(
                [match['film_transformation'] for match in match_list]
            )
            substrate_transformations = np.array(
                [match['substrate_transformation'] for match in match_list]
            )

            film_3x3_transformations = np.array(
                [np.eye(3,3) for _ in range(film_transformations.shape[0])]
            )
            substrate_3x3_transformations = np.array(
                [np.eye(3,3) for _ in range(substrate_transformations.shape[0])]
            )

            film_3x3_transformations[:,:2,:2] = film_transformations
            substrate_3x3_transformations[:,:2,:2] = substrate_transformations

            return [
                film_sl_vecs,
                sub_sl_vecs,
                match_area,
                film_vecs,
                sub_vecs,
                film_3x3_transformations,
                substrate_3x3_transformations,
            ]

    def _find_unique_reduced_vectors(self):
        film_lattice = self.film.slab_pmg.lattice.matrix
        reduced_film_vectors = reduce_vectors(film_lattice[0], film_lattice[1])
        reduced_film_lattice = np.vstack([reduced_film_vectors, film_lattice[-1]])

        substrate_lattice = self.substrate.slab_pmg.lattice.matrix
        reduced_substrate_vectors = reduce_vectors(substrate_lattice[0], substrate_lattice[1])
        reduced_substrate_lattice = np.vstack([reduced_substrate_vectors, substrate_lattice[-1]])

        reduced_sc_substrate_vecs = []
        for substrate_transformation in self.substrate_transformations:
            new_substrate_lattice = np.dot(substrate_transformation, reduced_substrate_lattice)
            new_reduced_vectors = reduce_vectors(new_substrate_lattice[0], new_substrate_lattice[1])
            reduced_sc_substrate_vecs.append(new_reduced_vectors)


        reduced_sc_film_vecs = []
        for film_transformation in self.film_transformations:
            new_film_lattice = np.dot(film_transformation, reduced_film_lattice)
            new_reduced_vectors = reduce_vectors(new_film_lattice[0], new_film_lattice[1])
            reduced_sc_film_vecs.append(new_reduced_vectors)

        reduced_sc_substrate_vecs = np.round(np.array(reduced_sc_substrate_vecs),4)
        reduced_sc_film_vecs = np.round(np.array(reduced_sc_film_vecs), 4)

        reduced_sc_substrate_vecs_a = reduced_sc_substrate_vecs[:,0]
        reduced_sc_film_vecs_a = reduced_sc_film_vecs[:,0]
        reduced_sc_substrate_vecs_b = reduced_sc_substrate_vecs[:,1]
        reduced_sc_film_vecs_b = reduced_sc_film_vecs[:,1]

        unique_substrate_vecs_a = np.unique(reduced_sc_substrate_vecs_a, axis=0)
        unique_film_vecs_a = np.unique(reduced_sc_film_vecs_a, axis=0)
        unique_substrate_vecs_b = np.unique(reduced_sc_substrate_vecs_b, axis=0)
        unique_film_vecs_b = np.unique(reduced_sc_film_vecs_b, axis=0)

        subs_ind_a = []
        for unique_vecs in unique_substrate_vecs_a:
            ind = np.where((reduced_sc_substrate_vecs_a==unique_vecs).all(axis=1))
            subs_ind_a.append(ind)

        subs_ind_b = []
        for unique_vecs in unique_substrate_vecs_b:
            ind = np.where((reduced_sc_substrate_vecs_b==unique_vecs).all(axis=1))
            subs_ind_b.append(ind)

        film_ind_a = []
        for unique_vecs in unique_film_vecs_a:
            ind = np.where((reduced_sc_film_vecs_a==unique_vecs).all(axis=1))
            film_ind_a.append(ind)

        film_ind_b = []
        for unique_vecs in unique_film_vecs_b:
            ind = np.where((reduced_sc_film_vecs_b==unique_vecs).all(axis=1))
            film_ind_b.append(ind)

        print(unique_substrate_vecs_a.shape)
        print(unique_film_vecs_a.shape)
        print(unique_substrate_vecs_b.shape)
        print(unique_film_vecs_b.shape)
#
        #  film_ind = []
        #  for unique_vecs in unique_reduced_supercell_film_vecs:
            #  ind = np.where((reduced_supercell_film_vecs==unique_vecs).all(axis=1).all(axis=1))[0]
            #  film_ind.append(ind)

    def _is_equal(self, structure1, structure2):
        structure_matcher = StructureMatcher(
            ltol=0.001,
            stol=0.001,
            angle_tol=0.001,
            primitive_cell=False,
            scale=False,
        )
        #  is_fit = structure_matcher.fit(structure1, structure2)
        match = structure_matcher._match(structure1, structure2, 1)
        if match is None:
            is_fit =  False
        else:
            is_fit = match[0] <= 0.001

        return is_fit

    def _find_exact_matches(self, structures):
        all_coords = np.array([i.interface.frac_coords for i in structures])
        all_species = np.array([i.interface.species for i in structures])

        for i in range(len(structures)):
            coords = np.round(all_coords[i], 6)
            coords[:,-1] = coords[:,-1] - np.min(coords[:,-1])
            coords.dtype = [('a', 'float64'), ('b', 'float64'), ('c', 'float64')]
            coords_inds = np.squeeze(np.argsort(coords, axis=0, order=('c', 'b', 'a')))
            coords.dtype = 'float64'
            
            coords_sorted = coords[coords_inds]
            species_sorted = np.array(all_species[i]).astype(str)[coords_inds]

            all_coords[i] = coords_sorted
            all_species[i] = species_sorted

        equal_coords = np.array(
            [np.isclose(all_coords[i], all_coords).all(axis=1).all(axis=1) for i in range(all_coords.shape[0])]
        )
        unique_eq = np.unique(equal_coords, axis=0)

        inds = [np.where(unique_eq[i])[0] for i in range(unique_eq.shape[0])]
        reduced_inds = [np.min(i) for i in inds]

        return reduced_inds



    def _is_equal_fast(self, structure1, structure2):
        if len(structure1) != len(structure2):
            return False
        else:
            coords1 = np.round(structure1.frac_coords, 4)
            coords1[:,-1] = coords1[:,-1] - np.min(coords1[:,-1])
            coords1.dtype = [('a', 'float64'), ('b', 'float64'), ('c', 'float64')]
            coords1_inds = np.squeeze(np.argsort(coords1, axis=0, order=('c', 'b', 'a')))
            coords1.dtype = 'float64'

            coords2 = np.round(structure2.frac_coords, 4)
            coords2[:,-1] = coords2[:,-1] - np.min(coords2[:,-1])
            coords2.dtype = [('a', 'float64'), ('b', 'float64'), ('c', 'float64')]
            coords2_inds = np.squeeze(np.argsort(coords2, axis=0, order=('c', 'b', 'a')))
            coords2.dtype = 'float64'

            coords1_sorted = coords1[coords1_inds]
            coords2_sorted = coords2[coords2_inds]
            species1_sorted = np.array(structure1.species).astype(str)[coords1_inds]
            species2_sorted = np.array(structure2.species).astype(str)[coords2_inds]

            coords = np.isclose(coords1_sorted, coords2_sorted, rtol=1e-2, atol=1e-2).all()
            species = (species1_sorted == species2_sorted).all()

            if coords and species:
                return True 
            else:
                return False


    def generate_interfaces_old(self):
        interfaces = []
        print('Generating Interfaces:')
        for i in tqdm(range(self.substrate_transformations.shape[0])):
            interface = Interface(
                substrate=self.substrate,
                film=self.film,
                film_transformation=self.film_transformations[i],
                substrate_transformation=self.substrate_transformations[i],
                strain=self.strain[i],
                angle_diff=self.angle_diff[i],
                sub_strain_frac=self.sub_strain_frac,
                interfacial_distance=self.interfacial_distance,
                vacuum=self.vacuum,
            )
            interfaces.append(interface)

        combos = combinations(range(len(interfaces)), 2)
        same_slab_indices = []
        print('Finding Symmetrically Equivalent Interfaces:')
        for combo in tqdm(combos):
            if self._is_equal(interfaces[combo[0]].interface, interfaces[combo[1]].interface):
                same_slab_indices.append(combo)

        to_delete = [np.min(same_slab_index) for same_slab_index in same_slab_indices]
        unique_slab_indices = [i for i in range(len(interfaces)) if i not in to_delete]
        unique_interfaces = [
            interfaces[i] for i in unique_slab_indices
        ]

        areas = []

        for interface in unique_interfaces:
            matrix = interface.interface.lattice.matrix
            area = self._get_area([matrix[0]], [matrix[1]])[0]
            areas.append(area)

        sort = np.argsort(areas)
        sorted_unique_interfaces = [unique_interfaces[i] for i in sort]

        return sorted_unique_interfaces

    def generate_interfaces(self):
        interfaces = []
        print('Generating Interfaces:')
        for i in tqdm(range(self.substrate_transformations.shape[0])):
            interface = Interface(
                substrate=self.substrate,
                film=self.film,
                film_transformation=self.film_transformations[i],
                substrate_transformation=self.substrate_transformations[i],
                strain=self.strain[i],
                angle_diff=self.angle_diff[i],
                sub_strain_frac=self.sub_strain_frac,
                interfacial_distance=self.interfacial_distance,
                vacuum=self.vacuum,
            )
            #  interface.shift_film([0.3, 0.6, 0])
            interfaces.append(interface)

        interfaces = np.array(interfaces)
        all_int = interfaces

        interface_sizes = np.array(
            [len(interfaces[i].interface) for i in range(len(interfaces))]
        )
        unique_inds = np.array(
            [np.isin(interface_sizes, i) for i in np.unique(interface_sizes)]
        )
        possible_alike_strucs = [interfaces[unique_inds[i]] for i in range(unique_inds.shape[0])]

        interfaces = []

        for strucs in possible_alike_strucs:
            inds = self._find_exact_matches(strucs)
            reduced_strucs = strucs[inds]
            interfaces.extend(reduced_strucs)

        combos = combinations(range(len(interfaces)), 2)
        same_slab_indices = []
        print('Finding Symmetrically Equivalent Interfaces:')
        for combo in tqdm(combos):
            if self._is_equal(interfaces[combo[0]].interface, interfaces[combo[1]].interface):
                same_slab_indices.append(combo)

        to_delete = [np.min(same_slab_index) for same_slab_index in same_slab_indices]
        unique_slab_indices = [i for i in range(len(interfaces)) if i not in to_delete]
        unique_interfaces = [
            interfaces[i] for i in unique_slab_indices
        ]

        areas = []

        for interface in unique_interfaces:
            matrix = interface.interface.lattice.matrix
            area = self._get_area([matrix[0]], [matrix[1]])[0]
            areas.append(area)

        sort = np.argsort(areas)
        sorted_unique_interfaces = [unique_interfaces[i] for i in sort]

        return sorted_unique_interfaces



if __name__ == "__main__":
    subs = SurfaceGenerator.from_file(
        './poscars/POSCAR_InSb_conv',
        miller_index=[1,0,0],
        layers=1,
        vacuum=3.5,
    )

    films = SurfaceGenerator.from_file(
        './poscars/POSCAR_bSn_conv',
        miller_index=[1,0,0],
        layers=8,
        vacuum=5,
    )

    Poscar(subs.slabs[0].primitive_slab_pmg).write_file('POSCAR_sub_100')
    #  Poscar(films.slabs[0].slab_pmg).write_file('POSCAR_film_121')

    #  subs.slabs[3].remove_layers(num_layers=5, top=False)
    #  films.slabs[0].remove_layers(num_layers=1, top=True)

    #  inter = InterfaceGenerator(
        #  substrate=subs.slabs[0],
        #  film=films.slabs[0],
        #  length_tol=0.01,
        #  angle_tol=0.01,
        #  area_tol=0.01,
        #  max_area=500,
        #  interfacial_distance=2,
        #  sub_strain_frac=0,
        #  vacuum=30,
    #  )
#
    #  interfaces = inter.generate_interfaces()
#  #
    #  import os
    #  from vaspvis.utils import passivator
    #  for i in range(len(interfaces)):
        #  print(interfaces[i].strain)
        #  #  Poscar(interfaces[i].interface).write_file(os.path.join('./test_bSn001-InSb1-10', f'POSCAR_{i}'))
#
        #  passivator(
            #  interfaces[i].interface,
            #  top=False,
            #  write_file=True,
            #  output=os.path.join('./test', f'POSCAR_{i}'),
            #  #  passivated_struc='./test/CONTCAR_FeInSb',
        #  )
    #  sub_layer = 5
    #  film_layer = 5
#
    #  subs = SurfaceGenerator.from_file(
        #  './poscars/POSCAR_InSb_conv',
        #  miller_index=[1,0,0],
        #  layers=sub_layer,
        #  vacuum=5,
    #  )
#
    #  films = SurfaceGenerator.from_file(
        #  './poscars/POSCAR_Fe_conv',
        #  miller_index=[1,0,0],
        #  layers=film_layer,
        #  vacuum=5,
    #  )
#
#
    #  #  films.slabs[1].remove_layers(num_layers=1, top=True)
    #  #  subs.slabs[3].remove_layers(num_layers=4, top=True)
#
    #  inter = InterfaceGenerator(
        #  substrate=subs.slabs[3],
        #  film=films.slabs[1],
        #  length_tol=0.015,
        #  angle_tol=0.015,
        #  area_tol=0.015,
        #  max_area=300,
        #  interfacial_distance=1.9,
        #  sub_strain_frac=0,
        #  vacuum=40,
    #  )
#
#
    #  interfaces = inter.generate_interfaces()
#
    #  import os
    #  from vaspvis.utils import passivator
#
    #  #  if not os.path.isdir(f'./test/small_InSb_{sub_layer}_Fe_{film_layer}'):
        #  #  os.mkdir(f'./test/small_InSb_{sub_layer}_Fe_{film_layer}')
#
    #  for i in range(len(interfaces)):
        #  print(interfaces[i].strain)
        #  #  film_vecs = interfaces[i].film_supercell.lattice.matrix[:2]
        #  #  sub_vecs = interfaces[i].substrate_supercell.lattice.matrix[:2]
        #  #  film_area = np.linalg.norm(np.cross(film_vecs[0], film_vecs[1]))
        #  #  sub_area = np.linalg.norm(np.cross(sub_vecs[0], sub_vecs[1]))
        #  #  print('Film', film_area)
        #  #  print('Sub', sub_area)
        #  #  print('Diff', (film_area / sub_area) - 1)
        #  #  #  print(interfaces[i].area_diff)
        #  #  passivator(
            #  #  interfaces[i].interface,
            #  #  top=False,
            #  #  write_file=True,
            #  #  output=os.path.join(
                #  #  'test',
                #  #  f'InSb_{sub_layer}_Fe_{film_layer}',
                #  #  f'POSCAR_{i}_InSb_{sub_layer}_Fe_{film_layer}'
            #  #  ),
            #  #  passivated_struc='./test/CONTCAR_FeInSb',
        #  #  )
        #  Poscar(interfaces[i].interface).write_file(os.path.join('test', f'POSCAR_{i}'))
            
