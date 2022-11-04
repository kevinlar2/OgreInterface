"""
This module will be used to construct the surfaces and interfaces used in this package.
"""
from pymatgen.core.structure import Structure 
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core.lattice import Lattice
from pymatgen.io.vasp.inputs import Poscar
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
# from pymatgen.analysis.substrate_analyzer import ZSLGenerator, SubstrateAnalyzer, reduce_vectors
from pymatgen.analysis.interfaces.zsl import ZSLGenerator, reduce_vectors
from pymatgen.analysis.interfaces.substrate_analyzer import SubstrateAnalyzer
from pymatgen.core.surface import get_slab_regions
from pymatgen.core.periodic_table import Element
from pymatgen.analysis.ewald import EwaldSummation
from pymatgen.transformations.standard_transformations import RotationTransformation

from ase import Atoms
from ase.io import read
from ase.build.surfaces_with_termination import surfaces_with_termination 
from ase.build.general_surface import surface
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
        generate_all=True,
        filter_ionic_slabs=False,
    ):
        if type(structure) == Atoms:
            self.structure = structure
            self.pmg_structure = AseAtomsAdaptor.get_structure(structure)
        elif type(structure) == Structure:
            self.structure = AseAtomsAdaptor.get_atoms(structure)
            self.pmg_structure = structure
        else:
            raise TypeError(f"structure accepts 'pymatgen.core.structure.Structure' or 'ase.Atoms' not '{type(structure).__name__}'")

        self.miller_index = miller_index
        self.layers = layers
        self.vacuum = vacuum
        self.generate_all = generate_all
        self.filter_ionic_slabs = filter_ionic_slabs
        self.slabs = self._generate_slabs()

    @classmethod
    def from_file(
        cls,
        filename,
        miller_index,
        layers,
        vacuum,
        generate_all=True,
        filter_ionic_slabs=False,
    ):
        structure = Structure.from_file(filename=filename)

        return cls(structure, miller_index, layers, vacuum, generate_all, filter_ionic_slabs)


    def _get_ewald_energy(self, slab):
        slab = copy.deepcopy(slab)
        bulk = copy.deepcopy(self.pmg_structure)
        slab.add_oxidation_state_by_guess()
        bulk.add_oxidation_state_by_guess()
        E_slab = EwaldSummation(slab).total_energy
        E_bulk = EwaldSummation(bulk).total_energy

        return E_slab, E_bulk

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

    def _generate_slabs(self):
        if self.generate_all:
            slabs = surfaces_with_termination(
                self.structure,
                self.miller_index,
                self.layers,
                vacuum=self.vacuum,
                return_all=True,
                verbose=False
            )

        else:
            slab = surface(
                self.structure,
                self.miller_index,
                self.layers,
                vacuum=self.vacuum
            )
            slabs = [slab]


        pmg_slabs = [AseAtomsAdaptor.get_structure(slab) for slab in slabs]

        if self.filter_ionic_slabs and self.generate_all:
            diffs = []
            for pmg_slab in pmg_slabs:
                s = time.time()
                E_vac, E_bulk = self._get_ewald_energy(pmg_slab)
                print(time.time() - s)
                diffs.append(E_vac - (self.layers * E_bulk))

            diffs = np.array(diffs)
            stable_inds = np.where(np.isclose(diffs, diffs.min()))[0]
            pmg_slabs = [pmg_slabs[i] for i in stable_inds]

        combos = combinations(range(len(pmg_slabs)), 2)
        same_slab_indices = []
        # rot = lambda x, a: RotationTransformation(axis=[0,0,1], angle=a).apply_transformation(x)
        for combo in combos:
            # if self._is_equal(pmg_slabs[combo[0]], pmg_slabs[combo[1]]):
            #     same_slab_indices.append(combo)

            if pmg_slabs[combo[0]] == pmg_slabs[combo[1]]:
                same_slab_indices.append(combo)
            # if pmg_slabs[combo[0]] == rot(pmg_slabs[combo[1]], 90):
            #     same = True
            # if pmg_slabs[combo[0]] == rot(pmg_slabs[combo[1]], 180):
            #     same = True
            # if pmg_slabs[combo[0]] == rot(pmg_slabs[combo[1]], 270):
            #     same = True
            # if same:
            #     same_slab_indices.append(combo)

        to_delete = [np.min(same_slab_index) for same_slab_index in same_slab_indices]
        unique_slab_indices = [i for i in range(len(pmg_slabs)) if i not in to_delete]
        unique_pmg_slabs = [
            pmg_slabs[i].get_sorted_structure() for i in unique_slab_indices
        ]

        surfaces = []
        for slab in unique_pmg_slabs:
            unique_surface = Surface(
                slab=slab,
                bulk=self.structure,
                miller_index=self.miller_index,
                layers=self.layers,
                vacuum=self.vacuum,
            )
            surfaces.append(unique_surface)


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
            center=False,
    ):
        if type(substrate) == Surface:
            self.substrate = substrate
        else:
            raise TypeError(f"InterfaceGenerator accepts 'ogre.core.Surface' not '{type(substrate).__name__}'")

        if type(film) == Surface:
            self.film = film
        else:
            raise TypeError(f"InterfaceGenerator accepts 'ogre.core.Surface' not '{type(film).__name__}'")

        self.center = center
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
        # zsl = ZSLGenerator(
        #     max_area_ratio_tol=self.area_tol,
        #     max_angle_tol=self.angle_tol,
        #     max_length_tol=self.length_tol,
        #     max_area=self.max_area,
        # )

        # sa = SubstrateAnalyzer(zslgen=zsl)
        sa = SubstrateAnalyzer(
            max_area_ratio_tol=self.area_tol,
            max_angle_tol=self.angle_tol,
            max_length_tol=self.length_tol,
            max_area=self.max_area,
        )

        matches = sa.calculate(
            film=self.film.bulk_pmg,
            substrate=self.substrate.bulk_pmg,
            film_millers=[self.film.miller_index],
            substrate_millers=[self.substrate.miller_index],
        )

        match_list = list(matches)

        # print(match_list[0])
        # print(match_list[0].__dict__.keys())
        
        if len(match_list) == 0:
            return None
        else:

            film_sl_vecs = np.array([match.film_sl_vectors for match in match_list])
            sub_sl_vecs = np.array([match.substrate_sl_vectors for match in match_list])
            match_area = np.array([match.match_area for match in match_list])
            film_vecs = np.array([match.film_vectors for match in match_list])
            sub_vecs = np.array([match.substrate_vectors for match in match_list])
            film_transformations = np.array(
                [match.film_transformation for match in match_list]
            )
            substrate_transformations = np.array(
                [match.substrate_transformation for match in match_list]
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
                center=self.center,
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
    sub_layer = 31
    film_layer = 13

    #  sub_layer = 13
    #  film_layer = 8

    subs = SurfaceGenerator.from_file(
        #  './poscars/POSCAR_InSb_conv',
        './poscars/POSCAR_InAs_conv',
        miller_index=[1,1,1],
        layers=sub_layer,
        vacuum=5,
    )

    films = SurfaceGenerator.from_file(
        #  './poscars/POSCAR_Fe_conv',
        './poscars/POSCAR_Al_conv',
        miller_index=[1,1,1],
        layers=film_layer,
        vacuum=5,
    )

    #  subs.slabs[3].remove_layers(num_layers=5)
    #  films.slabs[0].remove_layers(num_layers=1, top=True)

    inter = InterfaceGenerator(
        #  substrate=films.slabs[0],
        #  film=subs.slabs[1],
        substrate=subs.slabs[2],
        film=films.slabs[0],
        length_tol=0.01,
        angle_tol=0.01,
        area_tol=0.01,
        max_area=400,
        interfacial_distance=2.2,
        sub_strain_frac=1,
        vacuum=80,
    )

    #  inter = InterfaceGenerator(
        #  substrate=subs.slabs[3],
        #  film=films.slabs[0],
        #  length_tol=0.05,
        #  angle_tol=0.05,
        #  area_tol=0.05,
        #  max_area=700,
        #  interfacial_distance=2.2,
        #  sub_strain_frac=0,
        #  #  vacuum=2.2,
        #  vacuum=30,
    #  )

    r = {'In': 1.37, 'Sb': 1.33, 'Fe': 1.235}
    r3 = {'In': 1.582, 'As': 1.27, 'Al': 1.43}

    from pymatgen.analysis.molecule_structure_comparator import CovalentRadius
    r2 = {
        'In': CovalentRadius.radius['In'],
        'Sb': CovalentRadius.radius['Sb'],
        'As': CovalentRadius.radius['As'],
        'Fe': Element('Fe').metallic_radius,
        'Al': Element('Al').metallic_radius,
    }

    interfaces = inter.generate_interfaces()
    range_a = [-1, 1]
    range_b = [-1, 1]
    grid_size = 0.05
    grid_density = 15
    
    from vaspvis.utils import passivator

    for i, interface in enumerate(interfaces):
        pas = passivator(
            struc=interface.interface,
            #  top=True,
            #  bot=False,
            bot=True,
            top=False,
            symmetrize=False,
            passivated_struc='./test/Al-InAs111A/CONTCAR'
        )
        coords = pas.frac_coords
        shift_val = (coords[:,-1].max() - coords[:,-1].min()) / 2
        pas.translate_sites(
            indices=range(len(pas)),
            vector=[0,0,shift_val],
            frac_coords=True,
            to_unit_cell=True,
        )
        Poscar(pas).write_file(f'./test/Al-InAs111A/larger/POSCAR_{i}')
        #  Poscar(pas).write_file(f'./test/Al-InAs111B/POSCAR_{i}')
        #  ranking_score = interface.get_ranking_score(
            #  radius_dict=r,
        #  )
        #  print(ranking_score)

            
