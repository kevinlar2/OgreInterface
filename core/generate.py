"""
This module will be used to construct the surfaces and interfaces used in this package.
"""
from pymatgen.core.structure import Structure 
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
        [self.film_sl_vecs,
        self.sub_sl_vecs,
        self.match_area,
        self.film_vecs,
        self.sub_vecs,
        self.film_transformations,
        self.substrate_transformations] = self._generate_interfaces()
        self._film_norms = self._get_norm(self.film_sl_vecs, ein='ijk,ijk->ij')
        self._sub_norms = self._get_norm(self.sub_sl_vecs, ein='ijk,ijk->ij')
        self.strain = self._get_strain()
        self.angle_diff = self._get_angle_diff()
        self.area_diff = self._get_area_diff()
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


    def _generate_interfaces(self):
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



if __name__ == "__main__":
    subs = SurfaceGenerator.from_file(
        './POSCAR_InAs_conv',
        miller_index=[1,1,1],
        layers=5,
        vacuum=5,
    )
    films = SurfaceGenerator.from_file(
        './POSCAR_Al_conv',
        miller_index=[1,1,1],
        layers=5,
        vacuum=5,
    )
    inter = InterfaceGenerator(
        substrate=subs.slabs[0],
        film=films.slabs[0],
        length_tol=0.01,
        angle_tol=0.01,
        area_tol=0.01,
        max_area=500,
    )

    new = Interface(
        substrate=subs.slabs[0],
        film=films.slabs[0],
        film_transformation=inter.film_transformations[-1],
        substrate_transformation=inter.substrate_transformations[-1],
        #  film_sl_vecs=inter.film_sl_vecs[-1],
        #  sub_sl_vecs=inter.sub_sl_vecs[-1],
        strain=inter.strain[-1],
        angle_diff=inter.angle_diff[-1],
        sub_strain_frac=0.5,
        interfacial_distance=3
    )

    Poscar(new._stack_interface()).write_file('POSCAR_int')

    Poscar(new._strain_and_orient_sub()).write_file('POSCAR_sub_sc')
    Poscar(new._strain_and_orient_film()).write_file('POSCAR_film_sc')

    subs = new.substrate_supercell.lattice.matrix[:,:2]
    film = new.film_supercell.lattice.matrix[:,:2]
    new_vecs = new.interface_sl_vectors[:,:2]


    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot(
        [0,subs[0,0]],
        [0,subs[0,1]],
        color='red',
        linewidth=8,
        zorder=0,
        label='S',
    )
    ax.plot(
        [0,subs[1,0]],
        [0,subs[1,1]],
        color='red',
        linewidth=8,
        zorder=0,
        label='S',
    )
    ax.plot(
        [0,film[0,0]],
        [0,film[0,1]],
        color='blue',
        linewidth=4,
        zorder=1,
        label='F',
    )
    ax.plot(
        [0,film[1,0]],
        [0,film[1,1]],
        color='blue',
        linewidth=4,
        zorder=1,
        label='F',
    )
    ax.plot(
        [0,new_vecs[0,0]],
        [0,new_vecs[0,1]],
        color='green',
        linewidth=2,
        zorder=2,
        label='I',
    )
    ax.plot(
        [0,new_vecs[1,0]],
        [0,new_vecs[1,1]],
        color='green',
        linewidth=2,
        zorder=2,
        label='I',
    )
    plt.legend()
    plt.tight_layout()
    #  plt.show()
    #  print(new._align_film_a_to_sub_a())
    #  slab = subs.slabs[0].slab_pmg
#
    #  def remove_vacuum(slab):
        #  struc = copy.deepcopy(slab)
        #  bot, _ = get_slab_regions(struc)[0]
        #  struc.translate_sites(range(len(struc)), [0,0,-bot])
        #  frac_coords = struc.frac_coords
        #  cart_coords = struc.cart_coords
        #  max_z = np.max(frac_coords[:,-1])
        #  _, top_new = get_slab_regions(struc)[0]
        #  matrix = copy.deepcopy(struc.lattice.matrix)
        #  matrix[2,2] = top_new * matrix[2,2]
        #  new_lattice = Lattice(matrix)
        #  struc.lattice = new_lattice
#
        #  for i in range(len(struc)):
            #  struc.sites[i].frac_coords[-1] = struc.sites[i].frac_coords[-1] / max_z
            #  struc.sites[i].coords[-1] = cart_coords[i,-1]
#
        #  return struc

    #  Poscar(new.interface).write_file('POSCAR_int')
