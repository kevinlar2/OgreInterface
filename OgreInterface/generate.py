"""
This module will be used to construct the surfaces and interfaces used in this package.
"""
from OgreInterface.surfaces import Surface, Interface

from pymatgen.core.structure import Structure
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core.lattice import Lattice
from pymatgen.core.surface import SlabGenerator
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.interfaces.zsl import ZSLGenerator, reduce_vectors
from pymatgen.core.operations import SymmOp
from pymatgen.analysis.ewald import EwaldSummation

from tqdm import tqdm
import numpy as np
from copy import deepcopy
from functools import reduce
from typing import Union, List, Optional
from itertools import combinations
from ase import Atoms


class TolarenceError(RuntimeError):
    pass


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
        bulk: Union[Structure, Atoms],
        miller_index: List[int],
        layers: int,
        vacuum: float,
        generate_all: bool = True,
        filter_ionic_slabs: bool = False,
    ):
        self.bulk_structure, self.bulk_atoms = self._get_bulk(
            atoms_or_struc=bulk
        )

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

        return cls(
            structure,
            miller_index,
            layers,
            vacuum,
            generate_all,
            filter_ionic_slabs,
        )

    def _get_bulk(self, atoms_or_struc):
        if type(atoms_or_struc) == Atoms:
            init_structure = AseAtomsAdaptor.get_structure(atoms_or_struc)
        elif type(atoms_or_struc) == Structure:
            init_structure = atoms_or_struc
        else:
            raise TypeError(
                f"structure accepts 'pymatgen.core.structure.Structure' or 'ase.Atoms' not '{type(atoms_or_struc).__name__}'"
            )

        sg = SpacegroupAnalyzer(init_structure)
        conventional_structure = sg.get_conventional_standard_structure()
        conventional_atoms = AseAtomsAdaptor.get_atoms(conventional_structure)

        return conventional_structure, conventional_atoms

    # def _get_ewald_energy(self, slab):
    #     slab = deepcopy(slab)
    #     bulk = deepcopy(self.pmg_structure)
    #     slab.add_oxidation_state_by_guess()
    #     bulk.add_oxidation_state_by_guess()
    #     E_slab = EwaldSummation(slab).total_energy
    #     E_bulk = EwaldSummation(bulk).total_energy
    #     return E_slab, E_bulk

    def _float_gcd(self, a, b, rtol=1e-05, atol=1e-08):
        t = min(abs(a), abs(b))
        while abs(b) > rtol * t + atol:
            a, b = b, a % b
        return a

    def _check_oriented_cell(
        self, slab_generator: SlabGenerator, miller_index: np.ndarray
    ):
        """
        This function is used to ensure that the c-vector of the oriented bulk
        unit cell in the SlabGenerator matches with the given miller index.
        This is required to properly determine the in-plane lattice vectors for
        the epitaxial match.

        Parameters:
            slab_generator (SlabGenerator): SlabGenerator object from PyMatGen
            miller_index (np.ndarray): Miller index of the plane

        Returns:
            SlabGenerator with proper orientation of c-vector
        """
        if np.isclose(
            slab_generator.slab_scale_factor[-1],
            -miller_index / np.min(np.abs(miller_index[miller_index != 0])),
        ).all():
            slab_generator.slab_scale_factor *= -1
            single = self.bulk_structure.copy()
            single.make_supercell(slab_generator.slab_scale_factor)
            slab_generator.oriented_unit_cell = Structure.from_sites(
                single, to_unit_cell=True
            )

        return slab_generator

    def _get_reduced_basis(self, basis: np.ndarray):
        """
        This function is used to find the miller indices of the slab structure
        basis vectors in their most reduced form. i.e.

        |  2  4  0 |     | 1  2  0 |
        |  0 -2  4 | ==> | 0 -1  2 |
        | 10 10 10 |     | 1  1  1 |

        Parameters:
            basis (np.ndarray): 3x3 matrix defining the lattice vectors

        Returns:
            Reduced integer basis in the form of miller indices
        """
        basis /= np.linalg.norm(basis, axis=1)[:, None]

        for i, b in enumerate(basis):
            abs_b = np.abs(b)
            basis[i] /= abs_b[abs_b > 0.001].min()
            basis[i] /= np.abs(reduce(self._float_gcd, basis[i]))

        return basis

    def _get_properly_oriented_slab(
        self, basis: np.ndarray, miller_index: np.ndarray, slab: Structure
    ):
        """
        This function is used to flip the structure if the c-vector and miller
        index are negatives of each other. This happens during the process of
        making the primitive slab. To resolve this, the structure will be
        rotated 180 degrees.

        Parameters:
            basis (np.ndarray): 3x3 matrix defining the lattice vectors
            miller_index (np.ndarray): Miller index of surface
            slab (Structure): PyMatGen Structure object

        Return:
            Properly oriented slab
        """
        if (
            basis[-1]
            == -miller_index / np.min(np.abs(miller_index[miller_index != 0]))
        ).all():
            operation = SymmOp.from_origin_axis_angle(
                origin=[0.5, 0.5, 0.5],
                axis=[1, 1, 0],
                angle=180,
            )
            slab.apply_operation(operation, fractional=True)

        return slab

    def _generate_slabs(self):
        """
        This function is used to generate slab structures with all unique
        surface terminations.

        Returns:
            A list of Surface classes
        """
        # Initialize the SlabGenerator
        sg = SlabGenerator(
            initial_structure=self.bulk_structure,
            miller_index=self.miller_index,
            min_slab_size=self.layers,
            min_vacuum_size=self.vacuum,
            in_unit_planes=True,
            primitive=True,
            lll_reduce=False,
            reorient_lattice=False,
            max_normal_search=int(max(np.abs(self.miller_index))),
            center_slab=True,
        )
        # Convert miller index to a numpy array
        miller_index = np.array(self.miller_index)

        # Check if the oriented cell has the proper basis
        sg = self._check_oriented_cell(
            slab_generator=sg, miller_index=miller_index
        )

        # Determine if all possible terminations are generated
        if self.generate_all:
            slabs = sg.get_slabs(tol=0.25)
        else:
            possible_shifts = sg._calculate_possible_shifts()
            slabs = [sg.get_slab(shift=possible_shifts[0])]

        surfaces = []

        # Loop through slabs to ensure that they are all properly oriented and reduced
        # Return Surface objects
        for slab in slabs:
            # Get the inital miller-indices of the lattice
            basis = self._get_reduced_basis(
                basis=deepcopy(slab.lattice.matrix)
            )

            # Ensure that the slab is properly oriented w.r.t the given surface normal
            slab = self._get_properly_oriented_slab(
                basis=basis, miller_index=miller_index, slab=slab
            )

            # Reduce the vectors according to the Zur and McGill algorithm
            new_a, new_b = reduce_vectors(
                slab.lattice.matrix[0], slab.lattice.matrix[1]
            )

            # This is the lattice of the reduced surface
            reduced_matrix = np.hstack([new_a, new_b, slab.lattice.matrix[-1]])

            # Create the pymatgen structure of the reduced surface
            reduced_struc = Structure(
                lattice=Lattice(matrix=reduced_matrix),
                species=slab.species,
                coords=slab.cart_coords,
                to_unit_cell=True,
                coords_are_cartesian=True,
                site_properties=slab.site_properties,
            )
            reduced_struc.sort()

            # Get the final reduced miller-indices of the lattice
            reduced_basis = self._get_reduced_basis(
                basis=deepcopy(reduced_struc.lattice.matrix)
            )

            # Create the Surface object
            surface = Surface(
                slab=reduced_struc,
                bulk=self.bulk_structure,
                miller_index=self.miller_index,
                layers=self.layers,
                vacuum=self.vacuum,
                uvw_basis=np.round(reduced_basis).astype(int),
            )
            surfaces.append(surface)

        return surfaces

    def __len__(self):
        return len(self.slabs)

    @property
    def nslabs(self):
        """
        Return the number of slabs generated by the SurfaceGenerator
        """
        return self.__len__()

    @property
    def terminations(self):
        """
        Return the terminations of each slab generated by the SurfaceGenerator
        """
        raise NotImplementedError


class InterfaceGenerator:
    """
    This class will use the lattice matching algorithm from Zur and McGill to generate
    commensurate interface structures between two inorganic crystalline materials.
    """

    def __init__(
        self,
        substrate: Surface,
        film: Surface,
        area_tol: float = 0.01,
        angle_tol: float = 0.01,
        length_tol: float = 0.01,
        max_area: float = 500.0,
        interfacial_distance: float = 2.0,
        sub_strain_frac: float = 0.0,
        vacuum: float = 40.0,
        center: bool = False,
    ):
        if type(substrate) == Surface:
            self.substrate = substrate
        else:
            raise TypeError(
                f"InterfaceGenerator accepts 'ogre.core.Surface' not '{type(substrate).__name__}'"
            )

        if type(film) == Surface:
            self.film = film
        else:
            raise TypeError(
                f"InterfaceGenerator accepts 'ogre.core.Surface' not '{type(film).__name__}'"
            )

        self.center = center
        self.area_tol = area_tol
        self.angle_tol = angle_tol
        self.length_tol = length_tol
        self.max_area = max_area
        self.interfacial_distance = interfacial_distance
        self.sub_strain_frac = sub_strain_frac
        self.vacuum = vacuum
        # try:
        (
            self.film_sl_vecs,
            self.sub_sl_vecs,
            self.match_area,
            self.film_vecs,
            self.sub_vecs,
            self.film_transformations,
            self.substrate_transformations,
            self.stack_transformations,
        ) = self._generate_interface_props()
        # except TolarenceError:
        # print("No interfaces were found, please increase the tolarences.")

        self._film_norms = self._get_norm(self.film_sl_vecs, ein="ijk,ijk->ij")
        self._sub_norms = self._get_norm(self.sub_sl_vecs, ein="ijk,ijk->ij")
        self.strain = self._get_strain()
        self.angle_diff = self._get_angle_diff()
        self.area_diff = self._get_area_diff()
        self.area_ratio = self._get_area_ratios()
        self.substrate_areas = self._get_area(
            self.sub_sl_vecs[:, 0], self.sub_sl_vecs[:, 1]
        )
        self.rotation_mat = self._get_rotation_mat()

    def _get_norm(self, a, ein):
        a_norm = np.sqrt(np.einsum(ein, a, a))

        return a_norm

    def _get_angle(self, a, b):
        ein = "ij,ij->i"
        a_norm = self._get_norm(a, ein=ein)
        b_norm = self._get_norm(b, ein=ein)
        dot_prod = np.einsum("ij,ij->i", a, b)
        angles = np.arccos(dot_prod / (a_norm * b_norm))

        return angles

    def _get_area(self, a, b):
        cross_prod = np.cross(a, b)
        area = self._get_norm(cross_prod, ein="ij,ij->i")

        return area

    def _get_strain(self):
        a_strain = (self._film_norms[:, 0] / self._sub_norms[:, 0]) - 1
        b_strain = (self._film_norms[:, 1] / self._sub_norms[:, 1]) - 1

        return np.c_[a_strain, b_strain]

    def _get_angle_diff(self):
        sub_angles = self._get_angle(
            self.sub_sl_vecs[:, 0], self.sub_sl_vecs[:, 1]
        )
        film_angles = self._get_angle(
            self.film_sl_vecs[:, 0], self.film_sl_vecs[:, 1]
        )
        angle_diff = (film_angles / sub_angles) - 1

        return angle_diff

    def _get_area_diff(self):
        sub_areas = self._get_area(
            self.sub_sl_vecs[:, 0], self.sub_sl_vecs[:, 1]
        )
        film_areas = self._get_area(
            self.film_sl_vecs[:, 0], self.film_sl_vecs[:, 1]
        )
        area_diff = (film_areas / sub_areas) - 1

        return area_diff

    def _get_area_ratios(self):
        q = (
            self.film_transformations[:, 0, 0]
            * self.film_transformations[:, 1, 1]
        )
        p = (
            self.substrate_transformations[:, 0, 0]
            * self.substrate_transformations[:, 1, 1]
        )
        area_ratio = np.abs((p / q) - (self.film.area / self.substrate.area))

        return area_ratio

    def _get_rotation_mat(self):
        dot_prod = np.divide(
            np.einsum(
                "ij,ij->i", self.sub_sl_vecs[:, 0], self.film_sl_vecs[:, 0]
            ),
            np.multiply(self._sub_norms[:, 0], self._film_norms[:, 0]),
        )

        mag_cross = np.divide(
            self._get_area(self.sub_sl_vecs[:, 0], self.film_sl_vecs[:, 0]),
            np.multiply(self._sub_norms[:, 0], self._film_norms[:, 0]),
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
        ].reshape(-1, 3, 3)

        return rot_mat

    def _generate_interface_props(self):
        zsl = ZSLGenerator(
            max_area_ratio_tol=self.area_tol,
            max_angle_tol=self.angle_tol,
            max_length_tol=self.length_tol,
            max_area=self.max_area,
        )
        film_vectors = self.film.inplane_vectors
        substrate_vectors = self.substrate.inplane_vectors
        matches = zsl(film_vectors, substrate_vectors)
        match_list = list(matches)

        if len(match_list) == 0:
            raise TolarenceError(
                "No interfaces were found, please increase the tolarences."
            )
        else:
            film_sl_vecs = np.array(
                [match.film_sl_vectors for match in match_list]
            )
            sub_sl_vecs = np.array(
                [match.substrate_sl_vectors for match in match_list]
            )
            match_area = np.array([match.match_area for match in match_list])
            film_vecs = np.array([match.film_vectors for match in match_list])
            sub_vecs = np.array(
                [match.substrate_vectors for match in match_list]
            )
            film_transformations = np.array(
                [match.film_transformation for match in match_list]
            )
            substrate_transformations = np.array(
                [match.substrate_transformation for match in match_list]
            )

            film_3x3_transformations = np.array(
                [np.eye(3, 3) for _ in range(film_transformations.shape[0])]
            )
            substrate_3x3_transformations = np.array(
                [
                    np.eye(3, 3)
                    for _ in range(substrate_transformations.shape[0])
                ]
            )
            stack_transforms = np.array(
                [match.match_transformation for match in match_list]
            )

            film_3x3_transformations[:, :2, :2] = film_transformations
            substrate_3x3_transformations[
                :, :2, :2
            ] = substrate_transformations

            return [
                film_sl_vecs,
                sub_sl_vecs,
                match_area,
                film_vecs,
                sub_vecs,
                film_3x3_transformations,
                substrate_3x3_transformations,
                stack_transforms,
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
            is_fit = False
        else:
            is_fit = match[0] <= 0.001

        return is_fit

    def _find_exact_matches(self, structures):
        all_coords = np.array([i.interface.frac_coords for i in structures])
        all_species = np.array([i.interface.species for i in structures])

        for i in range(len(structures)):
            coords = np.round(all_coords[i], 6)
            coords[:, -1] = coords[:, -1] - np.min(coords[:, -1])
            coords.dtype = [
                ("a", "float64"),
                ("b", "float64"),
                ("c", "float64"),
            ]
            coords_inds = np.squeeze(
                np.argsort(coords, axis=0, order=("c", "b", "a"))
            )
            coords.dtype = "float64"

            coords_sorted = coords[coords_inds]
            species_sorted = np.array(all_species[i]).astype(str)[coords_inds]

            all_coords[i] = coords_sorted
            all_species[i] = species_sorted

        equal_coords = np.array(
            [
                np.isclose(all_coords[i], all_coords).all(axis=1).all(axis=1)
                for i in range(all_coords.shape[0])
            ]
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
            coords1[:, -1] = coords1[:, -1] - np.min(coords1[:, -1])
            coords1.dtype = [
                ("a", "float64"),
                ("b", "float64"),
                ("c", "float64"),
            ]
            coords1_inds = np.squeeze(
                np.argsort(coords1, axis=0, order=("c", "b", "a"))
            )
            coords1.dtype = "float64"

            coords2 = np.round(structure2.frac_coords, 4)
            coords2[:, -1] = coords2[:, -1] - np.min(coords2[:, -1])
            coords2.dtype = [
                ("a", "float64"),
                ("b", "float64"),
                ("c", "float64"),
            ]
            coords2_inds = np.squeeze(
                np.argsort(coords2, axis=0, order=("c", "b", "a"))
            )
            coords2.dtype = "float64"

            coords1_sorted = coords1[coords1_inds]
            coords2_sorted = coords2[coords2_inds]
            species1_sorted = np.array(structure1.species).astype(str)[
                coords1_inds
            ]
            species2_sorted = np.array(structure2.species).astype(str)[
                coords2_inds
            ]

            coords = np.isclose(
                coords1_sorted, coords2_sorted, rtol=1e-2, atol=1e-2
            ).all()
            species = (species1_sorted == species2_sorted).all()

            if coords and species:
                return True
            else:
                return False

    def generate_interfaces(self):
        interfaces = []
        print("Generating Interfaces:")
        for i in tqdm(range(self.substrate_transformations.shape[0])):
            interface = Interface(
                substrate=self.substrate,
                film=self.film,
                film_transformation=self.film_transformations[i],
                substrate_transformation=self.substrate_transformations[i],
                stack_transformation=self.stack_transformations[i],
                strain=self.strain[i],
                angle_diff=self.angle_diff[i],
                sub_strain_frac=self.sub_strain_frac,
                interfacial_distance=self.interfacial_distance,
                film_vecs=self.film_vecs[i],
                sub_vecs=self.sub_vecs[i],
                film_sl_vecs=self.film_sl_vecs[i],
                sub_sl_vecs=self.sub_sl_vecs[i],
                vacuum=self.vacuum,
                center=self.center,
            )
            #  interface.shift_film([0.3, 0.6, 0])
            interfaces.append(interface)

        interfaces = np.array(interfaces)

        interface_sizes = np.array(
            [len(interfaces[i].interface) for i in range(len(interfaces))]
        )
        unique_inds = np.array(
            [np.isin(interface_sizes, i) for i in np.unique(interface_sizes)]
        )
        possible_alike_strucs = [
            interfaces[unique_inds[i]] for i in range(unique_inds.shape[0])
        ]

        interfaces = []

        for strucs in possible_alike_strucs:
            inds = self._find_exact_matches(strucs)
            reduced_strucs = strucs[inds]
            interfaces.extend(reduced_strucs)

        combos = combinations(range(len(interfaces)), 2)
        same_slab_indices = []
        print("Finding Symmetrically Equivalent Interfaces:")
        for combo in tqdm(combos):
            if self._is_equal(
                interfaces[combo[0]].interface, interfaces[combo[1]].interface
            ):
                same_slab_indices.append(combo)

        to_delete = [
            np.min(same_slab_index) for same_slab_index in same_slab_indices
        ]
        unique_slab_indices = [
            i for i in range(len(interfaces)) if i not in to_delete
        ]
        unique_interfaces = [interfaces[i] for i in unique_slab_indices]

        areas = []

        for interface in unique_interfaces:
            # for interface in interfaces:
            matrix = interface.interface.lattice.matrix
            area = self._get_area([matrix[0]], [matrix[1]])[0]
            areas.append(area)

        sort = np.argsort(areas)
        sorted_unique_interfaces = [unique_interfaces[i] for i in sort]
        # sorted_unique_interfaces = [interfaces[i] for i in sort]

        return sorted_unique_interfaces
