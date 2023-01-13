"""
This module will be used to construct the surfaces and interfaces used in this package.
"""
from OgreInterface.surfaces import Surface, Interface
from OgreInterface.utils import (
    get_reduced_basis,
    reduce_vectors_zur_and_mcgill,
    get_primitive_structure,
    conv_a_to_b,
)
from OgreInterface.lattice_match import ZurMcGill

from pymatgen.core.structure import Structure
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core.lattice import Lattice
from pymatgen.io.vasp.inputs import Poscar
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.interfaces.zsl import ZSLGenerator
from pymatgen.core.operations import SymmOp

from tqdm import tqdm
import numpy as np
import math
from copy import deepcopy
from typing import Union, List
from itertools import combinations, product, groupby
from ase import Atoms

from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform


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
        convert_to_conventional: bool = True,
        generate_all: bool = True,
        filter_ionic_slabs: bool = False,
        lazy: bool = False,
    ):
        self.convert_to_conventional = convert_to_conventional
        self.bulk_structure, self.bulk_atoms = self._get_bulk(
            atoms_or_struc=bulk
        )

        self.miller_index = miller_index
        self.layers = layers
        self.vacuum = vacuum
        self.generate_all = generate_all
        self.filter_ionic_slabs = filter_ionic_slabs
        self.lazy = lazy
        (
            self.oriented_bulk_structure,
            self.oriented_bulk_atoms,
            self.uvw_basis,
            self.inplane_vectors,
        ) = self._get_oriented_bulk_structure()

        if not self.lazy:
            self.slabs = self._generate_slabs()

    def generate_slabs(self):
        if self.lazy:
            self.slabs = self._generate_slabs()
        else:
            print(
                "The slabs are already generated upon initialization. This function is only needed if lazy=True"
            )

    @classmethod
    def from_file(
        cls,
        filename,
        miller_index: List[int],
        layers: int,
        vacuum: float,
        convert_to_conventional: bool = True,
        generate_all: bool = True,
        filter_ionic_slabs: bool = False,
        lazy: bool = False,
    ):
        structure = Structure.from_file(filename=filename)

        return cls(
            structure,
            miller_index,
            layers,
            vacuum,
            convert_to_conventional,
            generate_all,
            filter_ionic_slabs,
            lazy,
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

        if self.convert_to_conventional:
            sg = SpacegroupAnalyzer(init_structure)
            conventional_structure = sg.get_conventional_standard_structure()
            conventional_atoms = AseAtomsAdaptor.get_atoms(
                conventional_structure
            )

            return conventional_structure, conventional_atoms
        else:
            init_atoms = AseAtomsAdaptor().get_atoms(init_structure)

            return init_structure, init_atoms

    def _get_oriented_bulk_structure(self):
        bulk = self.bulk_structure
        lattice = bulk.lattice
        recip_lattice = bulk.lattice.reciprocal_lattice_crystallographic
        miller_index = self.miller_index

        sg = SpacegroupAnalyzer(bulk)
        bulk.add_site_property(
            "bulk_wyckoff", sg.get_symmetry_dataset()["wyckoffs"]
        )
        bulk.add_site_property(
            "bulk_equivalent",
            sg.get_symmetry_dataset()["equivalent_atoms"].tolist(),
        )

        intercepts = np.array([1 / i if i != 0 else 0 for i in miller_index])
        non_zero_points = np.where(intercepts != 0)[0]

        d_hkl = lattice.d_hkl(miller_index)
        normal_vector = recip_lattice.get_cartesian_coords(miller_index)
        normal_vector /= np.linalg.norm(normal_vector)

        if len(non_zero_points) == 1:
            basis = np.eye(3)
            dot_products = basis.dot(normal_vector)
            sort_inds = np.argsort(dot_products)
            basis = basis[sort_inds]

            if np.linalg.det(basis) < 0:
                basis = basis[[1, 0, 2]]

            basis = basis

        if len(non_zero_points) == 2:
            points = intercepts * np.eye(3)
            vec1 = points[non_zero_points[1]] - points[non_zero_points[0]]
            vec2 = np.eye(3)[intercepts == 0]

            basis = np.vstack([vec1, vec2])

        if len(non_zero_points) == 3:
            points = intercepts * np.eye(3)
            possible_vecs = []
            for center_inds in [[0, 1, 2], [1, 0, 2], [2, 0, 1]]:
                vec1 = (
                    points[non_zero_points[center_inds[1]]]
                    - points[non_zero_points[center_inds[0]]]
                )
                vec2 = (
                    points[non_zero_points[center_inds[2]]]
                    - points[non_zero_points[center_inds[0]]]
                )
                cart_vec1 = lattice.get_cartesian_coords(vec1)
                cart_vec2 = lattice.get_cartesian_coords(vec2)
                angle = np.arccos(
                    np.dot(cart_vec1, cart_vec2)
                    / (np.linalg.norm(cart_vec1) * np.linalg.norm(cart_vec2))
                )
                possible_vecs.append((vec1, vec2, angle))

            chosen_vec1, chosen_vec2, angle = min(
                possible_vecs, key=lambda x: x[-1]
            )

            basis = np.vstack([chosen_vec1, chosen_vec2])

        basis = get_reduced_basis(basis)

        if len(basis) == 2:
            max_normal_search = 2

            index_range = sorted(
                reversed(range(-max_normal_search, max_normal_search + 1)),
                key=lambda x: abs(x),
            )
            candidates = []
            for uvw in product(index_range, index_range, index_range):
                if (not any(uvw)) or abs(
                    np.linalg.det(np.vstack([basis, uvw]))
                ) < 1e-8:
                    continue

                vec = lattice.get_cartesian_coords(uvw)
                proj = np.abs(np.dot(vec, normal_vector) - d_hkl)
                vec_length = np.linalg.norm(vec)
                cosine = np.dot(vec / vec_length, normal_vector)
                candidates.append((uvw, cosine, vec_length, proj))
                if abs(abs(cosine) - 1) < 1e-8:
                    # If cosine of 1 is found, no need to search further.
                    break
            # We want the indices with the maximum absolute cosine,
            # but smallest possible length.
            uvw, cosine, l, diff = max(
                candidates, key=lambda x: (-x[3], x[1], -x[2])
            )
            basis = np.vstack([basis, uvw])

        init_oriented_struc = bulk.copy()
        init_oriented_struc.make_supercell(basis)

        primitive_oriented_struc = get_primitive_structure(
            init_oriented_struc,
            constrain_latt={
                "c": init_oriented_struc.lattice.c,
                "alpha": init_oriented_struc.lattice.alpha,
                "beta": init_oriented_struc.lattice.beta,
            },
        )

        primitive_transformation = conv_a_to_b(
            init_oriented_struc, primitive_oriented_struc
        )

        primitive_basis = primitive_transformation.dot(basis)

        cart_basis = primitive_oriented_struc.lattice.matrix

        if np.linalg.det(cart_basis) < 0:
            ab_switch = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
            primitive_oriented_struc.make_supercell(ab_switch)
            primitive_basis = ab_switch.dot(primitive_basis)
            cart_basis = primitive_oriented_struc.lattice.matrix

        cross_ab = np.cross(cart_basis[0], cart_basis[1])
        cross_ab /= np.linalg.norm(cross_ab)
        cross_ac = np.cross(cart_basis[0], cross_ab)
        cross_ac /= np.linalg.norm(cross_ac)

        ortho_basis = np.vstack(
            [
                cart_basis[0] / np.linalg.norm(cart_basis[0]),
                cross_ac,
                cross_ab,
            ]
        )

        to_planar_operation = SymmOp.from_rotation_and_translation(
            ortho_basis, translation_vec=np.zeros(3)
        )

        planar_oriented_struc = primitive_oriented_struc.copy()
        planar_oriented_struc.apply_operation(to_planar_operation)

        planar_matrix = deepcopy(planar_oriented_struc.lattice.matrix)

        new_a, new_b, mat = reduce_vectors_zur_and_mcgill(
            planar_matrix[0, :2], planar_matrix[1, :2]
        )

        planar_oriented_struc.make_supercell(mat)

        a_norm = (
            planar_oriented_struc.lattice.matrix[0]
            / planar_oriented_struc.lattice.a
        )
        a_to_i = np.array(
            [[a_norm[0], -a_norm[1], 0], [a_norm[1], a_norm[0], 0], [0, 0, 1]]
        )

        a_to_i_operation = SymmOp.from_rotation_and_translation(
            a_to_i.T, translation_vec=np.zeros(3)
        )
        planar_oriented_struc.apply_operation(a_to_i_operation)
        planar_oriented_struc.sort()

        planar_oriented_atoms = AseAtomsAdaptor().get_atoms(
            planar_oriented_struc
        )

        final_matrix = deepcopy(planar_oriented_struc.lattice.matrix)

        final_basis = mat.dot(primitive_basis)

        oriented_primitive_struc = primitive_oriented_struc.copy()
        oriented_primitive_struc.make_supercell(mat)
        oriented_primitive_struc.sort()

        final_basis = get_reduced_basis(final_basis).astype(int)
        inplane_vectors = final_matrix[:2]

        return (
            planar_oriented_struc,
            planar_oriented_atoms,
            final_basis,
            inplane_vectors,
        )

    def _calculate_possible_shifts(self, tol: float = 0.1):
        frac_coords = self.oriented_bulk_structure.frac_coords
        n = len(frac_coords)

        if n == 1:
            # Clustering does not work when there is only one data point.
            shift = frac_coords[0][2] + 0.5
            return [shift - math.floor(shift)]

        # We cluster the sites according to the c coordinates. But we need to
        # take into account PBC. Let's compute a fractional c-coordinate
        # distance matrix that accounts for PBC.
        dist_matrix = np.zeros((n, n))
        h = self.oriented_bulk_structure.lattice.matrix[-1, -1]
        # Projection of c lattice vector in
        # direction of surface normal.
        for i, j in combinations(list(range(n)), 2):
            if i != j:
                cdist = frac_coords[i][2] - frac_coords[j][2]
                cdist = abs(cdist - round(cdist)) * h
                dist_matrix[i, j] = cdist
                dist_matrix[j, i] = cdist

        condensed_m = squareform(dist_matrix)
        z = linkage(condensed_m)
        clusters = fcluster(z, tol, criterion="distance")

        # Generate dict of cluster# to c val - doesn't matter what the c is.
        c_loc = {c: frac_coords[i][2] for i, c in enumerate(clusters)}

        # Put all c into the unit cell.
        possible_c = [c - math.floor(c) for c in sorted(c_loc.values())]

        # Calculate the shifts
        nshifts = len(possible_c)
        shifts = []
        for i in range(nshifts):
            if i == nshifts - 1:
                # There is an additional shift between the first and last c
                # coordinate. But this needs special handling because of PBC.
                shift = (possible_c[0] + 1 + possible_c[i]) * 0.5
                if shift > 1:
                    shift -= 1
            else:
                shift = (possible_c[i] + possible_c[i + 1]) * 0.5
            shifts.append(shift - math.floor(shift))
        shifts = sorted(shifts)

        return shifts

    def get_slab(self, shift=0, tol: float = 0.1, energy=None):
        """
        This method takes in shift value for the c lattice direction and
        generates a slab based on the given shift. You should rarely use this
        method. Instead, it is used by other generation algorithms to obtain
        all slabs.

        Arg:
            shift (float): A shift value in Angstrom that determines how much a
                slab should be shifted.
            tol (float): Tolerance to determine primitive cell.
            energy (float): An energy to assign to the slab.

        Returns:
            (Slab) A Slab object with a particular shifted oriented unit cell.
        """
        slab_base = self.oriented_bulk_structure.copy()
        slab_base.translate_sites(
            indices=range(len(slab_base)),
            vector=[0, 0, -shift],
            frac_coords=True,
            to_unit_cell=True,
        )
        slab_base.make_supercell([1, 1, self.layers])
        slab_base_c = slab_base.lattice.c

        vacuum_matrix = deepcopy(slab_base.lattice.matrix)
        c_norm = vacuum_matrix[-1] / np.linalg.norm(vacuum_matrix[-1])
        vacuum_scale = self.vacuum / c_norm[-1]
        vacuum_matrix[-1] += vacuum_scale * c_norm

        non_orthogonal_slab = Structure(
            lattice=Lattice(matrix=vacuum_matrix),
            species=slab_base.species,
            coords=slab_base.cart_coords,
            coords_are_cartesian=True,
            to_unit_cell=True,
            site_properties=slab_base.site_properties,
        )
        non_orthogonal_slab.sort()
        non_orthogonal_min_atom = non_orthogonal_slab.frac_coords[
            np.argmin(non_orthogonal_slab.frac_coords[:, -1])
        ]
        non_orthogonal_slab.translate_sites(
            indices=range(len(non_orthogonal_slab)),
            vector=[
                non_orthogonal_min_atom[0],
                non_orthogonal_min_atom[1],
                0.5 - (0.5 * slab_base_c / (vacuum_scale + slab_base_c)),
            ],
            frac_coords=True,
            to_unit_cell=True,
        )

        a, b, c = non_orthogonal_slab.lattice.matrix
        new_c = np.array([0.0, 0.0, 1.0])
        new_c = np.dot(c, new_c) * new_c

        orthogonal_matrix = np.vstack([a, b, new_c])
        orthogonal_slab = Structure(
            lattice=Lattice(matrix=orthogonal_matrix),
            species=non_orthogonal_slab.species,
            coords=non_orthogonal_slab.cart_coords,
            coords_are_cartesian=True,
            to_unit_cell=True,
            site_properties=non_orthogonal_slab.site_properties,
        )
        orthogonal_slab.sort()
        orthogonal_min_atom = orthogonal_slab.frac_coords[
            np.argmin(orthogonal_slab.frac_coords[:, -1])
        ]
        orthogonal_min_atom[-1] = 0
        orthogonal_slab.translate_sites(
            indices=range(len(orthogonal_slab)),
            vector=-orthogonal_min_atom,
            frac_coords=True,
            to_unit_cell=True,
        )

        return orthogonal_slab, non_orthogonal_slab

    # def _get_ewald_energy(self, slab):
    #     slab = deepcopy(slab)
    #     bulk = deepcopy(self.pmg_structure)
    #     slab.add_oxidation_state_by_guess()
    #     bulk.add_oxidation_state_by_guess()
    #     E_slab = EwaldSummation(slab).total_energy
    #     E_bulk = EwaldSummation(bulk).total_energy
    #     return E_slab, E_bulk

    def _generate_slabs(self):
        """
        This function is used to generate slab structures with all unique
        surface terminations.

        Returns:
            A list of Surface classes
        """
        # Determine if all possible terminations are generated
        possible_shifts = self._calculate_possible_shifts()
        orthogonal_slabs = []
        non_orthogonal_slabs = []
        if not self.generate_all:
            orthogonal_slab, non_orthogonal_slab = self.get_slab(
                shift=possible_shifts[0]
            )
            orthogonal_slab.sort_index = 0
            non_orthogonal_slab.sort_index = 0
            orthogonal_slabs.append(orthogonal_slab)
            non_orthogonal_slabs.append(non_orthogonal_slab)
        else:
            for i, possible_shift in enumerate(possible_shifts):
                orthogonal_slab, non_orthogonal_slab = self.get_slab(
                    shift=possible_shift
                )
                orthogonal_slab.sort_index = i
                non_orthogonal_slab.sort_index = i
                orthogonal_slabs.append(orthogonal_slab)
                non_orthogonal_slabs.append(non_orthogonal_slab)

        # TODO work on StructureMatcher when there is an inversion symmetry
        # m = StructureMatcher(
        #     ltol=0.1, stol=0.1, primitive_cell=False, scale=False
        # )

        # unique_orthogonal_slabs = []
        # for g in m.group_structures(orthogonal_slabs):
        #     unique_orthogonal_slabs.append(g[0])

        # match = StructureMatcher(
        #     ltol=0.1, stol=0.1, primitive_cell=False, scale=False
        # )
        # unique_orthogonal_slabs = [
        #     g[0] for g in match.group_structures(unique_orthogonal_slabs)
        # ]
        # unique_non_orthogonal_slabs = [
        #     non_orthogonal_slabs[slab.sort_index]
        #     for slab in unique_orthogonal_slabs
        # ]

        surfaces = []

        # Loop through slabs to ensure that they are all properly oriented and reduced
        # Return Surface objects
        for i, slab in enumerate(orthogonal_slabs):
            # Create the Surface object
            surface = Surface(
                orthogonal_slab=slab,
                non_orthogonal_slab=non_orthogonal_slabs[i],
                primitive_oriented_bulk=self.oriented_bulk_atoms,
                conventional_bulk=self.bulk_structure,
                miller_index=self.miller_index,
                layers=self.layers,
                vacuum=self.vacuum,
                uvw_basis=self.uvw_basis,
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
        return {i: slab.get_termination() for i, slab in enumerate(self.slabs)}


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
        self.match_list = self._generate_interface_props()

    def _generate_interface_props(self):
        zm = ZurMcGill(
            film_vectors=self.film.inplane_vectors,
            substrate_vectors=self.substrate.inplane_vectors,
            film_basis=self.film.uvw_basis,
            substrate_basis=self.substrate.uvw_basis,
            max_area=self.max_area,
            max_linear_strain=self.length_tol,
            max_angle_strain=self.angle_tol,
            max_area_mismatch=self.area_tol,
        )
        match_list = zm.run(return_all=True)

        if len(match_list) == 0:
            raise TolarenceError(
                "No interfaces were found, please increase the tolarences."
            )
        else:
            test_return = []
            for match in match_list:
                test_return.extend(match)

            return test_return
            # group_list = []
            # for i, match in enumerate(match_list):
            #     groups = groupby(
            #         match,
            #         key=lambda x: (
            #             round(x.substrate_a_norm, 3),
            #             round(x.substrate_b_norm, 3),
            #             round(x.substrate_angle, 3),
            #         ),
            #     )
            #     for group in groups:
            #         group_list.append(group[1])

            # return group_list

    def _is_equal(self, structure1, structure2):
        structure_matcher = StructureMatcher(
            ltol=0.01,
            stol=0.01,
            angle_tol=0.01,
            primitive_cell=False,
            scale=False,
        )
        is_fit = structure_matcher.fit(structure1, structure2)
        # match = structure_matcher._match(structure1, structure2, 1)
        # if match is None:
        #     is_fit = False
        # else:
        #     is_fit = match[0] <= 0.001

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

    def generate_interfaces(self):
        interfaces = []
        print("Generating Interfaces:")
        for group_matches in tqdm(self.match_list):
            for match in group_matches:
                interface = Interface(
                    substrate=self.substrate,
                    film=self.film,
                    interfacial_distance=self.interfacial_distance,
                    match=match,
                    vacuum=self.vacuum,
                    center=self.center,
                )
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
            area = interface.match.area
            areas.append(area)

        sort = np.argsort(areas)
        sorted_unique_interfaces = [unique_interfaces[i] for i in sort]

        return sorted_unique_interfaces
