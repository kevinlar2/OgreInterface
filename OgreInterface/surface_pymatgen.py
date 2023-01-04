# Copyright (c) Pymatgen Development Team.
# Distributed under the terms of the MIT License.

"""
This module implements representations of slabs and surfaces, as well as
algorithms for generating them. If you use this module, please consider
citing the following work::

    R. Tran, Z. Xu, B. Radhakrishnan, D. Winston, W. Sun, K. A. Persson,
    S. P. Ong, "Surface Energies of Elemental Crystals", Scientific Data,
    2016, 3:160080, doi: 10.1038/sdata.2016.80.

as well as::

    Sun, W.; Ceder, G. Efficient creation and convergence of surface slabs,
    Surface Science, 2013, 617, 53-59, doi:10.1016/j.susc.2013.05.016.
"""

import copy
import itertools
import json
import logging
import math
import os
import warnings
from functools import reduce
from math import gcd

import numpy as np
from monty.fractions import lcm
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core.lattice import Lattice
from pymatgen.core.periodic_table import get_el_sp
from pymatgen.core.sites import PeriodicSite
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.util.coord import in_coord_list
from pymatgen.core.surface import Slab
from pymatgen.core.operations import SymmOp
from pymatgen.io.vasp.inputs import Poscar

from OgreInterface.prmitive_pymatgen import get_primitive_structure

__author__ = "Richard Tran, Wenhao Sun, Zihan Xu, Shyue Ping Ong"


logger = logging.getLogger(__name__)


def _float_gcd(a, b, rtol=1e-05, atol=1e-08):
    t = min(abs(a), abs(b))
    while abs(b) > rtol * t + atol:
        a, b = b, a % b
    return a


def convert_float_to_int_vector(vec):
    return np.round(vec / np.abs(reduce(_float_gcd, vec))).astype(int)


def reduce_vectors_zur_and_mcgill(a, b):
    vecs = np.vstack([a, b])
    mat = np.eye(3)
    reduced = False

    while not reduced:
        if np.dot(vecs[0], vecs[1]) < 0:
            vecs[1] *= -1
            mat[1] *= -1
            continue

        if np.linalg.norm(vecs[0]) > np.linalg.norm(vecs[1]):
            vecs = vecs[[1, 0]]
            mat = mat[[1, 0, 2]]
            continue

        if np.linalg.norm(vecs[1]) > np.linalg.norm(vecs[1] + vecs[0]):
            vecs[1] = vecs[1] + vecs[0]
            mat[1] = mat[1] + mat[0]
            continue

        if np.linalg.norm(vecs[1]) > np.linalg.norm(vecs[1] - vecs[0]):
            vecs[1] = vecs[1] - vecs[0]
            mat[1] = mat[1] - mat[0]
            reduced = True
            continue

        reduced = True

    return vecs[0], vecs[1], mat


class SlabGenerator:
    """
    This class generates different slabs using shift values determined by where
    a unique termination can be found along with other criteria such as where a
    termination doesn't break a polyhedral bond. The shift value then indicates
    where the slab layer will begin and terminate in the slab-vacuum system.

    .. attribute:: oriented_unit_cell

        A unit cell of the parent structure with the miller
        index of plane parallel to surface

    .. attribute:: parent

        Parent structure from which Slab was derived.

    .. attribute:: lll_reduce

        Whether or not the slabs will be orthogonalized

    .. attribute:: center_slab

        Whether or not the slabs will be centered between
        the vacuum layer

    .. attribute:: slab_scale_factor

        Final computed scale factor that brings the parent cell to the
        surface cell.

    .. attribute:: miller_index

        Miller index of plane parallel to surface.

    .. attribute:: min_slab_size

        Minimum size in angstroms of layers containing atoms

    .. attribute:: min_vac_size

        Minimize size in angstroms of layers containing vacuum

    """

    def __init__(
        self,
        initial_structure,
        miller_index,
        min_slab_size,
        min_vacuum_size,
        lll_reduce=False,
        center_slab=False,
        in_unit_planes=False,
        primitive=True,
        max_normal_search=None,
        reorient_lattice=True,
    ):
        """
        Calculates the slab scale factor and uses it to generate a unit cell
        of the initial structure that has been oriented by its miller index.
        Also stores the initial information needed later on to generate a slab.

        Args:
            initial_structure (Structure): Initial input structure. Note that to
                ensure that the miller indices correspond to usual
                crystallographic definitions, you should supply a conventional
                unit cell structure.
            miller_index ([h, k, l]): Miller index of plane parallel to
                surface. Note that this is referenced to the input structure. If
                you need this to be based on the conventional cell,
                you should supply the conventional structure.
            min_slab_size (float): In Angstroms or number of hkl planes
            min_vacuum_size (float): In Angstroms or number of hkl planes
            lll_reduce (bool): Whether to perform an LLL reduction on the
                eventual structure.
            center_slab (bool): Whether to center the slab in the cell with
                equal vacuum spacing from the top and bottom.
            in_unit_planes (bool): Whether to set min_slab_size and min_vac_size
                in units of hkl planes (True) or Angstrom (False/default).
                Setting in units of planes is useful for ensuring some slabs
                have a certain nlayer of atoms. e.g. for Cs (100), a 10 Ang
                slab will result in a slab with only 2 layer of atoms, whereas
                Fe (100) will have more layer of atoms. By using units of hkl
                planes instead, we ensure both slabs
                have the same number of atoms. The slab thickness will be in
                min_slab_size/math.ceil(self._proj_height/dhkl)
                multiples of oriented unit cells.
            primitive (bool): Whether to reduce any generated slabs to a
                primitive cell (this does **not** mean the slab is generated
                from a primitive cell, it simply means that after slab
                generation, we attempt to find shorter lattice vectors,
                which lead to less surface area and smaller cells).
            max_normal_search (int): If set to a positive integer, the code will
                conduct a search for a normal lattice vector that is as
                perpendicular to the surface as possible by considering
                multiples linear combinations of lattice vectors up to
                max_normal_search. This has no bearing on surface energies,
                but may be useful as a preliminary step to generating slabs
                for absorption and other sizes. It is typical that this will
                not be the smallest possible cell for simulation. Normality
                is not guaranteed, but the oriented cell will have the c
                vector as normal as possible (within the search range) to the
                surface. A value of up to the max absolute Miller index is
                usually sufficient.
            reorient_lattice (bool): reorients the lattice parameters such that
                the c direction is the third vector of the lattice matrix
        """
        # pylint: disable=E1130
        # Add Wyckoff symbols of the bulk, will help with
        # identfying types of sites in the slab system
        sg = SpacegroupAnalyzer(initial_structure)
        initial_structure.add_site_property(
            "bulk_wyckoff", sg.get_symmetry_dataset()["wyckoffs"]
        )
        initial_structure.add_site_property(
            "bulk_equivalent",
            sg.get_symmetry_dataset()["equivalent_atoms"].tolist(),
        )

        intercepts = np.array([1 / i if i != 0 else 0 for i in miller_index])
        non_zero_points = np.where(intercepts != 0)[0]

        d_hkl = initial_structure.lattice.d_hkl(miller_index)
        recip_lattice = (
            initial_structure.lattice.reciprocal_lattice_crystallographic
        )
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
                cart_vec1 = initial_structure.lattice.get_cartesian_coords(
                    vec1
                )
                cart_vec2 = initial_structure.lattice.get_cartesian_coords(
                    vec2
                )
                angle = np.arccos(
                    np.dot(cart_vec1, cart_vec2)
                    / (np.linalg.norm(cart_vec1) * np.linalg.norm(cart_vec2))
                )
                possible_vecs.append((vec1, vec2, angle))

            chosen_vec1, chosen_vec2, angle = min(
                possible_vecs, key=lambda x: x[-1]
            )

            basis = np.vstack([chosen_vec1, chosen_vec2])

        for i in range(2):
            basis[i] = convert_float_to_int_vector(basis[i])

        if len(basis) == 2:
            max_normal_search = 2

            index_range = sorted(
                reversed(range(-max_normal_search, max_normal_search + 1)),
                key=lambda x: abs(x),
            )
            candidates = []
            for uvw in itertools.product(
                index_range, index_range, index_range
            ):
                if (not any(uvw)) or abs(
                    np.linalg.det(np.vstack([basis, uvw]))
                ) < 1e-8:
                    continue

                vec = initial_structure.lattice.get_cartesian_coords(uvw)
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

        init_oriented_struc = initial_structure.copy()
        init_oriented_struc.make_supercell(basis)

        Poscar(init_oriented_struc).write_file("POSCAR_ouc")

        primitive_oriented_struc = get_primitive_structure(
            init_oriented_struc,
            constrain_latt={
                "c": init_oriented_struc.lattice.c,
                "alpha": init_oriented_struc.lattice.alpha,
                "beta": init_oriented_struc.lattice.beta,
            },
        )

        Poscar(primitive_oriented_struc).write_file("POSCAR_prim")

        primitive_transformation = np.linalg.solve(
            init_oriented_struc.lattice.matrix,
            primitive_oriented_struc.lattice.matrix,
        )

        primitive_basis = basis.dot(primitive_transformation)

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

        if np.sign(np.linalg.det(ortho_basis)) != np.sign(
            np.linalg.det(basis)
        ):
            ortho_basis[1] *= -1

        op = SymmOp.from_rotation_and_translation(
            ortho_basis, translation_vec=np.zeros(3)
        )

        planar_oriented_struc = primitive_oriented_struc.copy()
        planar_oriented_struc.apply_operation(op)

        planar_matrix = copy.deepcopy(planar_oriented_struc.lattice.matrix)

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

        op2 = SymmOp.from_rotation_and_translation(
            a_to_i.T, translation_vec=np.zeros(3)
        )
        planar_oriented_struc.apply_operation(op2)
        final_matrix = copy.deepcopy(planar_oriented_struc.lattice.matrix)

        # a_to_i = np.array([[a_norm[0], -a_norm[1]], [a_norm[1], a_norm[0]]])
        # ai_vecs = orig_vecs.dot(a_to_i)

        final_basis = mat.dot(primitive_basis)

        oriented_primitive_struc = primitive_oriented_struc.copy()
        oriented_primitive_struc.make_supercell(mat)
        oriented_primitive_struc.sort()

        for i, b in enumerate(final_basis):
            final_basis[i] = convert_float_to_int_vector(b)

        # oriented_conventional_struc = struc.copy()
        # oriented_conventional_struc.make_supercell(final_basis)

        # When getting the OUC, lets return the most reduced
        # structure as possible to reduce calculations
        self.inplane_vectors = final_matrix[:2]
        self.basis = final_basis
        self.oriented_unit_cell = Structure.from_sites(
            planar_oriented_struc, to_unit_cell=True
        )
        self.max_normal_search = max_normal_search
        self.parent = initial_structure
        self.lll_reduce = lll_reduce
        self.center_slab = center_slab
        self.slab_scale_factor = final_basis
        self.miller_index = miller_index
        self.min_vac_size = min_vacuum_size
        self.min_slab_size = min_slab_size
        self.in_unit_planes = in_unit_planes
        self.primitive = primitive
        self._normal = normal_vector
        a, b, c = self.oriented_unit_cell.lattice.matrix
        self._proj_height = abs(np.dot(normal_vector, c))
        self.reorient_lattice = reorient_lattice

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
        h = self._proj_height
        p = round(h / self.parent.lattice.d_hkl(self.miller_index), 8)
        if self.in_unit_planes:
            nlayers_slab = int(math.ceil(self.min_slab_size / p))
            nlayers_vac = int(math.ceil(self.min_vac_size / p))
        else:
            nlayers_slab = int(math.ceil(self.min_slab_size / h))
            nlayers_vac = int(math.ceil(self.min_vac_size / h))
        nlayers = nlayers_slab + nlayers_vac

        species = self.oriented_unit_cell.species_and_occu
        props = self.oriented_unit_cell.site_properties
        props = {k: v * nlayers_slab for k, v in props.items()}
        frac_coords = self.oriented_unit_cell.frac_coords
        frac_coords = np.array(frac_coords) + np.array([0, 0, -shift])[None, :]
        frac_coords -= np.floor(frac_coords)
        a, b, c = self.oriented_unit_cell.lattice.matrix
        new_lattice = [a, b, nlayers * c]
        frac_coords[:, 2] = frac_coords[:, 2] / nlayers
        all_coords = []
        for i in range(nlayers_slab):
            fcoords = frac_coords.copy()
            fcoords[:, 2] += i / nlayers
            all_coords.extend(fcoords)

        slab = Structure(
            new_lattice,
            species * nlayers_slab,
            all_coords,
            site_properties=props,
        )

        scale_factor = self.slab_scale_factor
        # Whether or not to orthogonalize the structure
        if self.lll_reduce:
            lll_slab = slab.copy(sanitize=True)
            mapping = lll_slab.lattice.find_mapping(slab.lattice)
            scale_factor = np.dot(mapping[2], scale_factor)
            slab = lll_slab

        # Whether or not to center the slab layer around the vacuum
        if self.center_slab:
            avg_c = np.average([c[2] for c in slab.frac_coords])
            slab.translate_sites(list(range(len(slab))), [0, 0, 0.5 - avg_c])

        if self.primitive:
            prim = slab.get_primitive_structure(tolerance=tol)
            if energy is not None:
                energy = prim.volume / slab.volume * energy
            slab = prim

        # Reorient the lattice to get the correct reduced cell
        ouc = self.oriented_unit_cell.copy()
        if self.primitive:
            # find a reduced ouc
            slab_l = slab.lattice
            ouc = ouc.get_primitive_structure(
                constrain_latt={
                    "a": slab_l.a,
                    "b": slab_l.b,
                    "alpha": slab_l.alpha,
                    "beta": slab_l.beta,
                    "gamma": slab_l.gamma,
                }
            )
            # Check this is the correct oriented unit cell
            ouc = (
                self.oriented_unit_cell
                if slab_l.a != ouc.lattice.a or slab_l.b != ouc.lattice.b
                else ouc
            )

        return Slab(
            slab.lattice,
            slab.species_and_occu,
            slab.frac_coords,
            self.miller_index,
            ouc,
            shift,
            scale_factor,
            energy=energy,
            site_properties=slab.site_properties,
            reorient_lattice=self.reorient_lattice,
        )

    def _calculate_possible_shifts(self, tol: float = 0.1):
        frac_coords = self.oriented_unit_cell.frac_coords
        n = len(frac_coords)

        if n == 1:
            # Clustering does not work when there is only one data point.
            shift = frac_coords[0][2] + 0.5
            return [shift - math.floor(shift)]

        # We cluster the sites according to the c coordinates. But we need to
        # take into account PBC. Let's compute a fractional c-coordinate
        # distance matrix that accounts for PBC.
        dist_matrix = np.zeros((n, n))
        h = self._proj_height
        # Projection of c lattice vector in
        # direction of surface normal.
        for i, j in itertools.combinations(list(range(n)), 2):
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

    def _get_c_ranges(self, bonds):
        c_ranges = []
        bonds = {
            (get_el_sp(s1), get_el_sp(s2)): dist
            for (s1, s2), dist in bonds.items()
        }
        for (sp1, sp2), bond_dist in bonds.items():
            for site in self.oriented_unit_cell:
                if sp1 in site.species:
                    for nn in self.oriented_unit_cell.get_neighbors(
                        site, bond_dist
                    ):
                        if sp2 in nn.species:
                            c_range = tuple(
                                sorted(
                                    [site.frac_coords[2], nn.frac_coords[2]]
                                )
                            )
                            if c_range[1] > 1:
                                # Takes care of PBC when c coordinate of site
                                # goes beyond the upper boundary of the cell
                                c_ranges.append((c_range[0], 1))
                                c_ranges.append((0, c_range[1] - 1))
                            elif c_range[0] < 0:
                                # Takes care of PBC when c coordinate of site
                                # is below the lower boundary of the unit cell
                                c_ranges.append((0, c_range[1]))
                                c_ranges.append((c_range[0] + 1, 1))
                            elif c_range[0] != c_range[1]:
                                c_ranges.append((c_range[0], c_range[1]))
        return c_ranges

    def get_slabs(
        self,
        bonds=None,
        ftol=0.1,
        tol=0.1,
        max_broken_bonds=0,
        symmetrize=False,
        repair=False,
    ):
        """
        This method returns a list of slabs that are generated using the list of
        shift values from the method, _calculate_possible_shifts(). Before the
        shifts are used to create the slabs however, if the user decides to take
        into account whether or not a termination will break any polyhedral
        structure (bonds is not None), this method will filter out any shift
        values that do so.

        Args:
            bonds ({(specie1, specie2): max_bond_dist}: bonds are
                specified as a dict of tuples: float of specie1, specie2
                and the max bonding distance. For example, PO4 groups may be
                defined as {("P", "O"): 3}.
            tol (float): General tolerance parameter for getting primitive
                cells and matching structures
            ftol (float): Threshold parameter in fcluster in order to check
                if two atoms are lying on the same plane. Default thresh set
                to 0.1 Angstrom in the direction of the surface normal.
            max_broken_bonds (int): Maximum number of allowable broken bonds
                for the slab. Use this to limit # of slabs (some structures
                may have a lot of slabs). Defaults to zero, which means no
                defined bonds must be broken.
            symmetrize (bool): Whether or not to ensure the surfaces of the
                slabs are equivalent.
            repair (bool): Whether to repair terminations with broken bonds
                or just omit them. Set to False as repairing terminations can
                lead to many possible slabs as oppose to just omitting them.

        Returns:
            ([Slab]) List of all possible terminations of a particular surface.
            Slabs are sorted by the # of bonds broken.
        """
        c_ranges = [] if bonds is None else self._get_c_ranges(bonds)

        slabs = []
        for shift in self._calculate_possible_shifts(tol=ftol):
            bonds_broken = 0
            for r in c_ranges:
                if r[0] <= shift <= r[1]:
                    bonds_broken += 1
            slab = self.get_slab(shift, tol=tol, energy=bonds_broken)
            if bonds_broken <= max_broken_bonds:
                slabs.append(slab)
            elif repair:
                # If the number of broken bonds is exceeded,
                # we repair the broken bonds on the slab
                slabs.append(self.repair_broken_bonds(slab, bonds))

        # Further filters out any surfaces made that might be the same
        m = StructureMatcher(
            ltol=tol, stol=tol, primitive_cell=False, scale=False
        )

        new_slabs = []
        for g in m.group_structures(slabs):
            # For each unique termination, symmetrize the
            # surfaces by removing sites from the bottom.
            if symmetrize:
                slabs = self.nonstoichiometric_symmetrized_slab(g[0])
                new_slabs.extend(slabs)
            else:
                new_slabs.append(g[0])

        match = StructureMatcher(
            ltol=tol, stol=tol, primitive_cell=False, scale=False
        )
        new_slabs = [g[0] for g in match.group_structures(new_slabs)]

        return sorted(new_slabs, key=lambda s: s.energy)

    def repair_broken_bonds(self, slab, bonds):
        """
        This method will find undercoordinated atoms due to slab
        cleaving specified by the bonds parameter and move them
        to the other surface to make sure the bond is kept intact.
        In a future release of surface.py, the ghost_sites will be
        used to tell us how the repair bonds should look like.

        Arg:
            slab (structure): A structure object representing a slab.
            bonds ({(specie1, specie2): max_bond_dist}: bonds are
                specified as a dict of tuples: float of specie1, specie2
                and the max bonding distance. For example, PO4 groups may be
                defined as {("P", "O"): 3}.

        Returns:
            (Slab) A Slab object with a particular shifted oriented unit cell.
        """
        for pair in bonds:
            blength = bonds[pair]

            # First lets determine which element should be the
            # reference (center element) to determine broken bonds.
            # e.g. P for a PO4 bond. Find integer coordination
            # numbers of the pair of elements wrt to each other
            cn_dict = {}
            for i, el in enumerate(pair):
                cnlist = []
                for site in self.oriented_unit_cell:
                    poly_coord = 0
                    if site.species_string == el:

                        for nn in self.oriented_unit_cell.get_neighbors(
                            site, blength
                        ):
                            if nn[0].species_string == pair[i - 1]:
                                poly_coord += 1
                    cnlist.append(poly_coord)
                cn_dict[el] = cnlist

            # We make the element with the higher coordination our reference
            if max(cn_dict[pair[0]]) > max(cn_dict[pair[1]]):
                element1, element2 = pair
            else:
                element2, element1 = pair

            for i, site in enumerate(slab):
                # Determine the coordination of our reference
                if site.species_string == element1:
                    poly_coord = 0
                    for neighbor in slab.get_neighbors(site, blength):
                        poly_coord += (
                            1 if neighbor.species_string == element2 else 0
                        )

                    # suppose we find an undercoordinated reference atom
                    if poly_coord not in cn_dict[element1]:
                        # We get the reference atom of the broken bonds
                        # (undercoordinated), move it to the other surface
                        slab = self.move_to_other_side(slab, [i])

                        # find its NNs with the corresponding
                        # species it should be coordinated with
                        neighbors = slab.get_neighbors(
                            slab[i], blength, include_index=True
                        )
                        tomove = [
                            nn[2]
                            for nn in neighbors
                            if nn[0].species_string == element2
                        ]
                        tomove.append(i)
                        # and then move those NNs along with the central
                        # atom back to the other side of the slab again
                        slab = self.move_to_other_side(slab, tomove)

        return slab

    def move_to_other_side(self, init_slab, index_of_sites):
        """
        This method will Move a set of sites to the
        other side of the slab (opposite surface).

        Arg:
            init_slab (structure): A structure object representing a slab.
            index_of_sites (list of ints): The list of indices representing
                the sites we want to move to the other side.

        Returns:
            (Slab) A Slab object with a particular shifted oriented unit cell.
        """
        slab = init_slab.copy()

        # Determine what fraction the slab is of the total cell size
        # in the c direction. Round to nearest rational number.
        h = self._proj_height
        p = h / self.parent.lattice.d_hkl(self.miller_index)
        if self.in_unit_planes:
            nlayers_slab = int(math.ceil(self.min_slab_size / p))
            nlayers_vac = int(math.ceil(self.min_vac_size / p))
        else:
            nlayers_slab = int(math.ceil(self.min_slab_size / h))
            nlayers_vac = int(math.ceil(self.min_vac_size / h))
        nlayers = nlayers_slab + nlayers_vac
        slab_ratio = nlayers_slab / nlayers

        # Sort the index of sites based on which side they are on
        top_site_index = [
            i
            for i in index_of_sites
            if slab[i].frac_coords[2] > slab.center_of_mass[2]
        ]
        bottom_site_index = [
            i
            for i in index_of_sites
            if slab[i].frac_coords[2] < slab.center_of_mass[2]
        ]

        # Translate sites to the opposite surfaces
        slab.translate_sites(top_site_index, [0, 0, slab_ratio])
        slab.translate_sites(bottom_site_index, [0, 0, -slab_ratio])

        return Slab(
            init_slab.lattice,
            slab.species,
            slab.frac_coords,
            init_slab.miller_index,
            init_slab.oriented_unit_cell,
            init_slab.shift,
            init_slab.scale_factor,
            energy=init_slab.energy,
        )

    def nonstoichiometric_symmetrized_slab(self, init_slab):
        """
        This method checks whether or not the two surfaces of the slab are
        equivalent. If the point group of the slab has an inversion symmetry (
        ie. belong to one of the Laue groups), then it is assumed that the
        surfaces should be equivalent. Otherwise, sites at the bottom of the
        slab will be removed until the slab is symmetric. Note the removal of sites
        can destroy the stoichiometry of the slab. For non-elemental
        structures, the chemical potential will be needed to calculate surface energy.

        Arg:
            init_slab (Structure): A single slab structure

        Returns:
            Slab (structure): A symmetrized Slab object.
        """
        if init_slab.is_symmetric():
            return [init_slab]

        nonstoich_slabs = []
        # Build an equivalent surface slab for each of the different surfaces
        for top in [True, False]:
            asym = True
            slab = init_slab.copy()
            slab.energy = init_slab.energy

            while asym:
                # Keep removing sites from the bottom one by one until both
                # surfaces are symmetric or the number of sites removed has
                # exceeded 10 percent of the original slab

                c_dir = [site[2] for i, site in enumerate(slab.frac_coords)]

                if top:
                    slab.remove_sites([c_dir.index(max(c_dir))])
                else:
                    slab.remove_sites([c_dir.index(min(c_dir))])
                if len(slab) <= len(self.parent):
                    break

                # Check if the altered surface is symmetric
                if slab.is_symmetric():
                    asym = False
                    nonstoich_slabs.append(slab)

        if len(slab) <= len(self.parent):
            warnings.warn(
                "Too many sites removed, please use a larger slab size."
            )

        return nonstoich_slabs


def get_d(slab):
    """
    Determine the distance of space between
    each layer of atoms along c
    """
    sorted_sites = sorted(slab, key=lambda site: site.frac_coords[2])
    for i, site in enumerate(sorted_sites):
        if (
            not f"{site.frac_coords[2]:.6f}"
            == f"{sorted_sites[i + 1].frac_coords[2]:.6f}"
        ):
            d = abs(site.frac_coords[2] - sorted_sites[i + 1].frac_coords[2])
            break
    return slab.lattice.get_cartesian_coords([0, 0, d])[2]


def is_already_analyzed(
    miller_index: tuple, miller_list: list, symm_ops: list
) -> bool:
    """
    Helper function to check if a given Miller index is
    part of the family of indices of any index in a list

    Args:
        miller_index (tuple): The Miller index to analyze
        miller_list (list): List of Miller indices. If the given
            Miller index belongs in the same family as any of the
            indices in this list, return True, else return False
        symm_ops (list): Symmetry operations of a
            lattice, used to define family of indices
    """
    for op in symm_ops:
        if in_coord_list(miller_list, op.operate(miller_index)):
            return True
    return False


def get_symmetrically_equivalent_miller_indices(
    structure, miller_index, return_hkil=True
):
    """
    Returns all symmetrically equivalent indices for a given structure. Analysis
    is based on the symmetry of the reciprocal lattice of the structure.

    Args:
        miller_index (tuple): Designates the family of Miller indices
            to find. Can be hkl or hkil for hexagonal systems
        return_hkil (bool): If true, return hkil form of Miller
            index for hexagonal systems, otherwise return hkl
    """

    # Change to hkl if hkil because in_coord_list only handles tuples of 3
    miller_index = (
        (miller_index[0], miller_index[1], miller_index[3])
        if len(miller_index) == 4
        else miller_index
    )
    mmi = max(np.abs(miller_index))
    r = list(range(-mmi, mmi + 1))
    r.reverse()

    sg = SpacegroupAnalyzer(structure)
    # Get distinct hkl planes from the rhombohedral setting if trigonal
    if sg.get_crystal_system() == "trigonal":
        prim_structure = SpacegroupAnalyzer(
            structure
        ).get_primitive_standard_structure()
        symm_ops = prim_structure.lattice.get_recp_symmetry_operation()
    else:
        symm_ops = structure.lattice.get_recp_symmetry_operation()

    equivalent_millers = [miller_index]
    for miller in itertools.product(r, r, r):
        if miller == miller_index:
            continue
        if any(i != 0 for i in miller):
            if is_already_analyzed(miller, equivalent_millers, symm_ops):
                equivalent_millers.append(miller)

            # include larger Miller indices in the family of planes
            if all(mmi > i for i in np.abs(miller)) and not in_coord_list(
                equivalent_millers, miller
            ):
                if is_already_analyzed(
                    mmi * np.array(miller), equivalent_millers, symm_ops
                ):
                    equivalent_millers.append(miller)

    if return_hkil and sg.get_crystal_system() in ["trigonal", "hexagonal"]:
        return [
            (hkl[0], hkl[1], -1 * hkl[0] - hkl[1], hkl[2])
            for hkl in equivalent_millers
        ]
    return equivalent_millers


def get_symmetrically_distinct_miller_indices(
    structure, max_index, return_hkil=False
):
    """
    Returns all symmetrically distinct indices below a certain max-index for
    a given structure. Analysis is based on the symmetry of the reciprocal
    lattice of the structure.
    Args:
        structure (Structure): input structure.
        max_index (int): The maximum index. For example, a max_index of 1
            means that (100), (110), and (111) are returned for the cubic
            structure. All other indices are equivalent to one of these.
        return_hkil (bool): If true, return hkil form of Miller
            index for hexagonal systems, otherwise return hkl
    """

    r = list(range(-max_index, max_index + 1))
    r.reverse()

    # First we get a list of all hkls for conventional (including equivalent)
    conv_hkl_list = [
        miller
        for miller in itertools.product(r, r, r)
        if any(i != 0 for i in miller)
    ]

    sg = SpacegroupAnalyzer(structure)
    # Get distinct hkl planes from the rhombohedral setting if trigonal
    if sg.get_crystal_system() == "trigonal":
        transf = sg.get_conventional_to_primitive_transformation_matrix()
        miller_list = [
            hkl_transformation(transf, hkl) for hkl in conv_hkl_list
        ]
        prim_structure = SpacegroupAnalyzer(
            structure
        ).get_primitive_standard_structure()
        symm_ops = prim_structure.lattice.get_recp_symmetry_operation()
    else:
        miller_list = conv_hkl_list
        symm_ops = structure.lattice.get_recp_symmetry_operation()

    unique_millers, unique_millers_conv = [], []

    for i, miller in enumerate(miller_list):
        d = abs(reduce(gcd, miller))
        miller = tuple(int(i / d) for i in miller)
        if not is_already_analyzed(miller, unique_millers, symm_ops):
            if sg.get_crystal_system() == "trigonal":
                # Now we find the distinct primitive hkls using
                # the primitive symmetry operations and their
                # corresponding hkls in the conventional setting
                unique_millers.append(miller)
                d = abs(reduce(gcd, conv_hkl_list[i]))
                cmiller = tuple(int(i / d) for i in conv_hkl_list[i])
                unique_millers_conv.append(cmiller)
            else:
                unique_millers.append(miller)
                unique_millers_conv.append(miller)

    if return_hkil and sg.get_crystal_system() in ["trigonal", "hexagonal"]:
        return [
            (hkl[0], hkl[1], -1 * hkl[0] - hkl[1], hkl[2])
            for hkl in unique_millers_conv
        ]
    return unique_millers_conv


def hkl_transformation(transf, miller_index):
    """
    Returns the Miller index from setting
    A to B using a transformation matrix
    Args:
        transf (3x3 array): The transformation matrix
            that transforms a lattice of A to B
        miller_index ([h, k, l]): Miller index to transform to setting B
    """
    # Get a matrix of whole numbers (ints)

    def lcm(a, b):
        return a * b // math.gcd(a, b)

    reduced_transf = (
        reduce(lcm, [int(1 / i) for i in itertools.chain(*transf) if i != 0])
        * transf
    )
    reduced_transf = reduced_transf.astype(int)

    # perform the transformation
    t_hkl = np.dot(reduced_transf, miller_index)
    d = abs(reduce(gcd, t_hkl))
    t_hkl = np.array([int(i / d) for i in t_hkl])

    # get mostly positive oriented Miller index
    if len([i for i in t_hkl if i < 0]) > 1:
        t_hkl *= -1

    return tuple(t_hkl)


def generate_all_slabs(
    structure,
    max_index,
    min_slab_size,
    min_vacuum_size,
    bonds=None,
    tol=0.1,
    ftol=0.1,
    max_broken_bonds=0,
    lll_reduce=False,
    center_slab=False,
    primitive=True,
    max_normal_search=None,
    symmetrize=False,
    repair=False,
    include_reconstructions=False,
    in_unit_planes=False,
):
    """
    A function that finds all different slabs up to a certain miller index.
    Slabs oriented under certain Miller indices that are equivalent to other
    slabs in other Miller indices are filtered out using symmetry operations
    to get rid of any repetitive slabs. For example, under symmetry operations,
    CsCl has equivalent slabs in the (0,0,1), (0,1,0), and (1,0,0) direction.

    Args:
        structure (Structure): Initial input structure. Note that to
                ensure that the miller indices correspond to usual
                crystallographic definitions, you should supply a conventional
                unit cell structure.
        max_index (int): The maximum Miller index to go up to.
        min_slab_size (float): In Angstroms
        min_vacuum_size (float): In Angstroms
        bonds ({(specie1, specie2): max_bond_dist}: bonds are
            specified as a dict of tuples: float of specie1, specie2
            and the max bonding distance. For example, PO4 groups may be
            defined as {("P", "O"): 3}.
        tol (float): General tolerance parameter for getting primitive
            cells and matching structures
        ftol (float): Threshold parameter in fcluster in order to check
            if two atoms are lying on the same plane. Default thresh set
            to 0.1 Angstrom in the direction of the surface normal.
        max_broken_bonds (int): Maximum number of allowable broken bonds
            for the slab. Use this to limit # of slabs (some structures
            may have a lot of slabs). Defaults to zero, which means no
            defined bonds must be broken.
        lll_reduce (bool): Whether to perform an LLL reduction on the
            eventual structure.
        center_slab (bool): Whether to center the slab in the cell with
            equal vacuum spacing from the top and bottom.
        primitive (bool): Whether to reduce any generated slabs to a
            primitive cell (this does **not** mean the slab is generated
            from a primitive cell, it simply means that after slab
            generation, we attempt to find shorter lattice vectors,
            which lead to less surface area and smaller cells).
        max_normal_search (int): If set to a positive integer, the code will
            conduct a search for a normal lattice vector that is as
            perpendicular to the surface as possible by considering
            multiples linear combinations of lattice vectors up to
            max_normal_search. This has no bearing on surface energies,
            but may be useful as a preliminary step to generating slabs
            for absorption and other sizes. It is typical that this will
            not be the smallest possible cell for simulation. Normality
            is not guaranteed, but the oriented cell will have the c
            vector as normal as possible (within the search range) to the
            surface. A value of up to the max absolute Miller index is
            usually sufficient.
        symmetrize (bool): Whether or not to ensure the surfaces of the
            slabs are equivalent.
        repair (bool): Whether to repair terminations with broken bonds
            or just omit them
        include_reconstructions (bool): Whether to include reconstructed
            slabs available in the reconstructions_archive.json file.
    """
    all_slabs = []

    for miller in get_symmetrically_distinct_miller_indices(
        structure, max_index
    ):
        gen = SlabGenerator(
            structure,
            miller,
            min_slab_size,
            min_vacuum_size,
            lll_reduce=lll_reduce,
            center_slab=center_slab,
            primitive=primitive,
            max_normal_search=max_normal_search,
            in_unit_planes=in_unit_planes,
        )
        slabs = gen.get_slabs(
            bonds=bonds,
            tol=tol,
            ftol=ftol,
            symmetrize=symmetrize,
            max_broken_bonds=max_broken_bonds,
            repair=repair,
        )

        if len(slabs) > 0:
            logger.debug(f"{miller} has {len(slabs)} slabs... ")
            all_slabs.extend(slabs)

    if include_reconstructions:
        sg = SpacegroupAnalyzer(structure)
        symbol = sg.get_space_group_symbol()
        # enumerate through all posisble reconstructions in the
        # archive available for this particular structure (spacegroup)
        for name, instructions in reconstructions_archive.items():
            if "base_reconstruction" in instructions:
                instructions = reconstructions_archive[
                    instructions["base_reconstruction"]
                ]
            if instructions["spacegroup"]["symbol"] == symbol:
                # check if this reconstruction has a max index
                # equal or less than the given max index
                if max(instructions["miller_index"]) > max_index:
                    continue
                recon = ReconstructionGenerator(
                    structure, min_slab_size, min_vacuum_size, name
                )
                all_slabs.extend(recon.build_slabs())

    return all_slabs


def get_slab_regions(slab, blength=3.5):
    """
    Function to get the ranges of the slab regions. Useful for discerning where
    the slab ends and vacuum begins if the slab is not fully within the cell
    Args:
        slab (Structure): Structure object modelling the surface
        blength (float, Ang): The bondlength between atoms. You generally
            want this value to be larger than the actual bondlengths in
            order to find atoms that are part of the slab
    """

    fcoords, indices, all_indices = [], [], []
    for site in slab:
        # find sites with c < 0 (noncontiguous)
        neighbors = slab.get_neighbors(
            site, blength, include_index=True, include_image=True
        )
        for nn in neighbors:
            if nn[0].frac_coords[2] < 0:
                # sites are noncontiguous within cell
                fcoords.append(nn[0].frac_coords[2])
                indices.append(nn[-2])
                if nn[-2] not in all_indices:
                    all_indices.append(nn[-2])

    if fcoords:
        # If slab is noncontiguous, locate the lowest
        # site within the upper region of the slab
        while fcoords:
            last_fcoords = copy.copy(fcoords)
            last_indices = copy.copy(indices)
            site = slab[indices[fcoords.index(min(fcoords))]]
            neighbors = slab.get_neighbors(
                site, blength, include_index=True, include_image=True
            )
            fcoords, indices = [], []
            for nn in neighbors:
                if (
                    1 > nn[0].frac_coords[2] > 0
                    and nn[0].frac_coords[2] < site.frac_coords[2]
                ):
                    # sites are noncontiguous within cell
                    fcoords.append(nn[0].frac_coords[2])
                    indices.append(nn[-2])
                    if nn[-2] not in all_indices:
                        all_indices.append(nn[-2])

        # Now locate the highest site within the lower region of the slab
        upper_fcoords = []
        for site in slab:
            if all(
                nn.index not in all_indices
                for nn in slab.get_neighbors(site, blength)
            ):
                upper_fcoords.append(site.frac_coords[2])
        coords = copy.copy(last_fcoords) if not fcoords else copy.copy(fcoords)
        min_top = slab[last_indices[coords.index(min(coords))]].frac_coords[2]
        ranges = [[0, max(upper_fcoords)], [min_top, 1]]
    else:
        # If the entire slab region is within the slab cell, just
        # set the range as the highest and lowest site in the slab
        sorted_sites = sorted(slab, key=lambda site: site.frac_coords[2])
        ranges = [
            [sorted_sites[0].frac_coords[2], sorted_sites[-1].frac_coords[2]]
        ]

    return ranges


def miller_index_from_sites(
    lattice, coords, coords_are_cartesian=True, round_dp=4, verbose=True
):
    """
    Get the Miller index of a plane from a list of site coordinates.

    A minimum of 3 sets of coordinates are required. If more than 3 sets of
    coordinates are given, the best plane that minimises the distance to all
    points will be calculated.

    Args:
        lattice (list or Lattice): A 3x3 lattice matrix or `Lattice` object (for
            example obtained from Structure.lattice).
        coords (iterable): A list or numpy array of coordinates. Can be
            Cartesian or fractional coordinates. If more than three sets of
            coordinates are provided, the best plane that minimises the
            distance to all sites will be calculated.
        coords_are_cartesian (bool, optional): Whether the coordinates are
            in Cartesian space. If using fractional coordinates set to False.
        round_dp (int, optional): The number of decimal places to round the
            miller index to.
        verbose (bool, optional): Whether to print warnings.

    Returns:
        (tuple): The Miller index.
    """
    if not isinstance(lattice, Lattice):
        lattice = Lattice(lattice)

    return lattice.get_miller_index_from_coords(
        coords,
        coords_are_cartesian=coords_are_cartesian,
        round_dp=round_dp,
        verbose=verbose,
    )


def center_slab(slab):
    """
    The goal here is to ensure the center of the slab region
        is centered close to c=0.5. This makes it easier to
        find the surface sites and apply operations like doping.

    There are three cases where the slab in not centered:

    1. The slab region is completely between two vacuums in the
    box but not necessarily centered. We simply shift the
    slab by the difference in its center of mass and 0.5
    along the c direction.

    2. The slab completely spills outside the box from the bottom
    and into the top. This makes it incredibly difficult to
    locate surface sites. We iterate through all sites that
    spill over (z>c) and shift all sites such that this specific
    site is now on the other side. Repeat for all sites with z>c.

    3. This is a simpler case of scenario 2. Either the top or bottom
    slab sites are at c=0 or c=1. Treat as scenario 2.

    Args:
        slab (Slab): Slab structure to center
    Returns:
        Returns a centered slab structure
    """

    # get a reasonable r cutoff to sample neighbors
    bdists = sorted(
        nn[1] for nn in slab.get_neighbors(slab[0], 10) if nn[1] > 0
    )
    r = bdists[0] * 3

    all_indices = [i for i, site in enumerate(slab)]

    # check if structure is case 2 or 3, shift all the
    # sites up to the other side until it is case 1
    for site in slab:
        if any(nn[1] > slab.lattice.c for nn in slab.get_neighbors(site, r)):
            shift = 1 - site.frac_coords[2] + 0.05
            slab.translate_sites(all_indices, [0, 0, shift])

    # now the slab is case 1, shift the center of mass of the slab to 0.5
    weights = [s.species.weight for s in slab]
    center_of_mass = np.average(slab.frac_coords, weights=weights, axis=0)
    shift = 0.5 - center_of_mass[2]
    slab.translate_sites(all_indices, [0, 0, shift])

    return slab


def _reduce_vector(vector):
    # small function to reduce vectors

    d = abs(reduce(gcd, vector))
    vector = tuple(int(i / d) for i in vector)

    return vector
