"""
This module will be used to construct the surfaces and interfaces used in this package.
"""
from OgreInterface import utils
from OgreInterface.lattice_match import OgreMatch

from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from pymatgen.core.periodic_table import Element, Species
from pymatgen.io.vasp.inputs import Poscar
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SymmOp
from pymatgen.analysis.molecule_structure_comparator import CovalentRadius
from pymatgen.analysis.local_env import CrystalNN

from typing import Dict, Union, Iterable, List
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from itertools import combinations, groupby
import numpy as np
import copy
from copy import deepcopy
from functools import reduce
from ase import Atoms
import warnings

# supress warning from CrystallNN when ionic radii are not found.
warnings.filterwarnings("ignore", module=r"pymatgen.analysis.local_env")


class Surface:
    """Container for surfaces generated with the SurfaceGenerator

    The Surface class and will be used as an input to the InterfaceGenerator class,
    and it should be create exclusively using the SurfaceGenerator.

    Examples:
        Generating a surface with pseudo-hydrogen passivation where the atomic positions of the hydrogens need to be relaxed using DFT.
        >>> from OgreInterface.generate import SurfaceGenerator
        >>> surfaces = SurfaceGenerator.from_file(filename="POSCAR_bulk", miller_index=[1, 1, 1], layers=5, vacuum=60)
        >>> surface = surfaces.slabs[0] # OgreInterface.Surface object
        >>> surface.passivate(bot=True, top=True)
        >>> surface.write_file(output="POSCAR_slab", orthogonal=True, relax=True) # relax=True will automatically set selective dynamics=True for all passivating hydrogens

        Generating a surface with pseudo-hydrogen passivation that comes from a structure with pre-relaxed pseudo-hydrogens.
        >>> from OgreInterface.generate import SurfaceGenerator
        >>> surfaces = SurfaceGenerator.from_file(filename="POSCAR_bulk", miller_index=[1, 1, 1], layers=20, vacuum=60)
        >>> surface = surfaces.slabs[0] # OgreInterface.Surface object
        >>> surface.passivate(bot=True, top=True, passivated_struc="CONTCAR") # CONTCAR is the output of the structural relaxation
        >>> surface.write_file(output="POSCAR_slab", orthogonal=True, relax=False)

    Args:
        orthogonal_slab: Slab structure that is forced to have an c lattice vector that is orthogonal
            to the inplane lattice vectors
        non_orthogonal_slab: Slab structure that is not gaurunteed to have an orthogonal c lattice vector,
            and assumes the same basis as the primitive_oriented_bulk structure.
        oriented_bulk: Structure of the smallest building block of the slab, which was used to
            construct the non_orthogonal_slab supercell by creating a (1x1xN) supercell where N in the number
            of layers.
        bulk: Bulk structure that can be transformed into the slab basis using the transformation_matrix
        transformation_matrix: 3x3 integer matrix that used to change from the bulk basis to the slab basis.
        miller_index: Miller indices of the surface, with respect to the conventional bulk structure.
        layers: Number of unit cell layers in the surface
        vacuum: Size of the vacuum in Angstroms
        uvw_basis: Miller indices corresponding to the lattice vector directions of the slab
        point_group_operations: List of unique point group operations that will eventually be used to efficiently
            filter out symmetrically equivalent interfaces found using the lattice matching algorithm.
        bottom_layer_dist: z-distance of where the next atom should be if the slab structure were to continue downwards
            (This is used to automatically approximate the interfacial distance in interfacial_distance is set to None in the InterfaceGenerator)
        top_layer_dist: z-distance of where the next atom should be if the slab structure were to continue upwards
            (This is used to automatically approximate the interfacial distance in interfacial_distance is set to None in the InterfaceGenerator)
        termination_index: Index of the Surface in the list of Surfaces produced by the SurfaceGenerator
        surface_normal (np.ndarray): The normal vector of the surface
        c_projection (float): The projections of the c-lattice vector onto the surface normal

    Attributes:
        orthogonal_slab_structure (Structure): Pymatgen Structure that is forced to have an c lattice vector that is orthogonal
            to the inplane lattice vectors
        orthogonal_slab_atoms (Atoms): ASE Atoms that is forced to have an c lattice vector that is orthogonal
            to the inplane lattice vectors
        non_orthogonal_slab_structure (Structure): Pymatgen Structure that is not gaurunteed to have an orthogonal c lattice vector,
            and assumes the same basis as the primitive_oriented_bulk structure.
        non_orthogonal_slab_atoms (Atoms): ASE Atoms that is not gaurunteed to have an orthogonal c lattice vector,
            and assumes the same basis as the primitive_oriented_bulk structure.
        oriented_bulk_structure: Pymatgen Structure of the smallest building block of the slab, which was used to
            construct the non_orthogonal_slab supercell by creating a (1x1xN) supercell where N in the number
            of layers.
        oriented_bulk_atoms (Atoms): ASE Atoms of the smallest building block of the slab, which was used to
            construct the non_orthogonal_slab supercell by creating a (1x1xN) supercell where N in the number
            of layers.
        bulk_structure (Structure): Bulk Pymatgen Structure that can be transformed into the slab basis using the transformation_matrix
        bulk_atoms (Atoms): Bulk ASE Atoms that can be transformed into the slab basis using the transformation_matrix
        transformation_matrix (np.ndarray): 3x3 integer matrix that used to change from the bulk basis to the slab basis.
        miller_index (list): Miller indices of the surface, with respect to the conventional bulk structure.
        layers (int): Number of unit cell layers in the surface
        vacuum (float): Size of the vacuum in Angstroms
        uvw_basis (np.ndarray): Miller indices corresponding to the lattice vector directions of the slab
        point_group_operations (np.ndarray): List of unique point group operations that will eventually be used to efficiently
            filter out symmetrically equivalent interfaces found using the lattice matching algorithm.
        bottom_layer_dist (float): z-distance of where the next atom should be if the slab structure were to continue downwards
            (This is used to automatically approximate the interfacial distance in interfacial_distance is set to None in the InterfaceGenerator)
        top_layer_dist (float): z-distance of where the next atom should be if the slab structure were to continue upwards
            (This is used to automatically approximate the interfacial distance in interfacial_distance is set to None in the InterfaceGenerator)
        termination_index (int): Index of the Surface in the list of Surfaces produced by the SurfaceGenerator
        surface_normal (np.ndarray): The normal vector of the surface
        c_projection (float): The projections of the c-lattice vector onto the surface normal
    """

    def __init__(
        self,
        orthogonal_slab: Union[Structure, Atoms],
        non_orthogonal_slab: Union[Structure, Atoms],
        oriented_bulk: Union[Structure, Atoms],
        bulk: Union[Structure, Atoms],
        transformation_matrix: np.ndarray,
        miller_index: list,
        layers: int,
        vacuum: float,
        uvw_basis: np.ndarray,
        point_group_operations: np.ndarray,
        bottom_layer_dist: float,
        top_layer_dist: float,
        termination_index: int,
        surface_normal: np.ndarray,
        c_projection: int,
    ) -> None:
        (
            self._orthogonal_slab_structure,
            self._orthogonal_slab_atoms,
        ) = self._get_atoms_and_struc(orthogonal_slab)
        (
            self._non_orthogonal_slab_structure,
            self._non_orthogonal_slab_atoms,
        ) = self._get_atoms_and_struc(non_orthogonal_slab)
        (
            self.oriented_bulk_structure,
            self.oriented_bulk_atoms,
        ) = self._get_atoms_and_struc(oriented_bulk)
        (
            self.bulk_structure,
            self.bulk_atoms,
        ) = self._get_atoms_and_struc(bulk)

        self.surface_normal = surface_normal
        self.c_projection = c_projection
        self.transformation_matrix = transformation_matrix
        self.miller_index = miller_index
        self.layers = layers
        self.vacuum = vacuum
        self.uvw_basis = uvw_basis
        self.point_group_operations = point_group_operations
        self.bottom_layer_dist = bottom_layer_dist
        self.top_layer_dist = top_layer_dist
        self.termination_index = termination_index
        self._passivated = False

    def get_surface(
        self,
        orthogonal: bool = True,
        return_structure: bool = True,
        return_atoms: bool = False,
    ) -> Union[Atoms, Structure]:
        """
        This is a simple function for easier access to the surface structure generated from the SurfaceGenerator

        Args:
            orthogonal: Determines if the orthogonalized structure is returned
            return_structure: Determines if the Pymatgen Structure object is returned
            return_atoms: Determines if the ASE Atoms object is returned

        Returns:
            Either a Pymatgen Structure of ASE Atoms object of the surface structure
        """
        if orthogonal:
            if return_structure and not return_atoms:
                return self._orthogonal_slab_structure
            elif return_atoms and not return_structure:
                return self._orthogonal_slab_atoms
            else:
                raise ValueError(
                    "Please select either return_atoms=True OR return_structure=True to get an ASE atoms object or a Pymatgen Structure object."
                )
        else:
            if return_structure and not return_atoms:
                return self._non_orthogonal_slab_structure
            elif return_atoms and not return_structure:
                return self._non_orthogonal_slab_atoms
            else:
                raise ValueError(
                    "Please select either return_atoms=True OR return_structure=True to get an ASE atoms object or a Pymatgen Structure object."
                )

    @property
    def formula(self) -> str:
        """
        Reduced formula of the surface

        Examples:
            >>> surface.formula
            >>> "InAs"

        Returns:
            Reduced formula of the underlying bulk structure
        """
        return self.bulk_structure.composition.reduced_formula

    @property
    def area(self) -> float:
        """
        Cross section area of the slab in Angstroms^2

        Examples:
            >>> surface.area
            >>> 62.51234

        Returns:
            Cross-section area in Angstroms^2
        """
        area = np.linalg.norm(
            np.cross(
                self._orthogonal_slab_structure.lattice.matrix[0],
                self._orthogonal_slab_structure.lattice.matrix[1],
            )
        )

        return area

    @property
    def inplane_vectors(self) -> np.ndarray:
        """
        In-plane cartesian vectors of the slab structure

        Examples:
            >>> surface.inplane_vectors
            >>> [[4.0 0.0 0.0]
            ...  [2.0 2.0 0.0]]

        Returns:
            (2, 3) numpy array containing the cartesian coordinates of the in-place lattice vectors
        """
        matrix = deepcopy(self._orthogonal_slab_structure.lattice.matrix)
        return matrix[:2]

    @property
    def miller_index_a(self) -> np.ndarray:
        """
        Miller index of the a-lattice vector

        Examples:
            >>> surface.miller_index_a
            >>> [-1 1 0]

        Returns:
            (3,) numpy array containing the miller indices
        """
        return self.uvw_basis[0].astype(int)

    @property
    def miller_index_b(self) -> np.ndarray:
        """
        Miller index of the b-lattice vector

        Examples:
            >>> surface.miller_index_b
            >>> [1 -1 0]

        Returns:
            (3,) numpy array containing the miller indices
        """
        return self.uvw_basis[1].astype(int)

    def _get_atoms_and_struc(self, atoms_or_struc):
        if type(atoms_or_struc) == Atoms:
            init_structure = AseAtomsAdaptor.get_structure(atoms_or_struc)
            init_atoms = atoms_or_struc
        elif type(atoms_or_struc) == Structure:
            init_structure = atoms_or_struc
            init_atoms = AseAtomsAdaptor.get_atoms(atoms_or_struc)
        else:
            raise TypeError(
                f"Surface._get_atoms_and_struc() accepts 'pymatgen.core.structure.Structure' or 'ase.Atoms' not '{type(atoms_or_struc).__name__}'"
            )

        return init_structure, init_atoms

    def write_file(
        self,
        orthogonal: bool = True,
        output: str = "POSCAR_slab",
        relax: bool = False,
    ) -> None:
        """
        Writes a POSCAR file of the surface with important information about the slab such as the number of layers, the termination index, and pseudo-hydrogen charges

        Examples:
            Writing a POSCAR file for a static DFT calculation:
            >>> surface.write_file(orthogonal=True, output="POSCAR", relax=False)

            Writing a passivated POSCAR file that needs to be relaxed using DFT:
            >>> surface.write_file(orthogonal=True, output="POSCAR", relax=True)


        Args:
            orthogonal: Determines the the output slab is forced to have a c-vector that is orthogonal to the a and b lattice vectors
            output: File path of the POSCAR
            relax: Determines if selective dynamics should be set in the POSCAR
        """
        if orthogonal:
            slab = self._orthogonal_slab_structure
        else:
            slab = self._non_orthogonal_slab_structure

        comment = "|".join(
            [
                f"L={self.layers}",
                f"T={self.termination_index}",
                f"O={orthogonal}",
            ]
        )

        if not self._passivated:
            poscar_str = Poscar(slab, comment=comment).get_string()
        else:
            if relax:
                atomic_numbers = np.array(slab.atomic_numbers)
                selective_dynamics = np.repeat(
                    (atomic_numbers == 1).reshape(-1, 1),
                    repeats=3,
                    axis=1,
                )
            else:
                selective_dynamics = None

            syms = [site.specie.symbol for site in slab]

            syms = []
            for site in slab:
                if site.specie.symbol == "H":
                    if hasattr(site.specie, "oxi_state"):
                        oxi = site.specie.oxi_state

                        if oxi < 1.0:
                            H_str = "H" + f"{oxi:.2f}"[1:]
                        elif oxi > 1.0:
                            H_str = "H" + f"{oxi:.2f}"
                        else:
                            H_str = "H"

                        syms.append(H_str)
                else:
                    syms.append(site.specie.symbol)

            comp_list = [(a[0], len(list(a[1]))) for a in groupby(syms)]
            atom_types, n_atoms = zip(*comp_list)

            new_atom_types = []
            for atom in atom_types:
                if "H" == atom[0] and atom not in ["Hf", "Hs", "Hg", "He"]:
                    new_atom_types.append("H")
                else:
                    new_atom_types.append(atom)

            comment += "|potcar=" + " ".join(atom_types)

            poscar = Poscar(slab, comment=comment)

            if relax:
                poscar.selective_dynamics = selective_dynamics

            poscar_str = poscar.get_string().split("\n")
            poscar_str[5] = " ".join(new_atom_types)
            poscar_str[6] = " ".join(list(map(str, n_atoms)))
            poscar_str = "\n".join(poscar_str)

        with open(output, "w") as f:
            f.write(poscar_str)

    def remove_layers(
        self,
        num_layers: int,
        top: bool = False,
        atol: Union[float, None] = None,
    ) -> None:
        """
        Removes atomic layers from a specified side of the surface. Using this function will ruin the pseudo-hydrogen passivation
        for the side that has layers removed, so it would be prefered to just select a different termination from the list of Surfaces
        generated using the SurfaceGenerator instead of manually removing layers to get the termination you want.

        Examples:
            Removing 3 layers from the top of a surface:
            >>> surface.remove_layers(num_layers=3, top=True)

        Args:
            num_layers: Number of atomic layers to remove
            top: Determines of the layers are removed from the top of the slab or the bottom if False
            atol: Tolarence for grouping the layers, if None, it is automatically determined and usually performs well
        """
        group_inds_conv, _ = utils.group_layers(
            structure=self._orthogonal_slab_structure, atol=atol
        )
        if top:
            group_inds_conv = group_inds_conv[::-1]

        to_delete_conv = []
        for i in range(num_layers):
            to_delete_conv.extend(group_inds_conv[i])

        self._orthogonal_slab_structure.remove_sites(to_delete_conv)

    def _get_surface_atoms(self):
        obs = self.oriented_bulk_structure.copy()
        obs.add_oxidation_state_by_guess()

        layer_struc = utils.get_layer_supercelll(structure=obs, layers=3)
        layer_struc.sort()

        layer_inds = np.array(layer_struc.site_properties["layer_index"])

        bottom_inds = np.where(layer_inds == 0)[0]
        top_inds = np.where(layer_inds == np.max(layer_inds))[0]

        cnn = CrystalNN()
        top_neighborhood = []
        for i in top_inds:
            info_dict = cnn.get_nn_info(layer_struc, i)
            for neighbor in info_dict:
                if neighbor["image"][-1] > 0:
                    top_neighborhood.append((i, info_dict))
                    break

        bottom_neighborhood = []
        for i in bottom_inds:
            info_dict = cnn.get_nn_info(layer_struc, i)
            for neighbor in info_dict:
                if neighbor["image"][-1] < 0:
                    bottom_neighborhood.append((i, info_dict))
                    break

        return layer_struc, [bottom_neighborhood, top_neighborhood]

    def _get_pseudohydrogen_charge(self, site, coordination):
        electronic_struc = site.specie.electronic_structure.split(".")[1:]
        oxi_state = site.specie.oxi_state
        valence = 0
        for orb in electronic_struc:
            if orb[1] == "d":
                if int(orb[2:]) < 10:
                    valence += int(orb[2:])
            else:
                valence += int(orb[2:])

        if oxi_state >= 0:
            charge = (8 - valence) / coordination
        else:
            charge = ((2 * coordination) - valence) / coordination

        return charge

    def _get_bond_dict(self):
        image_map = {1: "+", 0: "=", -1: "-"}
        (
            layer_struc,
            surface_neighborhoods,
        ) = self._get_surface_atoms()

        labels = ["bottom", "top"]
        bond_dict = {"bottom": {}, "top": {}}
        H_len = 0.31

        for i, neighborhood in enumerate(surface_neighborhoods):
            for surface_atom in neighborhood:
                atom_index = surface_atom[0]
                center_atom_equiv_index = layer_struc[atom_index].properties[
                    "oriented_bulk_equivalent"
                ]

                try:
                    center_len = CovalentRadius.radius[
                        layer_struc[atom_index].specie.symbol
                    ]
                except KeyError:
                    center_len = layer_struc[atom_index].specie.atomic_radius

                oriented_bulk_equivalent = layer_struc[atom_index].properties[
                    "oriented_bulk_equivalent"
                ]
                neighbor_info = surface_atom[1]
                coordination = len(neighbor_info)
                charge = self._get_pseudohydrogen_charge(
                    layer_struc[atom_index], coordination
                )
                broken_atoms = [
                    neighbor
                    for neighbor in neighbor_info
                    if neighbor["image"][-1] != 0
                ]

                bonds = []
                bond_strs = []
                for atom in broken_atoms:
                    broken_site = atom["site"]
                    broken_atom_equiv_index = broken_site.properties[
                        "oriented_bulk_equivalent"
                    ]
                    broken_image = broken_site.image.astype(int)
                    broken_atom_cart_coords = broken_site.coords
                    center_atom_cart_coords = layer_struc[atom_index].coords
                    bond_vector = (
                        broken_atom_cart_coords - center_atom_cart_coords
                    )
                    norm_vector = bond_vector / np.linalg.norm(bond_vector)
                    H_vector = (H_len + center_len) * norm_vector

                    H_str = ",".join(
                        [
                            str(center_atom_equiv_index),
                            str(broken_atom_equiv_index),
                            "".join([image_map[i] for i in broken_image]),
                            str(i),
                        ]
                    )

                    bonds.append(H_vector)
                    bond_strs.append(H_str)

                bond_dict[labels[i]][oriented_bulk_equivalent] = {
                    "bonds": np.vstack(bonds),
                    "bond_strings": bond_strs,
                    "charge": charge,
                }

        return bond_dict

    def _get_passivation_atom_index(self, struc, bulk_equivalent, top=False):
        struc_layer_index = np.array(struc.site_properties["layer_index"])
        struc_bulk_equiv = np.array(
            struc.site_properties["oriented_bulk_equivalent"]
        )

        if top:
            layer_number = np.max(struc_layer_index)
        else:
            layer_number = 0

        atom_index = np.where(
            np.logical_and(
                struc_layer_index == layer_number,
                struc_bulk_equiv == bulk_equivalent,
            )
        )[0][0]

        return atom_index

    def _passivate(self, struc, index, bond, bond_str, charge):
        position = struc[index].coords + bond
        position = struc[index].coords + bond
        props = {k: -1 for k in struc[index].properties}
        # props["hydrogen_str"] = f"{index}," + bond_str

        struc.append(
            Species("H", oxidation_state=charge),
            coords=position,
            coords_are_cartesian=True,
            properties=props,
        )

    def _get_passivated_bond_dict(
        self,
        bond_dict: Dict[
            str, Dict[int, Dict[str, Union[np.ndarray, str, float]]]
        ],
        relaxed_structure_file: str,
    ):
        with open(relaxed_structure_file, "r") as f:
            poscar_str = f.read().split("\n")

        desc_str = poscar_str[0].split("|")

        layers = int(desc_str[0].split("=")[1])
        termination_index = int(desc_str[1].split("=")[1])

        if termination_index == self.termination_index:
            poscar = Poscar.from_string(
                "\n".join(poscar_str), read_velocities=False
            )

            structure = poscar.structure
            hydrogen_index = np.array(structure.atomic_numbers) == 1

            obs = self.oriented_bulk_structure.copy()
            obs.add_oxidation_state_by_guess()

            is_negative = np.linalg.det(obs.lattice.matrix) < 0

            if is_negative:
                structure = Structure(
                    lattice=Lattice(structure.lattice.matrix * -1),
                    species=structure.species,
                    coords=structure.frac_coords,
                )

            hydrogen_coords = structure.cart_coords[hydrogen_index]

            layer_struc = utils.get_layer_supercelll(
                structure=obs, layers=layers, vacuum_scale=2
            )
            layer_struc.sort()

            surface_atom_info = []

            for side, side_dict in bond_dict.items():
                is_top = True if side == "top" else False
                for bulk_equiv, bonds in side_dict.items():
                    index = self._get_passivation_atom_index(
                        struc=layer_struc,
                        bulk_equivalent=bulk_equiv,
                        top=is_top,
                    )
                    surface_atom_info.append((index, side, bulk_equiv))

            surface_atoms, _, _ = zip(*surface_atom_info)
            surface_atom_coords = structure.cart_coords[list(surface_atoms)]

            new_bonds = {i: [] for i in surface_atom_info}

            shifts = np.array(
                [
                    [0, 0, 0],
                    [1, 0, 0],
                    [0, 1, 0],
                    [1, 1, 0],
                    [-1, 1, 0],
                    [1, -1, 0],
                    [-1, -1, 0],
                    [-1, 0, 0],
                    [0, -1, 0],
                ]
            ).dot(structure.lattice.matrix)

            all_surface_atom_coords = (
                shifts[:, None] + surface_atom_coords[None, :]
            )

            for hydrogen in hydrogen_coords:
                all_dists = np.linalg.norm(
                    hydrogen[None, None, :] - all_surface_atom_coords, axis=2
                )
                center_ind = np.where(np.isclose(all_dists, np.min(all_dists)))
                center_atom_ind = center_ind[1][0]
                bond = hydrogen - all_surface_atom_coords[center_ind]
                new_bonds[surface_atom_info[center_atom_ind]].append(bond)

            for k, v in new_bonds.items():
                if k[1] in bond_dict:
                    bond_dict[k[1]][k[2]]["bonds"] = np.vstack(v)

            return bond_dict
        else:
            raise ValueError(
                f"This is not the same termination. The passivated structure has termination={termination_index}, and the current surface has termination={self.termination_index}"
            )

    def passivate(
        self,
        bottom: bool = True,
        top: bool = True,
        passivated_struc: Union[str, None] = None,
        inplace: bool = True,
    ) -> Union[List[Structure], None]:
        """
        This function will apply pseudohydrogen passivation to all broken bonds on the surface and assign charges to the pseudo-hydrogens based
        on the equations provided in https://doi.org/10.1103/PhysRevB.85.195328. The identification of the local coordination environments is
        provided using CrystalNN in Pymatgen which is based on https://doi.org/10.1021/acs.inorgchem.0c02996.

        Examples:
            Initial passivation:
            >>> surface.passivate(bottom=True, top=True, inplace=True)

            Relaxed passivation from a CONTCAR file:
            >>> surface.passivate(bottom=True, top=True, passivated_struc="CONTCAR", inplace=True)

            Non-In-place passivation:
            >>> orthogonal_pas_structure, non_orthogonal_pas_structure = surface.passivate(bottom=True, top=True, inplace=False)



        Args:
            bottom: Determines if the bottom of the structure should be passivated
            top: Determines of the top of the structure should be passivated
            passivated_struc: File path to the CONTCAR/POSCAR file that contains the relaxed atomic positions of the pseudo-hydrogens.
                This structure must have the same miller index and termination index.
            inplace: Determines if the pseudo-hydrogens should be added in-place to the structures or if new structures should be returned.

        Returns:
            If inplace = False, the passivated orthogonal and non_orthogonal structures are returned
        """
        bond_dict = self._get_bond_dict()

        if passivated_struc is not None:
            bond_dict = self._get_passivated_bond_dict(
                bond_dict=bond_dict, relaxed_structure_file=passivated_struc
            )

        ortho_slab = self._orthogonal_slab_structure.copy()
        non_ortho_slab = self._non_orthogonal_slab_structure.copy()

        # ortho_slab.add_site_property("hydrogen_str", [""] * len(ortho_slab))
        # non_ortho_slab.add_site_property(
        #     "hydrogen_str", [""] * len(non_ortho_slab)
        # )

        if top:
            for bulk_equiv, bonds in bond_dict["top"].items():
                ortho_index = self._get_passivation_atom_index(
                    struc=ortho_slab, bulk_equivalent=bulk_equiv, top=True
                )
                non_ortho_index = self._get_passivation_atom_index(
                    struc=non_ortho_slab, bulk_equivalent=bulk_equiv, top=True
                )

                for bond, bond_str in zip(
                    bonds["bonds"], bonds["bond_strings"]
                ):
                    self._passivate(
                        ortho_slab,
                        ortho_index,
                        bond,
                        bond_str,
                        bonds["charge"],
                    )
                    self._passivate(
                        non_ortho_slab,
                        non_ortho_index,
                        bond,
                        bond_str,
                        bonds["charge"],
                    )

        if bottom:
            for bulk_equiv, bonds in bond_dict["bottom"].items():
                ortho_index = self._get_passivation_atom_index(
                    struc=ortho_slab, bulk_equivalent=bulk_equiv, top=False
                )
                non_ortho_index = self._get_passivation_atom_index(
                    struc=non_ortho_slab, bulk_equivalent=bulk_equiv, top=False
                )

                for bond, bond_str in zip(
                    bonds["bonds"], bonds["bond_strings"]
                ):
                    self._passivate(
                        ortho_slab,
                        ortho_index,
                        bond,
                        bond_str,
                        bonds["charge"],
                    )
                    self._passivate(
                        non_ortho_slab,
                        non_ortho_index,
                        bond,
                        bond_str,
                        bonds["charge"],
                    )

        ortho_slab.sort()
        non_ortho_slab.sort()

        self._passivated = True

        if inplace:
            self._orthogonal_slab_structure = ortho_slab
            self._non_orthogonal_slab_structure = non_ortho_slab
        else:
            return ortho_slab, non_ortho_slab

    def get_termination(self):
        """
        Returns the termination of the surface as a dictionary

        Examples:
            >>> surface.get_termination()
            >>> {"bottom": "In", "top": "As"}
        """
        raise NotImplementedError


class Interface:
    """Container of Interfaces generated using the InterfaceGenerator

    The Surface class and will be used as an input to the InterfaceGenerator class,
    and it should be create exclusively using the SurfaceGenerator.

    Args:
        substrate: Surface class of the substrate material
        film: Surface class of the film material
        match: OgreMatch class of the matching interface
        interfacial_distance: Distance between the top atom of the substrate and the bottom atom of the film
        vacuum: Size of the vacuum in Angstroms
        center: Determines if the interface is centered in the vacuum

    Attributes:
        substrate (Surface): Surface class of the substrate material
        film (Surface): Surface class of the film material
        match (OgreMatch): OgreMatch class of the matching interface
        interfacial_distance (float): Distance between the top atom of the substrate and the bottom atom of the film
        vacuum (float): Size of the vacuum in Angstroms
        center (bool): Determines if the interface is centered in the vacuum
        orthogonal_structure (Structure): Pymatgen structure of the orthogonalized interface
        orthogonal_substrate_structure (Structure): Pymatgen structure of only the substrate half of the orthogonalized interface supercell
        orthogonal_film_structure (Structure): Pymatgen structure of only the film half of the orthogonalized interface supercell
        orthogonal_atoms (Atoms): ASE Atoms of the orthogonalized interface
        orthogonal_substrate_atoms (Atoms): ASE Atoms of only the substrate half of the orthogonalized interface supercell
        orthogonal_film_atoms (Atoms): ASE Atoms of only the film half of the orthogonalized interface supercell
        non_orthogonal_structure (Structure): Pymatgen structure of the interface structure that assumes the substrate basis
        non_orthogonal_substrate_structure (Structure): Pymatgen structure of only the substrate half of the non-orthogonalized interface supercell
        non_orthogonal_film_structure (Structure): Pymatgen structure of only the film half of the non-orthogonalized interface supercell
        non_orthogonal_atoms (Atoms): ASE Atoms of the interface structure that assumes the substrate basis
        non_orthogonal_substrate_atoms (Atoms): ASE Atoms of only the substrate half of the non-orthogonalized interface supercell
        non_orthogonal_film_atoms (Atoms): ASE Atoms of only the film half of the non-orthogonalized interface supercell
    """

    def __init__(
        self,
        substrate: Surface,
        film: Surface,
        match: OgreMatch,
        interfacial_distance: float,
        vacuum: float,
        center: bool = True,
    ) -> None:
        self.center = center
        self.substrate = substrate
        self.film = film
        self.match = match
        self.vacuum = vacuum
        (
            self._substrate_supercell,
            self._substrate_supercell_uvw,
            self._substrate_supercell_scale_factors,
        ) = self._prepare_substrate()
        (
            self._film_supercell,
            self._film_supercell_uvw,
            self._film_supercell_scale_factors,
        ) = self._prepare_film()
        self.interfacial_distance = interfacial_distance
        self.interface_height = None
        self._strained_sub = self._substrate_supercell
        (
            self._strained_film,
            self._stack_transformation,
        ) = self._strain_and_orient_film()

        (
            self._M_matrix,
            self._non_orthogonal_structure,
            self._non_orthogonal_substrate_structure,
            self._non_orthogonal_film_structure,
            self._non_orthogonal_atoms,
            self._non_orthogonal_substrate_atoms,
            self._non_orthogonal_film_atoms,
            self._orthogonal_structure,
            self._orthogonal_substrate_structure,
            self._orthogonal_film_structure,
            self._orthogonal_atoms,
            self._orthogonal_substrate_atoms,
            self._orthogonal_film_atoms,
        ) = self._stack_interface()
        # self.interface, self.sub_part, self.film_part = self._stack_interface()

    def get_interface(
        self,
        orthogonal: bool = True,
        return_structure: bool = True,
        return_atoms: bool = False,
    ) -> Union[Atoms, Structure]:
        """
        This is a simple function for easier access to the interface structure generated from the OgreMatch

        Args:
            orthogonal: Determines if the orthogonalized structure is returned
            return_structure: Determines if the Pymatgen Structure object is returned
            return_atoms: Determines if the ASE Atoms object is returned

        Returns:
            Either a Pymatgen Structure of ASE Atoms object of the interface structure
        """
        if orthogonal:
            if return_structure and not return_atoms:
                return self._orthogonal_structure
            elif return_atoms and not return_structure:
                return self._orthogonal_atoms
            else:
                raise ValueError(
                    "Please select either return_atoms=True OR return_structure=True to get an ASE atoms object or a Pymatgen Structure object."
                )
        else:
            if return_structure and not return_atoms:
                return self._non_orthogonal_structure
            elif return_atoms and not return_structure:
                return self._non_orthogonal_atoms
            else:
                raise ValueError(
                    "Please select either return_atoms=True OR return_structure=True to get an ASE atoms object or a Pymatgen Structure object."
                )

    def get_substrate_supercell(
        self,
        orthogonal: bool = True,
        return_structure: bool = True,
        return_atoms: bool = False,
    ):
        """
        This is a simple function for easier access to the substrate supercell generated from the OgreMatch
        (i.e. the interface structure with the film atoms removed)

        Args:
            orthogonal: Determines if the orthogonalized structure is returned
            return_structure: Determines if the Pymatgen Structure object is returned
            return_atoms: Determines if the ASE Atoms object is returned

        Returns:
            Either a Pymatgen Structure of ASE Atoms object of the substrate supercell structure
        """
        if orthogonal:
            if return_structure and not return_atoms:
                return self._orthogonal_substrate_structure
            elif return_atoms and not return_structure:
                return self._orthogonal_substrate_atoms
            else:
                raise ValueError(
                    "Please select either return_atoms=True OR return_structure=True to get an ASE atoms object or a Pymatgen Structure object."
                )
        else:
            if return_structure and not return_atoms:
                return self._non_orthogonal_substrate_structure
            elif return_atoms and not return_structure:
                return self._non_orthogonal_substrate_atoms
            else:
                raise ValueError(
                    "Please select either return_atoms=True OR return_structure=True to get an ASE atoms object or a Pymatgen Structure object."
                )

    def get_film_supercell(
        self,
        orthogonal: bool = True,
        return_structure: bool = True,
        return_atoms: bool = False,
    ):
        """
        This is a simple function for easier access to the film supercell generated from the OgreMatch
        (i.e. the interface structure with the substrate atoms removed)

        Args:
            orthogonal: Determines if the orthogonalized structure is returned
            return_structure: Determines if the Pymatgen Structure object is returned
            return_atoms: Determines if the ASE Atoms object is returned

        Returns:
            Either a Pymatgen Structure of ASE Atoms object of the film supercell structure
        """
        if orthogonal:
            if return_structure and not return_atoms:
                return self._orthogonal_film_structure
            elif return_atoms and not return_structure:
                return self._orthogonal_film_atoms
            else:
                raise ValueError(
                    "Please select either return_atoms=True OR return_structure=True to get an ASE atoms object or a Pymatgen Structure object."
                )
        else:
            if return_structure and not return_atoms:
                return self._non_orthogonal_film_structure
            elif return_atoms and not return_structure:
                return self._non_orthogonal_film_atoms
            else:
                raise ValueError(
                    "Please select either return_atoms=True OR return_structure=True to get an ASE atoms object or a Pymatgen Structure object."
                )

    @property
    def area(self) -> float:
        """
        Cross section area of the interface in Angstroms^2

        Examples:
            >>> interface.area
            >>> 205.123456

        Returns:
            Cross-section area in Angstroms^2
        """
        return self.match.area

    @property
    def _structure_volume(self):
        matrix = deepcopy(self._orthogonal_structure.lattice.matrix)
        vac_matrix = np.vstack(
            [
                matrix[:2],
                self.vacuum * (matrix[-1] / np.linalg.norm(matrix[-1])),
            ]
        )

        total_volume = np.abs(np.linalg.det(matrix))
        vacuum_volume = np.abs(np.linalg.det(vac_matrix))

        return total_volume - vacuum_volume

    @property
    def substrate_basis(self) -> np.ndarray:
        """
        Returns the miller indices of the basis vectors of the substrate supercell

        Examples:
            >>> interface.substrate_basis
            >>> [[3 1 0]
            ...  [-1 3 0]
            ...  [0 0 1]]

        Returns:
            (3, 3) numpy array containing the miller indices of each lattice vector
        """
        return self._substrate_supercell_uvw

    @property
    def substrate_a(self) -> np.ndarray:
        """
        Returns the miller indices of the a basis vector of the substrate supercell

        Examples:
            >>> interface.substrate_a
            >>> [3 1 0]

        Returns:
            (3,) numpy array containing the miller indices of the a lattice vector
        """
        return self._substrate_supercell_uvw[0]

    @property
    def substrate_b(self) -> np.ndarray:
        """
        Returns the miller indices of the b basis vector of the substrate supercell

        Examples:
            >>> interface.substrate_b
            >>> [-1 3 0]

        Returns:
            (3,) numpy array containing the miller indices of the b lattice vector
        """
        return self._substrate_supercell_uvw[1]

    @property
    def film_basis(self) -> np.ndarray:
        """
        Returns the miller indices of the basis vectors of the film supercell

        Examples:
            >>> interface.film_basis
            >>> [[1 -1 0]
            ...  [0 1 0]
            ...  [0 0 1]]

        Returns:
            (3, 3) numpy array containing the miller indices of each lattice vector
        """
        return self._film_supercell_uvw

    @property
    def film_a(self) -> np.ndarray:
        """
        Returns the miller indices of the a basis vector of the film supercell

        Examples:
            >>> interface.film_a
            >>> [1 -1 0]

        Returns:
            (3,) numpy array containing the miller indices of the a lattice vector
        """
        return self._film_supercell_uvw[0]

    @property
    def film_b(self) -> np.ndarray:
        """
        Returns the miller indices of the a basis vector of the film supercell

        Examples:
            >>> interface.film_b
            >>> [0 1 0]

        Returns:
            (3,) numpy array containing the miller indices of the b lattice vector
        """
        return self._film_supercell_uvw[1]

    def __str__(self):
        fm = self.film.miller_index
        sm = self.substrate.miller_index
        film_str = f"{self.film.formula}({fm[0]} {fm[1]} {fm[2]})"
        sub_str = f"{self.substrate.formula}({sm[0]} {sm[1]} {sm[2]})"
        s_uvw = self._substrate_supercell_uvw
        s_sf = self._substrate_supercell_scale_factors
        f_uvw = self._film_supercell_uvw
        f_sf = self._film_supercell_scale_factors
        match_a_film = (
            f"{f_sf[0]}*[{f_uvw[0][0]:2d} {f_uvw[0][1]:2d} {f_uvw[0][2]:2d}]"
        )
        match_a_sub = (
            f"{s_sf[0]}*[{s_uvw[0][0]:2d} {s_uvw[0][1]:2d} {s_uvw[0][2]:2d}]"
        )
        match_b_film = (
            f"{f_sf[1]}*[{f_uvw[1][0]:2d} {f_uvw[1][1]:2d} {f_uvw[1][2]:2d}]"
        )
        match_b_sub = (
            f"{s_sf[1]}*[{s_uvw[1][0]:2d} {s_uvw[1][1]:2d} {s_uvw[1][2]:2d}]"
        )
        return_info = [
            "Film: " + film_str,
            "Substrate: " + sub_str,
            "Epitaxial Match Along \\vec{a} (film || sub): "
            + f"({match_a_film} || {match_a_sub})",
            "Epitaxial Match Along \\vec{b} (film || sub): "
            + f"({match_b_film} || {match_b_sub})",
            "Strain Along \\vec{a} (%): "
            + f"{100*self.match.linear_strain[0]:.3f}",
            "Strain Along \\vec{b} (%): "
            + f"{100*self.match.linear_strain[1]:.3f}",
            "In-plane Angle Mismatch (%): "
            + f"{100*self.match.angle_strain:.3f}",
            "Cross Section Area (Ang^2): " + f"{self.area:.3f}",
        ]
        return_str = "\n".join(return_info)

        return return_str

    # def write_file_old(self, output: str = "POSCAR_interface"):
    #     """Write the POSCAR of the interface"""
    #     Poscar(self.interface).write_file(output)

    def write_file(
        self,
        output: str = "POSCAR_interface",
        orthogonal: bool = True,
        relax: bool = False,
        film_layers_to_relax: int = 1,
        substrate_layers_to_relax: int = 1,
    ):
        """
        Write the POSCAR of the interface

        Args:
            output: File path of the output POSCAR
            orthogonal: Determines of the orthogonal structure is written
            relax: Determines if selective dynamics is applied to the atoms at the interface
            film_layers_to_relax: Number of unit cell layers near the interface to relax
            substrate_layers_to_relax: Number of unit cell layers near the interface to relax
        """
        if orthogonal:
            slab = self._orthogonal_structure
        else:
            slab = self._non_orthogonal_structure

        comment = "|".join(
            [
                f"Lf={self.film.layers}",
                f"Ls={self.substrate.layers}",
                f"Tf={self.film.termination_index}",
                f"Ts={self.substrate.termination_index}",
                f"O={orthogonal}",
                f"d={self.interfacial_distance:.2f}",
            ]
        )

        if relax:
            comment += "|" + "|".join(
                [
                    f"Rf={film_layers_to_relax}",
                    f"Rs={substrate_layers_to_relax}",
                ]
            )
            film_layers = np.arange(film_layers_to_relax)
            sub_layers = np.arange(
                self.substrate.layers - substrate_layers_to_relax,
                self.substrate.layers,
            )
            layer_index = np.array(slab.site_properties["layer_index"])
            is_sub = np.array(slab.site_properties["is_sub"])
            is_film = np.array(slab.site_properties["is_film"])
            film_to_relax = np.logical_and(
                is_film, np.isin(layer_index, film_layers)
            )
            sub_to_relax = np.logical_and(
                is_sub, np.isin(layer_index, sub_layers)
            )
            to_relax = np.repeat(
                np.logical_or(sub_to_relax, film_to_relax).reshape(-1, 1),
                repeats=3,
                axis=1,
            )

        if not self.substrate._passivated and not self.film._passivated:
            poscar = Poscar(slab, comment=comment)

            if relax:
                poscar.selective_dynamics = to_relax

            poscar_str = poscar.get_string()

        else:
            syms = [site.specie.symbol for site in slab]

            syms = []
            for site in slab:
                if site.specie.symbol == "H":
                    if hasattr(site.specie, "oxi_state"):
                        oxi = site.specie.oxi_state

                        if oxi < 1.0:
                            H_str = "H" + f"{oxi:.2f}"[1:]
                        elif oxi > 1.0:
                            H_str = "H" + f"{oxi:.2f}"
                        else:
                            H_str = "H"

                        syms.append(H_str)
                else:
                    syms.append(site.specie.symbol)

            comp_list = [(a[0], len(list(a[1]))) for a in groupby(syms)]
            atom_types, n_atoms = zip(*comp_list)

            new_atom_types = []
            for atom in atom_types:
                if "H" == atom[0] and atom not in ["Hf", "Hs", "Hg", "He"]:
                    new_atom_types.append("H")
                else:
                    new_atom_types.append(atom)

            comment += "|potcar=" + " ".join(atom_types)

            poscar = Poscar(slab, comment=comment)

            if relax:
                poscar.selective_dynamics = to_relax

            poscar_str = poscar.get_string().split("\n")
            poscar_str[5] = " ".join(new_atom_types)
            poscar_str[6] = " ".join(list(map(str, n_atoms)))
            poscar_str = "\n".join(poscar_str)

        with open(output, "w") as f:
            f.write(poscar_str)

    def shift_film(
        self,
        shift: Iterable,
        fractional: bool = False,
        inplace: bool = False,
        return_atoms: bool = False,
    ) -> Union[Structure, None]:
        """
        Shifts the film over the substrate by a given shift vector.

        Examples:
            In-place shift using fractional coordinates:
            >>> interface.shift_film(shift=[0.5, 0.25, 0.0], fractional=True, inplace=True)

            Non in-place shift using cartesian coordinates:
            >>> shifted_interface = interface.shift_film(shift=[4.5, 0.0, 0.5], fractional=False, inplace=False)

        Args:
            shift: 3-element vector defining the shift in the x (a), y (b), and z (c) directions
            fractional: Determines if the shift is defined in fractional coordinates
            inplace: Determines of the shift happens inplace or if a new structure is returned
            return_atoms: Determine if the returned structure is an ASE Atoms object instead of a Structure object

        Returns:
            Shifted interface Structure if inplace=False or Shifted interface Atoms if inplace=False and return_atoms=True
        """
        if fractional:
            frac_shift = np.array(shift)
        else:
            shift = np.array(shift)

            if shift[-1] + self.interfacial_distance < 0.5:
                raise ValueError(
                    f"The film shift results in an interfacial distance of less than 0.5 Angstroms which is non-physical"
                )

            frac_shift = (
                self._orthogonal_structure.lattice.get_fractional_coords(shift)
            )

        film_ind = np.where(
            self._orthogonal_structure.site_properties["is_film"]
        )[0]

        if inplace:
            self._orthogonal_structure.translate_sites(
                film_ind,
                frac_shift,
            )
            self._orthogonal_film_structure.translate_sites(
                range(len(self._orthogonal_film_structure)),
                frac_shift,
            )
            self.interface_height += frac_shift[-1] / 2
            self.interfacial_distance += shift[-1]

        else:
            shifted_interface = self._orthogonal_structure.copy()
            shifted_interface.translate_sites(
                film_ind,
                frac_shift,
            )

            if return_atoms:
                return AseAtomsAdaptor().get_atoms(shifted_interface)
            else:
                return shifted_interface

    def _prepare_substrate(self):
        matrix = self.match.substrate_sl_transform
        supercell_slab = self.substrate._non_orthogonal_slab_structure.copy()
        supercell_slab.make_supercell(scaling_matrix=matrix)

        uvw_supercell = matrix @ self.substrate.uvw_basis
        scale_factors = []
        for i, b in enumerate(uvw_supercell):
            scale = np.abs(reduce(utils._float_gcd, b))
            uvw_supercell[i] = uvw_supercell[i] / scale
            scale_factors.append(scale)

        return supercell_slab, uvw_supercell, scale_factors

    def _prepare_substrate_old(self):
        matrix = self.match.substrate_sl_transform
        supercell_slab = self.substrate._orthogonal_slab_structure.copy()
        supercell_slab.make_supercell(scaling_matrix=matrix)

        uvw_supercell = matrix @ self.substrate.uvw_basis
        scale_factors = []
        for i, b in enumerate(uvw_supercell):
            scale = np.abs(reduce(utils._float_gcd, b))
            uvw_supercell[i] = uvw_supercell[i] / scale
            scale_factors.append(scale)

        return supercell_slab, uvw_supercell, scale_factors

    def _prepare_film(self):
        matrix = self.match.film_sl_transform
        supercell_slab = self.film._non_orthogonal_slab_structure.copy()
        supercell_slab.make_supercell(scaling_matrix=matrix)

        sc_matrix = supercell_slab.lattice.matrix
        sub_non_orth_c_vec = (
            self.substrate._non_orthogonal_slab_structure.lattice.matrix[-1]
        )
        sub_non_orth_c_norm = sub_non_orth_c_vec / np.linalg.norm(
            sub_non_orth_c_vec
        )

        norm = self.film.surface_normal
        proj = np.dot(norm, sub_non_orth_c_norm)
        scale = self.film._orthogonal_slab_structure.lattice.c / proj

        new_matrix = np.vstack([sc_matrix[:2], sub_non_orth_c_norm * scale])

        oriented_supercell_slab = Structure(
            lattice=Lattice(new_matrix),
            species=supercell_slab.species,
            coords=supercell_slab.cart_coords,
            coords_are_cartesian=True,
            to_unit_cell=True,
            site_properties=supercell_slab.site_properties,
        )

        uvw_supercell = matrix @ self.film.uvw_basis
        scale_factors = []
        for i, b in enumerate(uvw_supercell):
            scale = np.abs(reduce(utils._float_gcd, b))
            uvw_supercell[i] = uvw_supercell[i] / scale
            scale_factors.append(scale)

        return oriented_supercell_slab, uvw_supercell, scale_factors

    def _prepare_film_old(self):
        matrix = self.match.film_sl_transform
        supercell_slab = self.film._orthogonal_slab_structure.copy()
        supercell_slab.make_supercell(scaling_matrix=matrix)

        uvw_supercell = matrix @ self.film.uvw_basis
        scale_factors = []
        for i, b in enumerate(uvw_supercell):
            scale = np.abs(reduce(utils._float_gcd, b))
            uvw_supercell[i] = uvw_supercell[i] / scale
            scale_factors.append(scale)

        return supercell_slab, uvw_supercell, scale_factors

    def _strain_and_orient_film(self):
        sub_in_plane_vecs = self._substrate_supercell.lattice.matrix[:2]
        film_out_of_plane = self._film_supercell.lattice.matrix[-1]
        film_inv_matrix = self._film_supercell.lattice.inv_matrix
        new_matrix = np.vstack([sub_in_plane_vecs, film_out_of_plane])
        transform = (film_inv_matrix @ new_matrix).T
        op = SymmOp.from_rotation_and_translation(
            transform, translation_vec=np.zeros(3)
        )

        strained_film = deepcopy(self._film_supercell)
        strained_film.apply_operation(op)

        return strained_film, transform

    def _stack_interface(self):
        # Get the strained substrate and film
        strained_sub = self._strained_sub
        strained_film = self._strained_film

        # Get the oriented bulk structure of the substrate
        oriented_bulk_c = self.substrate.oriented_bulk_structure.lattice.c

        # Get the normalized projection of the substrate c-vector onto the normal vector,
        # This is used to determine the length of the non-orthogonal c-vector in order to get
        # the correct vacuum size.
        c_norm_proj = self.substrate.c_projection / oriented_bulk_c

        # Get the substrate matrix and c-vector
        sub_matrix = strained_sub.lattice.matrix
        sub_c = deepcopy(sub_matrix[-1])

        # Get the fractional and cartesian coordinates of the substrate and film
        strained_sub_coords = deepcopy(strained_sub.cart_coords)
        strained_film_coords = deepcopy(strained_film.cart_coords)
        strained_sub_frac_coords = deepcopy(strained_sub.frac_coords)
        strained_film_frac_coords = deepcopy(strained_film.frac_coords)

        # Find the min and max coordinates of the substrate and film
        min_sub_coords = np.min(strained_sub_frac_coords[:, -1])
        max_sub_coords = np.max(strained_sub_frac_coords[:, -1])
        min_film_coords = np.min(strained_film_frac_coords[:, -1])
        max_film_coords = np.max(strained_film_frac_coords[:, -1])

        # Get the lengths of the c-vetors of the substrate and film
        sub_c_len = strained_sub.lattice.c
        film_c_len = strained_film.lattice.c

        # Find the total length of the interface structure including the interfacial distance
        interface_structure_len = np.sum(
            [
                (max_sub_coords - min_sub_coords) * sub_c_len,
                (max_film_coords - min_film_coords) * film_c_len,
                self.interfacial_distance / c_norm_proj,
            ]
        )

        # Find the length of the vacuum region in the non-orthogonal basis
        interface_vacuum_len = self.vacuum / c_norm_proj

        # The total length of the interface c-vector should be the length of the structure + length of the vacuum
        # This will get changed in the next line to be exactly an integer multiple of the
        # oriented bulk cell of the substrate
        init_interface_c_len = interface_structure_len + interface_vacuum_len

        # Find the closest integer multiple of the substrate oriented bulk c-vector length
        n_unit_cell = int(np.ceil(init_interface_c_len / oriented_bulk_c))

        # Make the new interface c-vector length an integer multiple of the oriented bulk c-vector
        interface_c_len = oriented_bulk_c * n_unit_cell

        # Create the transformation matrix from the primtive bulk structure to the interface unit cell
        # this is only needed for band unfolding purposes
        sub_M = self.substrate.transformation_matrix
        layer_M = np.eye(3).astype(int)
        layer_M[-1, -1] = n_unit_cell
        interface_M = layer_M @ self.match.substrate_sl_transform @ sub_M

        # Create the new interface lattice vectors
        interface_matrix = np.vstack(
            [sub_matrix[:2], interface_c_len * (sub_c / sub_c_len)]
        )
        interface_lattice = Lattice(matrix=interface_matrix)

        # Convert the interfacial distance into fractional coordinated because they are easier to work with
        frac_int_distance_shift = np.array(
            [0, 0, self.interfacial_distance]
        ).dot(interface_lattice.inv_matrix)

        interface_inv_matrix = interface_lattice.inv_matrix

        # Convert the substrate cartesian coordinates into the interface fractional coordinates
        # and shift the bottom atom c-position to zero
        sub_interface_coords = strained_sub_coords.dot(interface_inv_matrix)
        sub_interface_coords[:, -1] -= sub_interface_coords[:, -1].min()

        # Convert the film cartesian coordinates into the interface fractional coordinates
        # and shift the bottom atom c-position to the top substrate c-position + interfacial distance
        film_interface_coords = strained_film_coords.dot(interface_inv_matrix)
        film_interface_coords[:, -1] -= film_interface_coords[:, -1].min()
        film_interface_coords[:, -1] += sub_interface_coords[:, -1].max()
        film_interface_coords += frac_int_distance_shift

        # Combine the coodinates, species, and site_properties to make the interface Structure
        interface_coords = np.r_[sub_interface_coords, film_interface_coords]
        interface_species = strained_sub.species + strained_film.species
        interface_site_properties = {
            key: strained_sub.site_properties[key]
            + strained_film.site_properties[key]
            for key in strained_sub.site_properties
        }
        interface_site_properties["is_sub"] = np.array(
            [True] * len(strained_sub) + [False] * len(strained_film)
        )
        interface_site_properties["is_film"] = np.array(
            [False] * len(strained_sub) + [True] * len(strained_film)
        )

        # Create the non-orthogonalized interface structure
        non_ortho_interface_struc = Structure(
            lattice=interface_lattice,
            species=interface_species,
            coords=interface_coords,
            to_unit_cell=True,
            coords_are_cartesian=False,
            site_properties=interface_site_properties,
        )
        non_ortho_interface_struc.sort()

        self.interface_height = sub_interface_coords[:, -1].max() + (
            0.5 * frac_int_distance_shift[-1]
        )

        if self.center:
            # Get the new vacuum length, needed for shifting
            vacuum_len = interface_c_len - interface_structure_len

            # Find the fractional coordinates of shifting the structure up by half the amount of vacuum cells
            center_shift = np.ceil(vacuum_len / oriented_bulk_c) // 2
            center_shift *= oriented_bulk_c / interface_c_len

            # Center the structure in the vacuum
            non_ortho_interface_struc.translate_sites(
                indices=range(len(non_ortho_interface_struc)),
                vector=[0.0, 0.0, center_shift],
                frac_coords=True,
                to_unit_cell=True,
            )
            self.interface_height += center_shift

        # Get the frac coords of the non-orthogonalized interface
        frac_coords = non_ortho_interface_struc.frac_coords

        # Find the max c-coord of the substrate
        # This is used to shift the x-y positions of the interface structure so the top atom of the substrate
        # is located at x=0, y=0. This will have no effect of the properties of the interface since all the
        # atoms are shifted, it is more of a visual thing to make the interfaces look nice.
        is_sub = np.array(non_ortho_interface_struc.site_properties["is_sub"])
        sub_frac_coords = frac_coords[is_sub]
        max_c = np.max(sub_frac_coords[:, -1])

        # Find the xy-shift in cartesian coordinates
        cart_shift = np.array([0.0, 0.0, max_c]).dot(
            non_ortho_interface_struc.lattice.matrix
        )
        cart_shift[-1] = 0.0

        # Get the projection of the non-orthogonal c-vector onto the surface normal
        proj_c = np.dot(
            self.substrate.surface_normal,
            non_ortho_interface_struc.lattice.matrix[-1],
        )

        # Get the orthogonalized c-vector of the interface (this conserves the vacuum, but breaks symmetries)
        ortho_c = self.substrate.surface_normal * proj_c

        # Create the orthogonalized lattice vectors
        new_matrix = np.vstack(
            [
                non_ortho_interface_struc.lattice.matrix[:2],
                ortho_c,
            ]
        )

        # Create the orthogonalized structure
        ortho_interface_struc = Structure(
            lattice=Lattice(new_matrix),
            species=non_ortho_interface_struc.species,
            coords=non_ortho_interface_struc.cart_coords,
            site_properties=non_ortho_interface_struc.site_properties,
            to_unit_cell=True,
            coords_are_cartesian=True,
        )

        # Shift the structure so the top substrate atom's x and y postions are zero, similar to the non-orthogonalized structure
        ortho_interface_struc.translate_sites(
            indices=range(len(ortho_interface_struc)),
            vector=-cart_shift,
            frac_coords=False,
            to_unit_cell=True,
        )

        # The next step is used extract on the film and substrate portions of the interface
        # These can be used for charge transfer calculation
        film_inds = np.where(
            non_ortho_interface_struc.site_properties["is_film"]
        )[0]
        sub_inds = np.where(
            non_ortho_interface_struc.site_properties["is_sub"]
        )[0]

        non_ortho_film_structure = non_ortho_interface_struc.copy()
        non_ortho_film_structure.remove_sites(sub_inds)

        non_ortho_sub_structure = non_ortho_interface_struc.copy()
        non_ortho_sub_structure.remove_sites(film_inds)

        ortho_film_structure = ortho_interface_struc.copy()
        ortho_film_structure.remove_sites(sub_inds)

        ortho_sub_structure = ortho_interface_struc.copy()
        ortho_sub_structure.remove_sites(film_inds)

        non_ortho_interface_atoms = AseAtomsAdaptor().get_atoms(
            non_ortho_interface_struc
        )
        ortho_interface_atoms = AseAtomsAdaptor().get_atoms(
            ortho_interface_struc
        )
        non_ortho_sub_atoms = AseAtomsAdaptor().get_atoms(
            non_ortho_sub_structure
        )
        non_ortho_film_atoms = AseAtomsAdaptor().get_atoms(
            non_ortho_film_structure
        )
        ortho_sub_atoms = AseAtomsAdaptor().get_atoms(ortho_sub_structure)
        ortho_film_atoms = AseAtomsAdaptor().get_atoms(ortho_film_structure)

        return (
            interface_M,
            non_ortho_interface_struc,
            non_ortho_sub_structure,
            non_ortho_film_structure,
            non_ortho_interface_atoms,
            non_ortho_sub_atoms,
            non_ortho_film_atoms,
            ortho_interface_struc,
            ortho_sub_structure,
            ortho_film_structure,
            ortho_interface_atoms,
            ortho_sub_atoms,
            ortho_film_atoms,
        )

    def _stack_interface_messy(self):
        strained_sub = self._strained_sub
        strained_film = self._strained_film

        oriented_bulk_c = self.substrate.oriented_bulk_structure.lattice.c

        c_norm_proj = self.substrate.c_projection / oriented_bulk_c

        sub_matrix = strained_sub.lattice.matrix
        sub_c = deepcopy(sub_matrix[-1])

        strained_sub_coords = deepcopy(strained_sub.cart_coords)
        strained_film_coords = deepcopy(strained_film.cart_coords)
        strained_sub_frac_coords = deepcopy(strained_sub.frac_coords)
        strained_film_frac_coords = deepcopy(strained_film.frac_coords)

        min_sub_coords = np.min(strained_sub_frac_coords[:, -1])
        max_sub_coords = np.max(strained_sub_frac_coords[:, -1])
        min_film_coords = np.min(strained_film_frac_coords[:, -1])
        max_film_coords = np.max(strained_film_frac_coords[:, -1])

        sub_c_len = np.linalg.norm(strained_sub.lattice.matrix[-1])
        film_c_len = np.linalg.norm(strained_film.lattice.matrix[-1])
        interface_structure_len = np.sum(
            [
                (max_sub_coords - min_sub_coords) * sub_c_len,
                (max_film_coords - min_film_coords) * film_c_len,
                self.interfacial_distance / c_norm_proj,
            ]
        )
        interface_vacuum_len = self.vacuum / c_norm_proj

        interface_c_len = interface_structure_len + interface_vacuum_len

        n_unit_cell = np.ceil(interface_c_len / oriented_bulk_c).astype(int)
        new_interface_c_len = oriented_bulk_c * n_unit_cell
        new_vacuum_len = new_interface_c_len - interface_structure_len

        center_shift = np.ceil(
            np.ceil(new_vacuum_len / oriented_bulk_c) / 2
        ).astpye(int)
        print(center_shift)
        center_shift *= oriented_bulk_c / new_interface_c_len

        sub_M = self.substrate.transformation_matrix
        layer_M = np.eye(3).astype(int)
        layer_M[-1, -1] = n_unit_cell
        interface_M = layer_M @ self.match.substrate_sl_transform @ sub_M

        print(layer_M @ sub_M)

        interface_matrix = np.vstack(
            [sub_matrix[:2], new_interface_c_len * (sub_c / sub_c_len)]
        )
        interface_lattice = Lattice(matrix=interface_matrix)
        frac_int_distance_shift = np.array(
            [0, 0, self.interfacial_distance]
        ).dot(interface_lattice.inv_matrix)
        interface_inv_matrix = interface_lattice.inv_matrix

        sub_interface_coords = strained_sub_coords.dot(interface_inv_matrix)
        sub_interface_coords[:, -1] -= sub_interface_coords[:, -1].min()

        film_interface_coords = strained_film_coords.dot(interface_inv_matrix)
        film_interface_coords[:, -1] -= film_interface_coords[:, -1].min()
        film_interface_coords[:, -1] += sub_interface_coords[:, -1].max()
        film_interface_coords += frac_int_distance_shift

        interface_coords = np.r_[sub_interface_coords, film_interface_coords]
        interface_species = strained_sub.species + strained_film.species
        interface_site_properties = {
            key: strained_sub.site_properties[key]
            + strained_film.site_properties[key]
            for key in strained_sub.site_properties
        }
        interface_site_properties["is_sub"] = np.array(
            [True] * len(strained_sub) + [False] * len(strained_film)
        )
        interface_site_properties["is_film"] = np.array(
            [False] * len(strained_sub) + [True] * len(strained_film)
        )

        self.interface_height = sub_interface_coords[:, -1].max() + (
            0.5 * frac_int_distance_shift[-1]
        )

        non_ortho_interface_struc = Structure(
            lattice=interface_lattice,
            species=interface_species,
            coords=interface_coords,
            to_unit_cell=True,
            coords_are_cartesian=False,
            site_properties=interface_site_properties,
        )
        non_ortho_interface_struc.sort()

        if self.center:
            non_ortho_interface_struc.translate_sites(
                indices=range(len(non_ortho_interface_struc)),
                vector=[0.0, 0.0, center_shift],
                frac_coords=True,
                to_unit_cell=True,
            )
            self.interface_height += center_shift

        frac_coords = non_ortho_interface_struc.frac_coords
        is_sub = np.array(non_ortho_interface_struc.site_properties["is_sub"])
        sub_frac_coords = frac_coords[is_sub]
        # max_c = sub_frac_coords[np.argmax(sub_frac_coords[:, -1])]
        max_c = np.max(sub_frac_coords[:, -1])

        # ortho_interface_struc = interface_struc.copy()
        cart_shift = np.array([0.0, 0.0, max_c]).dot(
            non_ortho_interface_struc.lattice.matrix
        )
        cart_shift[-1] = 0.0
        proj_c = np.dot(
            self.substrate.surface_normal,
            non_ortho_interface_struc.lattice.matrix[-1],
        )
        ortho_c = self.substrate.surface_normal * proj_c
        new_matrix = np.vstack(
            [
                non_ortho_interface_struc.lattice.matrix[:2],
                ortho_c,
            ]
        )

        ortho_interface_struc = Structure(
            lattice=Lattice(new_matrix),
            species=non_ortho_interface_struc.species,
            coords=non_ortho_interface_struc.cart_coords,
            site_properties=non_ortho_interface_struc.site_properties,
            to_unit_cell=True,
            coords_are_cartesian=True,
        )

        ortho_interface_struc.translate_sites(
            indices=range(len(ortho_interface_struc)),
            vector=-cart_shift,
            frac_coords=False,
            to_unit_cell=True,
        )
        # non_ortho_interface_struc.translate_sites(
        #     indices=range(len(non_ortho_interface_struc)),
        #     vector=-np.array([max_c[0], max_c[1], 0.0]),
        #     frac_coords=True,
        #     to_unit_cell=True,
        # )

        film_inds = np.where(
            non_ortho_interface_struc.site_properties["is_film"]
        )[0]
        sub_inds = np.where(
            non_ortho_interface_struc.site_properties["is_sub"]
        )[0]

        non_ortho_film_structure = non_ortho_interface_struc.copy()
        non_ortho_film_structure.remove_sites(sub_inds)

        non_ortho_sub_structure = non_ortho_interface_struc.copy()
        non_ortho_sub_structure.remove_sites(film_inds)

        ortho_film_structure = ortho_interface_struc.copy()
        ortho_film_structure.remove_sites(sub_inds)

        ortho_sub_structure = ortho_interface_struc.copy()
        ortho_sub_structure.remove_sites(film_inds)

        non_ortho_interface_atoms = AseAtomsAdaptor().get_atoms(
            non_ortho_interface_struc
        )
        ortho_interface_atoms = AseAtomsAdaptor().get_atoms(
            ortho_interface_struc
        )
        non_ortho_sub_atoms = AseAtomsAdaptor().get_atoms(
            non_ortho_sub_structure
        )
        non_ortho_film_atoms = AseAtomsAdaptor().get_atoms(
            non_ortho_film_structure
        )
        ortho_sub_atoms = AseAtomsAdaptor().get_atoms(ortho_sub_structure)
        ortho_film_atoms = AseAtomsAdaptor().get_atoms(ortho_film_structure)

        return (
            non_ortho_interface_struc,
            non_ortho_sub_structure,
            non_ortho_film_structure,
            non_ortho_interface_atoms,
            non_ortho_sub_atoms,
            non_ortho_film_atoms,
            ortho_interface_struc,
            ortho_sub_structure,
            ortho_film_structure,
            ortho_interface_atoms,
            ortho_sub_atoms,
            ortho_film_atoms,
        )

    def _stack_interface_old(self):
        strained_sub = self._strained_sub
        strained_film = self._strained_film

        sub_matrix = strained_sub.lattice.matrix
        sub_c = deepcopy(sub_matrix[-1])

        strained_sub_coords = deepcopy(strained_sub.cart_coords)
        strained_film_coords = deepcopy(strained_film.cart_coords)
        strained_sub_frac_coords = deepcopy(strained_sub.frac_coords)
        strained_film_frac_coords = deepcopy(strained_film.frac_coords)

        min_sub_coords = np.min(strained_sub_frac_coords[:, -1])
        max_sub_coords = np.max(strained_sub_frac_coords[:, -1])
        min_film_coords = np.min(strained_film_frac_coords[:, -1])
        max_film_coords = np.max(strained_film_frac_coords[:, -1])

        sub_c_len = np.linalg.norm(strained_sub.lattice.matrix[-1])
        film_c_len = np.linalg.norm(strained_film.lattice.matrix[-1])
        interface_c_len = np.sum(
            [
                (max_sub_coords - min_sub_coords) * sub_c_len,
                (max_film_coords - min_film_coords) * film_c_len,
                self.vacuum,
                self.interfacial_distance,
            ]
        )
        frac_int_distance = self.interfacial_distance / interface_c_len

        interface_matrix = np.vstack(
            [sub_matrix[:2], interface_c_len * (sub_c / sub_c_len)]
        )
        interface_lattice = Lattice(matrix=interface_matrix)
        interface_inv_matrix = interface_lattice.inv_matrix

        sub_interface_coords = strained_sub_coords.dot(interface_inv_matrix)
        sub_interface_coords[:, -1] -= sub_interface_coords[:, -1].min()

        film_interface_coords = strained_film_coords.dot(interface_inv_matrix)
        film_interface_coords[:, -1] -= film_interface_coords[:, -1].min()
        film_interface_coords[:, -1] += (
            sub_interface_coords[:, -1].max() + frac_int_distance
        )

        interface_coords = np.r_[sub_interface_coords, film_interface_coords]
        interface_species = strained_sub.species + strained_film.species
        interface_site_properties = {
            key: strained_sub.site_properties[key]
            + strained_film.site_properties[key]
            for key in strained_sub.site_properties
        }
        interface_site_properties["is_sub"] = np.array(
            [True] * len(strained_sub) + [False] * len(strained_film)
        )
        interface_site_properties["is_film"] = np.array(
            [False] * len(strained_sub) + [True] * len(strained_film)
        )

        self.interface_height = sub_interface_coords[:, -1].max() + (
            0.5 * frac_int_distance
        )

        interface_struc = Structure(
            lattice=interface_lattice,
            species=interface_species,
            coords=interface_coords,
            to_unit_cell=True,
            coords_are_cartesian=False,
            site_properties=interface_site_properties,
        )
        interface_struc.sort()

        if self.center:
            interface_struc.translate_sites(
                indices=range(len(interface_struc)),
                vector=[0, 0, 0.5 - self.interface_height],
            )
            self.interface_height = 0.5

        film_inds = np.where(interface_struc.site_properties["is_film"])[0]
        sub_inds = np.where(interface_struc.site_properties["is_sub"])[0]

        film_part = interface_struc.copy()
        film_part.remove_sites(sub_inds)

        sub_part = interface_struc.copy()
        sub_part.remove_sites(film_inds)

        return interface_struc, sub_part, film_part

    @property
    def _metallic_elements(self):
        elements_list = np.array(
            [
                "Li",
                "Be",
                "Na",
                "Mg",
                "Al",
                "K",
                "Ca",
                "Sc",
                "Ti",
                "V",
                "Cr",
                "Mn",
                "Fe",
                "Co",
                "Ni",
                "Cu",
                "Zn",
                "Ga",
                "Rb",
                "Sr",
                "Y",
                "Zr",
                "Nb",
                "Mo",
                "Tc",
                "Ru",
                "Rh",
                "Pd",
                "Ag",
                "Cd",
                "In",
                "Sn",
                "Cs",
                "Ba",
                "La",
                "Ce",
                "Pr",
                "Nd",
                "Pm",
                "Sm",
                "Eu",
                "Gd",
                "Tb",
                "Dy",
                "Ho",
                "Er",
                "Tm",
                "Yb",
                "Lu",
                "Hf",
                "Ta",
                "W",
                "Re",
                "Os",
                "Ir",
                "Pt",
                "Au",
                "Hg",
                "Tl",
                "Pb",
                "Bi",
                "Rn",
                "Fr",
                "Ra",
                "Ac",
                "Th",
                "Pa",
                "U",
                "Np",
                "Pu",
                "Am",
                "Cm",
                "Bk",
                "Cf",
                "Es",
                "Fm",
                "Md",
                "No",
                "Lr",
                "Rf",
                "Db",
                "Sg",
                "Bh",
                "Hs",
                "Mt",
                "Ds ",
                "Rg ",
                "Cn ",
                "Nh",
                "Fl",
                "Mc",
                "Lv",
            ]
        )
        return elements_list

    def _get_radii(self):
        sub_species = np.unique(
            np.array(self.substrate.bulk_structure.species, dtype=str)
        )
        film_species = np.unique(
            np.array(self.film.bulk_structure.species, dtype=str)
        )

        sub_elements = [Element(s) for s in sub_species]
        film_elements = [Element(f) for f in film_species]

        sub_metal = np.isin(sub_species, self._metallic_elements)
        film_metal = np.isin(film_species, self._metallic_elements)

        if sub_metal.all():
            sub_dict = {
                sub_species[i]: sub_elements[i].metallic_radius
                for i in range(len(sub_elements))
            }
        else:
            Xs = [e.X for e in sub_elements]
            X_diff = np.abs([c[0] - c[1] for c in combinations(Xs, 2)])
            if (X_diff >= 1.7).any():
                sub_dict = {
                    sub_species[i]: sub_elements[i].average_ionic_radius
                    for i in range(len(sub_elements))
                }
            else:
                sub_dict = {s: CovalentRadius.radius[s] for s in sub_species}

        if film_metal.all():
            film_dict = {
                film_species[i]: film_elements[i].metallic_radius
                for i in range(len(film_elements))
            }
        else:
            Xs = [e.X for e in film_elements]
            X_diff = np.abs([c[0] - c[1] for c in combinations(Xs, 2)])
            if (X_diff >= 1.7).any():
                film_dict = {
                    film_species[i]: film_elements[i].average_ionic_radius
                    for i in range(len(film_elements))
                }
            else:
                film_dict = {f: CovalentRadius.radius[f] for f in film_species}

        sub_dict.update(film_dict)

        return sub_dict

    def _generate_sc_for_interface_view(self, struc, transformation_matrix):
        plot_struc = Structure(
            lattice=struc.lattice,
            species=["H"],
            coords=np.zeros((1, 3)),
            to_unit_cell=True,
            coords_are_cartesian=True,
        )
        plot_struc.make_supercell(transformation_matrix)
        inv_matrix = plot_struc.lattice.inv_matrix

        return plot_struc, inv_matrix

    def _plot_interface_view(
        self,
        ax,
        zero_coord,
        supercell_shift,
        cell_vetices,
        slab_matrix,
        sc_inv_matrix,
        facecolor,
        edgecolor,
        is_film=False,
    ):
        cart_coords = (
            zero_coord + supercell_shift + cell_vetices.dot(slab_matrix)
        )
        fc = np.round(cart_coords.dot(sc_inv_matrix), 3)
        if is_film:
            plot_coords = cart_coords.dot(self._stack_transformation.T)
            linewidth = 1.0
        else:
            plot_coords = cart_coords
            linewidth = 2.0

        center = np.round(
            np.mean(cart_coords[:-1], axis=0).dot(sc_inv_matrix),
            3,
        )
        center_in = np.logical_and(-0.0001 <= center[:2], center[:2] <= 1.0001)

        x_in = np.logical_and(fc[:, 0] > 0.0, fc[:, 0] < 1.0)
        y_in = np.logical_and(fc[:, 1] > 0.0, fc[:, 1] < 1.0)
        point_in = np.logical_and(x_in, y_in)

        if point_in.any() or center_in.all():
            poly = Polygon(
                xy=plot_coords[:, :2],
                closed=True,
                facecolor=facecolor,
                edgecolor=edgecolor,
                linewidth=linewidth,
            )
            ax.add_patch(poly)

    def plot_interface(
        self,
        output: str = "interface_view.png",
        dpi: int = 400,
        show_in_colab: bool = False,
    ):
        """
        This function will show the relative alignment of the film and substrate supercells by plotting the in-plane unit cells on top of each other

        Args:
            output: File path for the output image
            dpi: dpi (dots per inch) of the output image.
                Setting dpi=100 gives reasonably sized images when viewed in colab notebook
            show_in_colab: Determines if the matplotlib figure is closed or not after the plot if made.
                if show_in_colab=True the plot will show up after you run the cell in colab/jupyter notebook.
        """
        sub_matrix = self.substrate._orthogonal_slab_structure.lattice.matrix
        film_matrix = self.film._orthogonal_slab_structure.lattice.matrix
        sub_sc_matrix = deepcopy(self._substrate_supercell.lattice.matrix)
        film_sc_matrix = deepcopy(self._film_supercell.lattice.matrix)

        coords = np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [1, 1, 0],
                [0, 1, 0],
                [0, 0, 0],
            ]
        )

        sc_shifts = np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [-1, 0, 0],
                [0, -1, 0],
                [1, 1, 0],
                [-1, -1, 0],
                [1, -1, 0],
                [-1, 1, 0],
            ]
        )

        sub_sc_shifts = sc_shifts.dot(sub_sc_matrix)
        film_sc_shifts = sc_shifts.dot(film_sc_matrix)
        sub_sl = coords.dot(sub_sc_matrix)

        sub_struc, sub_inv_matrix = self._generate_sc_for_interface_view(
            struc=self.substrate._orthogonal_slab_structure,
            transformation_matrix=self.match.substrate_sl_transform,
        )

        film_struc, film_inv_matrix = self._generate_sc_for_interface_view(
            struc=self.film._orthogonal_slab_structure,
            transformation_matrix=self.match.film_sl_transform,
        )

        fig, ax = plt.subplots(figsize=(4, 4), dpi=dpi)

        for c in sub_struc.cart_coords:
            for shift in sub_sc_shifts:
                self._plot_interface_view(
                    ax=ax,
                    zero_coord=c,
                    supercell_shift=shift,
                    cell_vetices=coords,
                    slab_matrix=sub_matrix,
                    sc_inv_matrix=sub_inv_matrix,
                    is_film=False,
                    facecolor=(0, 0, 1, 0.2),
                    edgecolor=(0, 0, 1, 1),
                )

        for c in film_struc.cart_coords:
            for shift in film_sc_shifts:
                self._plot_interface_view(
                    ax=ax,
                    zero_coord=c,
                    supercell_shift=shift,
                    cell_vetices=coords,
                    slab_matrix=film_matrix,
                    sc_inv_matrix=film_inv_matrix,
                    is_film=True,
                    facecolor=(200 / 255, 0, 0, 0.2),
                    edgecolor=(200 / 255, 0, 0, 1),
                )

        ax.plot(
            sub_sl[:, 0],
            sub_sl[:, 1],
            color="black",
            linewidth=3,
        )

        ax.set_aspect("equal")
        ax.axis("off")

        fig.tight_layout()
        fig.savefig(output, bbox_inches="tight")

        if not show_in_colab:
            plt.close()
