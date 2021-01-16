"""
This module will be used to construct the surfaces and interfaces used in this package.
"""
from pymatgen.core.structure import Structure 
from pymatgen.io.vasp.inputs import Poscar
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.substrate_analyzer import ZSLGenerator, SubstrateAnalyzer
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
            raise TypeError(f"structure accepts 'pymatgen.core.structure.Structure' or 'ase.Atoms' not '{type(slab).__name__}'")

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
        #  strain,
        #  angle_diff,
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
        #  self.strain = strain
        #  self.angle_diff = angle_diff
        #  self.area_diff = area_diff
        self.substrate_supercell = self._prepare_slab(
            self.substrate,
            self.substrate_transformation
        )
        self.film_supercell = self._prepare_slab(
            self.film,
            self.film_transformation
        )
        self.interface = self._stack_interface()

    def _prepare_slab(self, slab, matrix):
        supercell = make_supercell(
            slab.slab_ase,
            matrix,
        )

        return supercell

    def _stack_interface(self):
        interface = stack(
            atoms1=self.substrate_supercell,
            atoms2=self.film_supercell,
        )

        return interface

