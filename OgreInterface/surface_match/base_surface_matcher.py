from OgreInterface.score_function.generate_inputs import generate_dict_torch
from OgreInterface.surfaces import Interface
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core.periodic_table import Element
from pymatgen.analysis.local_env import CrystalNN
from ase.data import atomic_numbers, chemical_symbols
from typing import Dict, Optional, List
import numpy as np
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import RectBivariateSpline, CubicSpline
from copy import deepcopy
from itertools import groupby, combinations_with_replacement, product
from ase import Atoms


class BaseSurfaceMatcher:
    def __init__(
        self,
        interface: Interface,
        grid_density_x: int = 10,
        grid_density_y: int = 10,
    ):
        self.interface = interface
        self.matrix = deepcopy(interface._orthogonal_structure.lattice.matrix)
        self._vol = np.linalg.det(self.matrix)

        if self._vol < 0:
            self.matrix *= -1
            self._vol *= -1

        self.grid_density_x = grid_density_x
        self.grid_density_y = grid_density_y

        (
            self.shift_matrix,
            self.shift_images,
            self.shifts,
        ) = self._generate_shifts()

    def _generate_shifts(self) -> List[np.ndarray]:
        (
            sub_matrix,
            sub_images,
            film_matrix,
            film_images,
        ) = self.interface._get_oriented_cell_and_images(strain=True)

        if self.interface.substrate.area < self.interface.film.area:
            shift_matrix = sub_matrix
            shift_images = sub_images
        else:
            shift_matrix = film_matrix
            shift_images = film_images

        iface_inv_matrix = (
            self.interface._orthogonal_structure.lattice.inv_matrix
        )

        grid_x = np.linspace(0, 1, self.grid_density_x)
        grid_y = np.linspace(0, 1, self.grid_density_y)

        X, Y = np.meshgrid(grid_x, grid_y)

        prim_frac_shifts = (
            np.c_[X.ravel(), Y.ravel(), np.zeros(Y.shape).ravel()]
            + shift_images[0]
        )
        prim_cart_shifts = prim_frac_shifts.dot(shift_matrix)
        iface_frac_shifts = prim_cart_shifts.dot(iface_inv_matrix)

        return shift_matrix, shift_images, iface_frac_shifts
