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
        grid_density_x: int = 15,
        grid_density_y: int = 15,
        xlim: List[float] = [0.0, 1.0],
        ylim: List[float] = [0.0, 1.0],
    ):
        self.xlim = xlim
        self.ylim = ylim
        self.interface = interface
        self.matrix = deepcopy(interface._orthogonal_structure.lattice.matrix)
        self._vol = np.linalg.det(self.matrix)

        if self._vol < 0:
            self.matrix *= -1
            self._vol *= -1

        self.grid_density_x = grid_density_x
        self.grid_density_y = grid_density_y

        self.shifts, self.X, self.Y = self._generate_shifts()

    def _generate_shifts(self):
        pass
