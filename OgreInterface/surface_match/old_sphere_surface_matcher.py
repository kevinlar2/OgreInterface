from OgreInterface.score_function.overlap import SphereOverlap
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


class SphereSurfaceMatcher:
    def __init__(
        self,
        interface: Interface,
        radius_dict: Optional[Dict[str, float]] = None,
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

        self.radius_dict = self._get_radii(radius_dict)
        self.cutoff = self._get_cutoff()
        self.d_interface = self.interface.interfacial_distance
        self.film_part = self.interface._orthogonal_film_structure
        self.sub_part = self.interface._orthogonal_substrate_structure
        self.grid_density_x = grid_density_x
        self.grid_density_y = grid_density_y
        self.opt_xy_shift = np.zeros(2)

        self.shifts, self.X, self.Y = self._generate_shifts()
        self.z_PES_data = None

    def get_optmized_structure(self):
        opt_shift = self.opt_xy_shift

        self.interface.shift_film_inplane(
            x_shift=opt_shift[0], y_shift=opt_shift[1], fractional=True
        )

    def _get_radii(self, radius_dict):
        sub_radii = radius_dict["sub"]
        film_radii = radius_dict["film"]
        radii_dict = {(0, atomic_numbers[k]): v for k, v in sub_radii.items()}
        radii_dict.update(
            {(1, atomic_numbers[k]): v for k, v in film_radii.items()}
        )

        return radii_dict

    def _get_cutoff(self):
        max_radius = max(list(self.radius_dict.values()))
        cutoff_val = (2 * max_radius) / (1e-3) ** (1 / 6)

        return cutoff_val

    def _generate_shifts(self):
        grid_x = np.linspace(self.xlim[0], self.xlim[1], self.grid_density_x)
        grid_y = np.linspace(self.ylim[0], self.ylim[1], self.grid_density_y)
        X, Y = np.meshgrid(grid_x, grid_y)

        shifts = np.c_[X.ravel(), Y.ravel()]

        return shifts, X, Y

    def _get_shifted_atoms(self, shifts: np.ndarray) -> List[Atoms]:
        sub_atoms = self.interface.get_substrate_supercell(return_atoms=True)
        sub_atoms.set_array("is_film", np.zeros(len(sub_atoms)).astype(bool))

        film_atoms = self.interface.get_film_supercell(return_atoms=True)
        film_atoms.set_array("is_film", np.ones(len(film_atoms)).astype(bool))

        atoms = [sub_atoms, film_atoms]

        for shift in shifts:
            # Shift in-plane
            self.interface.shift_film_inplane(
                x_shift=shift[0], y_shift=shift[1], fractional=True
            )

            # Get inplane shifted atoms
            shifted_atoms = self.interface.get_interface(
                orthogonal=True, return_atoms=True
            )

            # Add the is_film property
            shifted_atoms.set_array(
                "is_film",
                self.interface._orthogonal_structure.site_properties[
                    "is_film"
                ],
            )

            self.interface.shift_film_inplane(
                x_shift=-shift[0], y_shift=-shift[1], fractional=True
            )

            # Add atoms to the list
            atoms.append(shifted_atoms)

        return atoms

    def _generate_inputs(self, atoms_list):
        inputs = generate_dict_torch(
            atoms=atoms_list,
            cutoff=self.cutoff,
        )

        return inputs

    def _calculate_overlap(self, inputs):
        sphere_overlap = SphereOverlap(cutoff=self.cutoff)
        overlap = sphere_overlap.forward(inputs, radius_dict=self.radius_dict)

        return overlap

    def _get_interpolated_data(self, X, Y, Z):
        x_grid = np.linspace(self.xlim[0], self.xlim[1], self.grid_density_x)
        y_grid = np.linspace(self.ylim[0], self.ylim[1], self.grid_density_y)
        spline = RectBivariateSpline(y_grid, x_grid, Z)

        x_grid_interp = np.linspace(self.xlim[0], self.xlim[1], 401)
        y_grid_interp = np.linspace(self.ylim[0], self.ylim[1], 401)

        X_interp, Y_interp = np.meshgrid(x_grid_interp, y_grid_interp)
        Z_interp = spline.ev(xi=Y_interp, yi=X_interp)
        frac_shifts = np.c_[
            X_interp.ravel(),
            Y_interp.ravel(),
            np.zeros(X_interp.shape).ravel(),
        ]

        cart_shifts = frac_shifts.dot(self.matrix)

        X_cart = cart_shifts[:, 0].reshape(X_interp.shape)
        Y_cart = cart_shifts[:, 1].reshape(Y_interp.shape)

        return X_cart, Y_cart, Z_interp

    def _plot_heatmap(
        self, fig, ax, X, Y, Z, borders, cmap, fontsize, show_max
    ):
        ax.set_xlabel(r"Shift in $x$ ($\AA$)", fontsize=fontsize)
        ax.set_ylabel(r"Shift in $y$ ($\AA$)", fontsize=fontsize)

        im = ax.contourf(
            X,
            Y,
            Z,
            cmap=cmap,
            levels=200,
            norm=Normalize(vmin=np.nanmin(Z), vmax=np.nanmax(Z)),
        )

        ax.plot(
            borders[:, 0],
            borders[:, 1],
            color="black",
            linewidth=2,
        )

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("top", size="5%", pad=0.1)
        cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
        cbar.ax.tick_params(labelsize=fontsize)
        cbar.ax.locator_params(nbins=3)

        if show_max:
            E_max = np.max(Z)
            label = (
                "$E_{adh}$ (eV/$\\AA^{2}$) : "
                + "$E_{max}$ = "
                + f"{E_max:.4f}"
            )
            cbar.set_label(label, fontsize=fontsize)
        else:
            label = "$E_{adh}$ (eV/$\\AA^{2}$)"
            cbar.set_label(label, fontsize=fontsize)

        cax.xaxis.set_ticks_position("top")
        cax.xaxis.set_label_position("top")
        ax.tick_params(labelsize=fontsize)
        ax.set_xlim(borders[:, 0].min(), borders[:, 0].max())
        ax.set_ylim(borders[:, 1].min(), borders[:, 1].max())
        ax.set_aspect("equal")

    def run_surface_matching(
        self,
        cmap: str = "jet",
        fontsize: int = 14,
        output: str = "PES.png",
        shift: bool = True,
        show_born_and_coulomb: bool = False,
        dpi: int = 400,
        show_max: bool = False,
    ) -> float:
        shifts = self.shifts
        atoms_list = self._get_shifted_atoms(shifts)

        inputs = self._generate_inputs(atoms_list)
        overlap = self._calculate_overlap(inputs)

        sub_overlap = overlap[0]
        film_overlap = overlap[1]

        interface_overlap = overlap[2:].reshape(self.X.shape)

        Z = (
            sub_overlap + film_overlap - interface_overlap
        ) / self.interface.area

        X_plot, Y_plot, Z_overlap = self._get_interpolated_data(
            self.X, self.Y, Z
        )

        a = self.matrix[0, :2]
        b = self.matrix[1, :2]
        borders = np.vstack(
            [
                self.xlim[0] * a + self.ylim[0] * b,
                self.xlim[1] * a + self.ylim[0] * b,
                self.xlim[1] * a + self.ylim[1] * b,
                self.xlim[0] * a + self.ylim[1] * b,
                self.xlim[0] * a + self.ylim[0] * b,
            ]
        )
        x_size = borders[:, 0].max() - borders[:, 0].min()
        y_size = borders[:, 1].max() - borders[:, 1].min()
        ratio = y_size / x_size

        if ratio < 1:
            figx = 5 / ratio
            figy = 5
        else:
            figx = 5
            figy = 5 * ratio

        fig, ax = plt.subplots(
            figsize=(figx, figy),
            dpi=dpi,
        )
        self._plot_heatmap(
            fig=fig,
            ax=ax,
            X=X_plot,
            Y=Y_plot,
            Z=Z_overlap,
            borders=borders,
            cmap=cmap,
            fontsize=fontsize,
            show_max=show_max,
        )

        frac_shifts = np.c_[
            X_plot.ravel(), Y_plot.ravel(), np.zeros(Y_plot.shape).ravel()
        ].dot(np.linalg.inv(self.matrix))
        opt_shift = frac_shifts[np.argmax(Z_overlap.ravel())]
        max_Z = np.max(Z_overlap)
        plot_shift = opt_shift.dot(self.matrix)

        ax.scatter(
            [plot_shift[0]],
            [plot_shift[1]],
            fc="white",
            ec="black",
            marker="X",
            s=100,
            zorder=10,
        )

        self.opt_xy_shift = opt_shift[:2]

        fig.tight_layout()
        fig.savefig(output, bbox_inches="tight")
        plt.close(fig)

        return max_Z
