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
        self.opt_xy_shift = np.zeros(3)
        self.opt_z_shift = np.zeros(3)

        self.shifts, self.X, self.Y = self._generate_shifts()
        self.z_PES_data = None

    def get_optmized_structure(self):
        opt_shift = self.opt_xy_shift + self.opt_z_shift
        self.interface.shift_film(
            shift=opt_shift, fractional=True, inplace=True
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

        shifts = np.c_[X.ravel(), Y.ravel(), np.zeros(X.shape).ravel()]

        return shifts, X, Y

    def _get_shifted_atoms(self, shifts):
        sub_atoms = AseAtomsAdaptor().get_atoms(self.sub_part)
        sub_atoms.set_array("is_film", np.zeros(len(sub_atoms)).astype(bool))
        film_atoms = AseAtomsAdaptor().get_atoms(self.film_part)
        film_atoms.set_array("is_film", np.ones(len(film_atoms)).astype(bool))

        atoms = [sub_atoms, film_atoms]

        for shift in shifts:
            shifted_atoms = self.interface.shift_film(
                shift + self.opt_z_shift,
                fractional=True,
                inplace=False,
                return_atoms=True,
            )
            shifted_atoms.set_array(
                "is_film",
                self.interface._orthogonal_structure.site_properties[
                    "is_film"
                ],
            )
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

        self.opt_xy_shift = opt_shift

        fig.tight_layout()
        fig.savefig(output, bbox_inches="tight")
        plt.close(fig)

        return max_Z

    def run_z_shifts_old(
        self,
        interfacial_distances: np.ndarray = np.linspace(2.0, 4.0, 101),
        fontsize: int = 12,
        output: str = "z_shift.png",
        shift: bool = True,
    ) -> float:
        z_shifts = interfacial_distances - self.d_interface
        shifts = np.c_[
            np.zeros((len(z_shifts), 2)), z_shifts
        ] / np.linalg.norm(self.matrix[-1])
        inputs = self._generate_inputs(shifts)

        coulomb_energy = self._calculate_coulomb(inputs)
        born_energy = self._calculate_born(inputs)

        sub_coulomb_energy = coulomb_energy[0]
        film_coulomb_energy = coulomb_energy[1]
        interface_coulomb_energy = coulomb_energy[2:]

        sub_born_energy = born_energy[0]
        film_born_energy = born_energy[1]
        interface_born_energy = born_energy[2:]

        # pmg_coulomb_adh_energy = (
        #     sub_pmg_coulomb_energy + film_pmg_coulomb_energy
        # ) - interface_pmg_coulomb_energy
        coulomb_adh_energy = (
            sub_coulomb_energy + film_coulomb_energy
        ) - interface_coulomb_energy
        born_adh_energy = (
            sub_born_energy + film_born_energy
        ) - interface_born_energy

        both_adh_energy = born_adh_energy + coulomb_adh_energy
        # both_pmg_adh_energy = born_adh_energy + pmg_coulomb_adh_energy

        new_x = np.linspace(
            interfacial_distances.min(), interfacial_distances.max(), 500
        )
        new_z_shifts = np.linspace(
            shifts[:, -1].min(), shifts[:, -1].max(), 500
        )
        cs = CubicSpline(interfacial_distances, both_adh_energy)
        new_y = cs(new_x)

        max_ind = np.argmax(new_y)
        max_x = new_x[max_ind]
        max_y = new_y[max_ind]
        max_shift = new_z_shifts[max_ind]

        if shift:
            self.interface.shift_film(
                shift=[0, 0, max_shift], fractional=True, inplace=True
            )

        fig, ax = plt.subplots(figsize=(4, 4), dpi=400)
        ax.set_xlabel("Interfacial Distance ($\\AA$)", fontsize=fontsize)
        ax.set_ylabel("$E_{adh}$", fontsize=fontsize)
        ax.plot(new_x, new_y, color="black")
        ax.scatter([max_x], [max_y], color="black", marker="o", s=20)
        ax.set_xlim(interfacial_distances.min(), interfacial_distances.max())

        fig.tight_layout(pad=0.4)
        fig.savefig(output)
        plt.close(fig)

        return max_y

    def run_z_shifts(
        self,
        interfacial_distances: np.ndarray = np.linspace(2.0, 4.0, 101),
        fontsize: int = 12,
        output: str = "z_shift.png",
        shift: bool = True,
    ) -> float:
        z_shifts = interfacial_distances - self.d_interface
        shifts = np.c_[
            np.zeros((len(z_shifts), 2)), z_shifts
        ] / np.linalg.norm(self.matrix[-1])
        atoms_list = self._get_shifted_atoms_z(shifts)
        inputs = self._generate_inputs(atoms_list)

        coulomb_energy = self._calculate_coulomb(inputs)
        born_energy = self._calculate_born(inputs)

        sub_film_coulomb_energy = coulomb_energy[0]
        interface_coulomb_energy = coulomb_energy[1:]

        sub_film_born_energy = born_energy[0]
        interface_born_energy = born_energy[1:]

        coulomb_adh_energy = sub_film_coulomb_energy - interface_coulomb_energy
        born_adh_energy = sub_film_born_energy - interface_born_energy

        both_adh_energy = born_adh_energy + coulomb_adh_energy

        max_ind = np.argmax(both_adh_energy)
        max_x = interfacial_distances[max_ind]
        max_y = both_adh_energy[max_ind]
        max_shift = shifts[max_ind]

        self.opt_z_shift = max_shift

        fig, ax = plt.subplots(figsize=(4, 4), dpi=400)
        ax.set_xlabel("Interfacial Distance ($\\AA$)", fontsize=fontsize)
        ax.set_ylabel("$E_{adh}$", fontsize=fontsize)
        ax.plot(
            interfacial_distances,
            both_adh_energy,
            color="black",
        )
        ax.scatter([max_x], [max_y], color="black", marker="o", s=20)
        ax.set_xlim(interfacial_distances.min(), interfacial_distances.max())

        fig.tight_layout(pad=0.4)
        fig.savefig(output)
        plt.close(fig)

        return max_y
