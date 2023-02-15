from OgreInterface.score_function.ewald import EnergyEwald
from OgreInterface.score_function.born import EnergyBorn
from OgreInterface.score_function.generate_inputs import generate_dict_torch
from OgreInterface.surfaces import Interface
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core.periodic_table import Element
from pymatgen.analysis.local_env import CrystalNN
from ase.data import atomic_numbers, chemical_symbols
from typing import Dict, List
from ase import Atoms
import numpy as np
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import RectBivariateSpline, CubicSpline
from copy import deepcopy
from itertools import groupby, combinations_with_replacement, product


class IonicSurfaceMatcher:
    def __init__(
        self,
        interface: Interface,
        grid_density_x: int = 15,
        grid_density_y: int = 15,
    ):
        self.interface = interface
        self.matrix = deepcopy(interface._orthogonal_structure.lattice.matrix)
        self._vol = np.linalg.det(self.matrix)

        if self._vol < 0:
            self.matrix *= -1
            self._vol *= -1

        self.cutoff, self.alpha, self.k_max = self._get_ewald_parameters()
        self.k_max = 10
        self.charge_dict = self._get_charges()
        self.r0_dict = self._get_r0s(
            sub=self.interface.substrate.bulk_structure,
            film=self.interface.film.bulk_structure,
            charge_dict=self.charge_dict,
        )
        self.ns_dict = {element: 6.0 for element in self.charge_dict}
        self.d_interface = self.interface.interfacial_distance
        self.film_part = self.interface._orthogonal_film_structure
        self.sub_part = self.interface._orthogonal_film_structure
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

    def _get_charges(self):
        sub = self.interface.substrate.bulk_structure
        film = self.interface.film.bulk_structure
        sub_oxidation_state = sub.composition.oxi_state_guesses()[0]
        film_oxidation_state = film.composition.oxi_state_guesses()[0]

        sub_oxidation_state.update(film_oxidation_state)

        return sub_oxidation_state

    def _get_neighborhood_info(self, struc, charge_dict):
        struc.add_oxidation_state_by_element(charge_dict)
        Zs = np.unique(struc.atomic_numbers)
        combos = combinations_with_replacement(Zs, 2)
        neighbor_dict = {c: None for c in combos}

        neighbor_list = []

        cnn = CrystalNN()
        for i, site in enumerate(struc.sites):
            info_dict = cnn.get_nn_info(struc, i)
            for neighbor in info_dict:
                dist = site.distance(neighbor["site"])
                species = tuple(
                    sorted([site.specie.Z, neighbor["site"].specie.Z])
                )
                neighbor_list.append([species, dist])

        sorted_neighbor_list = sorted(neighbor_list, key=lambda x: x[0])
        groups = groupby(sorted_neighbor_list, key=lambda x: x[0])

        for group in groups:
            nn = list(zip(*group[1]))[1]
            neighbor_dict[group[0]] = np.min(nn)

        for n, d in neighbor_dict.items():
            s1 = chemical_symbols[n[0]]
            s2 = chemical_symbols[n[1]]
            c1 = charge_dict[s1]
            c2 = charge_dict[s2]

            if d is None:
                try:
                    d1 = float(Element(s1).ionic_radii[c1])
                except KeyError:
                    print(
                        f"No ionic radius available for {s1}, using the atomic radius instead"
                    )
                    d1 = float(Element(s1).atomic_radius)

                try:
                    d2 = float(Element(s2).ionic_radii[c2])
                except KeyError:
                    print(
                        f"No ionic radius available for {s2}, using the atomic radius instead"
                    )
                    d2 = float(Element(s2).atomic_radius)

                neighbor_dict[n] = d1 + d2

        return neighbor_dict

    def _get_r0s(self, sub, film, charge_dict):
        sub_dict = self._get_neighborhood_info(sub, charge_dict)
        film_dict = self._get_neighborhood_info(film, charge_dict)

        interface_atomic_numbers = np.unique(
            np.concatenate([sub.atomic_numbers, film.atomic_numbers])
        )

        ionic_radius_dict = {
            n: Element(chemical_symbols[n]).ionic_radii[
                charge_dict[chemical_symbols[n]]
            ]
            for n in interface_atomic_numbers
        }
        interface_combos = product(interface_atomic_numbers, repeat=2)
        interface_neighbor_dict = {}
        for c in interface_combos:
            interface_neighbor_dict[(0, 0) + c] = None
            interface_neighbor_dict[(1, 1) + c] = None
            interface_neighbor_dict[(0, 1) + c] = None
            interface_neighbor_dict[(1, 0) + c] = None

        all_keys = np.array(list(sub_dict.keys()) + list(film_dict.keys()))
        unique_keys = np.unique(all_keys, axis=0)
        unique_keys = list(map(tuple, unique_keys))

        for key in unique_keys:
            rev_key = tuple(reversed(key))
            sum_d = ionic_radius_dict[key[0]] + ionic_radius_dict[key[1]]
            if key in sub_dict and key in film_dict:
                sub_d = sub_dict[key]
                film_d = film_dict[key]
                interface_neighbor_dict[(0, 0) + key] = sub_d
                interface_neighbor_dict[(1, 1) + key] = film_d
                interface_neighbor_dict[(0, 1) + key] = (sub_d + film_d) / 2
                interface_neighbor_dict[(1, 0) + key] = (sub_d + film_d) / 2
                interface_neighbor_dict[(0, 0) + rev_key] = sub_d
                interface_neighbor_dict[(1, 1) + rev_key] = film_d
                interface_neighbor_dict[(0, 1) + rev_key] = (
                    sub_d + film_d
                ) / 2
                interface_neighbor_dict[(1, 0) + rev_key] = (
                    sub_d + film_d
                ) / 2

            if key in sub_dict and key not in film_dict:
                sub_d = sub_dict[key]
                interface_neighbor_dict[(0, 0) + key] = sub_d
                interface_neighbor_dict[(1, 1) + key] = sum_d
                interface_neighbor_dict[(0, 1) + key] = sub_d
                interface_neighbor_dict[(1, 0) + key] = sub_d
                interface_neighbor_dict[(0, 0) + rev_key] = sub_d
                interface_neighbor_dict[(1, 1) + rev_key] = sum_d
                interface_neighbor_dict[(0, 1) + rev_key] = sub_d
                interface_neighbor_dict[(1, 0) + rev_key] = sub_d

            if key not in sub_dict and key in film_dict:
                film_d = film_dict[key]
                interface_neighbor_dict[(1, 1) + key] = film_d
                interface_neighbor_dict[(0, 0) + key] = sum_d
                interface_neighbor_dict[(0, 1) + key] = film_d
                interface_neighbor_dict[(1, 0) + key] = film_d
                interface_neighbor_dict[(1, 1) + rev_key] = film_d
                interface_neighbor_dict[(0, 0) + rev_key] = sum_d
                interface_neighbor_dict[(0, 1) + rev_key] = film_d
                interface_neighbor_dict[(1, 0) + rev_key] = film_d

            if key not in sub_dict and key not in film_dict:
                interface_neighbor_dict[(0, 0) + key] = sub_d
                interface_neighbor_dict[(1, 1) + key] = sum_d
                interface_neighbor_dict[(0, 1) + key] = sum_d
                interface_neighbor_dict[(1, 0) + key] = sum_d
                interface_neighbor_dict[(0, 0) + rev_key] = sub_d
                interface_neighbor_dict[(1, 1) + rev_key] = sum_d
                interface_neighbor_dict[(0, 1) + rev_key] = sum_d
                interface_neighbor_dict[(1, 0) + rev_key] = sum_d

        for key, val in interface_neighbor_dict.items():
            if val is None:
                sum_d = ionic_radius_dict[key[2]] + ionic_radius_dict[key[3]]
                interface_neighbor_dict[key] = sub_d

        return interface_neighbor_dict

    def _get_ewald_parameters(self):
        struc_vol = self.interface._structure_volume
        # struc_vol = self._vol
        accf = np.sqrt(np.log(10**4))
        w = 1 / 2**0.5
        alpha = np.pi * (
            len(self.interface._orthogonal_structure) * w / (struc_vol**2)
        ) ** (1 / 3)
        cutoff = accf / np.sqrt(alpha)
        k_max = 2 * np.sqrt(alpha) * accf

        return cutoff, alpha, k_max

    def _generate_shifts(self):
        grid_x = np.linspace(0, 1, self.grid_density_x)
        grid_y = np.linspace(0, 1, self.grid_density_y)
        X, Y = np.meshgrid(grid_x, grid_y)

        shifts = np.c_[X.ravel(), Y.ravel()]

        return shifts, X, Y

    def _get_shifted_atoms(self, shifts: np.ndarray) -> List[Atoms]:
        atoms = []

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
            charge_dict=self.charge_dict,
            ns_dict=self.ns_dict,
        )

        return inputs

    def _calculate_coulomb(self, inputs, z_shift=False):
        ewald = EnergyEwald(
            alpha=self.alpha, k_max=self.k_max, cutoff=self.cutoff
        )
        coulomb_energy = ewald.forward(inputs, z_shift)

        return coulomb_energy

    def _calculate_born(self, inputs, z_shift=False):
        born = EnergyBorn(cutoff=self.cutoff)
        born_energy = born.forward(
            inputs, z_shift=z_shift, r0_dict=self.r0_dict
        )

        return born_energy

    def _get_interpolated_data(self, X, Y, Z):
        x_grid = np.linspace(0, 1, self.grid_density_x)
        y_grid = np.linspace(0, 1, self.grid_density_y)
        spline = RectBivariateSpline(y_grid, x_grid, Z)

        x_grid_interp = np.linspace(0, 1, 401)
        y_grid_interp = np.linspace(0, 1, 401)

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

        if self.z_PES_data is None:
            z_coulomb_energy = self._calculate_coulomb(inputs, z_shift=True)
            z_born_energy = self._calculate_born(inputs, z_shift=True)
            z_interface_coulomb_energy = z_coulomb_energy.reshape(self.X.shape)
            z_interface_born_energy = z_born_energy.reshape(self.X.shape)
            self.z_PES_data = [
                z_interface_coulomb_energy,
                z_interface_born_energy,
            ]
        else:
            z_interface_coulomb_energy = self.z_PES_data[0]
            z_interface_born_energy = self.z_PES_data[1]

        coulomb_energy = self._calculate_coulomb(inputs, z_shift=False)
        born_energy = self._calculate_born(inputs, z_shift=False)

        interface_coulomb_energy = coulomb_energy.reshape(self.X.shape)
        interface_born_energy = born_energy.reshape(self.X.shape)

        coulomb_adh_energy = (
            z_interface_coulomb_energy - interface_coulomb_energy
        )
        born_adh_energy = z_interface_born_energy - interface_born_energy

        X_plot, Y_plot, Z_born = self._get_interpolated_data(
            self.X, self.Y, born_adh_energy
        )
        _, _, Z_coulomb = self._get_interpolated_data(
            self.X, self.Y, coulomb_adh_energy
        )

        Z_both = Z_born + Z_coulomb

        a = self.matrix[0, :2]
        b = self.matrix[1, :2]
        borders = np.vstack([np.zeros(2), a, a + b, b, np.zeros(2)])
        x_size = borders[:, 0].max() - borders[:, 0].min()
        y_size = borders[:, 1].max() - borders[:, 1].min()
        ratio = y_size / x_size

        if show_born_and_coulomb:
            fig, (ax1, ax2, ax3) = plt.subplots(
                figsize=(3 * 5, 5 * ratio),
                ncols=3,
                dpi=dpi,
            )

            self._plot_heatmap(
                fig=fig,
                ax=ax1,
                X=X_plot,
                Y=Y_plot,
                Z=Z_born / (self.interface.area),
                borders=borders,
                cmap=cmap,
                fontsize=fontsize,
                show_max=False,
            )
            self._plot_heatmap(
                fig=fig,
                ax=ax2,
                X=X_plot,
                Y=Y_plot,
                Z=Z_coulomb / (self.interface.area),
                borders=borders,
                cmap=cmap,
                fontsize=fontsize,
                show_max=False,
            )
            self._plot_heatmap(
                fig=fig,
                ax=ax3,
                X=X_plot,
                Y=Y_plot,
                Z=Z_both / (self.interface.area),
                borders=borders,
                cmap=cmap,
                fontsize=fontsize,
                show_max=show_max,
            )

            frac_shifts = np.c_[
                X_plot.ravel(), Y_plot.ravel(), np.zeros(Y_plot.shape).ravel()
            ].dot(np.linalg.inv(self.matrix))
            opt_shift = frac_shifts[np.argmax(Z_both.ravel())]
            max_Z = np.max(Z_both)
            plot_shift = opt_shift.dot(self.matrix)

            ax3.scatter(
                [plot_shift[0]],
                [plot_shift[1]],
                fc="white",
                ec="black",
                marker="X",
                s=100,
                zorder=10,
            )
        else:
            fig, ax = plt.subplots(
                figsize=(5, 5 * ratio),
                dpi=dpi,
            )
            self._plot_heatmap(
                fig=fig,
                ax=ax,
                X=X_plot,
                Y=Y_plot,
                Z=Z_both / (self.interface.area),
                borders=borders,
                cmap=cmap,
                fontsize=fontsize,
                show_max=show_max,
            )

            frac_shifts = np.c_[
                X_plot.ravel(), Y_plot.ravel(), np.zeros(Y_plot.shape).ravel()
            ].dot(np.linalg.inv(self.matrix))
            opt_shift = frac_shifts[np.argmax(Z_both.ravel())]
            max_Z = np.max(Z_both)
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
