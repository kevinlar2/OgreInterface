from OgreInterface.score_function.ewald import EnergyEwald
from OgreInterface.score_function.born import EnergyBorn
from OgreInterface.score_function.generate_inputs import generate_dict_torch
from OgreInterface.surfaces import Interface
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core.periodic_table import Element
from pymatgen.analysis.local_env import CrystalNN
from ase.data import atomic_numbers, chemical_symbols
from typing import Dict
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
        self.opt_xy_shift = np.zeros(3)
        self.opt_z_shift = np.zeros(3)

        self.shifts, self.X, self.Y = self._generate_shifts()
        self.z_PES_data = None

    def get_optmized_structure(self):
        opt_shift = self.opt_xy_shift + self.opt_z_shift
        self.interface.shift_film(
            shift=opt_shift, fractional=True, inplace=True
        )

    # def _get_charges_and_radii(self):
    #     sub = self.interface.substrate.conventional_bulk_structure
    #     film = self.interface.film.conventional_bulk_structure
    #     sub_oxidation_state = sub.composition.oxi_state_guesses()[0]
    #     film_oxidation_state = film.composition.oxi_state_guesses()[0]

    #     sub_oxidation_state.update(film_oxidation_state)

    #     radii = {
    #         element: Element(element).ionic_radii[charge]
    #         for element, charge in sub_oxidation_state.items()
    #     }

    #     return sub_oxidation_state, radii

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
                d1_ionic = float(Element(s1).ionic_radii[c1])
                d2_ionic = float(Element(s2).ionic_radii[c2])

                neighbor_dict[n] = d1_ionic + d2_ionic

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

        shifts = np.c_[X.ravel(), Y.ravel(), np.zeros(X.shape).ravel()]

        return shifts, X, Y

    def _get_shifted_atoms_old(self, shifts):

        atoms = [
            AseAtomsAdaptor().get_atoms(self.sub_part),
            AseAtomsAdaptor().get_atoms(self.film_part),
        ]

        for shift in shifts:
            shifted_atoms = self.interface.shift_film(
                shift, fractional=True, inplace=False, return_atoms=True
            )
            atoms.append(shifted_atoms)

        return atoms

    def _get_shifted_atoms(self, shifts):
        atoms = []
        z_atoms = []

        z_shift = np.array([0, 0, 10]) / np.linalg.norm(self.matrix[-1])

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
            z_shifted_atoms = self.interface.shift_film(
                shift + z_shift,
                fractional=True,
                inplace=False,
                return_atoms=True,
            )
            z_shifted_atoms.set_array(
                "is_film",
                self.interface._orthogonal_structure.site_properties[
                    "is_film"
                ],
            )
            atoms.append(shifted_atoms)
            z_atoms.append(z_shifted_atoms)

        return atoms, z_atoms

    def _get_shifted_atoms_z(self, shifts):
        z_shift = np.array([0, 0, 10]) / np.linalg.norm(self.matrix[-1])
        atoms = [
            self.interface.shift_film(
                z_shift + self.opt_xy_shift,
                fractional=True,
                inplace=False,
                return_atoms=True,
            )
        ]
        atoms[0].set_array(
            "is_film",
            self.interface._orthogonal_structure.site_properties["is_film"],
        )

        for shift in shifts:
            shifted_atoms = self.interface.shift_film(
                shift + self.opt_xy_shift,
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

    def _get_shifted_strucs(self, shifts):

        atoms = [
            self.sub_part,
            self.film_part,
        ]

        for shift in shifts:
            shifted_atoms = self.interface.shift_film(
                shift, fractional=True, inplace=False, return_atoms=False
            )
            atoms.append(shifted_atoms)

        return atoms

    def _generate_inputs(self, atoms_list):
        inputs = generate_dict_torch(
            atoms=atoms_list,
            cutoff=self.cutoff,
            charge_dict=self.charge_dict,
            # radius_dict=self.radius_dict,
            ns_dict=self.ns_dict,
        )

        return inputs

    def _calculate_coulomb(self, inputs):
        ewald = EnergyEwald(alpha=self.alpha, k_max=self.k_max)
        coulomb_energy = ewald.forward(inputs)

        return coulomb_energy

    def _calculate_born(self, inputs):
        born = EnergyBorn(cutoff=self.cutoff)
        born_energy = born.forward(inputs, r0_dict=self.r0_dict)

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
        atoms_list, z_atoms_list = self._get_shifted_atoms(shifts)

        inputs = self._generate_inputs(atoms_list)
        if self.z_PES_data is None:
            z_inputs = self._generate_inputs(z_atoms_list)
            z_coulomb_energy = self._calculate_coulomb(z_inputs)
            z_born_energy = self._calculate_born(z_inputs)
            z_interface_coulomb_energy = z_coulomb_energy.reshape(self.X.shape)
            z_interface_born_energy = z_born_energy.reshape(self.X.shape)
            self.z_PES_data = [
                z_interface_coulomb_energy,
                z_interface_born_energy,
            ]
        else:
            z_interface_coulomb_energy = self.z_PES_data[0]
            z_interface_born_energy = self.z_PES_data[1]

        coulomb_energy = self._calculate_coulomb(inputs)
        born_energy = self._calculate_born(inputs)

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
