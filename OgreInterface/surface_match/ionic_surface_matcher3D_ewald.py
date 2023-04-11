from OgreInterface.score_function.ewald import EnergyEwald
from OgreInterface.score_function.born import EnergyBorn
from OgreInterface.score_function.ionic import IonicPotential

# from OgreInterface.score_function.generate_inputs import generate_dict_torch
from OgreInterface.surfaces import Interface
from OgreInterface.surface_match.base_surface_matcher import BaseSurfaceMatcher
from pymatgen.core.periodic_table import Element
from pymatgen.analysis.local_env import CrystalNN
from ase.data import chemical_symbols, covalent_radii
from typing import List
from ase import Atoms
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline
from itertools import groupby, combinations_with_replacement, product


class IonicSurfaceMatcher3DEwald(BaseSurfaceMatcher):
    def __init__(
        self,
        interface: Interface,
        grid_density: float = 2.5,
    ):
        super().__init__(
            interface=interface,
            grid_density=grid_density,
        )
        self.cutoff, self.alpha, self.k_max = self._get_ewald_parameters()
        self.cutoff = 10.0
        self.k_max = 2
        self.charge_dict = self._get_charges()
        self.r0_dict = self._get_r0s(
            sub=self.interface.substrate.bulk_structure,
            film=self.interface.film.bulk_structure,
            charge_dict=self.charge_dict,
        )
        self.ns_dict = self._get_ns_dict()
        # self.ns_dict = {element: 6.0 for element in self.charge_dict}
        self.d_interface = self.interface.interfacial_distance
        self.film_part = self.interface._orthogonal_film_structure
        self.sub_part = self.interface._orthogonal_film_structure
        self.opt_xy_shift = np.zeros(2)

        self.z_PES_data = None

    def get_optmized_structure(self):
        opt_shift = self.opt_xy_shift

        self.interface.shift_film_inplane(
            x_shift=opt_shift[0], y_shift=opt_shift[1], fractional=True
        )

    def bo_function(self, a, b, z):
        x, y = self.get_cart_xy_shifts(a, b)
        z_shift = z - self.d_interface
        shift = np.array([a, b, z_shift])
        force = self._calculate_lj(inputs=self.inputs, shift=shift)

        return -force

    def optimize(self, z_bounds, max_iters):
        a_grid, b_grid, z_grid = np.meshgrid(
            np.linspace(0, 1, 5),
            np.linspace(0, 1, 5),
            np.linspace(z_bounds[0], z_bounds[1], 5),
        )
        probe_points = np.c_[a_grid.ravel(), b_grid.ravel(), z_grid.ravel()]

        interface_atoms = self.interface.get_interface(
            orthogonal=True, return_atoms=True
        )

        # TODO make the inputs better so I can pass in a shift
        # self.inputs = self._generate_inputs(
        #     atoms=interface_atoms, shifts=batch_shift, interface=True
        # )

        self._optimizer(
            func=self.bo_function,
            z_bounds=z_bounds,
            max_iters=max_iters,
            probe_points=probe_points,
        )

    def _get_ns_dict(self):
        ion_config_to_n_map = {
            "[He]": 5,
            "[Ne]": 7,
            "[Ar]": 9,
            "[Kr]": 10,
            "[Xe]": 12,
        }
        n_vals = {}

        Zs = np.unique(self.interface._orthogonal_structure.atomic_numbers)
        for z in Zs:
            element = Element(chemical_symbols[z])
            ion_config = element.electronic_structure.split(".")[0]
            n_val = ion_config_to_n_map[ion_config]
            n_vals[z] = n_val

        combos = combinations_with_replacement(Zs, 2)
        n = 12.0

        n_dict = {}
        for combo in combos:
            i = combo[0]
            j = combo[1]

            # q_i = self.charge_dict[chemical_symbols[i]]
            # q_j = self.charge_dict[chemical_symbols[j]]

            n_dict[(i, j)] = (n_vals[i] + n_vals[j]) / 2
            n_dict[(j, i)] = (n_vals[i] + n_vals[j]) / 2
            # n_dict[(i, j)] = n
            # n_dict[(j, i)] = n

            # if q_i * q_j < 0:
            # else:
            #     n_dict[(i, j)] = 6.0
            #     n_dict[(j, i)] = 6.0

        return n_dict

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

        cnn = CrystalNN(search_cutoff=7.0, cation_anion=True)
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

            # print(f"{s1}-{s2} ", neighbor_dict[n])

        return neighbor_dict

    def _get_r0s(self, sub, film, charge_dict):
        sub_dict = self._get_neighborhood_info(sub, charge_dict)
        film_dict = self._get_neighborhood_info(film, charge_dict)

        interface_atomic_numbers = np.unique(
            np.concatenate([sub.atomic_numbers, film.atomic_numbers])
        )

        ionic_radius_dict = {}
        cov_radius_dict = {
            n: covalent_radii[n] for n in interface_atomic_numbers
        }

        for n in interface_atomic_numbers:
            element = Element(chemical_symbols[n])

            try:
                d = element.ionic_radii[charge_dict[chemical_symbols[n]]]
            except KeyError:
                print(
                    f"No ionic radius available for {chemical_symbols[n]}, using the atomic radius instead"
                )
                d = float(element.atomic_radius)

            ionic_radius_dict[n] = d

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
            charge_sign = (
                charge_dict[chemical_symbols[key[0]]]
                * charge_dict[chemical_symbols[key[1]]]
            )

            if charge_sign < 0:
                ionic_sum_d = (
                    ionic_radius_dict[key[0]] + ionic_radius_dict[key[1]]
                )
            else:
                ionic_sum_d = cov_radius_dict[key[0]] + cov_radius_dict[key[1]]

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
                interface_neighbor_dict[(1, 1) + key] = ionic_sum_d
                interface_neighbor_dict[(0, 1) + key] = sub_d
                interface_neighbor_dict[(1, 0) + key] = sub_d
                interface_neighbor_dict[(0, 0) + rev_key] = sub_d
                interface_neighbor_dict[(1, 1) + rev_key] = ionic_sum_d
                interface_neighbor_dict[(0, 1) + rev_key] = sub_d
                interface_neighbor_dict[(1, 0) + rev_key] = sub_d

            if key not in sub_dict and key in film_dict:
                film_d = film_dict[key]
                interface_neighbor_dict[(1, 1) + key] = film_d
                interface_neighbor_dict[(0, 0) + key] = ionic_sum_d
                interface_neighbor_dict[(0, 1) + key] = film_d
                interface_neighbor_dict[(1, 0) + key] = film_d
                interface_neighbor_dict[(1, 1) + rev_key] = film_d
                interface_neighbor_dict[(0, 0) + rev_key] = ionic_sum_d
                interface_neighbor_dict[(0, 1) + rev_key] = film_d
                interface_neighbor_dict[(1, 0) + rev_key] = film_d

            if key not in sub_dict and key not in film_dict:
                interface_neighbor_dict[(0, 0) + key] = ionic_sum_d
                interface_neighbor_dict[(1, 1) + key] = ionic_sum_d
                interface_neighbor_dict[(0, 1) + key] = ionic_sum_d
                interface_neighbor_dict[(1, 0) + key] = ionic_sum_d
                interface_neighbor_dict[(0, 0) + rev_key] = ionic_sum_d
                interface_neighbor_dict[(1, 1) + rev_key] = ionic_sum_d
                interface_neighbor_dict[(0, 1) + rev_key] = ionic_sum_d
                interface_neighbor_dict[(1, 0) + rev_key] = ionic_sum_d

        for key, val in interface_neighbor_dict.items():
            if val is None:
                ionic_sum_d = (
                    ionic_radius_dict[key[2]] + ionic_radius_dict[key[3]]
                )
                interface_neighbor_dict[key] = ionic_sum_d

        # for key in interface_neighbor_dict:
        #     print(key, interface_neighbor_dict[key])
        # interface_neighbor_dict[key] *= 0.90

        return interface_neighbor_dict

    def _get_ewald_parameters(self):
        struc_vol = self.interface._structure_volume
        accf = np.sqrt(np.log(10**4))
        w = 1 / 2**0.5
        alpha = np.pi * (
            len(self.interface._orthogonal_structure) * w / (struc_vol**2)
        ) ** (1 / 3)
        cutoff = accf / np.sqrt(alpha)
        k_max = 2 * np.sqrt(alpha) * accf
        print(alpha)
        print(cutoff)
        print(k_max)

        return cutoff, alpha, k_max

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

    def _generate_inputs(self, atoms, shifts, interface=True):
        inputs = generate_dict_torch(
            atoms=atoms,
            shifts=shifts,
            cutoff=self.cutoff,
            interface=interface,
            charge_dict=self.charge_dict,
            ns_dict=self.ns_dict,
        )

        return inputs

    def _calculate(self, inputs, z_shift=False):
        ionic_potential = IonicPotential(
            alpha=self.alpha, k_max=self.k_max, cutoff=self.cutoff
        )
        outputs = ionic_potential.forward(
            inputs=inputs,
            r0_dict=self.r0_dict,
            ns_dict=self.ns_dict,
            z_shift=z_shift,
        )

        return outputs

    def _get_interpolated_data(self, Z, image):
        x_grid = np.linspace(0, 1, self.grid_density_x)
        y_grid = np.linspace(0, 1, self.grid_density_y)
        spline = RectBivariateSpline(y_grid, x_grid, Z)

        x_grid_interp = np.linspace(0, 1, 101)
        y_grid_interp = np.linspace(0, 1, 101)

        X_interp, Y_interp = np.meshgrid(x_grid_interp, y_grid_interp)
        Z_interp = spline.ev(xi=Y_interp, yi=X_interp)
        frac_shifts = (
            np.c_[
                X_interp.ravel(),
                Y_interp.ravel(),
                np.zeros(X_interp.shape).ravel(),
            ]
            + image
        )

        cart_shifts = frac_shifts.dot(self.shift_matrix)

        X_cart = cart_shifts[:, 0].reshape(X_interp.shape)
        Y_cart = cart_shifts[:, 1].reshape(Y_interp.shape)

        return X_cart, Y_cart, Z_interp

    def old_run_surface_matching(
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
        batch_atoms_list = [self._get_shifted_atoms(shift) for shift in shifts]
        batch_inputs = [self._generate_inputs(b) for b in batch_atoms_list]

        # atoms_list = self._get_shifted_atoms(shifts)
        # inputs = self._generate_inputs(atoms_list)

        x_grid = np.linspace(0, 1, self.grid_density_x)
        y_grid = np.linspace(0, 1, self.grid_density_y)
        X, Y = np.meshgrid(x_grid, y_grid)

        if self.z_PES_data is None:
            z_interface_coulomb_energy = np.vstack(
                [
                    self._calculate_coulomb(b, z_shift=True)
                    for b in batch_inputs
                ]
            )
            z_interface_born_energy = np.vstack(
                [self._calculate_born(b, z_shift=True) for b in batch_inputs]
            )
            # z_coulomb_energy = self._calculate_coulomb(inputs, z_shift=True)
            # z_born_energy = self._calculate_born(inputs, z_shift=True)
            # z_interface_coulomb_energy = z_coulomb_energy.reshape(X.shape)
            # z_interface_born_energy = z_born_energy.reshape(X.shape)
            self.z_PES_data = [
                z_interface_coulomb_energy,
                z_interface_born_energy,
            ]
        else:
            z_interface_coulomb_energy = self.z_PES_data[0]
            z_interface_born_energy = self.z_PES_data[1]

        interface_coulomb_energy = np.vstack(
            [self._calculate_coulomb(b, z_shift=False) for b in batch_inputs]
        )
        interface_born_energy = np.vstack(
            [self._calculate_born(b, z_shift=False) for b in batch_inputs]
        )

        # coulomb_energy = self._calculate_coulomb(inputs, z_shift=False)
        # born_energy = self._calculate_born(inputs, z_shift=False)
        # interface_coulomb_energy = coulomb_energy.reshape(X.shape)
        # interface_born_energy = born_energy.reshape(X.shape)

        coulomb_adh_energy = (
            z_interface_coulomb_energy - interface_coulomb_energy
        )
        born_adh_energy = z_interface_born_energy - interface_born_energy

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

            ax1.plot(
                borders[:, 0],
                borders[:, 1],
                color="black",
                linewidth=1,
                zorder=300,
            )

            ax2.plot(
                borders[:, 0],
                borders[:, 1],
                color="black",
                linewidth=1,
                zorder=300,
            )

            ax3.plot(
                borders[:, 0],
                borders[:, 1],
                color="black",
                linewidth=1,
                zorder=300,
            )

            self._plot_surface_matching(
                fig=fig,
                ax=ax1,
                X=X,
                Y=Y,
                Z=born_adh_energy / self.interface.area,
                dpi=dpi,
                cmap=cmap,
                fontsize=fontsize,
                show_max=show_max,
                shift=False,
            )

            self._plot_surface_matching(
                fig=fig,
                ax=ax2,
                X=X,
                Y=Y,
                Z=coulomb_adh_energy / self.interface.area,
                dpi=dpi,
                cmap=cmap,
                fontsize=fontsize,
                show_max=show_max,
                shift=False,
            )

            max_Z = self._plot_surface_matching(
                fig=fig,
                ax=ax3,
                X=X,
                Y=Y,
                Z=(born_adh_energy + coulomb_adh_energy) / self.interface.area,
                dpi=dpi,
                cmap=cmap,
                fontsize=fontsize,
                show_max=show_max,
                shift=True,
            )

            ax1.set_xlim(borders[:, 0].min(), borders[:, 0].max())
            ax1.set_ylim(borders[:, 1].min(), borders[:, 1].max())
            ax1.set_aspect("equal")

            ax2.set_xlim(borders[:, 0].min(), borders[:, 0].max())
            ax2.set_ylim(borders[:, 1].min(), borders[:, 1].max())
            ax2.set_aspect("equal")

            ax3.set_xlim(borders[:, 0].min(), borders[:, 0].max())
            ax3.set_ylim(borders[:, 1].min(), borders[:, 1].max())
            ax3.set_aspect("equal")
        else:
            fig, ax = plt.subplots(
                figsize=(5, 5 * ratio),
                dpi=dpi,
            )

            ax.plot(
                borders[:, 0],
                borders[:, 1],
                color="black",
                linewidth=1,
                zorder=300,
            )

            max_Z = self._plot_surface_matching(
                fig=fig,
                ax=ax,
                X=X,
                Y=Y,
                Z=(born_adh_energy + coulomb_adh_energy) / self.interface.area,
                dpi=dpi,
                cmap=cmap,
                fontsize=fontsize,
                show_max=show_max,
                shift=True,
            )

            ax.set_xlim(borders[:, 0].min(), borders[:, 0].max())
            ax.set_ylim(borders[:, 1].min(), borders[:, 1].max())
            ax.set_aspect("equal")

        fig.tight_layout()
        fig.savefig(output, bbox_inches="tight")
        plt.close(fig)

        return max_Z

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
        interface_atoms = self.interface.get_interface(
            orthogonal=True, return_atoms=True
        )
        interface_atoms.set_array(
            "is_film",
            self.interface._orthogonal_structure.site_properties["is_film"],
        )

        # batch_inputs = [
        #     for batch_shift in shifts
        # ]

        energies = []
        film_force_norms = []
        film_force_norm_grads = []
        # for inputs in batch_inputs:
        for batch_shift in shifts:
            inputs = self._generate_inputs(
                atoms=interface_atoms, shifts=batch_shift, interface=True
            )
            (
                batch_energies,
                _,
                _,
                batch_film_force_norms,
                batch_film_force_norm_grads,
            ) = self._calculate(inputs)
            energies.append(batch_energies)
            film_force_norms.append(batch_film_force_norms)
            film_force_norm_grads.append(batch_film_force_norm_grads)

        interface_energy = np.vstack(energies)
        interface_film_force_norms = np.vstack(film_force_norms)

        x_grid = np.linspace(0, 1, self.grid_density_x)
        y_grid = np.linspace(0, 1, self.grid_density_y)
        X, Y = np.meshgrid(x_grid, y_grid)

        # Z = interface_film_force_norms
        Z = interface_energy

        a = self.matrix[0, :2]
        b = self.matrix[1, :2]

        borders = np.vstack([np.zeros(2), a, a + b, b, np.zeros(2)])

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

        ax.plot(
            borders[:, 0],
            borders[:, 1],
            color="black",
            linewidth=1,
            zorder=300,
        )

        max_Z = self._plot_surface_matching(
            fig=fig,
            ax=ax,
            X=X,
            Y=Y,
            Z=Z,
            dpi=dpi,
            cmap=cmap,
            fontsize=fontsize,
            show_max=show_max,
            shift=True,
        )

        # opt_positions = self.run_surface_matching_grad()

        # inds = np.linspace(0, 1, len(opt_positions))
        # red = np.array([1, 0, 0])
        # blue = np.array([0, 0, 1])
        # colors = (inds[:, None] * blue[None, :]) + (
        #     (1 - inds)[:, None] * red[None, :]
        # )
        # ax.scatter(
        #     opt_positions[:, 0],
        #     opt_positions[:, 1],
        #     c=colors,
        # )

        # print(np.round(interface_lj_force_vec, 5))
        # for batch_shift, batch_force_vecs in zip(
        #     shifts, film_force_norm_grads
        # ):
        #     for shift, force_vec in zip(batch_shift, batch_force_vecs):
        #         norm_force_vec = (
        #             -(force_vec / np.linalg.norm(force_vec)) * 0.20
        #         )
        #         if norm_force_vec[-1] > 0:
        #             white = np.ones(3)
        #             green = np.array([1, 0, 0])
        #             z_frac = norm_force_vec[-1] / 0.21
        #             fc = (((1 - z_frac) * white) + (z_frac * green)).tolist()
        #         elif norm_force_vec[-1] < 0:
        #             white = np.ones(3)
        #             purple = np.array([0, 0, 1])
        #             z_frac = -norm_force_vec[-1] / 0.21
        #             fc = (((1 - z_frac) * white) + (z_frac * purple)).tolist()
        #         else:
        #             fc = "white"

        #         ax.arrow(
        #             x=shift[0],
        #             y=shift[1],
        #             dx=norm_force_vec[0],
        #             dy=norm_force_vec[1],
        #             width=0.04,
        #             fc=fc,
        #             ec="black",
        #         )

        ax.set_xlim(borders[:, 0].min(), borders[:, 0].max())
        ax.set_ylim(borders[:, 1].min(), borders[:, 1].max())
        ax.set_aspect("equal")

        fig.tight_layout()
        fig.savefig(output, bbox_inches="tight")
        plt.close(fig)

        return max_Z

    def run_z_shift(
        self,
        interfacial_distances,
        fontsize: int = 12,
        output: str = "PES.png",
        show_born_and_coulomb: bool = False,
        dpi: int = 400,
    ):
        zeros = np.zeros(len(interfacial_distances))
        shifts = np.c_[zeros, zeros, interfacial_distances - self.d_interface]

        interface_atoms = self.interface.get_interface(
            orthogonal=True, return_atoms=True
        )
        interface_atoms.set_array(
            "is_film",
            self.interface._orthogonal_structure.site_properties["is_film"],
        )

        inputs = self._generate_inputs(
            atoms=interface_atoms, shifts=shifts, interface=True
        )

        (
            interface_energy,
            coulomb,
            born,
            interface_film_force_norms,
            interface_film_force_norm_grads,
        ) = self._calculate(inputs)

        fig, axs = plt.subplots(figsize=(4 * 3, 3), dpi=dpi, ncols=3)

        print(interfacial_distances[np.argmin(interface_energy)])
        np.savez(
            "3d_ewald_energies.npz",
            total=interface_energy,
            born=born,
            coulomb=coulomb,
            dists=interfacial_distances,
        )

        axs[0].plot(
            interfacial_distances,
            interface_energy,
            color="black",
            linewidth=1,
            label="Born+Coulomb",
        )
        axs[1].plot(
            interfacial_distances,
            coulomb,
            color="red",
            linewidth=1,
            label="Coulomb",
        )
        axs[2].plot(
            interfacial_distances,
            born,
            color="blue",
            linewidth=1,
            label="Born",
        )

        for ax in axs:
            ax.tick_params(labelsize=fontsize)
            ax.set_ylabel("Energy", fontsize=fontsize)
            ax.set_xlabel("Interfacial Distance ($\\AA$)", fontsize=fontsize)
            ax.legend(fontsize=12)

        fig.tight_layout()
        fig.savefig(output, bbox_inches="tight")
        plt.close(fig)
