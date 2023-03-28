from OgreInterface.score_function.lj import LJ

# from OgreInterface.score_function.generate_inputs import generate_dict_torch
from OgreInterface.surfaces import Interface
from OgreInterface.surface_match.base_surface_matcher import BaseSurfaceMatcher
from pymatgen.analysis.local_env import CrystalNN
from ase.data import chemical_symbols, covalent_radii
from typing import List
from ase import Atoms
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline
from itertools import groupby, combinations_with_replacement, product
import torch


class LJSurfaceMatcher(BaseSurfaceMatcher):
    def __init__(
        self,
        interface: Interface,
        grid_density: float = 2.5,
        cutoff: float = 7.0,
    ):
        super().__init__(
            interface=interface,
            grid_density=grid_density,
        )
        self.cutoff = cutoff
        self.r0_dict = self._get_r0s(
            sub=self.interface.substrate.bulk_structure,
            film=self.interface.film.bulk_structure,
        )
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

    def _get_charges(self):
        sub = self.interface.substrate.bulk_structure
        film = self.interface.film.bulk_structure
        sub_oxidation_state = sub.composition.oxi_state_guesses()[0]
        film_oxidation_state = film.composition.oxi_state_guesses()[0]

        sub_oxidation_state.update(film_oxidation_state)

        return sub_oxidation_state

    def _get_neighborhood_info(self, struc):
        Zs = np.unique(struc.atomic_numbers)
        combos = combinations_with_replacement(Zs, 2)
        neighbor_dict = {c: None for c in combos}

        neighbor_list = []

        cnn = CrystalNN(search_cutoff=7.0)
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
            if d is None:
                neighbor_dict[n] = covalent_radii[n[0]] + covalent_radii[n[1]]
            else:
                if d < 4.0:
                    neighbor_dict[n] = (
                        covalent_radii[n[0]] + covalent_radii[n[1]]
                    )

        return neighbor_dict

    def _get_r0s(self, sub, film):
        sub_dict = self._get_neighborhood_info(sub)
        film_dict = self._get_neighborhood_info(film)

        interface_atomic_numbers = np.unique(
            np.concatenate([sub.atomic_numbers, film.atomic_numbers])
        )

        covalent_radius_dict = {
            n: covalent_radii[n] for n in interface_atomic_numbers
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
            covalent_sum_d = (
                covalent_radius_dict[key[0]] + covalent_radius_dict[key[1]]
            )
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
                interface_neighbor_dict[(1, 1) + key] = covalent_sum_d
                interface_neighbor_dict[(0, 1) + key] = sub_d
                interface_neighbor_dict[(1, 0) + key] = sub_d
                interface_neighbor_dict[(0, 0) + rev_key] = sub_d
                interface_neighbor_dict[(1, 1) + rev_key] = covalent_sum_d
                interface_neighbor_dict[(0, 1) + rev_key] = sub_d
                interface_neighbor_dict[(1, 0) + rev_key] = sub_d

            if key not in sub_dict and key in film_dict:
                film_d = film_dict[key]
                interface_neighbor_dict[(1, 1) + key] = film_d
                interface_neighbor_dict[(0, 0) + key] = covalent_sum_d
                interface_neighbor_dict[(0, 1) + key] = film_d
                interface_neighbor_dict[(1, 0) + key] = film_d
                interface_neighbor_dict[(1, 1) + rev_key] = film_d
                interface_neighbor_dict[(0, 0) + rev_key] = covalent_sum_d
                interface_neighbor_dict[(0, 1) + rev_key] = film_d
                interface_neighbor_dict[(1, 0) + rev_key] = film_d

            if key not in sub_dict and key not in film_dict:
                interface_neighbor_dict[(0, 0) + key] = covalent_sum_d
                interface_neighbor_dict[(1, 1) + key] = covalent_sum_d
                interface_neighbor_dict[(0, 1) + key] = covalent_sum_d
                interface_neighbor_dict[(1, 0) + key] = covalent_sum_d
                interface_neighbor_dict[(0, 0) + rev_key] = covalent_sum_d
                interface_neighbor_dict[(1, 1) + rev_key] = covalent_sum_d
                interface_neighbor_dict[(0, 1) + rev_key] = covalent_sum_d
                interface_neighbor_dict[(1, 0) + rev_key] = covalent_sum_d

        for key, val in interface_neighbor_dict.items():
            if val is None:
                covalent_sum_d = (
                    covalent_radius_dict[key[2]] + covalent_radius_dict[key[3]]
                )
                interface_neighbor_dict[key] = covalent_sum_d

        return interface_neighbor_dict

    # def _get_shifted_atoms(self, shifts: np.ndarray) -> List[Atoms]:
    #     atoms = []

    #     for shift in shifts:
    #         # Shift in-plane
    #         self.interface.shift_film_inplane(
    #             x_shift=shift[0], y_shift=shift[1], fractional=True
    #         )

    #         # Get inplane shifted atoms
    #         shifted_atoms = self.interface.get_interface(
    #             orthogonal=True, return_atoms=True
    #         )

    #         # Add the is_film property
    #         shifted_atoms.set_array(
    #             "is_film",
    #             self.interface._orthogonal_structure.site_properties[
    #                 "is_film"
    #             ],
    #         )

    #         self.interface.shift_film_inplane(
    #             x_shift=-shift[0], y_shift=-shift[1], fractional=True
    #         )

    #         # Add atoms to the list
    #         atoms.append(shifted_atoms)

    #     return atoms

    def _generate_inputs(self, atoms, shifts, interface=True):
        inputs = generate_dict_torch(
            atoms=atoms,
            shifts=shifts,
            cutoff=self.cutoff,
            interface=interface,
        )

        return inputs

    def _calculate_lj(
        self,
        inputs,
    ):
        lj = LJ(cutoff=self.cutoff)
        lj_energy = lj.forward(
            inputs,
            r0_dict=self.r0_dict,
        )

        return lj_energy

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

    def bo_function(self, a, b, z):
        x, y = self.get_cart_xy_shifts(a, b)
        z_shift = z - self.d_interface
        shift = torch.Tensor([a, b, z_shift])
        force = self._calculate_lj(inputs=self.inputs, shift=shift)

        return -force

    def optimize(self, z_bounds, max_iters):
        a_grid, b_grid, z_grid = np.meshgrid(
            np.linspace(0, 1, 5),
            np.linspace(0, 1, 5),
            np.linspace(z_bounds[0], z_bounds[1], 5),
        )
        probe_points = np.c_[a_grid.ravel(), b_grid.ravel(), z_grid.ravel()]
        self._optimizer(
            func=self.bo_function,
            z_bounds=z_bounds,
            max_iters=max_iters,
            probe_points=probe_points,
        )

    def get_inputs(self):
        interface_atoms = self.interface.get_interface(
            orthogonal=True, return_atoms=True
        )
        interface_atoms.set_array(
            "is_film",
            self.interface._orthogonal_structure.site_properties["is_film"],
        )

        interface_inputs = self._generate_inputs(
            [interface_atoms], interface=True
        )

        self.inputs = interface_inputs

    def run_surface_matching_grad(self):
        self.get_inputs()
        init_shift = np.ones(3) * 3.0
        init_shift[-1] = 0.0
        score_inputs = {
            "inputs": self.inputs,
            "r0_dict": self.r0_dict,
            "shift": torch.from_numpy(init_shift),
        }
        opt_positions = self._adam(
            score_func=LJ(cutoff=self.cutoff),
            score_func_inputs=score_inputs,
        )

        return opt_positions

    def run_surface_matching(
        self,
        cmap: str = "jet",
        fontsize: int = 14,
        output: str = "PES.png",
        shift: bool = True,
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

        batch_inputs = [
            self._generate_inputs(
                atoms=interface_atoms, shifts=batch_shift, interface=True
            )
            for batch_shift in shifts
        ]

        energies = []
        film_force_norms = []
        film_force_norm_grads = []
        for inputs in batch_inputs:
            (
                batch_energies,
                batch_film_force_norms,
                batch_film_force_norm_grads,
            ) = self._calculate_lj(inputs)
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
        # fig.savefig(output)
        plt.close(fig)

        return max_Z

    def run_z_shift(
        self,
        interfacial_distances,
        fontsize: int = 14,
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
            interface_film_force_norms,
            interface_film_force_norm_grads,
        ) = self._calculate_lj(inputs)

        fig, ax = plt.subplots(
            figsize=(4, 3),
            dpi=dpi,
        )
        ax.set_ylabel("Net Force", fontsize=fontsize)
        ax.set_xlabel("Interfacial Distance ($\\AA$)", fontsize=fontsize)

        ax.plot(
            interfacial_distances,
            interface_film_force_norms,
            color="black",
            linewidth=1,
        )

        fig.tight_layout()
        fig.savefig(output, bbox_inches="tight")
        plt.close(fig)
