from OgreInterface.score_function.ionic_shifted_force import (
    IonicShiftedForcePotential,
)

from OgreInterface.score_function.generate_inputs import create_batch
from OgreInterface.surfaces import Interface
from OgreInterface.surface_match.base_surface_matcher import BaseSurfaceMatcher
from pymatgen.core.periodic_table import Element
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.io.ase import AseAtomsAdaptor
from ase.data import chemical_symbols, covalent_radii
from typing import List, Dict
from ase import Atoms
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline, CubicSpline
from itertools import groupby, combinations_with_replacement, product
import torch
import time


class IonicSurfaceMatcherSF(BaseSurfaceMatcher):
    def __init__(
        self,
        interface: Interface,
        grid_density: float = 2.5,
    ):
        super().__init__(
            interface=interface,
            grid_density=grid_density,
        )
        self.cutoff = 18.0
        self.charge_dict = self._get_charges()
        self.r0_array = self._get_r0s(
            sub=self.interface.substrate.bulk_structure,
            film=self.interface.film.bulk_structure,
            charge_dict=self.charge_dict,
        )
        self._add_born_ns(self.iface)
        self._add_born_ns(self.sub_part)
        self._add_born_ns(self.film_part)
        self._add_charges(self.iface)
        self._add_charges(self.sub_part)
        self._add_charges(self.film_part)
        self.d_interface = self.interface.interfacial_distance
        self.opt_xy_shift = np.zeros(2)
        self.opt_d_interface = self.d_interface

        self.iface_inputs = self._generate_base_inputs(structure=self.iface)
        self.sub_inputs = self._generate_base_inputs(structure=self.sub_part)
        self.film_inputs = self._generate_base_inputs(structure=self.film_part)
        self.film_energy, self.sub_energy = self._get_film_sub_energies()

    def get_optimized_structure(self):
        opt_shift = self.opt_xy_shift

        self.interface.shift_film_inplane(
            x_shift=opt_shift[0], y_shift=opt_shift[1], fractional=True
        )
        self.interface.set_interfacial_distance(
            interfacial_distance=self.opt_d_interface
        )

        self.iface = self.interface.get_interface(orthogonal=True)
        self.iface_inputs = self._generate_base_inputs(structure=self.iface)

        self.opt_xy_shift[:2] = 0.0

    def _add_charges(self, struc):
        charges = [
            self.charge_dict[chemical_symbols[z]] for z in struc.atomic_numbers
        ]
        struc.add_site_property("charges", charges)

    def _add_born_ns(self, struc):
        ion_config_to_n_map = {
            "[He]": 5,
            "[Ne]": 7,
            "[Ar]": 9,
            "[Kr]": 10,
            "[Xe]": 12,
        }
        n_vals = {}

        Zs = np.unique(struc.atomic_numbers)
        for z in Zs:
            element = Element(chemical_symbols[z])
            ion_config = element.electronic_structure.split(".")[0]
            n_val = ion_config_to_n_map[ion_config]
            # n_vals[z] = n_val
            n_vals[z] = 12.0

        ns = [n_vals[z] for z in struc.atomic_numbers]
        struc.add_site_property("born_ns", ns)

        # sub_ns = [n_vals[z] for z in self.sub_part.atomic_numbers]
        # self.sub_part.add_site_property("born_ns", sub_ns)

        # film_ns = [n_vals[z] for z in self.film_part.atomic_numbers]
        # self.film_part.add_site_property("born_ns", film_ns)

    # def _get_ns_dict(self):
    #     ion_config_to_n_map = {
    #         "[He]": 5,
    #         "[Ne]": 7,
    #         "[Ar]": 9,
    #         "[Kr]": 10,
    #         "[Xe]": 12,
    #     }
    #     n_vals = {}

    #     Zs = np.unique(self.interface._orthogonal_structure.atomic_numbers)
    #     for z in Zs:
    #         element = Element(chemical_symbols[z])
    #         ion_config = element.electronic_structure.split(".")[0]
    #         n_val = ion_config_to_n_map[ion_config]
    #         n_vals[z] = n_val

    #     combos = combinations_with_replacement(Zs, 2)
    #     ns_array = np.zeros((118, 118))

    #     for combo in combos:
    #         i = combo[0]
    #         j = combo[1]

    #         # q_i = self.charge_dict[chemical_symbols[i]]
    #         # q_j = self.charge_dict[chemical_symbols[j]]

    #         ns_array[i, j] = (n_vals[i] + n_vals[j]) / 2
    #         ns_array[j, i] = (n_vals[i] + n_vals[j]) / 2
    #         # ns_array[i, j] = 12.0
    #         # ns_array[j, i] = 12.0
    #         # if q_i * q_j < 0:
    #         # else:
    #         #     n_dict[(i, j)] = 6.0
    #         #     n_dict[(j, i)] = 6.0

    #     return ns_array

    # def _get_ns_dict_old(self):
    #     ion_config_to_n_map = {
    #         "[He]": 5,
    #         "[Ne]": 7,
    #         "[Ar]": 9,
    #         "[Kr]": 10,
    #         "[Xe]": 12,
    #     }
    #     n_vals = {}

    #     Zs = np.unique(self.interface._orthogonal_structure.atomic_numbers)
    #     for z in Zs:
    #         element = Element(chemical_symbols[z])
    #         ion_config = element.electronic_structure.split(".")[0]
    #         n_val = ion_config_to_n_map[ion_config]
    #         n_vals[z] = n_val

    #     combos = combinations_with_replacement(Zs, 2)

    #     n_dict = {}
    #     for combo in combos:
    #         i = combo[0]
    #         j = combo[1]

    #         # q_i = self.charge_dict[chemical_symbols[i]]
    #         # q_j = self.charge_dict[chemical_symbols[j]]

    #         n_dict[(i, j)] = (n_vals[i] + n_vals[j]) / 2
    #         n_dict[(j, i)] = (n_vals[i] + n_vals[j]) / 2

    #         # if q_i * q_j < 0:
    #         # else:
    #         #     n_dict[(i, j)] = 6.0
    #         #     n_dict[(j, i)] = 6.0

    #     return n_dict

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

        return neighbor_dict

    def _get_r0s(self, sub, film, charge_dict):
        r0_array = np.zeros((3, 118, 118))
        sub_dict = self._get_neighborhood_info(sub, charge_dict)
        film_dict = self._get_neighborhood_info(film, charge_dict)
        # sub_dict[(35, 55)] = 3.63
        # film_dict[(35, 55)] = 3.63

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
        for key in interface_combos:
            charge_sign = (
                charge_dict[chemical_symbols[key[0]]]
                * charge_dict[chemical_symbols[key[1]]]
            )

            if charge_sign < 0:
                ionic_sum_d = (
                    ionic_radius_dict[key[0]] + ionic_radius_dict[key[1]]
                )
                cov_sum_d = cov_radius_dict[key[0]] + cov_radius_dict[key[1]]
                ionic_sum_d = min(ionic_sum_d, cov_sum_d)
            else:
                ionic_sum_d = (
                    ionic_radius_dict[key[0]] + ionic_radius_dict[key[1]]
                )
                cov_sum_d = cov_radius_dict[key[0]] + cov_radius_dict[key[1]]
                ionic_sum_d = min(ionic_sum_d, cov_sum_d)

            r0_array[:, key[0], key[1]] = ionic_sum_d
            r0_array[:, key[1], key[0]] = ionic_sum_d

        all_keys = np.array(list(sub_dict.keys()) + list(film_dict.keys()))
        unique_keys = np.unique(all_keys, axis=0)
        unique_keys = list(map(tuple, unique_keys))

        for key in unique_keys:
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
                sub_d = min(sub_dict[key], ionic_sum_d)
                film_d = min(film_dict[key], ionic_sum_d)
                r0_array[0, key[0], key[1]] = film_d
                r0_array[1, key[0], key[1]] = (sub_d + film_d) / 2
                r0_array[2, key[0], key[1]] = sub_d
                r0_array[0, key[1], key[0]] = film_d
                r0_array[1, key[1], key[0]] = (sub_d + film_d) / 2
                r0_array[2, key[1], key[0]] = sub_d

            if key in sub_dict and key not in film_dict:
                sub_d = min(sub_dict[key], ionic_sum_d)
                r0_array[0, key[0], key[1]] = sub_d
                r0_array[1, key[0], key[1]] = sub_d
                r0_array[2, key[0], key[1]] = ionic_sum_d
                r0_array[0, key[1], key[0]] = sub_d
                r0_array[1, key[1], key[0]] = sub_d
                r0_array[2, key[1], key[0]] = ionic_sum_d

            if key not in sub_dict and key in film_dict:
                film_d = min(film_dict[key], ionic_sum_d)
                r0_array[0, key[0], key[1]] = ionic_sum_d
                r0_array[1, key[0], key[1]] = film_d
                r0_array[2, key[0], key[1]] = film_d
                r0_array[0, key[1], key[0]] = ionic_sum_d
                r0_array[1, key[1], key[0]] = film_d
                r0_array[2, key[1], key[0]] = film_d

            if key not in sub_dict and key not in film_dict:
                r0_array[0, key[0], key[1]] = ionic_sum_d
                r0_array[1, key[0], key[1]] = ionic_sum_d
                r0_array[2, key[0], key[1]] = ionic_sum_d
                r0_array[0, key[1], key[0]] = ionic_sum_d
                r0_array[1, key[1], key[0]] = ionic_sum_d
                r0_array[2, key[1], key[0]] = ionic_sum_d

        return torch.from_numpy(r0_array).to(dtype=torch.float32)

    # def _get_r0s_old(self, sub, film, charge_dict):
    #     sub_dict = self._get_neighborhood_info(sub, charge_dict)
    #     film_dict = self._get_neighborhood_info(film, charge_dict)

    #     interface_atomic_numbers = np.unique(
    #         np.concatenate([sub.atomic_numbers, film.atomic_numbers])
    #     )

    #     ionic_radius_dict = {}
    #     cov_radius_dict = {
    #         n: covalent_radii[n] for n in interface_atomic_numbers
    #     }

    #     for n in interface_atomic_numbers:
    #         element = Element(chemical_symbols[n])

    #         try:
    #             d = element.ionic_radii[charge_dict[chemical_symbols[n]]]
    #         except KeyError:
    #             print(
    #                 f"No ionic radius available for {chemical_symbols[n]}, using the atomic radius instead"
    #             )
    #             d = float(element.atomic_radius)

    #         ionic_radius_dict[n] = d

    #     interface_combos = product(interface_atomic_numbers, repeat=2)
    #     interface_neighbor_dict = {}
    #     for c in interface_combos:
    #         interface_neighbor_dict[(0, 0) + c] = None
    #         interface_neighbor_dict[(1, 1) + c] = None
    #         interface_neighbor_dict[(0, 1) + c] = None
    #         interface_neighbor_dict[(1, 0) + c] = None

    #     all_keys = np.array(list(sub_dict.keys()) + list(film_dict.keys()))
    #     unique_keys = np.unique(all_keys, axis=0)
    #     unique_keys = list(map(tuple, unique_keys))

    #     for key in unique_keys:
    #         rev_key = tuple(reversed(key))
    #         charge_sign = (
    #             charge_dict[chemical_symbols[key[0]]]
    #             * charge_dict[chemical_symbols[key[1]]]
    #         )

    #         if charge_sign < 0:
    #             ionic_sum_d = (
    #                 ionic_radius_dict[key[0]] + ionic_radius_dict[key[1]]
    #             )
    #         else:
    #             ionic_sum_d = cov_radius_dict[key[0]] + cov_radius_dict[key[1]]

    #         if key in sub_dict and key in film_dict:
    #             sub_d = sub_dict[key]
    #             film_d = film_dict[key]
    #             interface_neighbor_dict[(0, 0) + key] = sub_d
    #             interface_neighbor_dict[(1, 1) + key] = film_d
    #             interface_neighbor_dict[(0, 1) + key] = (sub_d + film_d) / 2
    #             interface_neighbor_dict[(1, 0) + key] = (sub_d + film_d) / 2
    #             interface_neighbor_dict[(0, 0) + rev_key] = sub_d
    #             interface_neighbor_dict[(1, 1) + rev_key] = film_d
    #             interface_neighbor_dict[(0, 1) + rev_key] = (
    #                 sub_d + film_d
    #             ) / 2
    #             interface_neighbor_dict[(1, 0) + rev_key] = (
    #                 sub_d + film_d
    #             ) / 2

    #         if key in sub_dict and key not in film_dict:
    #             sub_d = sub_dict[key]
    #             interface_neighbor_dict[(0, 0) + key] = sub_d
    #             interface_neighbor_dict[(1, 1) + key] = ionic_sum_d
    #             interface_neighbor_dict[(0, 1) + key] = sub_d
    #             interface_neighbor_dict[(1, 0) + key] = sub_d
    #             interface_neighbor_dict[(0, 0) + rev_key] = sub_d
    #             interface_neighbor_dict[(1, 1) + rev_key] = ionic_sum_d
    #             interface_neighbor_dict[(0, 1) + rev_key] = sub_d
    #             interface_neighbor_dict[(1, 0) + rev_key] = sub_d

    #         if key not in sub_dict and key in film_dict:
    #             film_d = film_dict[key]
    #             interface_neighbor_dict[(1, 1) + key] = film_d
    #             interface_neighbor_dict[(0, 0) + key] = ionic_sum_d
    #             interface_neighbor_dict[(0, 1) + key] = film_d
    #             interface_neighbor_dict[(1, 0) + key] = film_d
    #             interface_neighbor_dict[(1, 1) + rev_key] = film_d
    #             interface_neighbor_dict[(0, 0) + rev_key] = ionic_sum_d
    #             interface_neighbor_dict[(0, 1) + rev_key] = film_d
    #             interface_neighbor_dict[(1, 0) + rev_key] = film_d

    #         if key not in sub_dict and key not in film_dict:
    #             interface_neighbor_dict[(0, 0) + key] = ionic_sum_d
    #             interface_neighbor_dict[(1, 1) + key] = ionic_sum_d
    #             interface_neighbor_dict[(0, 1) + key] = ionic_sum_d
    #             interface_neighbor_dict[(1, 0) + key] = ionic_sum_d
    #             interface_neighbor_dict[(0, 0) + rev_key] = ionic_sum_d
    #             interface_neighbor_dict[(1, 1) + rev_key] = ionic_sum_d
    #             interface_neighbor_dict[(0, 1) + rev_key] = ionic_sum_d
    #             interface_neighbor_dict[(1, 0) + rev_key] = ionic_sum_d

    #     for key, val in interface_neighbor_dict.items():
    #         if val is None:
    #             ionic_sum_d = (
    #                 ionic_radius_dict[key[2]] + ionic_radius_dict[key[3]]
    #             )
    #             interface_neighbor_dict[key] = ionic_sum_d

    #     # for key in interface_neighbor_dict:
    #     #     print(key, interface_neighbor_dict[key])
    #     # interface_neighbor_dict[key] *= 0.90

    #     return interface_neighbor_dict

    # def _get_ewald_parameters(self):
    #     struc = self.interface._orthogonal_structure
    #     max_len = max(struc.lattice.a, struc.lattice.b)
    #     struc_vol = self.interface._structure_volume
    #     accf = np.sqrt(np.log(10**4))
    #     struc_len = len(self.interface._orthogonal_structure)
    #     w = 1 / 2**0.5
    #     alpha = np.sqrt(np.pi * (struc_len * w / (struc_vol**2)) ** (1 / 3))
    #     cutoff = accf / np.sqrt(alpha)
    #     r_max = int(np.ceil(cutoff / max_len))
    #     k_max = int(np.ceil(2 * alpha * accf))
    #     print(alpha)
    #     print(cutoff)
    #     print(k_max)

    #     return cutoff, alpha, k_max, r_max

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

    # def _generate_inputs(self, atoms, shifts, interface=True):
    #     inputs = generate_dict_torch(
    #         atoms=atoms,
    #         shifts=shifts,
    #         cutoff=self.cutoff,
    #         interface=interface,
    #         charge_dict=self.charge_dict,
    #         ns_dict=self.ns_dict,
    #     )

    #     return inputs

    def bo_function(self, a, b, z):
        frac_ab = np.array([a, b]).reshape(1, 2)
        cart_xy = self.get_cart_xy_shifts(frac_ab)
        z_shift = z - self.d_interface
        shift = np.c_[cart_xy, z_shift * np.ones(len(cart_xy))]
        batch_inputs = create_batch(
            inputs=self.iface_inputs,
            batch_size=1,
        )
        E, _, _, _, _ = self._calculate(inputs=batch_inputs, shifts=shift)

        return -E[0]

    def pso_function(self, x):
        # frac_ab = np.array([a, b]).reshape(1, 2)
        cart_xy = self.get_cart_xy_shifts(x[:, :2])
        z_shift = x[:, -1] - self.d_interface
        shift = np.c_[cart_xy, z_shift]
        batch_inputs = create_batch(
            inputs=self.iface_inputs,
            batch_size=len(x),
        )
        E, _, _, _, _ = self._calculate(inputs=batch_inputs, shifts=shift)
        E_adh = (
            -((self.film_energy + self.sub_energy) - E) / self.interface.area
        )

        return E_adh

    def _get_film_sub_energies(self):
        sub_inputs = create_batch(inputs=self.sub_inputs, batch_size=1)
        film_inputs = create_batch(inputs=self.film_inputs, batch_size=1)

        sub_energy, _, _, _, _ = self._calculate(
            sub_inputs,
            shifts=np.zeros((1, 3)),
        )
        film_energy, _, _, _, _ = self._calculate(
            film_inputs,
            shifts=np.zeros((1, 3)),
        )

        return film_energy, sub_energy

    def optimizePSO(
        self, z_bounds, max_iters: int = 200, n_particles: int = 15
    ):

        opt_score, opt_pos = self._optimizerPSO(
            func=self.pso_function,
            z_bounds=z_bounds,
            max_iters=max_iters,
            n_particles=n_particles,
        )

        opt_cart = self.get_cart_xy_shifts(opt_pos[:2].reshape(1, -1))
        opt_cart = np.c_[opt_cart, np.zeros(1)]
        opt_frac = opt_cart.dot(self.inv_matrix)[0]

        self.opt_xy_shift = opt_frac[:2]
        self.opt_d_interface = opt_pos[-1]

        return opt_score

    def optimize(self, z_bounds, max_iters):
        probe_xy = self._get_gd_init_points()
        probe_xy = np.vstack([probe_xy, probe_xy])
        probe_ab = self.get_frac_xy_shifts(probe_xy)
        probe_z = np.random.uniform(
            z_bounds[0],
            z_bounds[1],
            len(probe_ab),
        )
        probe_points = np.c_[probe_ab, probe_z]

        opt_score, opt_pos = self._optimizer_old(
            func=self.bo_function,
            z_bounds=z_bounds,
            max_iters=max_iters,
            probe_points=probe_points,
        )

        self.opt_xy_shift = opt_pos[:2]
        self.opt_d_interface = opt_pos[-1]

        return opt_score

    def _calculate(self, inputs: Dict, shifts: np.ndarray):
        ionic_potential = IonicShiftedForcePotential(
            cutoff=self.cutoff,
        )
        outputs = ionic_potential.forward(
            inputs=inputs,
            shift=torch.from_numpy(shifts).to(dtype=torch.float32),
            r0_array=self.r0_array,
        )

        return outputs

    # def old_run_surface_matching(
    #     self,
    #     cmap: str = "jet",
    #     fontsize: int = 14,
    #     output: str = "PES.png",
    #     shift: bool = True,
    #     show_born_and_coulomb: bool = False,
    #     dpi: int = 400,
    #     show_max: bool = False,
    # ) -> float:
    #     shifts = self.shifts
    #     batch_atoms_list = [self._get_shifted_atoms(shift) for shift in shifts]
    #     batch_inputs = [self._generate_inputs(b) for b in batch_atoms_list]

    #     # atoms_list = self._get_shifted_atoms(shifts)
    #     # inputs = self._generate_inputs(atoms_list)

    #     x_grid = np.linspace(0, 1, self.grid_density_x)
    #     y_grid = np.linspace(0, 1, self.grid_density_y)
    #     X, Y = np.meshgrid(x_grid, y_grid)

    #     if self.z_PES_data is None:
    #         z_interface_coulomb_energy = np.vstack(
    #             [
    #                 self._calculate_coulomb(b, z_shift=True)
    #                 for b in batch_inputs
    #             ]
    #         )
    #         z_interface_born_energy = np.vstack(
    #             [self._calculate_born(b, z_shift=True) for b in batch_inputs]
    #         )
    #         # z_coulomb_energy = self._calculate_coulomb(inputs, z_shift=True)
    #         # z_born_energy = self._calculate_born(inputs, z_shift=True)
    #         # z_interface_coulomb_energy = z_coulomb_energy.reshape(X.shape)
    #         # z_interface_born_energy = z_born_energy.reshape(X.shape)
    #         self.z_PES_data = [
    #             z_interface_coulomb_energy,
    #             z_interface_born_energy,
    #         ]
    #     else:
    #         z_interface_coulomb_energy = self.z_PES_data[0]
    #         z_interface_born_energy = self.z_PES_data[1]

    #     interface_coulomb_energy = np.vstack(
    #         [self._calculate_coulomb(b, z_shift=False) for b in batch_inputs]
    #     )
    #     interface_born_energy = np.vstack(
    #         [self._calculate_born(b, z_shift=False) for b in batch_inputs]
    #     )

    #     # coulomb_energy = self._calculate_coulomb(inputs, z_shift=False)
    #     # born_energy = self._calculate_born(inputs, z_shift=False)
    #     # interface_coulomb_energy = coulomb_energy.reshape(X.shape)
    #     # interface_born_energy = born_energy.reshape(X.shape)

    #     coulomb_adh_energy = (
    #         z_interface_coulomb_energy - interface_coulomb_energy
    #     )
    #     born_adh_energy = z_interface_born_energy - interface_born_energy

    #     a = self.matrix[0, :2]
    #     b = self.matrix[1, :2]
    #     borders = np.vstack([np.zeros(2), a, a + b, b, np.zeros(2)])
    #     x_size = borders[:, 0].max() - borders[:, 0].min()
    #     y_size = borders[:, 1].max() - borders[:, 1].min()
    #     ratio = y_size / x_size

    #     if show_born_and_coulomb:
    #         fig, (ax1, ax2, ax3) = plt.subplots(
    #             figsize=(3 * 5, 5 * ratio),
    #             ncols=3,
    #             dpi=dpi,
    #         )

    #         ax1.plot(
    #             borders[:, 0],
    #             borders[:, 1],
    #             color="black",
    #             linewidth=1,
    #             zorder=300,
    #         )

    #         ax2.plot(
    #             borders[:, 0],
    #             borders[:, 1],
    #             color="black",
    #             linewidth=1,
    #             zorder=300,
    #         )

    #         ax3.plot(
    #             borders[:, 0],
    #             borders[:, 1],
    #             color="black",
    #             linewidth=1,
    #             zorder=300,
    #         )

    #         self._plot_surface_matching(
    #             fig=fig,
    #             ax=ax1,
    #             X=X,
    #             Y=Y,
    #             Z=born_adh_energy / self.interface.area,
    #             dpi=dpi,
    #             cmap=cmap,
    #             fontsize=fontsize,
    #             show_max=show_max,
    #             shift=False,
    #         )

    #         self._plot_surface_matching(
    #             fig=fig,
    #             ax=ax2,
    #             X=X,
    #             Y=Y,
    #             Z=coulomb_adh_energy / self.interface.area,
    #             dpi=dpi,
    #             cmap=cmap,
    #             fontsize=fontsize,
    #             show_max=show_max,
    #             shift=False,
    #         )

    #         max_Z = self._plot_surface_matching(
    #             fig=fig,
    #             ax=ax3,
    #             X=X,
    #             Y=Y,
    #             Z=(born_adh_energy + coulomb_adh_energy) / self.interface.area,
    #             dpi=dpi,
    #             cmap=cmap,
    #             fontsize=fontsize,
    #             show_max=show_max,
    #             shift=True,
    #         )

    #         ax1.set_xlim(borders[:, 0].min(), borders[:, 0].max())
    #         ax1.set_ylim(borders[:, 1].min(), borders[:, 1].max())
    #         ax1.set_aspect("equal")

    #         ax2.set_xlim(borders[:, 0].min(), borders[:, 0].max())
    #         ax2.set_ylim(borders[:, 1].min(), borders[:, 1].max())
    #         ax2.set_aspect("equal")

    #         ax3.set_xlim(borders[:, 0].min(), borders[:, 0].max())
    #         ax3.set_ylim(borders[:, 1].min(), borders[:, 1].max())
    #         ax3.set_aspect("equal")
    #     else:
    #         fig, ax = plt.subplots(
    #             figsize=(5, 5 * ratio),
    #             dpi=dpi,
    #         )

    #         ax.plot(
    #             borders[:, 0],
    #             borders[:, 1],
    #             color="black",
    #             linewidth=1,
    #             zorder=300,
    #         )

    #         max_Z = self._plot_surface_matching(
    #             fig=fig,
    #             ax=ax,
    #             X=X,
    #             Y=Y,
    #             Z=(born_adh_energy + coulomb_adh_energy) / self.interface.area,
    #             dpi=dpi,
    #             cmap=cmap,
    #             fontsize=fontsize,
    #             show_max=show_max,
    #             shift=True,
    #         )

    #         ax.set_xlim(borders[:, 0].min(), borders[:, 0].max())
    #         ax.set_ylim(borders[:, 1].min(), borders[:, 1].max())
    #         ax.set_aspect("equal")

    #     fig.tight_layout()
    #     fig.savefig(output, bbox_inches="tight")
    #     plt.close(fig)

    #     return max_Z

    # def run_surface_matching(
    #     self,
    #     cmap: str = "jet",
    #     fontsize: int = 14,
    #     output: str = "PES.png",
    #     shift: bool = True,
    #     show_born_and_coulomb: bool = False,
    #     dpi: int = 400,
    #     show_max: bool = False,
    # ) -> float:
    #     shifts = self.shifts
    #     interface_atoms = self.interface.get_interface(
    #         orthogonal=True,
    #         return_atoms=True,
    #     )
    #     sub_atoms = self.interface.get_substrate_supercell(
    #         orthogonal=True,
    #         return_atoms=True,
    #     )
    #     film_atoms = self.interface.get_film_supercell(
    #         orthogonal=True,
    #         return_atoms=True,
    #     )

    #     sub_inputs = self._generate_inputs(
    #         atoms=sub_atoms,
    #         shifts=[np.zeros(3)],
    #         interface=True,
    #     )

    #     film_inputs = self._generate_inputs(
    #         atoms=film_atoms,
    #         shifts=[np.zeros(3)],
    #         interface=True,
    #     )

    #     sub_energy, _, _, _, _ = self._calculate(sub_inputs)
    #     film_energy, _, _, _, _ = self._calculate(film_inputs)

    #     # print("Interface is_film:", interface_atoms.get_array("is_film"))
    #     # print("Sub only is_film:", sub_atoms.get_array("is_film"))
    #     # print("Film only is_film:", film_atoms.get_array("is_film"))
    #     # interface_atoms.set_array(
    #     #     "is_film",
    #     #     self.interface._orthogonal_structure.site_properties["is_film"],
    #     # )

    #     energies = []
    #     grads = []
    #     # for inputs in batch_inputs:
    #     for batch_shift in shifts:
    #         inputs = self._generate_inputs(
    #             atoms=interface_atoms, shifts=batch_shift, interface=True
    #         )
    #         (
    #             batch_energies,
    #             _,
    #             _,
    #             batch_grads,
    #             _,
    #         ) = self._calculate(inputs)
    #         # print("Calculate:", round(time.time() - s, 4))
    #         energies.append(batch_energies)
    #         grads.append(batch_grads)
    #         # film_force_norm_grads.append(batch_film_force_norm_grads)

    #     interface_energy = np.vstack(energies)

    #     x_grid = np.linspace(0, 1, self.grid_density_x)
    #     y_grid = np.linspace(0, 1, self.grid_density_y)
    #     X, Y = np.meshgrid(x_grid, y_grid)

    #     # Z = interface_film_force_norms
    #     Z = (
    #         -((film_energy + sub_energy) - interface_energy)
    #         / self.interface.area
    #     )

    #     a = self.matrix[0, :2]
    #     b = self.matrix[1, :2]

    #     borders = np.vstack([np.zeros(2), a, a + b, b, np.zeros(2)])

    #     x_size = borders[:, 0].max() - borders[:, 0].min()
    #     y_size = borders[:, 1].max() - borders[:, 1].min()

    #     ratio = y_size / x_size

    #     if ratio < 1:
    #         figx = 5 / ratio
    #         figy = 5
    #     else:
    #         figx = 5
    #         figy = 5 * ratio

    #     fig, ax = plt.subplots(
    #         figsize=(figx, figy),
    #         dpi=dpi,
    #     )

    #     ax.plot(
    #         borders[:, 0],
    #         borders[:, 1],
    #         color="black",
    #         linewidth=1,
    #         zorder=300,
    #     )

    #     max_Z = self._plot_surface_matching(
    #         fig=fig,
    #         ax=ax,
    #         X=X,
    #         Y=Y,
    #         Z=Z,
    #         dpi=dpi,
    #         cmap=cmap,
    #         fontsize=fontsize,
    #         show_max=show_max,
    #         shift=True,
    #     )

    #     # opt_positions = self.run_surface_matching_grad()

    #     # inds = np.linspace(0, 1, len(opt_positions))
    #     # red = np.array([1, 0, 0])
    #     # blue = np.array([0, 0, 1])
    #     # colors = (inds[:, None] * blue[None, :]) + (
    #     #     (1 - inds)[:, None] * red[None, :]
    #     # )
    #     # ax.scatter(
    #     #     opt_positions[:, 0],
    #     #     opt_positions[:, 1],
    #     #     c=colors,
    #     # )
    #     # gd_start, gd_colors = self._get_gd_init_points()
    #     # ax.scatter(gd_start[:, 0], gd_start[:, 1], c=gd_colors, zorder=1000)

    #     # print(np.round(interface_lj_force_vec, 5))
    #     # for batch_shift, batch_grads in zip(shifts, grads):
    #     #     for shift, grad in zip(batch_shift, batch_grads):
    #     #         grad_norm = -(grad / np.linalg.norm(grad)) * 0.50
    #     #         if grad_norm[-1] > 0:
    #     #             white = np.ones(3)
    #     #             green = np.array([1, 0, 0])
    #     #             z_frac = grad_norm[-1] / 0.51
    #     #             fc = (((1 - z_frac) * white) + (z_frac * green)).tolist()
    #     #         elif grad_norm[-1] < 0:
    #     #             white = np.ones(3)
    #     #             purple = np.array([0, 0, 1])
    #     #             z_frac = -grad_norm[-1] / 0.51
    #     #             fc = (((1 - z_frac) * white) + (z_frac * purple)).tolist()
    #     #         else:
    #     #             fc = "white"

    #     #         ax.arrow(
    #     #             x=shift[0],
    #     #             y=shift[1],
    #     #             dx=grad_norm[0],
    #     #             dy=grad_norm[1],
    #     #             width=0.04,
    #     #             fc=fc,
    #     #             ec="black",
    #     #         )

    #     ax.set_xlim(borders[:, 0].min(), borders[:, 0].max())
    #     ax.set_ylim(borders[:, 1].min(), borders[:, 1].max())
    #     ax.set_aspect("equal")

    #     fig.tight_layout()
    #     fig.savefig(output, bbox_inches="tight")
    #     plt.close(fig)

    #     return max_Z

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

        interface_energy = []
        coulomb = []
        born = []
        for shift in shifts:
            inputs = create_batch(self.iface_inputs, batch_size=1)

            (
                _interface_energy,
                _coulomb,
                _born,
                _,
                _,
            ) = self._calculate(inputs, shifts=shift.reshape(1, -1))
            interface_energy.append(_interface_energy)
            coulomb.append(_coulomb)
            born.append(_born)

        interface_energy = (
            -(
                (self.film_energy + self.sub_energy)
                - np.array(interface_energy)
            )
            / self.interface.area
        )

        fig, axs = plt.subplots(
            figsize=(3, 3),
            dpi=dpi,
            # ncols=3,
        )

        cs = CubicSpline(interfacial_distances, interface_energy)
        new_x = np.linspace(
            interfacial_distances.min(),
            interfacial_distances.max(),
            201,
        )
        new_y = cs(new_x)

        opt_d = new_x[np.argmin(new_y)]
        opt_E = np.min(new_y)
        self.opt_d_interface = opt_d

        axs.plot(
            new_x,
            new_y,
            color="black",
            linewidth=1,
            label="Born+Coulomb",
        )
        axs.scatter(
            [opt_d],
            [opt_E],
            color="black",
            marker="x",
        )
        axs.tick_params(labelsize=fontsize)
        axs.set_ylabel("Energy", fontsize=fontsize)
        axs.set_xlabel("Interfacial Distance ($\\AA$)", fontsize=fontsize)
        axs.legend(fontsize=12)

        fig.tight_layout()
        fig.savefig(output, bbox_inches="tight")
        plt.close(fig)

        return opt_E

    def run_bulk(
        self,
        strains,
        fontsize: int = 12,
        output: str = "PES.png",
        show_born_and_coulomb: bool = False,
        dpi: int = 400,
    ):
        # sub = self.interface.substrate.bulk_structure
        sub = self.interface.film.bulk_structure
        is_film = True

        strained_atoms = []
        for strain in strains:
            strain_struc = sub.copy()
            strain_struc.apply_strain(strain)
            strain_struc.add_site_property(
                "is_film", [is_film] * len(strain_struc)
            )
            self._add_charges(strain_struc)
            self._add_born_ns(strain_struc)
            strained_atoms.append(strain_struc)

        total_energy = []
        coulomb = []
        born = []
        for i, atoms in enumerate(strained_atoms):
            inputs = self._generate_base_inputs(
                structure=atoms,
            )
            batch_inputs = create_batch(inputs, 1)

            (
                _total_energy,
                _coulomb,
                _born,
                _,
                _,
            ) = self._calculate(batch_inputs, shifts=np.zeros((1, 3)))
            total_energy.append(_total_energy)
            coulomb.append(_coulomb)
            born.append(_born)

        total_energy = np.array(total_energy)
        coulomb = np.array(coulomb)
        born = np.array(born)

        fig, axs = plt.subplots(figsize=(4 * 3, 3), dpi=dpi, ncols=3)
        print("Min Strain:", strains[np.argmin(total_energy)])

        axs[0].plot(
            strains,
            total_energy,
            color="black",
            linewidth=1,
            label="Born+Coulomb",
        )
        axs[1].plot(
            strains,
            coulomb,
            color="red",
            linewidth=1,
            label="Coulomb",
        )
        axs[2].plot(
            strains,
            born,
            color="blue",
            linewidth=1,
            label="Born",
        )

        for ax in axs:
            ax.tick_params(labelsize=fontsize)
            ax.set_ylabel("Energy", fontsize=fontsize)
            ax.set_xlabel("Strain ($\\AA$)", fontsize=fontsize)
            ax.legend(fontsize=12)

        fig.tight_layout()
        fig.savefig(output, bbox_inches="tight")
        plt.close(fig)

    def run_scale(
        self,
        scales,
        fontsize: int = 12,
        output: str = "scale.png",
        show_born_and_coulomb: bool = False,
        dpi: int = 400,
    ):
        # sub = self.interface.substrate.bulk_structure
        sub = self.interface.film.bulk_structure

        strains = np.linspace(-0.1, 0.1, 21)
        strained_atoms = []
        # for strain in [-0.02, -0.01, 0.0, 0.01, 0.02]:
        for strain in strains:
            strain_struc = sub.copy()
            strain_struc.apply_strain(strain)
            strain_atoms = AseAtomsAdaptor().get_atoms(strain_struc)
            strain_atoms.set_array(
                "is_film", np.zeros(len(strain_atoms)).astype(bool)
            )
            strained_atoms.append(strain_atoms)

        total_energy = []
        for scale in scales:
            strain_energy = []
            for atoms in strained_atoms:
                inputs = self._generate_inputs(
                    atoms=atoms, shifts=[np.zeros(3)], interface=False
                )
                ionic_potential = IonicShiftedForcePotential(
                    cutoff=self.cutoff,
                )
                (_total_energy, _, _, _, _,) = ionic_potential.forward(
                    inputs=inputs,
                    r0_dict=scale * self.r0_array,
                    ns_dict=self.ns_dict,
                    z_shift=False,
                )
                strain_energy.append(_total_energy)
            total_energy.append(strain_energy)

        total_energy = np.array(total_energy)
        # coulomb = np.array(coulomb)
        # born = np.array(born)

        fig, axs = plt.subplots(figsize=(6, 3), dpi=dpi, ncols=2)

        colors = plt.cm.jet
        color_list = [colors(i) for i in np.linspace(0, 1, len(total_energy))]

        min_strains = []
        min_Es = []
        for i, E in enumerate(total_energy):
            E -= E.min()
            E /= E.max()
            axs[0].plot(
                strains,
                E,
                color=color_list[i],
                linewidth=1,
                # marker=".",
                # alpha=0.3,
            )
            min_strain = strains[np.argmin(E)]
            min_E = E.min()
            min_strains.append(min_strain)
            min_Es.append(min_E)
            axs[0].scatter(
                [min_strain],
                [min_E],
                c=[color_list[i]],
                s=2,
            )

        axs[1].plot(
            scales, np.array(min_strains) ** 2, color="black", marker="."
        )

        fig.tight_layout()
        fig.savefig(output, bbox_inches="tight")
        plt.close(fig)
