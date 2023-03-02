from OgreInterface.score_function.overlap import SphereOverlap
from OgreInterface.surface_match.base_surface_matcher import BaseSurfaceMatcher
from OgreInterface.score_function.generate_inputs import generate_dict_torch
from OgreInterface.surfaces import Interface
from ase.data import atomic_numbers
from typing import Dict, Optional, List
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline
from ase import Atoms


class SphereSurfaceMatcher(BaseSurfaceMatcher):
    def __init__(
        self,
        interface: Interface,
        radius_dict: Optional[Dict[str, float]] = None,
        grid_density: float = 2.5,
    ):
        super().__init__(
            interface=interface,
            grid_density=grid_density,
        )

        self.radius_dict = self._get_radii(radius_dict)
        self.cutoff = self._get_cutoff()
        self.d_interface = self.interface.interfacial_distance
        self.film_part = self.interface._orthogonal_film_structure
        self.sub_part = self.interface._orthogonal_substrate_structure
        self.opt_xy_shift = np.zeros(2)

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

    def _get_shifted_atoms(self, shifts: np.ndarray) -> List[Atoms]:
        atoms_list = []

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
            atoms_list.append(shifted_atoms)

        return atoms_list

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
        batch_atoms_list = [self._get_shifted_atoms(shift) for shift in shifts]
        batch_inputs = [self._generate_inputs(b) for b in batch_atoms_list]

        sub_atoms = self.interface.get_substrate_supercell(return_atoms=True)
        sub_atoms.set_array("is_film", np.zeros(len(sub_atoms)).astype(bool))

        film_atoms = self.interface.get_film_supercell(return_atoms=True)
        film_atoms.set_array("is_film", np.ones(len(film_atoms)).astype(bool))

        sub_film_atoms = [sub_atoms, film_atoms]
        sub_film_inputs = self._generate_inputs(sub_film_atoms)
        sub_film_overlap = self._calculate_overlap(sub_film_inputs)

        interface_overlap = np.vstack(
            [self._calculate_overlap(b) for b in batch_inputs]
        )

        sub_overlap = sub_film_overlap[0]
        film_overlap = sub_film_overlap[1]

        x_grid = np.linspace(0, 1, self.grid_density_x)
        y_grid = np.linspace(0, 1, self.grid_density_y)
        X, Y = np.meshgrid(x_grid, y_grid)

        # interface_overlap = overlap[2:].reshape(X.shape)

        Z = (
            sub_overlap + film_overlap - interface_overlap
        ) / self.interface.area

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

        ax.set_xlim(borders[:, 0].min(), borders[:, 0].max())
        ax.set_ylim(borders[:, 1].min(), borders[:, 1].max())
        ax.set_aspect("equal")

        fig.tight_layout()
        fig.savefig(output, bbox_inches="tight")
        plt.close(fig)

        return max_Z
