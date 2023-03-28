from OgreInterface.surfaces import Interface
from OgreInterface.score_function.generate_inputs import (
    generate_input_dict,
    create_batch,
)
from typing import List
import numpy as np
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Polygon
from scipy.interpolate import RectBivariateSpline, CubicSpline
from copy import deepcopy
from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt import SequentialDomainReductionTransformer
import torch
from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from pymatgen.core.periodic_table import Element
from pymatgen.symmetry.analyzer import SymmOp, SpacegroupAnalyzer
from pymatgen.io.vasp.inputs import Poscar
from ase.data import chemical_symbols
import itertools
import time
from pyswarms.single.global_best import GlobalBestPSO


class BaseSurfaceMatcher:
    def __init__(
        self,
        interface: Interface,
        grid_density: float = 2.5,
    ):
        self.interface = interface

        self.iface = self.interface.get_interface(orthogonal=True)
        self.film_part = self.interface.get_film_supercell(orthogonal=True)
        self.sub_part = self.interface.get_substrate_supercell(orthogonal=True)

        self.matrix = deepcopy(interface._orthogonal_structure.lattice.matrix)
        self._vol = np.linalg.det(self.matrix)

        if self._vol < 0:
            self.matrix *= -1
            self._vol *= -1

        self.inv_matrix = np.linalg.inv(self.matrix)

        self.grid_density = grid_density

        (
            self.shift_matrix,
            self.shift_images,
        ) = self._get_shift_matrix_and_images()

        self.shifts = self._generate_shifts()

    def _generate_base_inputs(self, structure: Structure):
        inputs = generate_input_dict(
            structure=structure,
            cutoff=self.cutoff,
            interface=True,
        )

        return inputs

    def _optimizerPSO(self, func, z_bounds, max_iters, n_particles: int = 15):
        bounds = (
            np.array([0.0, 0.0, z_bounds[0]]),
            np.array([1.0, 1.0, z_bounds[1]]),
        )
        options = {"c1": 0.5, "c2": 0.3, "w": 0.9}
        optimizer = GlobalBestPSO(
            n_particles=n_particles,
            dimensions=3,
            options=options,
            bounds=bounds,
        )
        cost, pos = optimizer.optimize(func, iters=max_iters)

        return cost, pos

    def _optimizer(self, func, z_bounds, max_iters, probe_points):
        from bayes_opt import UtilityFunction
        from bayes_opt.event import Events

        pbounds = {"a": (0, 1), "b": (0, 1), "z": z_bounds}
        bounds_transformer = SequentialDomainReductionTransformer(
            minimum_window=0.5
        )
        optimizer = BayesianOptimization(
            f=func,
            pbounds=pbounds,
            verbose=1,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
            random_state=1,
            allow_duplicate_points=False,
            bounds_transformer=bounds_transformer,
        )
        # utility = UtilityFunction(kind="ucb", kappa=2.5, xi=0.0)

        for point in probe_points:
            optimizer.probe(
                params={"a": point[0], "b": point[1], "z": point[2]},
                lazy=True,
            )

        optimizer._prime_subscriptions()
        optimizer.dispatch(Events.OPTIMIZATION_START)
        optimizer._prime_queue(0)

        util = UtilityFunction(
            kind="ucb",
            kappa=2.576,
            xi=0.0,
            kappa_decay=1,
            kappa_decay_delay=0,
        )

        iteration = 0
        while not optimizer._queue.empty or iteration < max_iters:
            try:
                x_probe = next(optimizer._queue)
            except StopIteration:
                util.update_params()
                x_probe = optimizer.suggest(util)
                iteration += 1
            optimizer.probe(x_probe, lazy=False)

            if optimizer._bounds_transformer and iteration > 0:
                optimizer.set_bounds(
                    optimizer._bounds_transformer.transform(optimizer._space)
                )

        optimizer.dispatch(Events.OPTIMIZATION_END)

        # optimizer.maximize(
        #     init_points=0,
        #     n_iter=1000,
        # )

    def _optimizer_old(self, func, z_bounds, max_iters, probe_points):
        pbounds = {"a": (0, 1), "b": (0, 1), "z": z_bounds}
        bounds_transformer = SequentialDomainReductionTransformer(
            minimum_window=0.5
        )
        optimizer = BayesianOptimization(
            f=func,
            pbounds=pbounds,
            verbose=1,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
            random_state=1,
            allow_duplicate_points=False,
            bounds_transformer=bounds_transformer,
        )

        for point in probe_points:
            optimizer.probe(
                params={"a": point[0], "b": point[1], "z": point[2]},
                lazy=True,
            )

        # logger = JSONLogger(path="./logs.json")
        # optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

        optimizer.maximize(
            init_points=0,
            n_iter=1000,
        )

        max_vals = optimizer.max
        max_score = max_vals["target"]
        max_shift = np.array(list(max_vals["params"].values()))

        return -max_score, max_shift

    def _get_gd_init_points(self):
        sub_struc = self.interface.substrate.oriented_bulk_structure.copy()
        is_top = sub_struc.site_properties["is_top"]
        to_del = np.where(np.logical_not(is_top))[0]
        sub_struc.remove_sites(to_del)

        sub_a_to_i_op = SymmOp.from_rotation_and_translation(
            rotation_matrix=self.interface._substrate_a_to_i,
            translation_vec=np.zeros(3),
        )
        sub_struc.apply_operation(sub_a_to_i_op)

        film_struc = self.interface.film.oriented_bulk_structure.copy()
        is_bottom = film_struc.site_properties["is_bottom"]
        to_del = np.where(np.logical_not(is_bottom))[0]
        film_struc.remove_sites(to_del)

        film_a_to_i_op = SymmOp.from_rotation_and_translation(
            rotation_matrix=self.interface._film_a_to_i,
            translation_vec=np.zeros(3),
        )
        film_struc.apply_operation(film_a_to_i_op)

        unstrained_film_matrix = film_struc.lattice.matrix
        strain_matrix = (
            self.interface._film_supercell.lattice.inv_matrix
            @ self.interface._strained_sub.lattice.matrix
        )
        strain_matrix[-1] = np.array([0, 0, 1])
        strained_matrix = unstrained_film_matrix.dot(strain_matrix.T)
        film_struc = Structure(
            lattice=Lattice(strained_matrix),
            species=film_struc.species,
            coords=film_struc.frac_coords,
            to_unit_cell=True,
            coords_are_cartesian=False,
            site_properties=film_struc.site_properties,
        )

        Poscar(sub_struc).write_file("POSCAR_sub_top")
        Poscar(film_struc).write_file("POSCAR_film_bot")

        # sub_sg = SpacegroupAnalyzer(sub_struc)
        # film_sg = SpacegroupAnalyzer(film_struc)

        # sub_dataset = sub_sg.get_symmetry_dataset()
        # film_dataset = film_sg.get_symmetry_dataset()
        # sub_equivs = sub_dataset["equivalent_atoms"]
        # film_equivs = film_dataset["equivalent_atoms"]

        sub_equivs = sub_struc.site_properties["bulk_equivalent"]
        film_equivs = film_struc.site_properties["bulk_equivalent"]

        _, sub_unique = np.unique(sub_equivs, return_index=True)
        _, film_unique = np.unique(film_equivs, return_index=True)

        print(sub_unique)
        print(film_unique)

        film_cart_coords = film_struc.cart_coords[:, :2]
        sub_cart_coords = sub_struc.cart_coords[:, :2]
        # inds = itertools.product(
        #     # film_unique,
        #     # sub_unique,
        #     # [1, 4]
        #     range(len(film_cart_coords)),
        #     range(len(sub_cart_coords)),
        # )

        shifts = []
        unique_inds = []
        for i, film_coords in enumerate(film_cart_coords):
            for j, sub_coords in enumerate(sub_cart_coords):
                shift = sub_coords - film_coords
                shifts.append(shift)
                unique_inds.append((film_equivs[i], sub_equivs[j]))
                # unique_inds.append(film_equivs[i])

        shifts = np.c_[shifts, np.zeros(len(shifts))]
        inv_matrix = np.linalg.inv(self.shift_matrix)
        frac_shifts = shifts.dot(inv_matrix)
        frac_shifts = np.round(np.mod(frac_shifts, 1), 5)
        # unique_shifts = np.unique(frac_shifts, axis=0)
        points = (frac_shifts + self.shift_images[0]).dot(self.shift_matrix)

        colors = [
            "red",
            "green",
            "blue",
            "magenta",
            "orange",
            "purple",
            "white",
            "yellow",
        ]
        color_dict = {s: c for s, c in zip(set(unique_inds), colors)}
        plot_colors = [color_dict[s] for s in unique_inds]

        return points[:, :2]
        # , plot_colors

    def _get_shift_matrix_and_images(self) -> List[np.ndarray]:
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

        return shift_matrix, shift_images

    def _generate_shifts(self) -> List[np.ndarray]:
        iface_inv_matrix = (
            self.interface._orthogonal_structure.lattice.inv_matrix
        )

        grid_density_x = int(
            np.round(np.linalg.norm(self.shift_matrix[0]) * self.grid_density)
        )
        grid_density_y = int(
            np.round(np.linalg.norm(self.shift_matrix[1]) * self.grid_density)
        )

        self.grid_density_x = grid_density_x
        self.grid_density_y = grid_density_y

        grid_x = np.linspace(0, 1, grid_density_x)
        grid_y = np.linspace(0, 1, grid_density_y)

        X, Y = np.meshgrid(grid_x, grid_y)
        self.X_shape = X.shape

        prim_frac_shifts = (
            np.c_[X.ravel(), Y.ravel(), np.zeros(Y.shape).ravel()]
            + self.shift_images[0]
        )
        prim_cart_shifts = prim_frac_shifts.dot(self.shift_matrix)
        # iface_frac_shifts = prim_cart_shifts.dot(iface_inv_matrix).reshape(
        #     X.shape + (-1,)
        # )

        return prim_cart_shifts.reshape(X.shape + (-1,))

    def get_cart_xy_shifts(self, ab):
        frac_abc = np.c_[ab, np.zeros(len(ab))]
        cart_xyz = (frac_abc + self.shift_images[0]).dot(self.shift_matrix)

        return cart_xyz[:, :2]

    def get_frac_xy_shifts(self, xy):
        cart_xyz = np.c_[xy, np.zeros(len(xy))]
        inv_shift = np.linalg.inv(self.shift_matrix)
        frac_abc = cart_xyz.dot(inv_shift)
        frac_abc = np.mod(frac_abc, 1)

        return frac_abc[:, :2]

    def get_optmized_structure(self):
        opt_shift = self.opt_xy_shift

        self.interface.shift_film_inplane(
            x_shift=opt_shift[0], y_shift=opt_shift[1], fractional=True
        )

    def _plot_heatmap(
        self, fig, ax, X, Y, Z, cmap, fontsize, show_max, add_color_bar
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

        if add_color_bar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("top", size="5%", pad=0.1)
            cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
            cbar.ax.tick_params(labelsize=fontsize)
            cbar.ax.locator_params(nbins=3)

            if show_max:
                E_max = np.min(Z)
                label = (
                    "$E_{adh}$ (eV/$\\AA^{2}$) : "
                    + "$E_{min}$ = "
                    + f"{E_max:.4f}"
                )
                # label = (
                #     "$|F_{film}|$ (eV/$\\AA$) : "
                #     + "$|F|_{min}$ = "
                #     + f"{E_max:.4f}"
                # )
                cbar.set_label(label, fontsize=fontsize)
            else:
                label = "$E_{adh}$ (eV/$\\AA^{2}$)"
                cbar.set_label(label, fontsize=fontsize)

            cax.xaxis.set_ticks_position("top")
            cax.xaxis.set_label_position("top")
            ax.tick_params(labelsize=fontsize)

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

    def _plot_surface_matching(
        self,
        fig,
        ax,
        X,
        Y,
        Z,
        dpi,
        cmap,
        fontsize,
        show_max,
        shift,
    ):
        for i, image in enumerate(self.shift_images):
            X_plot, Y_plot, Z_plot = self._get_interpolated_data(Z, image)

            if i == 0:
                self._plot_heatmap(
                    fig=fig,
                    ax=ax,
                    X=X_plot,
                    Y=Y_plot,
                    Z=Z_plot,
                    cmap=cmap,
                    fontsize=fontsize,
                    show_max=show_max,
                    add_color_bar=True,
                )

                frac_shifts = np.c_[
                    X_plot.ravel(),
                    Y_plot.ravel(),
                    np.zeros(Y_plot.shape).ravel(),
                ].dot(np.linalg.inv(self.matrix))

                opt_shift = frac_shifts[np.argmin(Z_plot.ravel())]
                opt_shift = np.mod(opt_shift, 1)
                max_Z = np.min(Z_plot)
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

                if shift:
                    self.opt_xy_shift = opt_shift[:2]
            else:
                self._plot_heatmap(
                    fig=fig,
                    ax=ax,
                    X=X_plot,
                    Y=Y_plot,
                    Z=Z_plot,
                    cmap=cmap,
                    fontsize=fontsize,
                    show_max=show_max,
                    add_color_bar=False,
                )

            coords = np.array(
                [
                    [0, 0, 0],
                    [1, 0, 0],
                    [1, 1, 0],
                    [0, 1, 0],
                    [0, 0, 0],
                ]
            )

            sc_shifts = np.array(
                [
                    [1, 0, 0],
                    [0, 1, 0],
                    [-1, 0, 0],
                    [0, -1, 0],
                    [1, 1, 0],
                    [-1, -1, 0],
                    [1, -1, 0],
                    [-1, 1, 0],
                ]
            )

            for shift in sc_shifts:
                shift_coords = (coords + shift).dot(self.matrix)
                poly = Polygon(
                    xy=shift_coords[:, :2],
                    closed=True,
                    facecolor="white",
                    edgecolor="white",
                    linewidth=1,
                    zorder=200,
                )
                ax.add_patch(poly)

        return max_Z

    def _adam(
        self,
        score_func,
        score_func_inputs,
        beta1=0.9,
        beta2=0.999,
        eta=0.01,
        epsilon=1e-7,
        iterations=300,
    ):
        inv_shift_matrix = np.linalg.inv(self.shift_matrix)
        init_position = score_func_inputs["shift"]
        opt_position = [np.copy(init_position)]
        m = np.zeros(init_position.shape)
        v = np.zeros(init_position.shape)

        for i in range(iterations):
            print(opt_position[i])
            force_norm, gradient = score_func.forward(**score_func_inputs)
            m = beta1 * m + (1 - beta1) * gradient
            v = beta2 * v + (1 - beta2) * gradient**2
            m_hat = m / (1 - beta1)
            v_hat = v / (1 - beta2)
            update = m_hat / (np.sqrt(v_hat) + epsilon)
            new_opt_position = opt_position[i] - eta * update

            new_opt_frac_coords = (
                np.array([new_opt_position[0], new_opt_position[1], 0.0]).dot(
                    inv_shift_matrix
                )
                - self.shift_images[0]
            )
            new_opt_frac_coords = np.mod(new_opt_frac_coords, 1)
            new_opt_cart_coords = (
                new_opt_frac_coords + self.shift_images[0]
            ).dot(self.shift_matrix)
            new_opt_position[:2] = new_opt_cart_coords[:2]
            opt_position.append(new_opt_position)
            score_func_inputs["shift"] = torch.from_numpy(new_opt_position)

        opt_position = np.vstack(opt_position)

        return opt_position

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

        energies = []
        grads = []
        for batch_shift in shifts:
            batch_inputs = create_batch(
                inputs=self.iface_inputs,
                batch_size=len(batch_shift),
            )
            (
                batch_energies,
                _,
                _,
                batch_grads,
                _,
            ) = self._calculate(batch_inputs, shifts=batch_shift)
            energies.append(batch_energies)
            grads.append(batch_grads)

        interface_energy = np.vstack(energies)

        x_grid = np.linspace(0, 1, self.grid_density_x)
        y_grid = np.linspace(0, 1, self.grid_density_y)
        X, Y = np.meshgrid(x_grid, y_grid)

        Z = (
            -((film_energy + sub_energy) - interface_energy)
            / self.interface.area
        )

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
