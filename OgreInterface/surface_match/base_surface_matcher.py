from OgreInterface.surfaces import Interface
from typing import List
import numpy as np
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Polygon
from copy import deepcopy
from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
import torch


class BaseSurfaceMatcher:
    def __init__(
        self,
        interface: Interface,
        grid_density: float = 2.5,
    ):
        self.interface = interface
        self.matrix = deepcopy(interface._orthogonal_structure.lattice.matrix)
        self._vol = np.linalg.det(self.matrix)

        if self._vol < 0:
            self.matrix *= -1
            self._vol *= -1

        self.grid_density = grid_density

        (
            self.shift_matrix,
            self.shift_images,
        ) = self._get_shift_matrix_and_images()

        self.shifts = self._generate_shifts()

    def _optimizer(self, func, z_bounds, max_iters, probe_points):
        pbounds = {"a": (0, 1), "b": (0, 1), "z": z_bounds}
        optimizer = BayesianOptimization(
            f=func,
            pbounds=pbounds,
            verbose=1,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
            random_state=1,
            allow_duplicate_points=True,
        )

        # for point in probe_points:
        #     optimizer.probe(
        #         params={"a": point[0], "b": point[1], "z": point[2]},
        #         lazy=True,
        #     )

        # logger = JSONLogger(path="./logs.json")
        # optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

        optimizer.maximize(
            init_points=200,
            n_iter=1000,
        )

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

        return prim_cart_shifts

    def get_cart_xy_shifts(self, a, b):
        cart_xyz = (np.array([a, b, 0.0]) + self.shift_images[0]).dot(
            self.shift_matrix
        )

        return cart_xyz[0], cart_xyz[1]

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
                # label = (
                #     "$E_{adh}$ (eV/$\\AA^{2}$) : "
                #     + "$E_{max}$ = "
                #     + f"{E_max:.4f}"
                # )
                label = (
                    "$|F_{film}|$ (eV/$\\AA$) : "
                    + "$|F|_{min}$ = "
                    + f"{E_max:.4f}"
                )
                cbar.set_label(label, fontsize=fontsize)
            else:
                label = "$E_{adh}$ (eV/$\\AA^{2}$)"
                cbar.set_label(label, fontsize=fontsize)

            cax.xaxis.set_ticks_position("top")
            cax.xaxis.set_label_position("top")
            ax.tick_params(labelsize=fontsize)

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
                opt_shift = frac_shifts[np.argmax(Z_plot.ravel())]
                max_Z = np.min(Z_plot)
                plot_shift = opt_shift.dot(self.matrix)

                # ax.scatter(
                #     [plot_shift[0]],
                #     [plot_shift[1]],
                #     fc="white",
                #     ec="black",
                #     marker="X",
                #     s=100,
                #     zorder=10,
                # )

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
