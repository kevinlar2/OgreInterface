from OgreInterface.generate import SurfaceGenerator
from OgreInterface.lattice_match import ZurMcGill

from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.interfaces.zsl import ZSLGenerator

from ase import Atoms
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

from functools import reduce
from itertools import product


class MillerSearch(object):

    """Docstring for MillerSearch."""

    def __init__(
        self,
        substrate,
        film,
        max_substrate_index=1,
        max_film_index=1,
        area_tol=0.01,
        angle_tol=0.01,
        length_tol=0.01,
        max_area=500,
        convert_to_conventional=True,
    ):
        if type(substrate) == str:
            self.substrate = Structure.from_file(substrate)
        elif type(substrate) == Structure:
            self.substrate = substrate
        elif type(substrate) == Atoms:
            self.substrate = AseAtomsAdaptor().get_structure(substrate)
        else:
            raise TypeError(
                f"MillerSearch accepts 'pymatgen.core.structure.Structure', 'ase.Atoms', or 'str', not '{type(substrate).__name__}'"
            )

        if type(film) == str:
            self.film = Structure.from_file(film)
        elif type(film) == Structure:
            self.film = film
        elif type(film) == Atoms:
            self.film = AseAtomsAdaptor().get_structure(film)
        else:
            raise TypeError(
                f"MillerSearch accepts 'pymatgen.core.structure.Structure', 'ase.Atoms', or 'str', not '{type(film).__name__}'"
            )

        self.max_film_index = max_film_index
        self.max_substrate_index = max_substrate_index
        self.area_tol = area_tol
        self.angle_tol = angle_tol
        self.length_tol = length_tol
        self.max_area = max_area
        self.convert_to_conventional = convert_to_conventional
        self.substrate_inds = self._get_unique_miller_indices(
            self.substrate, self.max_substrate_index
        )
        self.film_inds = self._get_unique_miller_indices(
            self.film, self.max_film_index
        )
        # self.substrate_inds, self.film_inds = self._get_unique_miller_indices()
        self.misfit_data = None
        self.area_data = None
        self.count_data = None

    def _float_gcd(self, a, b, rtol=1e-05, atol=1e-08):
        t = min(abs(a), abs(b))
        while abs(b) > rtol * t + atol:
            a, b = b, a % b
        return a

    def _hex_to_cubic(self, uvtw):
        u = 2 * uvtw[0] + uvtw[1]
        v = 2 * uvtw[1] + uvtw[0]
        w = uvtw[-1]

        output = np.array([u, v, w])
        gcd = np.abs(reduce(self._float_gcd, output))
        output = output / gcd

        return output.astype(int)

    def _cubic_to_hex(self, uvw):
        u = (1 / 3) * ((2 * uvw[0]) - uvw[1])
        v = (1 / 3) * ((2 * uvw[1]) - uvw[0])
        t = -(u + v)
        w = uvw[-1]

        output = np.array([u, v, t, w])
        gcd = np.abs(reduce(self._float_gcd, output))
        output /= gcd

        return output.astype(int)

    def _get_unique_miller_indices(self, struc, max_index):
        struc_sg = SpacegroupAnalyzer(struc)
        # struc_conv_cell = struc_sg.get_conventional_standard_structure()
        symmops = struc_sg.get_symmetry_operations(cartesian=False)
        planes = set(list(product(range(-max_index, max_index + 1), repeat=3)))
        planes.remove((0, 0, 0))

        reduced_planes = []
        for plane in planes:
            gcd = np.abs(reduce(self._float_gcd, plane))
            reduced_plane = tuple((plane / gcd).astype(int))
            reduced_planes.append(reduced_plane)

        reduced_planes = set(reduced_planes)

        planes_dict = {p: [] for p in reduced_planes}

        for plane in reduced_planes:
            if plane in planes_dict.keys():
                for i, symmop in enumerate(symmops):
                    origin = symmop.operate((0, 0, 0))
                    point_out = symmop.operate(plane) - origin
                    gcd = np.abs(reduce(self._float_gcd, point_out))
                    point_out = tuple((point_out / gcd).astype(int))
                    planes_dict[plane].append(point_out)
                    if point_out != plane:
                        if point_out in planes_dict.keys():
                            del planes_dict[point_out]

        unique_planes = []

        for k in planes_dict:
            equivalent_planes = np.array(list(set(planes_dict[k])))
            diff = np.abs(np.sum(np.sign(equivalent_planes), axis=1))
            like_signs = equivalent_planes[diff == np.max(diff)]
            if len(like_signs) == 1:
                unique_planes.append(like_signs[0])
            else:
                first_max = like_signs[
                    np.abs(like_signs)[:, 0]
                    == np.max(np.abs(like_signs)[:, 0])
                ]
                if len(first_max) == 1:
                    unique_planes.append(first_max[0])
                else:
                    second_max = first_max[
                        np.abs(first_max)[:, 1]
                        == np.max(np.abs(first_max)[:, 1])
                    ]
                    if len(second_max) == 1:
                        unique_planes.append(second_max[0])
                    else:
                        unique_planes.append(
                            second_max[
                                np.argmax(np.sign(second_max).sum(axis=1))
                            ]
                        )

        unique_planes = np.vstack(unique_planes)
        sorted_planes = sorted(
            unique_planes, key=lambda x: (np.linalg.norm(x), -np.sign(x).sum())
        )

        return np.vstack(sorted_planes)

    def run_scan(self):
        substrates = []
        films = []

        for inds in self.substrate_inds:
            sg_sub = SurfaceGenerator(
                bulk=self.substrate,
                miller_index=inds,
                layers=5,
                vacuum=10,
                generate_all=False,
                lazy=True,
                convert_to_conventional=self.convert_to_conventional,
            )
            sub_inplane_vectors = sg_sub.inplane_vectors
            sub_area = np.linalg.norm(
                np.cross(sub_inplane_vectors[0], sub_inplane_vectors[1])
            )
            substrates.append([sub_inplane_vectors, sub_area])

        for inds in self.film_inds:
            sg_film = SurfaceGenerator(
                bulk=self.film,
                miller_index=inds,
                layers=5,
                vacuum=10,
                generate_all=False,
                lazy=True,
                convert_to_conventional=self.convert_to_conventional,
            )
            film_inplane_vectors = sg_film.inplane_vectors
            film_area = np.linalg.norm(
                np.cross(film_inplane_vectors[0], film_inplane_vectors[1])
            )

            films.append([film_inplane_vectors, film_area])

        misfits = np.ones((len(substrates), len(films))) * np.nan
        areas = np.ones((len(substrates), len(films))) * np.nan
        counts = np.ones((len(substrates), len(films))) * np.nan

        for i, substrate in enumerate(substrates):
            for j, film in enumerate(films):
                zm = ZurMcGill(
                    film_vectors=film[0],
                    substrate_vectors=substrate[0],
                    max_area=self.max_area,
                    max_linear_strain=self.length_tol,
                    max_angle_strain=self.angle_tol,
                    max_area_mismatch=self.area_tol,
                )
                matches = zm.run()

                if len(matches) > 0:
                    min_area_match = matches[0]
                    area = min_area_match.area
                    strain = min_area_match.linear_strain
                    misfits[i, j] = strain.max()
                    areas[i, j] = area / np.sqrt(substrate[1] * film[1])

        self.misfits = np.round(misfits.T, 8)
        self.areas = areas.T
        self.counts = counts.T

    def plot_misfits(
        self,
        cmap="rainbow",
        dpi=400,
        output="misfit_plot.png",
        fontsize=12,
        figure_scale=1,
        # figsize=(5.5, 4),
        labelrotation=20,
        substrate_label=None,
        film_label=None,
        show_in_colab=False,
    ):
        ylabels = []
        for ylabel in self.film_inds:
            tmp_label = [
                str(i) if i >= 0 else "$\\overline{" + str(-i) + "}$"
                for i in ylabel
            ]
            ylabels.append(f'({"".join(tmp_label)})')

        xlabels = []
        for xlabel in self.substrate_inds:
            tmp_label = [
                str(i) if i >= 0 else "$\\overline{" + str(-i) + "}$"
                for i in xlabel
            ]
            xlabels.append(f'({"".join(tmp_label)})')

        N = len(self.film_inds)
        M = len(self.substrate_inds)
        x, y = np.meshgrid(np.arange(M), np.arange(N))
        s = self.areas
        c = self.misfits * 100

        figsize = (figure_scale * 5 * (M / N), figure_scale * 4)

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax_divider = make_axes_locatable(ax)

        cax = ax_divider.append_axes(
            "right",
            size=np.min(figsize) * 0.04,
            pad=np.min(figsize) * 0.01,
        )

        if film_label is None:
            film_label = self.film.composition.reduced_formula

        ax.set_ylabel(film_label + " Miller Index", fontsize=fontsize)

        if substrate_label is None:
            substrate_label = self.substrate.composition.reduced_formula

        ax.set_xlabel(substrate_label + " Miller Index", fontsize=fontsize)

        R = 0.85 * s / np.nanmax(s) / 2
        circles = [
            plt.Circle((i, j), radius=r, edgecolor="black", lw=3)
            for r, i, j in zip(R.flat, x.flat, y.flat)
        ]
        col = PatchCollection(
            circles,
            array=c.flatten(),
            cmap=cmap,
            norm=Normalize(
                vmin=np.max([np.nanmin(c) - 0.01 * np.nanmin(c), 0]),
                vmax=np.nanmax(c) + 0.01 * np.nanmax(c),
            ),
            edgecolor="black",
            linewidth=1,
        )
        ax.add_collection(col)

        ax.set(
            xticks=np.arange(M),
            yticks=np.arange(N),
            xticklabels=xlabels,
            yticklabels=ylabels,
        )
        ax.set_xticks(np.arange(M + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(N + 1) - 0.5, minor=True)
        ax.tick_params(axis="x", labelrotation=labelrotation)
        ax.tick_params(labelsize=fontsize)
        ax.grid(which="minor", linestyle=":", linewidth=0.75)

        cbar = fig.colorbar(col, cax=cax)
        cbar.set_label("Misfit Percentage", fontsize=fontsize)
        cbar.ax.tick_params(labelsize=fontsize)
        cbar.ax.ticklabel_format(
            style="sci", scilimits=(-3, 3), useMathText=True
        )
        cbar.ax.yaxis.set_offset_position("left")

        ax.set_aspect("equal")
        fig.tight_layout(pad=0.4)
        fig.savefig(output, bbox_inches="tight")

        if not show_in_colab:
            plt.close(fig)


if __name__ == "__main__":
    ms = MillerSearch(
        substrate="./dd-poscars/POSCAR_InAs_conv",
        film="./dd-poscars/POSCAR_Al_conv",
        max_film_index=2,
        max_substrate_index=2,
        length_tol=0.01,
        angle_tol=0.01,
        area_tol=0.01,
        max_area=500,
    )
    ms.run_scan()
    ms.plot_misfits(figsize=(6.5, 5), fontsize=17, labelrotation=0)
