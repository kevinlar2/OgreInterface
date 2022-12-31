from OgreInterface.generate import InterfaceGenerator, SurfaceGenerator
from pymatgen.core.surface import get_symmetrically_distinct_miller_indices
from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from ase import Atoms
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

from functools import reduce
from itertools import combinations_with_replacement



class MillerSearch(object):

    """Docstring for MillerSearch. """

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
    ):
        if type(substrate) == str:
            self.substrate = Structure.from_file(substrate)
        elif type(substrate) == Structure:
            self.substrate = substrate
        elif type(substrate) == Atoms:
            self.substrate = AseAtomsAdaptor().get_structure(substrate)
        else:
            raise TypeError(f"MillerSearch accepts 'pymatgen.core.structure.Structure', 'ase.Atoms', or 'str', not '{type(substrate).__name__}'")

        if type(film) == str:
            self.film = Structure.from_file(film)
        elif type(film) == Structure:
            self.film = film
        elif type(film) == Atoms:
            self.film = AseAtomsAdaptor().get_structure(film)
        else:
            raise TypeError(f"MillerSearch accepts 'pymatgen.core.structure.Structure', 'ase.Atoms', or 'str', not '{type(film).__name__}'")

        self.max_film_index = max_film_index
        self.max_substrate_index = max_substrate_index
        self.area_tol = area_tol
        self.angle_tol = angle_tol
        self.length_tol = length_tol
        self.max_area = max_area
        self.substrate_inds, self.film_inds = self._get_unique_miller_indices()
        self.misfit_data = None
        self.area_data = None
        self.count_data = None 

    def _float_gcd(self, a, b, rtol = 1e-05, atol = 1e-08):
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
        u = (1/3) * ((2 * uvw[0]) - uvw[1])
        v = (1/3) * ((2 * uvw[1]) - uvw[0])
        t = -(u + v)
        w = uvw[-1]

        output = np.array([u, v, t, w])
        gcd = np.abs(reduce(self._float_gcd, output))
        output /= gcd

        return output.astype(int)

    def _get_unique_miller_indices(self):
        sub_sg = SpacegroupAnalyzer(self.substrate)
        film_sg = SpacegroupAnalyzer(self.film)

        sub_conventional_cell = sub_sg.get_conventional_standard_structure()
        film_conventional_cell = film_sg.get_conventional_standard_structure()

        unique_sub_inds = np.array(
            get_symmetrically_distinct_miller_indices(sub_conventional_cell, max_index=self.max_substrate_index)
        )
        unique_film_inds = np.array(
            get_symmetrically_distinct_miller_indices(film_conventional_cell, max_index=self.max_film_index)
        )

        sub_norms = np.linalg.norm(unique_sub_inds, axis=1)
        film_norms = np.linalg.norm(unique_film_inds, axis=1)
        
        sub_sort = np.argsort(sub_norms)
        film_sort = np.argsort(film_norms)

        sorted_unique_sub_inds = unique_sub_inds[sub_sort]
        sorted_unique_film_inds = unique_film_inds[film_sort]


        return sorted_unique_sub_inds, sorted_unique_film_inds

    def run_scan(self):
        substrates = []
        films = []

        for inds in self.substrate_inds:
            substrate = SurfaceGenerator(
                structure=self.substrate,
                miller_index=inds,
                layers=3,
                vacuum=20,
                generate_all=False,
            )
            substrates.append(substrate.slabs[0])

        for inds in self.film_inds:
            film = SurfaceGenerator(
                structure=self.film,
                miller_index=inds,
                layers=3,
                vacuum=20,
                generate_all=False,
            )
            films.append(film.slabs[0])

        misfits = np.ones((len(substrates), len(films))) * np.nan
        areas = np.ones((len(substrates), len(films))) * np.nan
        counts = np.ones((len(substrates), len(films))) * np.nan

        for i, substrate in enumerate(substrates):
            for j, film in enumerate(films):
                interface = InterfaceGenerator(
                    substrate=substrate,
                    film=film,
                    length_tol=self.length_tol,
                    angle_tol=self.angle_tol,
                    area_tol=self.area_tol,
                    max_area=self.max_area,
                )
                if interface.interface_output is not None:
                    strain = interface.strain
                    angle_diff = interface.angle_diff
                    strains = np.c_[strain, angle_diff]
                    max_misfits = strains[:, np.argmax(np.abs(strains), axis=1)]
                    min_strain = np.min(np.abs(max_misfits))
                    #  min_strain = np.min(np.abs(interface.area_ratio))
                    #  min_strain = np.min(np.abs(strain))
                    misfits[i,j] = min_strain 
                    areas[i,j] = np.min(interface.substrate_areas)
                    counts[i,j] = len(max_misfits)

        self.misfits = np.round(misfits.T, 8)
        self.areas = areas.T
        self.counts = counts.T

    def plot_misfits(
        self,
        cmap='rainbow',
        dpi=400,
        output='misfit_plot.png',
        fontsize=12,
        figsize=(5.5,4),
        labelrotation=20,
        substrate_label=None,
        film_label=None,
    ):
        ylabels = []
        for ylabel in self.film_inds:
            tmp_label = [str(i) if i >= 0 else '$\\overline{' + str(-i) + '}$' for i in ylabel]
            ylabels.append(f'({"".join(tmp_label)})')
            # ylabels.append(str(tmp_label).replace('[', '(').replace(']', ')').replace(' ', ''))

        xlabels = []
        for xlabel in self.substrate_inds:
            tmp_label = [str(i) if i >= 0 else '$\\overline{' + str(-i) + '}$' for i in xlabel]
            xlabels.append(f'({"".join(tmp_label)})')

        # ylabels = [f'{i}'.replace('[', '(').replace(']', ')').replace(' ', '') for i in self.film_inds]
        # xlabels = [f'{i}'.replace('[', '(').replace(']', ')').replace(' ', '') for i in self.substrate_inds]

        N = len(self.film_inds)
        M = len(self.substrate_inds)
        x, y = np.meshgrid(np.arange(M), np.arange(N))
        s = self.areas
        c = self.misfits* 100
        
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax_divider = make_axes_locatable(ax)

        cax = ax_divider.append_axes(
            "right",
            size=np.min(figsize)*0.04,
            pad=np.min(figsize)*0.01,
        )

        if film_label is None:
            ax.set_ylabel('Film Miller Index', fontsize=fontsize)
        else:
            ax.set_ylabel(film_label + ' Miller Index', fontsize=fontsize)

        if substrate_label is None:
            ax.set_xlabel('Substrate Miller Index', fontsize=fontsize)
        else:
            ax.set_xlabel(substrate_label + ' Miller Index', fontsize=fontsize)

        R = 0.9 * s/np.nanmax(s)/2
        circles = [plt.Circle((i,j), radius=r, edgecolor='black', lw=3) for r, i, j in zip(R.flat, x.flat, y.flat)]
        col = PatchCollection(
            circles,
            array=c.flatten(),
            cmap=cmap,
            norm=Normalize(
                vmin=np.max([np.nanmin(c) - 0.01 * np.nanmin(c), 0]),
                vmax=np.nanmax(c) + 0.01 * np.nanmax(c),
            ),
            edgecolor='black',
            linewidth=1,
        )
        ax.add_collection(col)

        ax.set(
            xticks=np.arange(M),
            yticks=np.arange(N),
            xticklabels=xlabels,
            yticklabels=ylabels,
        )
        ax.set_xticks(np.arange(M+1)-0.5, minor=True)
        ax.set_yticks(np.arange(N+1)-0.5, minor=True)
        ax.tick_params(axis='x', labelrotation=labelrotation)
        ax.tick_params(labelsize=fontsize)
        ax.grid(which='minor', linestyle=':', linewidth=0.75)

        cbar = fig.colorbar(col, cax=cax)
        cbar.set_label('Misfit Percentage', fontsize=fontsize)
        cbar.ax.tick_params(labelsize=fontsize)
        cbar.ax.ticklabel_format(style='sci', scilimits=(-3,3), useMathText=True)
        cbar.ax.yaxis.set_offset_position('left')

        fig.tight_layout(pad=0.4)
        fig.savefig(output)
        plt.close(fig)


if __name__ == "__main__":
    ms = MillerSearch(
        substrate='./dd-poscars/POSCAR_InAs_conv',
        film='./dd-poscars/POSCAR_Al_conv',
        max_film_index=2,
        max_substrate_index=2,
        length_tol=0.01,
        angle_tol=0.01,
        area_tol=0.01,
        max_area=500,
    )
    ms.run_scan()
    ms.plot_misfits(figsize=(6.5,5), fontsize=17, labelrotation=0)
    


