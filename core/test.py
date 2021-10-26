from generate import InterfaceGenerator, SurfaceGenerator
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from sklearn.metrics import pairwise_distances
from pymatgen.io.vasp.inputs import Poscar
from scipy.spatial.distance import cdist
from vaspvis.utils import passivator
from itertools import product
import copy
import numpy as np
import time

def kl_overlap(coords, r):
    x1 = coords[:,0].reshape(-1,1) @ np.ones((1, len(coords)))
    x2 = x1.T

    y1 = coords[:,1].reshape(-1,1) @ np.ones((1, len(coords)))
    y2 = y1.T

    z1 = coords[:,2].reshape(-1,1) @ np.ones((1, len(coords)))
    z2 = z1.T

    r1 = r.reshape(-1,1) @ np.ones(r.shape[0]).reshape(1,-1)
    r2 = r1.T

    xs = x1 - x2
    ys = y1 - y2
    zs = z1 - z2

    t1 = np.log(r2/r1)
    t2 = 3
    t3 = (xs**2 + ys**2 + zs**2) / r2
    t4 = 3 * (r1 / r2) 

    kl = (t1 - t2 + t3 + t4) / 2

    kl_up = kl[np.triu_indices(kl.shape[0], 1)]
    kl_down = kl[np.tril_indices(kl.shape[0], -1)]

    means = np.mean([np.abs(kl_up), np.abs(kl_down)], axis=1)
    
    return 1 / np.linalg.norm(np.abs(kl))

def overlap(coords, r):
    d = cdist(coords, coords)
    r1 = r.reshape(-1,1) @ np.ones(r.shape[0]).reshape(1,-1)
    r2 = r1.T
    d[d - r1 - r2 > 0] = 0
    n1 = np.pi * (r1 + r2 - d)**2
    n2 = d**2 + (2*d*r1) + (2*d*r2) + (6*r1*r2) - (3*(r1**2)) - (3*(r2**2))
    volume = np.divide((n1 * n2), (12 * d), out=np.zeros_like(d), where=(d != 0))

    return volume

sub_layer = 5
film_layer = 5

subs = SurfaceGenerator.from_file(
    './poscars/POSCAR_InAs_conv',
    miller_index=[1,1,1],
    layers=sub_layer,
    vacuum=5,
)

films = SurfaceGenerator.from_file(
    './poscars/POSCAR_Al_conv',
    miller_index=[1,1,1],
    layers=film_layer,
    vacuum=5,
)

inter = InterfaceGenerator(
    substrate=subs.slabs[0],
    film=films.slabs[0],
    length_tol=0.01,
    angle_tol=0.01,
    area_tol=0.01,
    max_area=500,
    interfacial_distance=2,
    vacuum=30,
)

interface = inter.generate_interfaces()[0]

#  interface.run_surface_matching([0,0.5], [0,0.5])

interface_ref = copy.deepcopy(interface)
interface_ref.shift_film([0,0,5])

struc = interface.interface

#  Poscar(struc).write_file('POSCAR_AlInAs')

matrix = struc.lattice.matrix
test_array = np.c_[1 + np.arange(4), np.zeros(4), np.zeros(4)]

struc_ref = interface_ref.interface

interface_height = interface.interface_height
species = np.array(struc.species, dtype=str)
film_inds = np.where(struc.frac_coords[:,-1] > interface_height)[0]

coords_ref = struc_ref.cart_coords
coords = struc.cart_coords

rs = {'As': 1.19, 'In': 1.42, 'Al': 1.43}
#  rs = {'In': 1.582, 'As': 1.27, 'Al': 1.43}
radii = np.array([rs[i] for i in species])

overlap_ref = kl_overlap(coords_ref, radii).sum()

grid = 200
shift_dist = 1
#  shift_dist = 1

frac_shifts = np.array(list(product(np.linspace(0,shift_dist,grid), np.linspace(0,shift_dist,grid))))
indices = list(product(range(grid), range(grid)))

frac_shifts = np.c_[frac_shifts, np.zeros(frac_shifts.shape[0])]

overlaps = np.zeros((grid,grid))

film_i = list(range(0,10))
sub_i = list(range(61,65)) + list(range(81,85))
ids = film_i + sub_i

for shift, inds in zip(frac_shifts, indices):
    struc.translate_sites(indices=film_inds, vector=shift)
    overlaps[inds[0], inds[1]] = kl_overlap(struc.cart_coords[ids], 2*radii[ids])
    #  .sum()
    struc.translate_sites(indices=film_inds, vector=-shift)

#  score = (overlaps - overlap_ref)**2
score = overlaps


fig, ax = plt.subplots(figsize=(4,4.5), dpi=400)
ax.set_xlabel(r"Fractional Shift in $\vec{a}$ Direction", fontsize=12)
ax.set_ylabel(r"Fractional Shift in $\vec{b}$ Direction", fontsize=12)

score -= score.min()
score /= score.max()
score = gaussian_filter(score.T, 2)

im = ax.pcolormesh(
#  im = ax.contourf(
    np.linspace(0, shift_dist, grid),
    np.linspace(0, shift_dist, grid),
    score,
    cmap='jet',
    shading='gouraud',
    #  levels=200,
    norm=Normalize(vmin=score.min(), vmax=score.max()),
)

cbar = fig.colorbar(im, ax=ax, orientation='horizontal')
cbar.ax.tick_params(labelsize=12)
cbar.ax.locator_params(nbins=4)
cbar.set_label('Score', fontsize=12)
ax.tick_params(labelsize=12)
fig.tight_layout()
fig.savefig('graph.png')




