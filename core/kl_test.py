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

    means = np.mean([np.abs(kl_up), np.abs(kl_down)])
    
    return 1 / means.sum()
    #  return np.divide(1, kl, out=np.zeros_like(kl), where=(kl != 0))

def overlap(coords, r):
    d = cdist(coords, coords)
    r1 = r.reshape(-1,1) @ np.ones(r.shape[0]).reshape(1,-1)
    r2 = r1.T
    d[d - r1 - r2 > 0] = 0
    n1 = np.pi * (r1 + r2 - d)**2
    n2 = d**2 + (2*d*r1) + (2*d*r2) + (6*r1*r2) - (3*(r1**2)) - (3*(r2**2))
    volume = np.divide((n1 * n2), (12 * d), out=np.zeros_like(d), where=(d != 0))
    #  print(volume)

    return volume

sub_layer = 5
film_layer = 5

subs = SurfaceGenerator.from_file(
    './poscars/GaAs.cif',
    miller_index=[1,1,0],
    layers=sub_layer,
    vacuum=5,
)

films = SurfaceGenerator.from_file(
    './poscars/MnAs.cif',
    miller_index=[1,-1,0],
    layers=film_layer,
    vacuum=5,
)

inter = InterfaceGenerator(
    substrate=subs.slabs[0],
    film=films.slabs[0],
    length_tol=0.04,
    angle_tol=0.04,
    area_tol=0.04,
    max_area=300,
    interfacial_distance=1.7,
    vacuum=30,
)

interface = inter.generate_interfaces()[0]
Poscar(interface.interface).write_file('POSCAR_test')

lattice_vecs = np.linalg.norm(interface.interface.lattice.matrix, axis=1)
a_b_ratio = lattice_vecs[0] / lattice_vecs[1]
print(a_b_ratio)

#  interface.run_surface_matching(a_range=[0,1], b_range=[0,1])

interface_ref = copy.deepcopy(interface)
interface_ref.shift_film([0,0,5])

struc = interface.interface
matrix = struc.lattice.matrix
test_array = np.c_[1 + np.arange(4), np.zeros(4), np.zeros(4)]

struc_ref = interface_ref.interface

interface_height = interface.interface_height
print(interface_height)
species = np.array(struc.species, dtype=str)
film_inds = np.where(struc.frac_coords[:,-1] > interface_height)[0]

sub = list(range(290,300)) + list(range(190,200))
film = list(range(310,321))
ids = sub + film

coords_ref = struc_ref.cart_coords[[9,10,20,21,320]]
coords = struc.cart_coords[[9,10,20,21,320]]



rs = {'As': 1.19, 'Ga': 1.22, 'Mn': 1.5}
#  rs = {'In': 1.582, 'As': 1.27, 'Al': 1.43}
radii = np.array([rs[i] for i in species]) 

#  overlap_ref = overlap(coords_ref, radii).sum()

#  grid_a = 40
#  grid_b = 10
grid_a = 100
grid_b = 100
#  shift_dist_a = [-0.1, 0.1]
#  shift_dist_b = [-0.7, 0.7]
#  shift_dist_b = [-0.001, 0.001]
#  shift_dist_a = [-0.7, 0.7]
#  shift_dist_a = [0.04242, 2/15]
#  shift_dist_b = [0, 1]
shift_dist_a = [-1/7,1/7]
shift_dist_b = [-1,1]

frac_shifts = np.array(list(product(
    np.linspace(shift_dist_a[0], shift_dist_a[1], grid_a),
    np.linspace(shift_dist_b[0], shift_dist_b[1], grid_b),
)))
indices = list(product(range(grid_a), range(grid_b)))

frac_shifts = np.c_[frac_shifts, np.zeros(frac_shifts.shape[0])]

overlaps = np.zeros((grid_a, grid_b))

for shift, inds in zip(frac_shifts, indices):
    struc.translate_sites(indices=film_inds, vector=shift)
    overlaps[inds[0], inds[1]] = kl_overlap(
        struc.cart_coords[ids], radii[ids]
    ).sum()
    struc.translate_sites(indices=film_inds, vector=-shift)

print(overlaps)
#  score = (overlaps - overlap_ref)
score = overlaps

fig, ax = plt.subplots(figsize=(4, 5), dpi=400)
ax.set_xlabel(r"Fractional Shift in $\vec{a}$ Direction", fontsize=12)
ax.set_ylabel(r"Fractional Shift in $\vec{b}$ Direction", fontsize=12)

score -= score.min()
score /= score.max()
score = gaussian_filter(score, 2)

im = ax.pcolormesh(
#  im = ax.contourf(
    np.linspace(shift_dist_a[0], shift_dist_a[1], grid_a),
    np.linspace(shift_dist_b[0], shift_dist_b[1], grid_b),
    score.T,
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




