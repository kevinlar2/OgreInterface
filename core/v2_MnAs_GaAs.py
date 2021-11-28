from generate import InterfaceGenerator, SurfaceGenerator
from pymatgen.io.vasp.inputs import Poscar
from scipy.ndimage import gaussian_filter
from matplotlib.colors import Normalize
from vaspvis.utils import group_layers
import matplotlib.pyplot as plt
from itertools import product
import numpy as np
import copy
import time

def kl_overlap(coords, r, si, fi, matrix):
    x1 = coords[np.repeat(si, len(fi)), 0]
    x2 = coords[np.tile(fi, len(si)), 0]

    y1 = coords[np.repeat(si, len(fi)), 1]
    y2 = coords[np.tile(fi, len(si)), 1]

    z1 = coords[np.repeat(si, len(fi)), 2]
    z2 = coords[np.tile(fi, len(si)), 2]

    r1 = r[np.repeat(si, len(fi))]
    r2 = r[np.tile(fi, len(si))]

    #  x1_p = np.c_[[x1 for _ in range(9)]].ravel()
    #  y1_p = np.c_[[y1 for _ in range(9)]].ravel()
    #  z1_p = np.c_[[z1 for _ in range(9)]].ravel()
    #  z2_p = np.c_[[z2 for _ in range(9)]].ravel()
    #  r1_p = np.c_[[r1 for _ in range(9)]].ravel()
    #  r2_p = np.c_[[r2 for _ in range(9)]].ravel()

    x1_p = np.hstack([x1.reshape(-1,1) for _ in range(9)]).ravel()
    y1_p = np.hstack([y1.reshape(-1,1) for _ in range(9)]).ravel()
    z1_p = np.hstack([z1.reshape(-1,1) for _ in range(9)]).ravel()
    z2_p = np.hstack([z2.reshape(-1,1) for _ in range(9)]).ravel()
    r1_p = np.hstack([r1.reshape(-1,1) for _ in range(9)]).ravel()
    r2_p = np.hstack([r2.reshape(-1,1) for _ in range(9)]).ravel()

    #  x1_p = np.c_[x1, x1, x1, x1, x1, x1, x1, x1, x1].ravel()
    #  y1_p = np.c_[y1, y1, y1, y1, y1, y1, y1, y1, y1].ravel()
    #  z1_p = np.c_[z1, z1, z1, z1, z1, z1, z1, z1, z1].ravel()
    #  z2_p = np.c_[z2, z2, z2, z2, z2, z2, z2, z2, z2].ravel()
    #  r1_p = np.c_[r1, r1, r1, r1, r1, r1, r1, r1, r1].ravel()
    #  r2_p = np.c_[r2, r2, r2, r2, r2, r2, r2, r2, r2].ravel()

    x2_p = np.c_[x2-1, x2-1, x2-1, x2, x2, x2, x2+1, x2+1, x2+1].ravel()
    y2_p = np.c_[y2-1, y2, y2+1, y2-1, y2, y2+1, y2-1, y2, y2+1].ravel()

    fc_1 = np.c_[x1_p, y1_p, z1_p]
    fc_2 = np.c_[x2_p, y2_p, z2_p]

    cc_1 = fc_1.dot(matrix)
    cc_2 = fc_2.dot(matrix)

    xs = cc_1[:, 0] - cc_2[:, 0]
    ys = cc_1[:, 1] - cc_2[:, 1]
    zs = cc_1[:, 2] - cc_2[:, 2]

    #  xs = x1_p - x2_p
    #  ys = y1_p - y2_p
    #  zs = z1_p - z2_p

    t1 = np.log(r2_p/r1_p)
    t2 = 3
    t3 = (xs**2 + ys**2 + zs**2) / r2_p
    t4 = 3 * (r1_p / r2_p) 

    kl = (t1 - t2 + t3 + t4) / 2
    kl = kl.reshape(x1.shape[0], -1)
    
    return np.exp(-(kl**2))


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

#  films.slabs[0].remove_layers(1)

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

rs = {'As': 1.19, 'Ga': 1.22, 'Mn': 1.5}

interface = inter.generate_interfaces()[2]


struc = interface.interface
interface_height = interface.interface_height
layer_inds, heights = group_layers(struc)

film_shift_inds = layer_inds[np.min(np.where(heights > interface_height))]
sub_shift_inds = layer_inds[np.max(np.where(heights < interface_height))]

species = np.array(struc.species, dtype=str)
film_inds = np.where(struc.frac_coords[:,-1] > interface_height)[0]

#  rs = {'As': 1.19, 'In': 1.42, 'Al': 1.43}
#  rs = {'In': 1.582, 'As': 1.27, 'Al': 1.43}
radii = np.array([rs[i] for i in species]) 

grid_a = 200
grid_b = 200
shift_dist_a = [-0.07, 0.07]
shift_dist_b = [-0.5, 0.5]


frac_shifts = np.array(list(product(
    np.linspace(shift_dist_a[0], shift_dist_a[1], grid_a),
    np.linspace(shift_dist_b[0], shift_dist_b[1], grid_b),
)))
ix, iy = np.meshgrid(np.arange(grid_a), np.arange(grid_b))
indices = np.c_[ix.ravel(), iy.ravel()]
frac_shifts = np.c_[frac_shifts, np.zeros(frac_shifts.shape[0])]
cart_shifts = frac_shifts.dot(struc.lattice.matrix)
overlaps = []

for shift, inds in zip(frac_shifts, indices):
    struc.translate_sites(indices=film_inds, vector=shift, to_unit_cell=True)
    kl = kl_overlap(
        struc.frac_coords,
        radii,
        sub_shift_inds,
        film_shift_inds,
        struc.lattice.matrix,
    )
    #  print(kl.shape)
    overlaps.append(kl)
    struc.translate_sites(indices=film_inds, vector=-shift, to_unit_cell=True)

Poscar(struc).write_file('POSCAR_No')
struc.translate_sites(indices=film_inds, vector=[0,3,0], frac_coords=False)
Poscar(struc).write_file('POSCAR_KL')

score = np.dstack(overlaps).reshape(-1, 9, grid_a, grid_b).max(axis=1)
#  score = score[1:] - score[0][None, :, :]
score = -score.sum(axis=0)
print(score.shape)

fig, ax = plt.subplots(figsize=(4, 5), dpi=400)
ax.set_xlabel(r"Fractional Shift in $\vec{a}$ Direction", fontsize=12)
ax.set_ylabel(r"Fractional Shift in $\vec{b}$ Direction", fontsize=12)

im = ax.pcolormesh(
    cart_shifts[:,1].reshape(grid_a, grid_b).T,
    cart_shifts[:,0].reshape(grid_a, grid_b).T,
    score.T,
    cmap='jet',
    shading='gouraud',
    norm=Normalize(vmin=score.min(), vmax=score.max()),
)

cbar = fig.colorbar(im, ax=ax, orientation='horizontal')
cbar.ax.tick_params(labelsize=12)
cbar.ax.locator_params(nbins=4)
cbar.set_label('Score', fontsize=12)
ax.tick_params(labelsize=12)
#  ax.set_xlim(-1.5,1.5)
#  ax.set_ylim(-1.5,1.5)
fig.tight_layout()
fig.savefig('graph_MnAs.png')
