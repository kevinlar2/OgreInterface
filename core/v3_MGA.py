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

def kl_overlap(coords, grid_density, a_range, b_range, r, si, fi, matrix):
    X, Y = np.meshgrid(
        np.linspace(a_range[0], a_range[1], grid_density[0]),
        np.linspace(b_range[0], b_range[1], grid_density[1]),
    )

    frac_shifts = np.c_[X.ravel(), Y.ravel(), np.zeros(len(Y.ravel()))]
    cart_shifts = frac_shifts.dot(matrix)

    x1 = coords[np.repeat(si, len(fi)), 0]
    x2 = coords[np.tile(fi, len(si)), 0]

    y1 = coords[np.repeat(si, len(fi)), 1]
    y2 = coords[np.tile(fi, len(si)), 1]

    z1 = coords[np.repeat(si, len(fi)), 2]
    z2 = coords[np.tile(fi, len(si)), 2]

    r1 = r[np.repeat(si, len(fi))]
    r2 = r[np.tile(fi, len(si))]

    x1p = np.concatenate([x1 for _ in range(9)])
    y1p = np.concatenate([y1 for _ in range(9)])
    z1p = np.concatenate([z1 for _ in range(9)])
    z2p = np.concatenate([z2 for _ in range(9)])
    r1p = np.concatenate([r1 for _ in range(9)])
    r2p = np.concatenate([r2 for _ in range(9)])
    x2p = np.concatenate([x2-1, x2-1, x2-1, x2, x2, x2, x2+1, x2+1, x2+1])
    y2p = np.concatenate([y2-1, y2, y2+1, y2-1, y2, y2+1, y2-1, y2, y2+1])

    f1 = np.c_[x1p, y1p, z1p]
    f2 = np.c_[x2p, y2p, z2p]
    c1 = f1.dot(matrix)
    c2 = f2.dot(matrix)

    x1p, y1p, z1p = c1[:,0], c1[:,1], c1[:,2] 
    x2p, y2p, z2p = c2[:,0], c2[:,1], c2[:,2] 

    x1s = (x1p + (0*cart_shifts[:,0])[:, None]).T.reshape((len(x1p),) + X.shape)
    y1s = (y1p + (0*cart_shifts[:,1])[:, None]).T.reshape((len(x1p),) + X.shape)
    z1s = (z1p + (0*cart_shifts[:,2])[:, None]).T.reshape((len(x1p),) + X.shape)
    x2s = (x2p + cart_shifts[:,0][:, None]).T.reshape((len(x1p),) + X.shape)
    y2s = (y2p + cart_shifts[:,1][:, None]).T.reshape((len(x1p),) + X.shape)
    z2s = (z2p + cart_shifts[:,2][:, None]).T.reshape((len(x1p),) + X.shape)
    r1s = (r1p + (0*cart_shifts[:,2])[:, None]).T.reshape((len(x1p),) + X.shape)
    r2s = (r2p + (0*cart_shifts[:,2])[:, None]).T.reshape((len(x1p),) + X.shape)

    xs = x1s - x2s
    ys = y1s - y2s
    zs = z1s - z2s

    print(xs.shape)

    t1 = np.log(r2s/r1s)
    t2 = 3
    t3 = (xs**2 + ys**2 + zs**2) / r2s
    t4 = 3 * (r1s / r2s)

    kl = (t1 - t2 + t3 + t4) / 2
    kl = np.array(np.split(kl, 9)).min(axis=0)
    kl -= np.min(kl, axis=(1,2))[:,None,None]
    #  kl /= np.std(kl, axis=(1,2))[:,None,None]
    #  kl = kl.sum(axis=0)
    #  kl = (1 / (kl+0.1)**2).sum(axis=0)
    kl = -np.exp(-(10*kl)**2).sum(axis=0)
    kl[:10,:] = np.median(kl[15:35, 15:35])
    kl[:,:10] = np.median(kl[15:35, 15:35])
    kl[-10:,:] = np.median(kl[15:35, 15:35])
    kl[:,-10:] = np.median(kl[15:35, 15:35])
    kl -= kl.min()
    kl /= kl.max()
    #  kl = kl.sum(axis=0)
    #  mean = np.mean(kl)
    #  std = np.std(kl)
    #  print(np.mean(kl))
    #  print(np.std(kl))
    #  kl = (kl - mean) / std
    #  print(np.mean(kl))
    #  print(np.std(kl))
    #  print(np.exp(kl[0,0]**2))
    #  kl = np.exp(-(kl**2) / 1)

    return kl


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


interface = inter.generate_interfaces()[2]

#  subs = SurfaceGenerator.from_file(
    #  './poscars/POSCAR_InSb_conv',
    #  miller_index=[0,0,1],
    #  layers=sub_layer,
    #  vacuum=10,
#  )
#
#  films = SurfaceGenerator.from_file(
    #  './poscars/POSCAR_Fe_conv',
    #  miller_index=[0,0,1],
    #  layers=film_layer,
    #  vacuum=10,
#  )
#
#  inter = InterfaceGenerator(
    #  substrate=subs.slabs[1],
    #  film=films.slabs[0],
    #  length_tol=0.02,
    #  angle_tol=0.02,
    #  area_tol=0.02,
    #  max_area=200,
    #  interfacial_distance=2,
    #  vacuum=40,
#  )

#  subs = SurfaceGenerator.from_file(
    #  './poscars/POSCAR_InAs_conv',
    #  miller_index=[0,0,1],
    #  layers=sub_layer,
    #  vacuum=10,
#  )
#
#  films = SurfaceGenerator.from_file(
    #  './poscars/POSCAR_Al_conv',
    #  miller_index=[0,0,1],
    #  layers=film_layer,
    #  vacuum=10,
#  )
#
#  inter = InterfaceGenerator(
    #  substrate=subs.slabs[3],
    #  film=films.slabs[0],
    #  length_tol=0.01,
    #  angle_tol=0.01,
    #  area_tol=0.01,
    #  max_area=300,
    #  interfacial_distance=2,
    #  vacuum=40,
#  )
#
#  interface = inter.generate_interfaces()[0]


struc = interface.interface
interface_height = interface.interface_height
layer_inds, heights = group_layers(struc)

Poscar(struc).write_file('POSCAR_t')

film_shift_inds = layer_inds[np.min(np.where(heights > interface_height))]
sub_shift_inds = layer_inds[np.max(np.where(heights < interface_height))]

species = np.array(struc.species, dtype=str)
film_inds = np.where(struc.frac_coords[:,-1] > interface_height)[0]

rs = {'As': 1.19, 'Ga': 1.22, 'Mn': 1.5}
#  rs = {'As': 1.19, 'In': 1.42, 'Al': 1.43}
#  rs = {'In': 1.42, 'Sb': 1.39, 'Fe': 1.277}
#  rs = {'In': 1.582, 'As': 1.27, 'Al': 1.43}
radii = np.array([rs[i] for i in species]) 

grid_a = 200
grid_b = 200
#  shift_dist_a = [-0.3, 0.3]
#  shift_dist_b = [-0.3, 0.3]
shift_dist_a = [-0.15, 0.15]
shift_dist_b = [-0.15, 0.15]


frac_shifts = np.array(list(product(
    np.linspace(shift_dist_a[0], shift_dist_a[1], grid_a),
    np.linspace(shift_dist_b[0], shift_dist_b[1], grid_b),
)))
ix, iy = np.meshgrid(np.arange(grid_a), np.arange(grid_b))
indices = np.c_[ix.ravel(), iy.ravel()]
frac_shifts = np.c_[frac_shifts, np.zeros(frac_shifts.shape[0])]
cart_shifts = frac_shifts.dot(struc.lattice.matrix)
overlaps = []

print(struc.lattice.matrix.round(3))

kl = kl_overlap(
    coords=struc.frac_coords,
    grid_density=[grid_a,grid_b],
    a_range=shift_dist_a,
    b_range=shift_dist_b,
    r=radii,
    si=sub_shift_inds,
    fi=film_shift_inds,
    matrix=struc.lattice.matrix,
)
print(kl.shape)

score = kl

#  score = kl.reshape(kl.shape[0], kl.shape[1], grid_a, grid_b).max(axis=1)
#  #  score = score[1:] - score[0][None, :, :]
#  score = -score.sum(axis=0)
#  print(score.shape)
#
fig, ax = plt.subplots(figsize=(4.5, 5), dpi=600)
ax.set_xlabel(r"Shift in $x$ Direction", fontsize=16)
ax.set_ylabel(r"Shift in $y$ Direction", fontsize=16)

im = ax.pcolormesh(
    #  np.flip(cart_shifts[:,0].reshape(grid_a, grid_b).T, axis=0),
    cart_shifts[:,0].reshape(grid_a, grid_b),
    cart_shifts[:,1].reshape(grid_a, grid_b),
    score.T,
    cmap='jet',
    shading='gouraud',
    #  norm=Normalize(vmin=score.min(), vmax=score.max()),
    norm=Normalize(vmin=score.min(), vmax=score.max()),
)

cbar = fig.colorbar(im, ax=ax, orientation='horizontal')
cbar.ax.tick_params(labelsize=16)
cbar.ax.locator_params(nbins=4)
cbar.set_label('Score', fontsize=16)
ax.tick_params(labelsize=16)
ax.set_xlim(-2,2)
ax.set_ylim(-2,2)
fig.tight_layout()
fig.savefig('graph2.png')
