from generate import InterfaceGenerator, SurfaceGenerator
from pymatgen.io.vasp.inputs import Poscar
from scipy.ndimage import gaussian_filter
from scipy.stats import multivariate_normal
from matplotlib.colors import Normalize
from vaspvis.utils import group_layers
import matplotlib.pyplot as plt
from itertools import product
import plotly.graph_objects as go
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

    t1 = np.log(r2s/r1s)
    t2 = 3
    t3 = (xs**2 + ys**2 + zs**2) / r2s
    t4 = 3 * (r1s / r2s)

    kl = (t1 - t2 + t3 + t4) / 2
    kl = np.array(np.split(kl, 9)).min(axis=0)
    kl = np.exp(-(kl)**2).sum(axis=0)
    kl -= kl.min()
    kl /= kl.max()

    return kl

def sphere_overlap(coords, grid_density, a_range, b_range, r, si, fi, matrix):
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

    d = np.sqrt((x1s - x2s)**2 + (y1s - y2s)**2 + (z1s - z2s)**2)
    d = np.array(np.split(d, 9)).min(axis=0)

    d[d - r1[:, None, None] - r2[:, None, None] > 0] = 0
    n1 = np.pi * (r1[:, None, None] + r2[:, None, None] - d)**2
    n2 = d**2 + (2*d*r1[:, None, None]) + (2*d*r2[:, None, None]) + (6*r1*r2)[:, None, None] - (3*(r1**2))[:, None, None] - (3*(r2**2))[:, None, None]
    volume = np.divide((n1 * n2), (12 * d), out=np.zeros_like(d), where=(d != 0))

    return volume

def norm_overlap(coords, grid_density, repeat, r, si, fi, matrix):
    X, Y = np.meshgrid(
        np.linspace(0, 1, grid_density[0]),
        np.linspace(0, 1, grid_density[1]),
    )
    plot_coords = np.c_[X.ravel(), Y.ravel(), np.zeros(len(Y.ravel()))].dot(matrix)
    X = plot_coords[:,0].reshape(X.shape)
    Y = plot_coords[:,1].reshape(Y.shape)

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

    x_shift = x1p - x2p
    y_shift = y1p - y2p
    z_shift = z1p - z2p

    frac_shifts = -np.c_[x_shift, y_shift, z_shift]
    cart_shifts = frac_shifts.dot(matrix)
    new_r = np.sqrt((r1p + r2p)**2 - cart_shifts[:,-1]**2)

    mus = cart_shifts[:,:2]
    sigmas = [(nr/2.5) * np.eye(2) for nr in new_r]
    vol = 0.75 * np.pi * new_r**3
    vol /= vol.max()
    x = np.c_[X.ravel(), Y.ravel()]
    ns = np.sum(
        [v*multivariate_normal.pdf(x, m, s**2) for m, s, v in zip(mus, sigmas, vol)],
        axis=0,
    ).reshape(X.shape)

    X_new, Y_new = np.meshgrid(
        np.linspace(-1*repeat, 1*repeat, 2*grid_density[0]*repeat),
        np.linspace(-1*repeat, 1*repeat, 2*grid_density[1]*repeat),
    )
    coords_new = np.c_[X_new.ravel(), Y_new.ravel(), np.zeros(len(Y_new.ravel()))].dot(matrix)
    X_new = coords_new[:,0].reshape(X_new.shape)
    Y_new = coords_new[:,1].reshape(Y_new.shape)
    ns_new = np.tile(ns, (2*repeat,2*repeat))
    print(ns_new.shape)

    return X_new, Y_new, ns_new

sub_layer = 3
film_layer = 3

subs = SurfaceGenerator.from_file(
    './poscars/POSCAR_InAs_conv',
    miller_index=[0,0,1],
    layers=sub_layer,
    vacuum=10,
)

films = SurfaceGenerator.from_file(
    './poscars/POSCAR_Ni2MnIn_conv',
    miller_index=[0,0,1],
    layers=film_layer,
    vacuum=10,
)

inter = InterfaceGenerator(
    substrate=subs.slabs[1],
    film=films.slabs[1],
    length_tol=0.01,
    angle_tol=0.01,
    area_tol=0.01,
    max_area=100,
    interfacial_distance=2.0,
    vacuum=40,
)

interface = inter.generate_interfaces()

for i, inter in enumerate(interface):
    Poscar(inter.interface).write_file(f'POSCAR_{i}')

interface = interface[0]

struc = interface.interface
interface_height = interface.interface_height

si = np.where(struc.frac_coords[:, -1] < interface_height)[0]
fi = np.where(struc.frac_coords[:, -1] > interface_height)[0]
struc.translate_sites(si, [0,0.5,0])

layer_inds, heights = group_layers(struc)

Poscar(struc).write_file('POSCAR_t')

film_shift_inds = layer_inds[np.min(np.where(heights > interface_height))]
sub_shift_inds = layer_inds[np.max(np.where(heights < interface_height))]

species = np.array(struc.species, dtype=str)
film_inds = np.where(struc.frac_coords[:,-1] > interface_height)[0]

#  rs = {'As': 1.19, 'Ga': 1.22, 'Mn': 1.5}
#  rs = {'As': 1.19, 'In': 1.42, 'Al': 1.43}
rs = {'As': 1.19, 'In': 1.67, 'Mn': 1.292, 'Ni': 1.246}
#  rs = {'In': 1.42, 'Sb': 1.39, 'Fe': 1.277}
#  rs = {'In': 1.582, 'As': 1.27, 'Al': 1.43}
radii = np.array([rs[i] for i in species]) 

grid_a = 250
grid_b = 250
shift_dist_a = [0, 1]
shift_dist_b = [0, 1]


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

X, Y, n = norm_overlap(
    coords=struc.frac_coords,
    grid_density=[grid_a,grid_b],
    repeat=1,
    r=radii,
    si=sub_shift_inds,
    fi=film_shift_inds,
    matrix=struc.lattice.matrix,
)

#  vol = sphere_overlap(
    #  coords=struc.frac_coords,
    #  grid_density=[grid_a,grid_b],
    #  a_range=shift_dist_a,
    #  b_range=shift_dist_b,
    #  r=radii,
    #  si=sub_shift_inds,
    #  fi=film_shift_inds,
    #  matrix=struc.lattice.matrix,
#  )
#
#  smear_vol = np.zeros(vol.shape)
#
#  for i, v in enumerate(vol):
    #  smear_vol[i] = gaussian_filter(v, sigma=2, mode='wrap')
#
#  #  score = vol.sum(axis=0)
#  print(vol.shape)
score = n
score -= score.min()
score /= score.max()


fig = go.Figure(
    data=[go.Surface(z=score, x=X, y=Y, colorscale='jet')])
fig.show()

fig, ax = plt.subplots(figsize=(4.5, 5), dpi=600)
ax.set_xlabel(r"Shift in $x$ Direction", fontsize=16)
ax.set_ylabel(r"Shift in $y$ Direction", fontsize=16)

im = ax.pcolormesh(
    X,
    Y,
    score.T,
    cmap='jet',
    shading='gouraud',
    norm=Normalize(vmin=score.min(), vmax=score.max()),
)

cbar = fig.colorbar(im, ax=ax, orientation='horizontal')
cbar.ax.tick_params(labelsize=16)
cbar.ax.locator_params(nbins=4)
cbar.set_label('Score', fontsize=16)
ax.tick_params(labelsize=16)
fig.tight_layout()
fig.savefig('circ.png')
#
#  struc.translate_sites(fi, [0,0.5,0])
#  Poscar(struc).write_file('POSCAR_opt')
