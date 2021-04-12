import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import plotly.graph_objects as go
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import time
import copy


"""
Given a unit cell with side lengths: a, b, c:
    the cube size is: np.max(a, b, c) / 0.1
    the voxel width in the x-direction is: a / cube size
    the voxel width in the y-direction is: b / cube size
    the voxel width in the z-direction is: c / cube size

"""

r = 1.6

print((4 / 3) * np.pi * r**3)

s = Structure.from_file('./POSCAR_sub_100')
sg = SpacegroupAnalyzer(s)
#  ps = sg.get_primitive_standard_structure()

lattice = s.lattice.matrix
lattice_norm = np.linalg.norm(lattice, axis=1)
frac_coords = s.frac_coords
print(len(frac_coords))
print(lattice)


def generate_unit_cell_tensor(lattice, grid_size=0.1):
    cube_size = np.floor(np.max(np.linalg.norm(lattice, axis=1)) / grid_size).astype(np.int16)
    unit_cell = np.zeros((cube_size, cube_size, cube_size), dtype=bool)
    X, Y, Z = np.meshgrid(
        np.arange(cube_size, dtype=np.int16),
        np.arange(cube_size, dtype=np.int16),
        np.arange(cube_size, dtype=np.int16),
    )

    return unit_cell, X, Y, Z

def get_radii(lattice, radius, grid_size=0.1):
    a, b, c = np.linalg.norm(lattice, axis=1)
    cube_size = np.floor(np.max([a, b, c]) / grid_size).astype(np.int16)
    r_a = np.floor(radius / (a / cube_size)).astype(np.int16)
    r_b = np.floor(radius / (b / cube_size)).astype(np.int16)
    r_c = np.floor(radius / (c / cube_size)).astype(np.int16)

    return r_a, r_b, r_c

def get_atoms_volume(unit_cell, lattice):
    abc = np.linalg.norm(lattice, axis=1)
    side_length = unit_cell.shape[0]
    voxel_widths = abc / side_length
    voxel_volume = np.product(voxel_widths)
    volume = np.sum(unit_cell) * voxel_volume

    return volume


def generate_elipsoid(radii):
    r_a, r_b, r_c = radii
    cube_len_a = (r_a * 2) + 1
    cube_len_b = (r_b * 2) + 1
    cube_len_c = (r_c * 2) + 1

    sphere = np.zeros(
        (cube_len_a, cube_len_b, cube_len_c),
        dtype=bool
    )
    X, Y, Z = np.meshgrid(
        np.arange(cube_len_a, dtype=np.int16),
        np.arange(cube_len_b, dtype=np.int16),
        np.arange(cube_len_c, dtype=np.int16),
    )
    s = time.time()
    sl = np.s_[r_a:, r_b:, r_c:]
    x = ((X[sl] - r_a) / (r_a + 0.90))**2
    y = ((Y[sl] - r_b) / (r_b + 0.90))**2
    z = ((Z[sl] - r_c) / (r_c + 0.90))**2
    sphere[sl] =  x + y + z < 1
    sphere += np.flip(sphere, axis=2)
    sphere += np.fliplr(sphere)
    sphere += np.flip(sphere)
    e = time.time()
    
    return sphere, X, Y, Z

def append_atoms(frac_coords, unit_cell, sphere):
    center_inds = np.int16(np.array(unit_cell.shape) / 2)
    final_coords = np.int16(frac_coords * unit_cell.shape)
    roll_coords = final_coords - center_inds[None, :]
    temp_array = np.zeros(unit_cell.shape, dtype=bool)
    temp_overlap_array = np.zeros(unit_cell.shape, dtype=np.int8)
    overlap_unit_cell = np.zeros(unit_cell.shape, dtype=np.int8)
    sl = np.s_[
        center_inds[0] - int(sphere.shape[0] / 2):center_inds[0] + int(sphere.shape[0] / 2) + 1,
        center_inds[1] - int(sphere.shape[1] / 2):center_inds[1] + int(sphere.shape[1] / 2) + 1,
        center_inds[2] - int(sphere.shape[2] / 2):center_inds[2] + int(sphere.shape[2] / 2) + 1,
    ]
    for roll_coord in roll_coords:
        s = time.time()
        #  temp_array[sl] = sphere
        temp_overlap_array[sl] = sphere.astype(np.int8)
        #  rolled_array = np.roll(temp_array, roll_coord, axis=[0,1,2])
        rolled_overlap_array = np.roll(temp_overlap_array, roll_coord, axis=[0,1,2])
        #  temp_array[sl] = False
        temp_overlap_array[sl] = np.int8(0)
        #  unit_cell += rolled_array
        overlap_unit_cell += rolled_overlap_array
        e = time.time()
        print('LOOP TIME =', e - s)

    unit_cell = overlap_unit_cell > 0

    return unit_cell, overlap_unit_cell


unit_cell, X, Y, Z = generate_unit_cell_tensor(lattice=lattice, grid_size=0.02)

s = time.time()
ellipsiod, _, _, _ = generate_elipsoid(get_radii(lattice, r, grid_size=0.02))
e = time.time()
print('Sphere Gen = ', e - s)


s = time.time()
unit_cell, overlap_unit_cell = append_atoms(
    frac_coords=frac_coords,
    unit_cell=unit_cell,
    sphere=ellipsiod,
)
e = time.time()
print('Append atoms = ', e - s)
print(unit_cell.shape)

print(get_atoms_volume(unit_cell, lattice) / 4)
print('Overlap =', get_atoms_volume(overlap_unit_cell > 1, lattice))

    

#  fig = go.Figure(data=go.Volume(
    #  x=X.flatten(),
    #  y=Y.flatten(),
    #  z=Z.flatten(),
    #  value=unit_cell.flatten(),
    #  isomin=0.1,
    #  isomax=0.8,
    #  opacity=0.2, # needs to be small to see through all surfaces
    #  surface_count=10, # needs to be a large number for good volume rendering
    #  ))
#  #  fig.write_image("fig1.png")
#  fig.show()
