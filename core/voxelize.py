import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import plotly.graph_objects as go
import time


def generate_sphere(spacing_size, r_x, r_y, r_z):
    r_x_s = np.floor(r_x / spacing_size).astype(np.int16)
    r_y_s = np.floor(r_y / spacing_size).astype(np.int16)
    r_z_s = np.floor(r_z / spacing_size).astype(np.int16)

    cube_len_x = (r_x_s * 2) + 1
    cube_len_y = (r_y_s * 2) + 1
    cube_len_z = (r_z_s * 2) + 1

    sphere = np.zeros(
        (cube_len_x, cube_len_y, cube_len_z),
        dtype=bool
    )
    X, Y, Z = np.meshgrid(
        np.arange(cube_len_x, dtype=np.int16),
        np.arange(cube_len_y, dtype=np.int16),
        np.arange(cube_len_z, dtype=np.int16),
    )
    X = X.transpose(np.argsort(sphere.shape))
    Y = Y.transpose(np.argsort(sphere.shape))
    Z = Z.transpose(np.argsort(sphere.shape))
    s = time.time()
    sl = np.s_[r_x_s:, r_y_s:, r_z_s:]
    x = ((X[sl] - r_x_s) / (r_x_s + 0.5))**2
    y = ((Y[sl] - r_y_s) / (r_y_s + 0.5))**2
    z = ((Z[sl] - r_z_s) / (r_z_s + 0.5))**2
    sphere[sl] =  x + y + z < 1
    sphere = np.logical_or(sphere, np.flip(sphere, axis=2))
    sphere = np.logical_or(sphere, np.fliplr(sphere))
    sphere = np.logical_or(sphere, np.flip(sphere))
    e = time.time()
    print(e-s)
    area = sphere.sum() * spacing_size**3
    
    return sphere, X, Y, Z, area

def append_sphere(center_inds, box, sphere):
    sl = np.s_[
        center_inds[0] - int(sphere.shape[0] / 2):center_inds[0] + int(sphere.shape[0] / 2)+1,
        center_inds[1] - int(sphere.shape[1] / 2):center_inds[1] + int(sphere.shape[1] / 2)+1,
        center_inds[2] - int(sphere.shape[2] / 2):center_inds[2] + int(sphere.shape[2] / 2)+1,
    ]
    box[sl] = sphere
    
def generate_box(spacing_size, side_length):
    segments = int(side_length / spacing_size)
    box = np.zeros((segments, segments, segments), dtype=bool)
    X, Y, Z = np.meshgrid(
        np.arange(segments, dtype=np.int16),
        np.arange(segments, dtype=np.int16),
        np.arange(segments, dtype=np.int16),
    )
    #  X = X.transpose(np.argsort(box.shape))
    #  Y = Y.transpose(np.argsort(box.shape))
    #  Z = Z.transpose(np.argsort(box.shape))

    return box, X, Y, Z

sphere, X, Y, Z, area = generate_sphere(0.06, 1.25, 1.25, 1.25)
#  sphere = np.roll(sphere, [int(3 * sphere.shape[0]/4),int(3 * sphere.shape[0]/4),int(3 * sphere.shape[0]/4)], [0,1,2])

box, Xbox, Ybox, Zbox = generate_box(0.06, 5)
box2, _, _, _  = generate_box(0.06, 5)
append_sphere(
    [int(box.shape[0] / 3),int(box.shape[1] / 2),int(box.shape[2] / 2)],
    box,
    sphere,
)
append_sphere(
    [int(2 * box.shape[0] / 3),int(box.shape[1] / 2),int(box.shape[2] / 2)],
    box2,
    sphere,
)




#
#  #  for size in [100, 110, 120, 130, 140, 150, 160, 170, 180, 190]:
#  r_x = 3
#  r_y = 3
#  r_z = 3
#  side_length = 8
#  #  segments = int(size) + 1
#  segments = 51
#  spacing_size = side_length / segments
#  r_x_s = np.int16(r_x / spacing_size)
#  r_y_s = np.int16(r_y / spacing_size)
#  r_z_s = np.int16(r_z / spacing_size)
#  array = np.zeros((segments,segments,segments), dtype=bool)
#  X, Y, Z = np.meshgrid(np.arange(segments, dtype=np.int16), np.arange(segments, dtype=np.int16), np.arange(segments, dtype=np.int16))
#  #  X, Y, Z = np.meshgrid(np.arange(segments), np.arange(segments), np.arange(segments))
#
#  sl = np.s_[int(segments / 2) -1:,int(segments / 2) -1:,int(segments / 2) -1:]
#
#  sym = []
#  full = []
#
#  s1 = time.time()
#  #  array[sl] = np.sum([
    #  #  ((X[sl] - np.int16(segments / 2)) / r_x_s)**2,
    #  #  ((Y[sl] - np.int16(segments / 2)) / r_y_s)**2,
    #  #  ((Z[sl] - np.int16(segments / 2)) / r_z_s)**2,
#  #  ], dtype=np.float32) < 1
#  array[sl] = np.sum([
    #  np.square((X[sl] - np.int16(segments / 2)) / r_x_s, dtype=np.float32),
    #  np.square((Y[sl] - np.int16(segments / 2)) / r_y_s, dtype=np.float32),
    #  np.square((Z[sl] - np.int16(segments / 2)) / r_z_s, dtype=np.float32),
#  ], axis=2, dtype=np.float32) < 1
#  #  array[sl] = ((X[sl] - np.int16(segments / 2)) / r_x_s)**2 + ((Y[sl] - np.int16(segments / 2)) / r_y_s)**2 + ((Z[sl] - np.int16(segments / 2)) / r_z_s)**2 < 1
#  array = np.logical_or(array, np.rot90(array))
#  array = np.logical_or(array, np.fliplr(array))
#  array = np.logical_or(array, np.flip(array))
#  #  elipsoid1 = ((X - np.int16(segments / 2)) / r_x_s)**2 + ((Y - np.int16(segments / 4)) / r_y_s)**2 + ((Z - np.int16(segments / 2)) / r_z_s)**2 < 1
#  #  elipsoid2 = ((X - np.int16(segments / 2)) / r_x_s)**2 + ((Y - np.int16(3 * segments / 4)) / r_y_s)**2 + ((Z - np.int16(segments / 2)) / r_z_s)**2 < 1
#  e1 = time.time()
#  #  print(type(np.square((X[0,0,0] - np.int16(segments / 2)) / r_x_s, dtype=np.float32)))
#
#  #  print('Sym opp = ', e - s)
#
#  s2 = time.time()
#  elipsoid = ((X - np.int16(segments / 2)) / r_x_s)**2 + ((Y - np.int16(segments / 2)) / r_y_s)**2 + ((Z - np.int16(segments / 2)) / r_z_s)**2 < 1
#  #  elipsoid1 = ((X - np.int16(segments / 2)) / r_x_s)**2 + ((Y - np.int16(segments / 4)) / r_y_s)**2 + ((Z - np.int16(segments / 2)) / r_z_s)**2 < 1
#  #  elipsoid2 = ((X - np.int16(segments / 2)) / r_x_s)**2 + ((Y - np.int16(3 * segments / 4)) / r_y_s)**2 + ((Z - np.int16(segments / 2)) / r_z_s)**2 < 1
#  e2 = time.time()
#  #  print('Full = ', e - s)
#
#  #  print('Sym Time =', np.mean(sym))
#  #  print('Full Time =', np.mean(full))
#  #  print('Diff = ', np.mean(full) / np.mean(sym))
#  #
#  #  print(elipsoid[21,15:25])

fig = go.Figure(data=go.Volume(
    x=Xbox.flatten(),
    y=Ybox.flatten(),
    z=Zbox.flatten(),
    value=np.logical_and(box, box2).flatten(),
    isomin=0.1,
    isomax=0.8,
    opacity=0.2, # needs to be small to see through all surfaces
    surface_count=10, # needs to be a large number for good volume rendering
    ))
#  fig.write_image("fig1.png")
fig.show()
