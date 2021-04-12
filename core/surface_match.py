#Iterator
import os
from itertools import product
import numpy as np
import math
from pymatgen.io.cif import CifWriter
from ase.build import general_surface
from ase.spacegroup import crystal
from ase.visualize import view
from ase.lattice.surface import *
from ase.io import *
import pymatgen as mg
from pymatgen.io.vasp.inputs import Poscar
import argparse
import pymatgen as mg
from pymatgen.core.structure import Structure
from pymatgen.io.vasp.inputs import Poscar
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core.surface import Slab, SlabGenerator, ReconstructionGenerator
from pymatgen.analysis.substrate_analyzer import SubstrateAnalyzer,ZSLGenerator
from pymatgen.symmetry.analyzer import *
from random import uniform
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import *
from mayavi.mlab import *
import scipy.optimize as optimize
import scipy.interpolate
import matplotlib.tri as tri
import scipy.spatial as ss
import matplotlib as mpl
import scipy.interpolate as interp
import time
from scipy.spatial import Delaunay
from scipy.interpolate import griddata
from scipy.interpolate import LinearNDInterpolator
import math as m
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
from bayes_opt import BayesianOptimization

############################################################################
############################################################################
#Inputs

working_dir = os.getcwd() + "/"
sub_struc = Structure.from_file( "POSCAR_sub")
film_struc = Structure.from_file("POSCAR_film")
BO_iterations = 200
z_shift = 0

rad_dic =  {'In': 1.377, 'As': 1.245 , 'Eu':1.805, 'S':1.22 }
############################################################################
############################################################################


def PBC_coord_gen(Struc, dir, k, ii, jj, kk, file_gen=True):

    Struc_frac = np.round(Struc.frac_coords, 6)
    if (Struc_frac[0,0] < 0) and (Struc_frac[0,1] < 0) and (Struc_frac[0,2] < 0):
        Struc_frac = - Struc_frac
    Struc_sp = Struc.species
    CC_dic = {}
    counter = 0
    adding_frac_coords = []
    adding_sps = []

    for i in Struc_frac:

        if ((i[0] == 0) and (i[1] == 0)):
            adding_frac_coords.append([i[0], i[1], i[2]])
            adding_sps.append(Struc_sp[counter])
            adding_frac_coords.append([i[0], 1 + i[1], i[2]])
            adding_frac_coords.append([1 + i[0], i[1], i[2]])
            adding_frac_coords.append([1 + i[0], 1 + i[1], i[2]])
            adding_sps.append(Struc_sp[counter])
            adding_sps.append(Struc_sp[counter])
            adding_sps.append(Struc_sp[counter])

        elif ((i[0] == 1) and (i[1] == 0)):
            adding_frac_coords.append([i[0], i[1], i[2]])
            adding_sps.append(Struc_sp[counter])
            adding_frac_coords.append([i[0], 1 + i[1], i[2]])
            adding_frac_coords.append([i[0] - 1, i[1], i[2]])
            adding_frac_coords.append([i[0] - 1, 1 + i[1], i[2]])
            adding_sps.append(Struc_sp[counter])
            adding_sps.append(Struc_sp[counter])
            adding_sps.append(Struc_sp[counter])

        elif ((i[0] == 0) and (i[1] == 1)):
            adding_frac_coords.append([i[0], i[1], i[2]])
            adding_sps.append(Struc_sp[counter])
            adding_frac_coords.append([i[0], i[1] - 1, i[2]])
            adding_frac_coords.append([i[0] + 1, i[1], i[2]])
            adding_frac_coords.append([i[0] + 1, i[1] - 1, i[2]])
            adding_sps.append(Struc_sp[counter])
            adding_sps.append(Struc_sp[counter])
            adding_sps.append(Struc_sp[counter])

        elif ((i[0] == 1) and (i[1] == 1)):
            adding_frac_coords.append([i[0], i[1], i[2]])
            adding_sps.append(Struc_sp[counter])
            adding_frac_coords.append([i[0], i[1] - 1, i[2]])
            adding_frac_coords.append([i[0] - 1, i[1], i[2]])
            adding_frac_coords.append([i[0] - 1, i[1] - 1, i[2]])
            adding_sps.append(Struc_sp[counter])
            adding_sps.append(Struc_sp[counter])
            adding_sps.append(Struc_sp[counter])

        elif ((i[0] <= 0) or (i[0] >= 1)):

            # print([i[0] , i[1]])
            if i[0] <= 0:
                adding_frac_coords.append([i[0], i[1], i[2]])
                adding_frac_coords.append([1 + i[0], i[1], i[2]])
                adding_sps.append(Struc_sp[counter])
                adding_sps.append(Struc_sp[counter])
            elif i[0] >= 1:
                adding_frac_coords.append([i[0] , i[1], i[2]])
                adding_frac_coords.append([i[0] - 1, i[1], i[2]])
                adding_sps.append(Struc_sp[counter])
                adding_sps.append(Struc_sp[counter])

        elif ((i[1] <= 0) or (i[1] >= 1)):
            if i[1] <= 0:
                adding_frac_coords.append([i[0], i[1], i[2]])
                adding_frac_coords.append([i[0], 1 + i[1], i[2]])
                adding_sps.append(Struc_sp[counter])
                adding_sps.append(Struc_sp[counter])
            elif i[1] >= 1:
                adding_frac_coords.append([i[0], i[1] , i[2]])
                adding_frac_coords.append([i[0], i[1] - 1, i[2]])
                adding_sps.append(Struc_sp[counter])
                adding_sps.append(Struc_sp[counter])
        else:
            adding_frac_coords.append([i[0], i[1], i[2]])
            adding_sps.append(Struc_sp[counter])


        counter += 1

    adding_frac_coords = np.array(adding_frac_coords)
    # if len(adding_frac_coords) != 0:
    #     Struc_frac = np.concatenate((Struc_frac, adding_frac_coords))

    Struc_frac = adding_frac_coords
    # Struc_sp = Struc_sp + adding_sps
    Struc_sp = adding_sps
    PBC_struc = Structure(Struc.lattice.matrix, Struc_sp, Struc_frac, coords_are_cartesian= False)
    # Poscar(PBC_struc).write_file(working_dir + "POSCAR_PBC", direct=False)

    PBC_latt = PBC_struc.lattice.matrix
    Struc_cart = np.array(PBC_struc.cart_coords)
    Struc_sp = PBC_struc.species
    if PBC_latt[2,2] < 0 :
        for k_index in range(len(Struc_cart)):
            Struc_cart[k_index, 2] = - Struc_cart[k_index, 2]

    # print(Struc_cart)
    Struc_frac2 = np.round(Struc.frac_coords, 6)
    Struc_sp2 = Struc.species
    Struc_len2 = len(Struc_sp2)
    ini_list = [[0, 0], [0, 1], [1, 0], [1, 1]]
    z_frac = Struc_frac[0, 2]
    adding_frac = np.array([[0, 0, z_frac], [0, 1, z_frac], [1, 0, z_frac], [1, 1, z_frac]])
    Struc_frac2 = np.concatenate((Struc_frac2, adding_frac))

    for i3 in range(4):
        Struc_sp2.append("Zn")
    CC_struc = Structure(Struc.lattice.matrix, Struc_sp2, Struc_frac2, coords_are_cartesian=False)
    # CC_struc = CC_struc.get_primitive_structure()
    # Poscar(CC_struc).write_file(working_dir+"POSCAR_PBC_CC", direct = False)
    # CC_struc_cart = CC_struc.frac_coords
    CC_struc_cart = CC_struc.cart_coords
    CC_dic['0_0'] = CC_struc_cart[Struc_len2]
    CC_dic['0_1'] = CC_struc_cart[Struc_len2 + 1]
    CC_dic['1_0'] = CC_struc_cart[Struc_len2 + 2]
    CC_dic['1_1'] = CC_struc_cart[Struc_len2 + 3]

    if file_gen:
        PBC_struc = open(dir + "/PBC_" + str(k) + "_x_" + str(ii) +
                         "_y_" + str(jj) + "_z_" + str(kk) + ".xyz", "w")
        print(len(Struc_sp), file=PBC_struc)
        # print(len(Struc.species) , file= PBC_struc)
        print("PBC", file=PBC_struc)
        for i in range(len(Struc_sp)):
            print(Struc_sp[i], end="\t", file=PBC_struc)
            for j in range(3):
                if j == 2:
                    print(Struc_cart[i, j], end="\n", file=PBC_struc)
                else:
                    print(Struc_cart[i, j], end="\t", file=PBC_struc)
    # corner_coords = np.array(corner_coords)
    return (Struc_cart, Struc_sp, CC_dic)
    # return (CC_struc.cart_coords, Struc_sp, CC_dic)


def triangel_area(r1, r2, r3):

    a = np.linalg.norm(r2 - r1)
    b = np.linalg.norm(r3 - r1)
    c = np.linalg.norm(r3 - r2)
    s = (a + b + c) / 2
    area_square = (s * (s - a) * (s - b) * (s - c))
    if (area_square < 0) and (area_square > -0.000001):
        area = 0
    else:
        area = area_square ** 0.5
    return area

def In_triangel(p, r1, r2,r3):
    tot_area = triangel_area(r1,r2,r3)
    area1 = triangel_area(p, r1, r2)
    area2 = triangel_area(p, r1, r3)
    area3 = triangel_area(p, r2, r3)
    # if np.round(area1) == 0 or np.round(area2) == 0 or np.round(area3) == 0:
    #     return True

    if np.round(tot_area, 3) == np.round(area1 + area2 + area3 , 3):
        return True
    else:
        return False

def tri_area(a, b, c):
    # calculate the sides
    s = (a + b + c) / 2
    # calculate the area
    area = (s*(s-a)*(s-b)*(s-c)) ** 0.5
    return area
def fast_norm(a):
    """
    Much faster variant of numpy linalg norm
    """
    return np.sqrt(np.dot(a, a))
def vec_area(a, b):
    """
    Area of lattice plane defined by two vectors
    """
    return fast_norm(np.cross(a, b))
def vec_angle(a, b):
    """
    Calculate angle between two vectors
    """
    cosang = np.dot(a, b)
    sinang = fast_norm(np.cross(a, b))
    return np.arctan2(sinang, cosang)
def reduce_vectors(a, b):
    """
    Generate independent and unique basis vectors based on the
    methodology of Zur and McGill
    """
    if np.dot(a, b) < 0:
        return reduce_vectors(a, -b)

    if fast_norm(a) > fast_norm(b):
        return reduce_vectors(b, a)

    if fast_norm(b) > fast_norm(np.add(b, a)):
        return reduce_vectors(a, np.add(b, a))

    if fast_norm(b) > fast_norm(np.subtract(b, a)):
        return reduce_vectors(a, np.subtract(b, a))

    return [a, b]
def get_dis(p1, p2):
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2)**0.5
def sphe_vol(r):
    return (4 / 3 * math.pi * (r ** 3))


def interface_voxel(sub_struc, film_struc, probe_rad = 1.2, scan_step = 0.1, visualize = False,
                        surface_matching = False, grid_search = False, int_distance = 2, potential_curve = True,
                        delta_z = 2):

    sub_uc_coords, film_uc_coords = sub_struc.cart_coords, film_struc.cart_coords
    sub_uc_species, film_uc_species = sub_struc.species, film_struc.species

    sub_pbc_output = PBC_coord_gen(sub_struc, working_dir, 1, 1, 1, 1, file_gen=False)
    film_pbc_output = PBC_coord_gen(film_struc, working_dir, 2, 1, 1, 1, file_gen=False)

    sub_CC_dic, film_CC_dic = sub_pbc_output[2], film_pbc_output[2]
    sub_cart_coords, film_cart_coords = sub_pbc_output[0], film_pbc_output[0]
    sub_struc_species, film_struc_species = sub_pbc_output[1], film_pbc_output[1]

    sub_struc_sc = sub_struc.copy()
    sub_struc_sc.make_supercell([3, 3, 1])
    sub_sc_pbc_output = PBC_coord_gen(sub_struc_sc, working_dir, 3, 1, 1, 1, file_gen=False)
    sub_sc_CC_dic = sub_sc_pbc_output[2]

    sub_sc_cart_coords, sub_sc_struc_species = sub_sc_pbc_output[0], sub_sc_pbc_output[1]

    film_struc_sc = film_struc.copy()
    film_struc_sc.make_supercell([3, 3, 1])
    film_sc_pbc_output = PBC_coord_gen(film_struc_sc, working_dir, 4, 1, 1, 1, file_gen=False)
    film_sc_CC_dic = film_sc_pbc_output[2]
    film_sc_cart_coords, film_sc_struc_species  = film_sc_pbc_output[0], film_sc_pbc_output[1]

    ini_max_z_sub = np.max(sub_cart_coords[:, 2])
    ini_min_z_film = np.min(film_cart_coords[:, 2])

    ini_slab_distance = ini_min_z_film - ini_max_z_sub
    if ini_slab_distance < 0:
        ini_slab_distance = 2
    # print(ini_max_z_sub, ini_min_z_film)

    film_cart_coords[:, 2] += (- ini_min_z_film + ini_slab_distance + ini_max_z_sub)
    film_sc_cart_coords[:, 2] += (- ini_min_z_film + ini_slab_distance + ini_max_z_sub)

    sec_max_z_sub = np.max(sub_cart_coords[:,2])
    sec_min_z_film = np.min(film_cart_coords[:,2])
    # print(sec_max_z_sub, sec_min_z_film)

    p_d = np.array([sub_CC_dic['0_0'][0] , sub_CC_dic['0_0'][1] ])
    p_u = np.array([sub_CC_dic['1_1'][0] , sub_CC_dic['1_1'][1] ])
    p_r = np.array([sub_CC_dic['1_0'][0] , sub_CC_dic['1_0'][1] ])
    p_l = np.array([sub_CC_dic['0_1'][0] , sub_CC_dic['0_1'][1] ])

    p_d_sc = np.array([film_sc_CC_dic['0_0'][0], film_sc_CC_dic['0_0'][1]])
    p_u_sc = np.array([film_sc_CC_dic['1_1'][0], film_sc_CC_dic['1_1'][1]])
    p_r_sc = np.array([film_sc_CC_dic['1_0'][0], film_sc_CC_dic['1_0'][1]])
    p_l_sc = np.array([film_sc_CC_dic['0_1'][0], film_sc_CC_dic['0_1'][1]])

    p_d_c = np.array([p_u[0] + 0, p_u[1] + 0])
    p_u_c = np.array([p_u[0] + (p_u[0] - p_d[0]), p_u[1] + (p_u[1] - p_d[1])])
    p_r_c = np.array([p_u[0] + (p_r[0] - p_d[0]), p_u[1] + (p_r[1] - p_d[1])])
    p_l_c = np.array([p_u[0] + (p_l[0] - p_d[0]), p_u[1] + (p_l[1] - p_d[1])])


    lattice_vecs = sub_struc.lattice.matrix
    lat_area = vec_area(lattice_vecs[0], lattice_vecs[1])
    len_vec_a, len_vec_b, len_vec_c = np.linalg.norm(lattice_vecs[0]), np.linalg.norm(lattice_vecs[1]), \
                                      np.linalg.norm(lattice_vecs[2])

    lat_angle = math.pi - vec_angle(lattice_vecs[0], lattice_vecs[1])

    sub_max_z = np.max(sub_cart_coords[:, 2])
    film_min_z = np.min(film_cart_coords[:, 2])

    target_sub_coords, target_film_coords, target_sub_sc_coords, target_film_sc_coords = [], [], [], []
    target_sp_sub, target_sp_film, target_sp_sub_sc, target_sp_film_sc = [], [], [], []

    total_atoms_vol = 0

    for i in range(len(sub_cart_coords)):
        if (sub_max_z - sub_cart_coords[i, 2]) <= delta_z:
            target_sub_coords.append([sub_cart_coords[i, 0], sub_cart_coords[i, 1], sub_cart_coords[i, 2]])
            target_sp_sub.append(str(sub_struc_species[i]))

    for i in range(len(film_cart_coords)):
        if (film_cart_coords[i, 2] - film_min_z) <= delta_z:
            target_film_coords.append([film_cart_coords[i, 0], film_cart_coords[i, 1], film_cart_coords[i, 2]])
            target_sp_film.append(str(film_struc_species[i]))

    for i in range(len(sub_sc_cart_coords)):
        if (sub_max_z - sub_sc_cart_coords[i, 2]) <= delta_z:
            target_sub_sc_coords.append([sub_sc_cart_coords[i, 0], sub_sc_cart_coords[i, 1], sub_sc_cart_coords[i, 2]])
            target_sp_sub_sc.append(str(sub_sc_struc_species[i]))

    for i in range(len(film_sc_cart_coords)):
        if (film_sc_cart_coords[i, 2] - film_min_z) <= delta_z:
            target_film_sc_coords.append([film_sc_cart_coords[i, 0], film_sc_cart_coords[i, 1], film_sc_cart_coords[i, 2]])
            target_sp_film_sc.append(str(film_sc_struc_species[i]))

    # print("TTT: ", len(target_sub_sc_coords), len(target_film_sc_coords))

    total_atom_counter = 0
    for i in range(len(sub_cart_coords)):
        if (sub_max_z - sub_cart_coords[i, 2]) <= 1:
            total_atoms_vol += sphe_vol(rad_dic[str(sub_struc_species[i])])
            total_atom_counter += 1
    for i in range(len(film_cart_coords)):
        if (film_cart_coords[i, 2] - film_min_z) <= 1:
            total_atoms_vol += sphe_vol(rad_dic[str(film_struc_species[i])])
            total_atom_counter += 1

    total_atoms_vol = total_atoms_vol / 2

    # print("Total counter:", total_atom_counter, "Total atoms:", len(sub_struc_species) + len(film_struc_species))


    target_sub_coords = np.array(target_sub_coords)
    target_film_coords = np.array(target_film_coords)
    target_sub_sc_coords = np.array(target_sub_sc_coords)
    target_film_sc_coords = np.array(target_film_sc_coords)


    min_x_val = min([p_d_sc[0], p_u_sc[0], p_l_sc[0], p_r_sc[0]])
    max_x_val = max([p_d_sc[0], p_u_sc[0], p_l_sc[0], p_r_sc[0]])
    min_y_val = min([p_d_sc[1], p_u_sc[1], p_l_sc[1], p_r_sc[1]])
    max_y_val = max([p_d_sc[1], p_u_sc[1], p_l_sc[1], p_r_sc[1]])

    min_z_val = np.min(target_sub_coords[:, 2])
    max_z_val = np.max(target_film_coords[:, 2])

    xi = np.arange(min_x_val, max_x_val + 0.01, scan_step)
    yi = np.arange(min_y_val, max_y_val + 0.01, scan_step)
    zi = np.arange(min_z_val, max_z_val + 0.01, scan_step)

    voxel_array_sub, voxel_array_film = np.zeros((len(xi), len(yi), len(zi))), np.zeros((len(xi), len(yi), len(zi)))
    voxel_array_ref_inside = np.zeros((len(xi), len(yi), len(zi)))
    voxel_array_ref_outside = np.zeros((len(xi), len(yi), len(zi)))

    t1 = time.time()

    ii_counter = 0
    for ii in xi:
        jj_counter = 0
        for jj in yi:
            if In_triangel(np.array([ii, jj]), p_l_c, p_r_c, p_u_c) or In_triangel(np.array([ii, jj]), p_l_c, p_r_c,
                                                                                   p_d_c):
                voxel_array_ref_inside[ii_counter, jj_counter, :] = 1
            else:
                voxel_array_ref_outside[ii_counter, jj_counter, :] = 1
            jj_counter += 1
        ii_counter += 1
    t2 = time.time()
    # print("Time ref: ", t2 - t1)

    t1_inter = time.time()
    percision = 30

    for i in range(len(target_sub_sc_coords)):
        rad = probe_rad + rad_dic[target_sp_sub_sc[i]]
        theta_array = np.linspace(0, (math.pi / 2), percision)
        angle_counter = 0
        org_ind_x = int((target_sub_sc_coords[i, 0] - min_x_val) / scan_step)
        org_ind_y = int((target_sub_sc_coords[i, 1] - min_y_val) / scan_step)
        org_ind_z = int((target_sub_sc_coords[i, 2] + rad - min_z_val) / scan_step)

        if (org_ind_x >= 1) and (org_ind_y >= 1):
        # if True:
            voxel_array_sub[org_ind_x - 1, org_ind_y - 1, :org_ind_z] = 1

        for theta in theta_array:
            phi_array = np.linspace(0, (math.pi * 2), angle_counter * 4)
            for phi in phi_array:
                ind_x = int((target_sub_sc_coords[i, 0] + rad * np.sin(theta) * np.cos(phi) - min_x_val)/scan_step)
                ind_y = int((target_sub_sc_coords[i, 1] + rad * np.sin(theta) * np.sin(phi) - min_y_val)/scan_step)
                ind_z = int((target_sub_sc_coords[i, 2] + rad * cos(theta) - min_z_val)/scan_step)
                if (ind_x >= 1 and ind_y >= 1):
                # if True:
                    try:
                        voxel_array_sub[ind_x -1, ind_y-1, :ind_z] = 1
                    except:
                        pass
            angle_counter += 1


    for i in range(len(target_film_sc_coords)):
        rad = probe_rad + rad_dic[target_sp_film_sc[i]]
        theta_array = np.linspace((math.pi / 2), math.pi  , percision)
        angle_counter = 0
        org_ind_x = int((target_film_sc_coords[i, 0] - min_x_val) / scan_step)
        org_ind_y = int((target_film_sc_coords[i, 1] - min_y_val) / scan_step)
        org_ind_z = int((target_film_sc_coords[i, 2] - rad - min_z_val) / scan_step)

        if (org_ind_x >= 1) and (org_ind_y >= 1):
        # if True:
            voxel_array_film[org_ind_x - 1, org_ind_y - 1, org_ind_z:] = 1

        for theta in theta_array:
            phi_array = np.linspace(0, (math.pi * 2), angle_counter * 4)
            for phi in phi_array:
                ind_x = int((target_film_sc_coords[i, 0] + rad * np.sin(theta) * np.cos(phi) - min_x_val)/scan_step)
                ind_y = int((target_film_sc_coords[i, 1] + rad * np.sin(theta) * np.sin(phi) - min_y_val)/scan_step)
                ind_z = int((target_film_sc_coords[i, 2] + rad * cos(theta) - min_z_val)/scan_step)
                if (ind_x >= 1) and (ind_y >= 1):
                # if True:
                    try:
                        voxel_array_film[ind_x -1, ind_y-1, ind_z:] = 1
                    except:
                        pass
            angle_counter += 1



    voxel_array_sub = np.logical_and(voxel_array_sub, voxel_array_ref_inside)
    t3 = time.time()
    print("Initialization time:", t3 - t1, " s")

    cell_height = (np.max(target_film_coords[:, 2]) - np.min(target_sub_coords[:, 2]))
    voxel_volume = scan_step ** 3

    # film_struc_ref_coords = film_cart_coords.copy()
    film_struc_ref_coords = film_struc.cart_coords
    film_struc_ref_species = film_struc.species
    sub_struc_ref_coords = sub_struc.cart_coords
    sub_struc_ref_species = sub_struc.species
    film_struc_ref_lat = film_struc.lattice.matrix

    def shift_film_voxel(x_shift, y_shift, z_shift):

        film_new_coords = film_struc_ref_coords.copy()
        film_new_coords = np.array(film_new_coords)
        film_new_coords[:, 0] += x_shift
        film_new_coords[:, 1] += y_shift
        film_new_coords[:, 2] += z_shift
        film_new_struc = Structure(film_struc_ref_lat, film_struc_ref_species, film_new_coords, coords_are_cartesian=
        True)


        int_sps = sub_struc_ref_species + film_struc_ref_species
        int_coords = np.concatenate((sub_struc_ref_coords, film_new_coords))
        int_coords[:,2] += 20
        interface_struc = Structure(film_struc_ref_lat, int_sps, int_coords, coords_are_cartesian=True)

        Poscar(interface_struc).write_file("grid_search_output/POSCAR_Iface_" + "_x_" + str(int(x_shift *
                                                                                                          10)) +
                                           "_y_" + str(int(y_shift * 10)) + "_z_" + str(int(z_shift * 10)),
                                           direct=False)

        # film_tensor = np.copy(voxel_array_sub)
        film_tensor = np.copy(voxel_array_film)
        x_shift =  x_shift
        y_shift =  y_shift

        x_shift_step = int(x_shift / scan_step)
        y_shift_step = int(y_shift / scan_step)


        if x_shift > 0:
            x_shift_step = int(x_shift / scan_step)
            film_tensor = np.roll(film_tensor, shift=x_shift_step, axis=0)
        elif x_shift < 0:
            x_shift_step = int(x_shift / scan_step)
            film_tensor = np.roll(film_tensor, shift=x_shift_step, axis=0)
        #
        if y_shift > 0:
            y_shift_step = int(y_shift / scan_step)
            film_tensor = np.roll(film_tensor, shift=y_shift_step, axis=1)
        elif y_shift < 0:
            y_shift_step = int(y_shift / scan_step)
            film_tensor = np.roll(film_tensor, shift=y_shift_step, axis=1)

        film_tensor = np.logical_and(film_tensor, voxel_array_ref_inside)
        z_shift_step = int(z_shift / scan_step)
        film_tensor = np.roll(film_tensor, shift=z_shift_step, axis=2)

        if z_shift_step > 0:
            film_tensor[:, :, 0: z_shift_step] = 0

        if z_shift_step < 0:
            film_tensor[:, :, film_tensor.shape[2] + z_shift_step : film_tensor.shape[2] ] = 1
            film_tensor = np.logical_and(film_tensor, voxel_array_ref_inside)

        return film_tensor

    cell_volume = lat_area * (cell_height - 2)

    if visualize:

        xlist, ylist, zlist, slist = [], [], [], []
        shift_x = 0
        shift_y = 0
        shift_z = 0
        t00 = time.time()
        shifted_film_voxel = shift_film_voxel(shift_x, shift_y, shift_z)
        t11 = time.time()
        print("Shift time:", t11 - t00)

        voxel_dim = voxel_array_sub.shape
        or_array = np.logical_or(voxel_array_sub, shifted_film_voxel)
        and_array = np.logical_and(voxel_array_sub, shifted_film_voxel)
        nor_array = np.logical_not(or_array)

        nor_array = np.logical_and(nor_array, voxel_array_ref_inside)
        interface_tensor = np.logical_not(np.logical_or(voxel_array_sub, shifted_film_voxel))
        new_dims = voxel_array_sub.shape
        voxel_array_inside_not = np.logical_not(voxel_array_ref_inside)

        for i in range(len(voxel_array_sub[:, 0, 0])):
            for j in range(len(voxel_array_sub[0, :, 0])):
                # if np.count_nonzero(or_array[i, j, :]) > 0 :
                for k in range(len(voxel_array_sub[0, 0, :])):
                    if shifted_film_voxel[i, j, k]:
                        xlist.append(i)
                        ylist.append(j)
                        zlist.append(k)

        fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        ax = fig.gca(projection='3d')
        ax.view_init(-45, -45)
        ax.set_xlabel("X index" , labelpad= 16, fontsize = 24)
        ax.set_ylabel("Y index", labelpad= 16, fontsize = 24)
        # ax.set_zlabel("Z index", labelpad= 8, fontsize = 24)
        tite_str = "dX: " + str(shift_x) + "  dY: " + str(shift_y)
        # plt.title(tite_str)
        ax.scatter(xlist, ylist, zlist, c=zlist, cmap="jet")
        cax = plt.gca().figure.axes[-1]
        cax.tick_params(labelsize=24)
        plt.show()

    cell_volume = np.sum(voxel_array_ref_inside)* voxel_volume
    def score_calculator(x_shift, y_shift, z_shift, coeff = 1):

        sub_tensor = np.copy(voxel_array_sub)
        film_tensor = shift_film_voxel(x_shift, y_shift, z_shift)

        tensor_dim = sub_tensor.shape

        interface_tensor = np.logical_or(sub_tensor, film_tensor)
        interface_tensor = np.logical_and(interface_tensor, voxel_array_ref_inside)

        overlap_tensor = np.logical_and(sub_tensor, film_tensor)

        overlap_count = np.sum(overlap_tensor)
        empt_count = np.sum(interface_tensor)
        overlap_space = overlap_count * voxel_volume
        array_dimension = sub_tensor.shape
        total_count = array_dimension[0] * array_dimension[1] * array_dimension[2]
        empt_space = cell_volume - np.sum(interface_tensor) * voxel_volume

        overlap_space_rel = overlap_space / total_atoms_vol
        empty_space_rel = empt_space / cell_volume

        score =  (1 + overlap_space_rel) ** 2 + coeff * empty_space_rel
        voxel_ref_dims = voxel_array_ref_inside.shape
        vol_ref_count = voxel_ref_dims[0] * voxel_ref_dims[1] * voxel_ref_dims[2]

        return [-score,overlap_space]




    if surface_matching:


        pbounds = {'x_shift': (0, len_vec_a), 'y_shift': (0, len_vec_b), 'z_shift': (-3, 0)}
        optimizer = BayesianOptimization(f=score_calculator, pbounds=pbounds, verbose=2)
        optimizer.maximize(kappa= 8, n_iter= BO_iterations, random_state=1)

        coord_diffs, score_targets = [], []
        x_diffs, y_diffs, z_diffs = [], [], []

        for bo_index in range(len(optimizer.res)):
            x_diffs.append(optimizer.res[bo_index]['params']['x_shift'])
            y_diffs.append(optimizer.res[bo_index]['params']['y_shift'])
            z_diffs.append(optimizer.res[bo_index]['params']['z_shift'])
            coord_diffs.append([optimizer.res[bo_index]['params']['x_shift'], optimizer.res[bo_index][
                'params']['y_shift'], optimizer.res[bo_index]['params']['z_shift']])
            score_targets.append(-optimizer.res[bo_index]['target'])
        # Best 3D

        targets = np.round(np.array(score_targets), 5)
        sorted_targets = sorted(targets, reverse=False)
        print(sorted_targets)
        lowest_indices = []
        for score_value in sorted_targets[0:20]:
            lowest_indices.append(np.where(targets == score_value))

        lowest_score_coords = []
        for score_index in lowest_indices:
            lowest_score_coords.append(
                [np.round(x_diffs[score_index[0][0]], 5), np.round(y_diffs[score_index[0][0]], 5),
                 np.round(z_diffs[score_index[0][0]], 5)])
            print("x:", np.round(x_diffs[score_index[0][0]], 3), "  y:", np.round(y_diffs[score_index[0][0]],
                                                                                  3), "  z:",
                  np.round(z_diffs[score_index[0][0]], 3), " target:", np.round(score_targets[score_index[0][0]], 5))



    if grid_search:
        print("Grid search score contour started")
        x_grid_range = np.linspace(0, 3, 11)
        y_grid_range = np.linspace(0, 3, 11)
        z_grid_range = np.linspace( z_shift, z_shift, 1)
        score_array = np.zeros((len(x_grid_range), len(y_grid_range)))

        x_diffs, y_diffs, z_diffs, score_targets = [], [], [], []

        film_shifted_coords = target_film_coords

        # 2D
        scptDir = os.getcwd()
        ext = "grid_search_output"
        ext_Dir = os.path.join(scptDir, ext)
        os.mkdir(ext_Dir)

        for ii in range(len(x_grid_range)):
            for jj in range(len(y_grid_range)):
                for kk in range(len(z_grid_range)):
                    print("x shift:", np.round(x_grid_range[ii], 2)  , "y shift:", np.round(y_grid_range[jj],2) ,
                                                                                          "z shift:" ,
                                                 np.round(z_grid_range[kk], 2) )
                    score_array[ii, jj] = -score_calculator(x_grid_range[ii], y_grid_range[jj], z_grid_range[
                        kk], coeff=1)[0]

        xxi, yyi = np.meshgrid(x_grid_range, y_grid_range)
        # plt.figure()
        fig = plt.figure(figsize=(13.5, 10))
        ax1 = fig.subplots()
        for label in (ax1.get_xticklabels() + ax1.get_yticklabels()):
            label.set_fontsize(48)
        # plt.plot(x_grid_range, score_array[:,0,0])
        score_array = np.transpose(score_array)
        contour = plt.contourf(xxi, yyi, score_array, 200, cmap='jet')
        cbar = fig.colorbar(contour, orientation = 'vertical')
        # plt.xticks([0, 0.5, 1, 1.5, 2])
        # plt.yticks([0, 0.5, 1, 1.5, 2])
        cax = plt.gca().figure.axes[-1]
        cax.yaxis.label.set_size(20)
        cax.yaxis.labelpad = 20

        np.savetxt(working_dir + '\\scores.txt', score_array)

        cax.tick_params(labelsize=40)
        ax1.set_xlabel("Shift in X direction (Å)", fontsize='48')
        ax1.set_ylabel("Shift in Y direction (Å)", fontsize='48')
        # plt.title("Config 5", fontsize = 20)
        plt.savefig(working_dir + "contour.png")
        plt.show()

    if potential_curve:
        x_shift = 0
        y_shift = 2.1419325
        z_grid_range = np.linspace(-2, 4, 31)
        label_x_range = np.linspace(0, 6, 31)
        score_array = np.zeros(len(z_grid_range))

        line_width_range = np.linspace(-2, 0, 11)

        color = 'tab:red'
        counter = 0

        N = 4
        coeff_range = np.linspace(0, 1, N)

        HSV_tuples = [(1 - x * 1.0 / N, 0, x * 1.0 / N) for x in range(N)]

        fig = plt.figure(figsize=(10, 7))
        fig = plt.figure(figsize=(13, 11))
        ax1 = fig.subplots()

        for label in (ax1.get_xticklabels() + ax1.get_yticklabels()):
            label.set_fontsize(48)
        col_ind = 0

        for coeff in coeff_range:

            for kk in range(len(z_grid_range)):

                score_res = score_calculator(x_shift, y_shift, z_grid_range[kk], coeff)
                score = -score_res[0]
                if score_res[1] > 0.01:
                    score_array[kk] = score
                else:
                    score_array[kk] = score_array[kk - 1]

            print(coeff, "Percentage: ", (score_array[-1] - min(score_array)) / (score_array[0] - min(score_array)) *
                  100)
            print(score_array)

            ax1.plot(label_x_range, score_array, color="black", linestyle='dashed', linewidth='1', label="DFT")

            ax1.set_xlabel('Interface distance (Å)', fontsize=48)
            ax1.set_ylabel('Score', fontsize=48)

            counter += 1
            ax2 = ax1.twinx()
            for label in (ax2.get_xticklabels() + ax2.get_yticklabels()):
                label.set_fontsize(48)
            ax2.plot(label_x_range, score_array, color=HSV_tuples[col_ind], label=str(coeff))
            col_ind += 1

        col_ind = 0
        # col_ind = 2
        for i in coeff_range:
            ax2.plot([], [], color=HSV_tuples[col_ind], label="C: " + str(np.round(i, 1)))
            col_ind += 1
        ax2.legend(prop={'size': 15})
        ax2.set_xlabel('Slabs distance', fontsize=16)

        # plt.rcParams['xtick.labelsize'] = 26
        plt.show()



    return 1

interface_voxel(sub_struc, film_struc,probe_rad= 0,  surface_matching= False, visualize = False , grid_search =True,
                scan_step= 0.1, potential_curve = False)
