from pymatgen.core.structure import Structure
import numpy as np

old_structure = Structure.from_file('../CONTCAR')
new_structure = Structure.from_file('./POSCAR_0')

old_c = np.linalg.norm(old_structure.lattice.matrix[-1])
new_c = np.linalg.norm(new_structure.lattice.matrix[-1])

In_ref_small = 0.4598089999999999
In_ref_large = 0.408161

As_ref_small = 0.4535970000000020
As_ref_large = 0.404079

Al_ref_small = 0.5934460000000001
Al_ref_large = 0.495980

In_ref_small *= old_c
In_ref_large *= new_c

As_ref_small *= old_c
As_ref_large *= new_c

Al_ref_small *= old_c
Al_ref_large *= new_c


In_pos_small = old_c * np.array([
    0.5097555972231007,
    0.5087179027317783,
    0.5087179027317783,
    0.5087179027317783,
    0.4847665286214843,
    0.4842442019567196,
    0.4842442019567196,
    0.4842442019567196,
])

As_pos_small = old_c * np.array([
    0.5284978073640235, 
    0.5274496020765578,
    0.5274496020765578, 
    0.5274496020765578, 
    0.5034100310282323, 
    0.5028075789007257, 
    0.5028075789007257, 
    0.5028075789007257, 
    0.4784066418328549, 
    0.4783536094545932, 
    0.4783536094545932, 
    0.4783536094545932, 
])

Al_pos_small = old_c * np.array([
    0.5764069939743725,
    0.5760421330003925,
    0.5764069939743725,
    0.5760421330003925,
    0.5760421330003925,
    0.5751155041625736,
    0.5764069939743725,
    0.5751155041625736,
    0.5751155041625736,
    0.5584480010454462,
    0.5584480010454462,
    0.5559628319248601,
    0.5584480010454462,
    0.5600122328185387,
    0.5600122328185387,
    0.5559628319248601,
    0.5600122328185387,
    0.5559628319248601,
    0.5404170461681593,
    0.5404170461681593,
    0.5455756105127594,
    0.5343684285848682,
    0.5404170461681593,
    0.5404170461681593,
    0.5404170461681593,
    0.5430029608201012,
    0.5404170461681593,
])

Al_pos_small_rel = (Al_pos_small - Al_ref_small) 
Al_pos_large_new = (Al_ref_large + Al_pos_small_rel) / new_c

print('\nAl Positions:')
[print(np.round(i,16)) for i in Al_pos_large_new]

In_pos_small_rel = (In_pos_small - In_ref_small) 
In_pos_large_new = (In_ref_large + In_pos_small_rel) / new_c

print('\nIn Positions:')
[print(np.round(i,16)) for i in In_pos_large_new]

As_pos_small_rel = (As_pos_small - As_ref_small) 
As_pos_large_new = (As_ref_large + As_pos_small_rel) / new_c

print('\nAs Positions:')
[print(np.round(i,16)) for i in As_pos_large_new]

