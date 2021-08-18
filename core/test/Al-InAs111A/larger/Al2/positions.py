from pymatgen.core.structure import Structure
import numpy as np

old_structure = Structure.from_file('./CONTCAR')
new_structure = Structure.from_file('./POSCAR_Al2')

old_c = np.linalg.norm(old_structure.lattice.matrix[-1])
new_c = np.linalg.norm(new_structure.lattice.matrix[-1])

In_ref_small = 0.4349640000000008
In_ref_large = 0.391834

As_ref_small = 0.4287530000000004
As_ref_large = 0.387753

Al_ref_small = 0.5934460000000001
Al_ref_large = 0.495980

In_ref_small *= old_c
In_ref_large *= new_c

As_ref_small *= old_c
As_ref_large *= new_c

Al_ref_small *= old_c
Al_ref_large *= new_c


In_pos_small = old_c * np.array([
    0.4605383679369138,
    0.4605010650641386,
    0.4605010650641386,
    0.4605010650641386,
])

As_pos_small = old_c * np.array([
    0.5259334340475259,
    0.5251232782556856,
    0.5251232782556856,
    0.5251232782556856,
    0.5025882885514399,
    0.5026191713420448,
    0.5026191713420448,
    0.5026191713420448,
    0.4798299995955230,
    0.4797709936432461,
    0.4797709936432461,
    0.4797709936432461,
    0.4538777546539497,
    0.4538945822147976,
    0.4538945822147976,
    0.4538945822147976,
])

Al_pos_small = old_c * np.array([
    0.5577264831294584,
    0.5577264831294584,
    0.5558898210378654,
    0.5577264831294584,
    0.5586859657398328,
    0.5586859657398328,
    0.5558898210378654,
    0.5586859657398328,
    0.5558898210378654,
    0.5389154769100175,
    0.5389154769100175,
    0.5430510934080448,
    0.5285036851785456,
    0.5389154769100175,
    0.5389154769100175,
    0.5389154769100175,
    0.5424083651523401,
    0.5389154769100175,
    0.5084579479683802,
    0.5075187621454156,
    0.5075187621454156,
    0.5075187621454156,
    0.4849049811073587,
    0.4849550162842302,
    0.4849550162842302,
    0.4849550162842302,
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

