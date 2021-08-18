from pymatgen.core.structure import Structure
import numpy as np

old_structure = Structure.from_file('./CONTCAR')
new_structure = Structure.from_file('./POSCAR_Al1')

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
    0.4852645841400617,
    0.4850892905538094,
    0.4850892905538094,
    0.4850892905538094,
])

As_pos_small = old_c * np.array([
    0.5271122899238264, 
    0.5263471515458250,
    0.5263471515458250, 
    0.5263471515458250, 
    0.5044700443574486, 
    0.5042113602530042, 
    0.5042113602530042, 
    0.5042113602530042, 
    0.4787189735718023, 
    0.4786667983754435, 
    0.4786667983754435, 
    0.4786667983754435, 
])

Al_pos_small = old_c * np.array([
    0.5761039117173603,
    0.5757250205855095,
    0.5761039117173603,
    0.5757250205855095,
    0.5757250205855095,
    0.5750765264217126,
    0.5761039117173603,
    0.5750765264217126,
    0.5750765264217126,
    0.5582061477176291,
    0.5582061477176291,
    0.5557463392192731,
    0.5582061477176291,
    0.5592323020204951,
    0.5592323020204951,
    0.5557463392192731,
    0.5592323020204951,
    0.5557463392192731,
    0.5397061039625655,
    0.5397061039625655,
    0.5441163581666700,
    0.5321535736511727,
    0.5397061039625655,
    0.5397061039625655,
    0.5397061039625655,
    0.5429725252660864,
    0.5397061039625655,
    0.5097138292198224,
    0.5089217107028088,
    0.5089217107028088,
    0.5089217107028088,
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

